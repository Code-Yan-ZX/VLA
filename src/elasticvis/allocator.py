"""ElasticVis admission-time per-request visual-token budget allocator (§1, §5).

§1 formalization:
  Decision  k_i in [k_min, k_max] (LLaVA-1.5 discrete grid {144,288,576}).
  Objective goodput@SLO = Sum 1{meets_SLO} * accuracy(k_i).
  Gate      LatPred(own_k=k_i, sum_k, num_running, kv_occupancy) <= slo_ms.
  Solvers   Greedy (threshold), Lagrangian (shadow price), Oracle (upper bound),
            Fixed-r (degenerate constant baseline).

Policy-only: CONSUMES `LatencyPred` / `AccuracyTerm` (built in `predictors.py`)
and `LoadReading` (`load_controller.py`) via duck typing. No import of torch /
vLLM / predictors.py — callers inject `lat` and `acc` (annotations are strings).
"""

from __future__ import annotations
from typing import Optional, Sequence

# LLaVA-1.5: only these k have measured accuracy in v2 probes (§2). Other arches
# (Qwen3-VL) pass a different `k_grid` to the constructor.
DEFAULT_K_GRID: tuple[int, ...] = (144, 288, 576)
DEFAULT_N_NATIVE: int = 576  # LLaVA-1.5 native visual-token count (fixed-r basis)


def _lat_value(est, slo_type: str, percentile: str) -> float:
    """Pull gated latency attr, e.g. ('ttft','p99') -> est.ttft_p99."""
    attr = f"{slo_type}_{percentile}"  # ttft_p50 | ttft_p99 | e2e_p50 | e2e_p99
    if not hasattr(est, attr):
        raise ValueError(f"LatencyEstimate has no {attr!r}")
    return float(getattr(est, attr))


def _candidates(k_range: tuple[int, int], k_grid: Optional[Sequence[int]]) -> list[int]:
    k_min, k_max = k_range
    grid = k_grid if k_grid is not None else DEFAULT_K_GRID
    return sorted(int(k) for k in grid if k_min <= int(k) <= k_max)


def _req_id(req_features: Optional[dict]):
    if not req_features:
        return None
    return req_features.get("req_id") or req_features.get("id")


class Allocator:
    """Base. Subclasses implement `allocate` (shared §5 signature + `sum_k` +
    `percentile`). `sum_k` = batch's current total visual tokens (caller/sim
    computes it). `percentile` ∈ {"p99","p50"} selects the gate attribute."""

    def allocate(self, load, slo_ms: float, slo_type: str,
                 req_features: Optional[dict], k_range: tuple[int, int],
                 lat, acc, sum_k: int, percentile: str = "p99") -> int:
        raise NotImplementedError

    # ---- public gate helper (§5) -------------------------------------------
    def would_meet_slo(self, k: int, sum_k: int, num_running: int,
                       slo_ms: float, slo_type: str, lat,
                       percentile: str = "p99", kv_occupancy: float = 0.0) -> bool:
        """Gate check: LatPred(own_k=k, ...) <= slo_ms on chosen percentile.

        kv_occupancy is keyword-only with 0.0 default — §5's signature omits it
        but `LatencyPred.predict` requires it; pass `load.kv_occupancy` here.
        """
        est = lat.predict(own_k=int(k), sum_k=int(sum_k),
                          num_running=int(num_running) if num_running is not None else 0,
                          kv_occupancy=float(kv_occupancy) if kv_occupancy is not None else 0.0)
        return _lat_value(est, slo_type, percentile) <= float(slo_ms)

    @staticmethod
    def _gate_ok(k, load, sum_k, slo_ms, slo_type, lat, percentile) -> bool:
        """Internal gate using a `LoadReading` for num_running/kv_occupancy."""
        nr = getattr(load, "num_running", None) if load is not None else None
        kv = getattr(load, "kv_occupancy", None) if load is not None else None
        est = lat.predict(own_k=int(k), sum_k=int(sum_k),
                          num_running=int(nr) if nr is not None else 0,
                          kv_occupancy=float(kv) if kv is not None else 0.0)
        return _lat_value(est, slo_type, percentile) <= float(slo_ms)


class FixedAllocator(Allocator):
    """Degenerate constant baseline (fixed-{r0,r25,r50,r75}).

    Returns round(N*(1-rate)) clamped to k_range. rate=0 -> N (max tokens),
    rate=0.75 -> k=144 with N=576."""

    def __init__(self, rate: float, N: int = DEFAULT_N_NATIVE):
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"rate must be in [0,1], got {rate}")
        self.rate, self.N = float(rate), int(N)

    def allocate(self, load, slo_ms, slo_type, req_features, k_range, lat, acc,
                 sum_k, percentile="p99") -> int:
        k_min, k_max = k_range
        return int(max(k_min, min(k_max, round(self.N * (1.0 - self.rate)))))


class GreedyAllocator(Allocator):
    """Threshold baseline (§1 solver 1): default k_min; raise toward k_max on
    the discrete grid WHILE (a) gate holds at the higher k AND (b) marginal
    accuracy gain is strictly positive. Return the highest feasible k."""

    def __init__(self, k_grid: Optional[Sequence[int]] = None):
        self.k_grid = tuple(k_grid) if k_grid is not None else DEFAULT_K_GRID

    def allocate(self, load, slo_ms, slo_type, req_features, k_range, lat, acc,
                 sum_k, percentile="p99") -> int:
        cands = _candidates(k_range, self.k_grid)
        if not cands:
            return int(k_range[0])
        rid = _req_id(req_features)
        k = cands[0]  # start at k_min
        for nk in cands[1:]:
            if acc.utility(nk, rid, req_features) - acc.utility(k, rid, req_features) <= 0:
                break  # (b) no marginal accuracy gain
            if not self._gate_ok(nk, load, sum_k, slo_ms, slo_type, lat, percentile):
                break  # (a) gate fails at the higher k
            k = nk
        return int(k)


class LagrangianAllocator(Allocator):
    """Shadow-price policy (§1 solver 2). Price `lam` on SLO violations,
    updated externally from realized violation rate.

        k_i* = argmax_{k in grid} [ acc.utility(k) - lam * violation_cost(k) ]
        violation_cost(k) = max(0, lat_{slo_type,pct}(k) - slo_ms)   (ms over budget)
        update_lambda(vr): lam <- lam * (1 + eta * (vr - target))    (floored at 0)
    """

    def __init__(self, eta: float = 0.1, target_violation: float = 0.0,
                 lam: float = 0.0, k_grid: Optional[Sequence[int]] = None):
        self.eta = float(eta)
        self.target_violation = float(target_violation)
        self.lam = float(lam)
        self.k_grid = tuple(k_grid) if k_grid is not None else DEFAULT_K_GRID

    def update_lambda(self, observed_violation_rate: float) -> float:
        """Multiplicative update on the SLO-violation shadow price."""
        self.lam = max(0.0, self.lam * (1.0 + self.eta *
                         (float(observed_violation_rate) - self.target_violation)))
        return self.lam

    def allocate(self, load, slo_ms, slo_type, req_features, k_range, lat, acc,
                 sum_k, percentile="p99") -> int:
        cands = _candidates(k_range, self.k_grid)
        if not cands:
            return int(k_range[0])
        rid = _req_id(req_features)
        nr = int(getattr(load, "num_running", 0) or 0)
        kv = float(getattr(load, "kv_occupancy", 0.0) or 0.0)
        best_k, best_score = cands[0], float("-inf")
        for k in cands:  # ascending; strict > keeps lower k on tie
            est = lat.predict(own_k=k, sum_k=int(sum_k), num_running=nr, kv_occupancy=kv)
            cost = max(0.0, _lat_value(est, slo_type, percentile) - float(slo_ms))
            score = acc.utility(k, rid, req_features) - self.lam * cost
            if score > best_score:
                best_score, best_k = score, k
        return int(best_k)


class OracleAllocator(Allocator):
    """Offline UPPER BOUND. Uses ground-truth per-image accuracy(k) (via
    `acc.per_image_curve(req_id)` when available) under the SAME gate as
    Greedy; picks max-accuracy gate-feasible k. Requires req_id in
    req_features; falls back to aggregate `acc.utility(k, rid)` if absent."""

    def __init__(self, k_grid: Optional[Sequence[int]] = None):
        self.k_grid = tuple(k_grid) if k_grid is not None else DEFAULT_K_GRID
        self._warned = False

    def allocate(self, load, slo_ms, slo_type, req_features, k_range, lat, acc,
                 sum_k, percentile="p99") -> int:
        cands = _candidates(k_range, self.k_grid)
        if not cands:
            return int(k_range[0])
        rid = _req_id(req_features)
        curve = None
        if rid is not None and hasattr(acc, "per_image_curve"):
            try:
                curve = acc.per_image_curve(rid)  # dict[int,float]
            except Exception:
                curve = None
        if curve is None:
            if rid is None and not self._warned:
                import sys
                print("OracleAllocator: no req_id -> aggregate accuracy curve", file=sys.stderr)
                self._warned = True
            curve = {k: acc.utility(k, rid, req_features) for k in cands}
        best_k, best_u = cands[0], float("-inf")
        for k in cands:  # highest accuracy among gate-feasible k
            if not self._gate_ok(k, load, sum_k, slo_ms, slo_type, lat, percentile):
                continue
            u = float(curve.get(k, curve.get(str(k), 0.0)))
            if u > best_u:
                best_u, best_k = u, k
        return int(best_k)  # if even k_min violates, returns lowest candidate


# ---------------------------------------------------------------------------
# Smoke test: inline mocks (no predictors.py / torch / vLLM needed).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class _LatEst:
        ttft_p50: float; ttft_p99: float; e2e_p50: float; e2e_p99: float

    class _FakeLat:
        def predict(self, own_k, sum_k, num_running, kv_occupancy):
            nr = num_running or 0
            t99 = 50.0 + 0.5 * own_k + 0.03 * sum_k + 10.0 * nr
            e99 = 300.0 + 1.2 * own_k + 0.05 * sum_k + 40.0 * nr
            return _LatEst(0.6 * t99, t99, 0.6 * e99, e99)

    class _FakeAcc:
        def utility(self, k, req_id=None, features=None):
            return 0.5 + 0.4 * (k / 576.0)
        def per_image_curve(self, req_id):
            return {144: 0.60, 288: 0.78, 576: 0.90}

    @dataclass
    class _Load:
        num_running: int = 1
        kv_occupancy: float = 0.0
        max_num_seqs: int = 64

    lat, acc, k_range = _FakeLat(), _FakeAcc(), (144, 576)
    # Three regimes (ttft_p99 gate): slack->576 feasible, medium->288, tight->floor.
    cases = [("slack", _Load(num_running=1), 2000.0, 576),
             ("medium", _Load(num_running=4), 400.0, 288),
             ("tight", _Load(num_running=8), 150.0, 144)]
    sum_k_for = {1: 576, 4: 4 * 288, 8: 8 * 288}
    fixed, greedy = FixedAllocator(0.5), GreedyAllocator()
    lagr, oracle = LagrangianAllocator(eta=0.5, lam=0.5), OracleAllocator()
    print(f"{'case':>6}  sum_k  slo   fixed  greedy  lagr  oracle")
    for name, load, slo, expect in cases:
        sk = sum_k_for[load.num_running]
        feat = {"req_id": "gqa_0001"}
        kf = fixed.allocate(load, slo, "ttft", feat, k_range, lat, acc, sk)
        kg = greedy.allocate(load, slo, "ttft", feat, k_range, lat, acc, sk)
        kl = lagr.allocate(load, slo, "ttft", feat, k_range, lat, acc, sk)
        ko = oracle.allocate(load, slo, "ttft", feat, k_range, lat, acc, sk)
        print(f"{name:>6}  {sk:>5}  {slo:>5.0f}  {kf:>5}  {kg:>6}  {kl:>4}  {ko:>6}   (expect {expect})")

    print("\nlambda update (eta=0.5, target=0.2, fresh lam=1.0 each):")
    print("  vr=0.3 ->", round(LagrangianAllocator(0.5, 0.2, 1.0).update_lambda(0.3), 3), "  (rises, above target)")
    print("  vr=0.1 ->", round(LagrangianAllocator(0.5, 0.2, 1.0).update_lambda(0.1), 3), "  (falls, below target)")
