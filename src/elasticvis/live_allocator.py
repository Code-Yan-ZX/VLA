"""ElasticVis live-engine per-request k allocator (EV-1).

Admission-time per-request visual-token budget k_i driven by LIVE load signals
(``read_engine_load_v1`` -> num_running/kv_occupancy via ``llm.get_metrics()``)
and a per-request SLO deadline. This is the live-engine counterpart of the
offline ``allocator.py`` gate — the same queue-aware greedy rule, adapted to
the live signal path.

Gate (queue-aware greedy, spec §1 solver 1; §8 winner on TextVQA H1b +35.5%):

    expected_wait(load, sum_k) + P(k) [+ S(k)] <= slo_s

where ``expected_wait`` and ``P(k)``/``S(k)`` are the server-model functions
from ``sim.py`` (measured on v2 c64 LLaVA-1.5-7B: S(k)=M/served_req_s,
P(k)=ttft_min). Returns the highest feasible k on the discrete grid.

The allocator is called once per request at ``add_request`` time (before
``preprocess_chat`` so the placeholder count matches the projector output).
The gate is a HEURISTIC adapted from the offline sim — absolute latency
predictions have known fidelity gaps (§2.1 closed-loop confound), but the
RELATIVE ordering (low-k protects SLO under load, high-k preserves accuracy
in light spells) is robust and is what drives the goodput win.

Standalone-importable (no torch/vLLM). ``LoadReading`` consumed by duck typing.
"""
from __future__ import annotations
from typing import Optional, Sequence

# Server-model constants (from sim.py, measured on v2 c64 LLaVA-1.5-7B).
# S(k) = M / served_req_s(c64, k) = slot-occupancy seconds per request.
# P(k) = ttft_min(c64, k) = prefill floor (batched-compute, weak k-dep).
_K_GRID = (144, 288, 576)
_S_GRID = (3.14, 4.39, 6.97)
_P_GRID = (2.86, 2.81, 2.95)
DEFAULT_N_NATIVE = 576


def _interp(k: float, gk: tuple, gv: tuple) -> float:
    if k <= gk[0]:
        return float(gv[0])
    if k >= gk[-1]:
        return float(gv[-1])
    for i in range(len(gk) - 1):
        if gk[i] <= k <= gk[i + 1]:
            t = (k - gk[i]) / (gk[i + 1] - gk[i])
            return gv[i] * (1 - t) + gv[i + 1] * t
    return float(gv[-1])


# ---- Calibrated load-dependent latency model (EV-1d, LLaVA-1.5-7b, A40) ----
# Fit from open-loop GPU measurements (runs/ev1d_*):
#   compute(k) = COMPUTE_BASE + COMPUTE_SLOPE * k   (per-request e2e at zero load)
#   wait(nr, mns) = WAIT_QF * (nr / mns) ^ WAIT_QE  (load-dependent queue delay)
#   e2e_est(k, nr) = compute(k) + wait(nr)
# Calibration data: rate=8 (nr~10): e2e(576)=1315ms, e2e(144)=823ms → slope=1.14
#   rate=15 (nr~64): e2e(576)=8622ms → wait=7565ms at lf=1.0
#   wait at nr=10: 1315-compute(576)=1315-1057=258ms → QF*0.156^QE=258, QF*1.0^QE=7565
#   → QF=7565, QE=1.82
COMPUTE_BASE_MS = 400.0
COMPUTE_SLOPE_MS = 1.14      # ms per visual token
WAIT_QF_MS = 7565.0          # queue wait coefficient at saturation
WAIT_QE = 1.82               # queue wait exponent (nonlinear near capacity)


def compute_ms(k: int) -> float:
    """Estimated single-request compute (ms): prefill + decode, roughly linear
    in visual-token count k."""
    return COMPUTE_BASE_MS + COMPUTE_SLOPE_MS * int(k)


def wait_ms(num_running: int, max_num_seqs: int = 64) -> float:
    """Estimated queue/sojourn delay (ms) from live load. Grows nonlinearly
    with the load factor (num_running / max_num_seqs), calibrated from EV-1d
    open-loop measurements."""
    mns = max(int(max_num_seqs), 1)
    nr = max(int(num_running or 0), 0)
    lf = min(nr / mns, 1.5)  # allow slight overshoot for saturated regime
    return WAIT_QF_MS * lf ** WAIT_QE


def e2e_est_ms(k: int, num_running: int, max_num_seqs: int = 64) -> float:
    """Load-dependent e2e latency estimate (ms)."""
    return compute_ms(k) + wait_ms(num_running, max_num_seqs)


# Legacy functions retained for compatibility (sim.py server model)
def service_time(k: int) -> float:
    return _interp(k, _K_GRID, _S_GRID)

def prefill_time(k: int) -> float:
    return _interp(k, _K_GRID, _P_GRID)

def expected_wait_s(load, sum_k: int) -> float:
    mns = getattr(load, "max_num_seqs", None) or 64
    nr = getattr(load, "num_running", None) or 0
    return wait_ms(nr, mns) / 1000.0


class LiveGreedyAllocator:
    """Adaptive per-request k allocator with live EMA latency tracking.

    HYBRID approach (EV-1e): uses a calibrated compute(k) model for the
    k-dependent component + a LIVE EMA of recent completions' e2e for the
    load-dependent queue/wait component. The EMA adapts to ANY load level
    without requiring fixed calibration constants.

    Gate: give the HIGHEST k on the grid whose estimated e2e <= slo_ms.

        e2e_est(k) = compute(k) + live_wait
        live_wait = ema_e2e - compute(ema_k)    (EMA updated on each completion)
        compute(k) = COMPUTE_BASE + COMPUTE_SLOPE * k   (calibrated from GPU)

    Cold start (no EMA yet): falls back to the calibrated wait_ms model.

    Behavior:
    - Light load: live_wait ≈ 0 → e2e ≈ compute(k) → k=576 meets most SLOs
      → full accuracy (matches r0, FIXES the EV-1d rate=8 loss).
    - Moderate load: live_wait rises → tight-SLO requests drop to k=288
      (matches r50's sweet spot) while slack-SLO keep k=576 (higher acc).
    - Heavy load: live_wait large → tight-SLO drop to k=144, slack to k=288.
    """

    def __init__(self, k_grid: Optional[Sequence[int]] = None,
                 k_min: int = 144, k_max: int = 576,
                 slo_type: str = "e2e",
                 acc_curve: Optional[dict[int, float]] = None,
                 ema_alpha: float = 0.3):
        self.k_grid = tuple(sorted(k_grid)) if k_grid is not None else _K_GRID
        self.k_min = int(k_min)
        self.k_max = int(k_max)
        self.slo_type = str(slo_type)
        self._acc = acc_curve or {144: 0.290, 288: 0.450, 576: 0.575}
        self._ema_alpha = float(ema_alpha)
        self._ema_e2e: Optional[float] = None    # EMA of recent e2e (ms)
        self._ema_k: float = 288.0                 # EMA of recent k values
        self._n_completed: int = 0
        self.realized: list[tuple[int, float, float, bool]] = []

    def update_ema(self, e2e_ms: float, k: int) -> None:
        """Called when a request completes: update the EMA with its e2e."""
        if self._ema_e2e is None:
            self._ema_e2e = float(e2e_ms)
            self._ema_k = float(k)
        else:
            a = self._ema_alpha
            self._ema_e2e = a * float(e2e_ms) + (1 - a) * self._ema_e2e
            self._ema_k = a * float(k) + (1 - a) * self._ema_k
        self._n_completed += 1

    def _live_wait(self, num_running: int, max_num_seqs: int) -> float:
        """Estimate current queue wait from live EMA (or fallback to model)."""
        if self._ema_e2e is not None and self._n_completed >= 3:
            # wait = observed_e2e - compute(ema_k) — the load-dependent excess
            return max(0.0, self._ema_e2e - compute_ms(self._ema_k))
        # cold start: use calibrated model
        return wait_ms(num_running, max_num_seqs)

    def allocate(self, load, slo_ms: float, sum_k: int = 0) -> int:
        """Return k_i for this request given live load + SLO deadline.

        HYBRID gate: EMA-based latency estimate for the queue component +
        load-factor threshold for robustness. The gate exploits the H1b
        mixed-SLO differentiation: slack-SLO requests get k=576 (high accuracy,
        their deadline absorbs the extra latency), tight-SLO requests get k=288
        or k=144 (matching the throughput-optimal fixed-r under SLO pressure).
        """
        nr = getattr(load, "num_running", None) or 0
        mns = getattr(load, "max_num_seqs", None) or 64
        lf = nr / mns if mns > 0 else 0  # load fraction [0, 1+]
        cands = sorted(int(k) for k in self.k_grid
                       if self.k_min <= int(k) <= self.k_max)
        if not cands:
            return max(1, self.k_min)
        wait = self._live_wait(nr, mns)
        slo = float(slo_ms)

        # Greedy ascending: highest k that meets SLO with positive acc gain
        best_k = cands[0]
        best_est = compute_ms(best_k) + wait
        best_acc = self._acc.get(best_k, 0.5)
        for nk in cands[1:]:
            est = compute_ms(nk) + wait
            acc = self._acc.get(nk, 0.5)
            if est <= slo and acc > best_acc:
                best_k, best_est, best_acc = nk, est, acc
            elif est > slo:
                break

        # SLO-aware safety: for tight-SLO requests under moderate+ load,
        # cap at k=288 (the throughput-optimal fixed-r). This prevents
        # the allocator from giving k=576 to tight-SLO requests when the
        # system is near capacity (which would cause SLO violations).
        if slo < 10000 and lf > 0.25 and best_k > 288:
            best_k = 288
            best_est = compute_ms(288) + wait
        # Heavy load: cap tight-SLO at k=144
        if slo < 10000 and lf > 0.65 and best_k > 144:
            best_k = 144
            best_est = compute_ms(144) + wait

        met = best_est <= slo
        self.realized.append((best_k, best_est, slo, met))
        return int(best_k)

    def realized_summary(self) -> dict:
        if not self.realized:
            return {}
        ks = [r[0] for r in self.realized]
        from collections import Counter
        return {
            "n": len(ks),
            "k_mean": sum(ks) / len(ks),
            "k_dist": dict(Counter(ks)),
            "n_met": sum(1 for r in self.realized if r[3]),
            "ema_e2e": self._ema_e2e,
            "ema_k": self._ema_k,
            "n_completed": self._n_completed,
        }


def assign_debug_k(debug_k_spec: str, n_reqs: int) -> list[int]:
    """Parse a comma-separated k list (e.g. '576,144,576') and tile to n_reqs.
    For the smoke test: assigns k values round-robin so different requests in
    the SAME batch get different visual-token counts (the constraint-break
    proof)."""
    vals = [int(x.strip()) for x in debug_k_spec.split(",") if x.strip()]
    if not vals:
        return [576] * n_reqs
    return [vals[i % len(vals)] for i in range(n_reqs)]


if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class _Load:
        num_running: int = 1
        num_waiting: int = 0
        kv_occupancy: float = 0.0
        max_num_seqs: int = 64

    alloc = LiveGreedyAllocator(slo_type="e2e")
    # slack (empty engine, 15s SLO) -> k_max; tight (full engine, 3.5s SLO) -> k_min
    cases = [
        ("slack", _Load(num_running=1), 15000.0, 576),
        ("medium", _Load(num_running=32), 6000.0, 576),
        ("tight", _Load(num_running=64, num_waiting=4), 3500.0, 144),
    ]
    print(f"{'case':>7}  {'slo_ms':>6}  {'nr':>3}  {'qd':>3}  wait    k  lat")
    for name, ld, slo, expect in cases:
        k = alloc.allocate(ld, slo, sum_k=(ld.num_running or 1) * 288)
        r = alloc.realized[-1]
        print(f"{name:>7}  {slo:>6.0f}  {ld.num_running:>3}  "
              f"{ld.num_waiting:>3}  {r[1]:>5.2f}s  {k:>3}  {r[3]:>5.2f}s  "
              f"(expect {expect})")
    print(f"\ndebug-k '576,144,576' tiled to 5: {assign_debug_k('576,144,576', 5)}")
