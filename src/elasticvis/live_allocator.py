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


def service_time(k: int) -> float:
    """Slot-occupancy S(k) seconds (M / served_req_s at c64)."""
    return _interp(k, _K_GRID, _S_GRID)


def prefill_time(k: int) -> float:
    """Prefill P(k) seconds (ttft floor at c64)."""
    return _interp(k, _K_GRID, _P_GRID)


def expected_wait_s(load, sum_k: int) -> float:
    """Predicted queue wait at admission. 0 if a free slot; else
    (qd+1) * S(mean_k) / M (queue drains at M/S_avg; position qd+1 waits)."""
    mns = getattr(load, "max_num_seqs", None) or 64
    nr = getattr(load, "num_running", None) or 0
    if nr < mns:
        return 0.0
    qd = getattr(load, "num_waiting", None) or 0
    mean_k = (sum_k / nr) if nr else 288
    return (qd + 1) * service_time(int(mean_k)) / mns


class LiveGreedyAllocator:
    """Queue-aware greedy per-request k allocator (live-engine counterpart of
    ``allocator.GreedyAllocator``).

    Returns the HIGHEST k on the discrete grid such that:

        expected_wait(load, sum_k) + P(k) [+ S(k)] <= slo_s

    Falls back to k_min if even the cheapest k violates. The gate is monotone-
    nondecreasing in k (higher k -> higher P+S), so greedy ascending + track-
    highest-feasible is optimal on the grid.

    Args:
        k_grid: discrete k candidates (default {144,288,576} from v2 probes).
        k_min/k_max: clamp range.
        slo_type: "ttft" gates on wait+P(k); "e2e" gates on wait+P(k)+S(k).
    """

    def __init__(self, k_grid: Optional[Sequence[int]] = None,
                 k_min: int = 144, k_max: int = 576,
                 slo_type: str = "e2e"):
        self.k_grid = tuple(k_grid) if k_grid is not None else _K_GRID
        self.k_min = int(k_min)
        self.k_max = int(k_max)
        self.slo_type = str(slo_type)
        self.realized: list[tuple[int, float, float, float, float, bool]] = []
        # (k_i, wait_s, p_s, total_lat_s, slo_s, met)

    def allocate(self, load, slo_ms: float, sum_k: int = 0) -> int:
        """Return k_i for this request given live load + SLO deadline."""
        slo_s = float(slo_ms) / 1000.0
        cands = sorted(int(k) for k in self.k_grid
                       if self.k_min <= int(k) <= self.k_max)
        if not cands:
            return max(1, self.k_min)
        wait = expected_wait_s(load, sum_k)
        best_k = cands[0]  # floor
        best_lat = self._lat(wait, best_k)
        for nk in cands[1:]:
            lat = self._lat(wait, nk)
            if lat <= slo_s:
                best_k, best_lat = nk, lat
            else:
                break  # greedy: stop at first violation (monotone in k)
        # if even the floor violates, best_k = cands[0] (over-deadline, but
        # k_min is the accuracy floor — serve with lowest visual budget).
        met = best_lat <= slo_s
        self.realized.append((best_k, wait, prefill_time(best_k), best_lat,
                              slo_s, met))
        return int(best_k)

    def _lat(self, wait: float, k: int) -> float:
        p = prefill_time(k)
        if self.slo_type == "ttft":
            return wait + p
        return wait + p + service_time(k)

    def realized_summary(self) -> dict:
        if not self.realized:
            return {}
        ks = [r[0] for r in self.realized]
        from collections import Counter
        return {
            "n": len(ks),
            "k_mean": sum(ks) / len(ks),
            "k_dist": dict(Counter(ks)),
            "n_met": sum(1 for r in self.realized if r[5]),
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
