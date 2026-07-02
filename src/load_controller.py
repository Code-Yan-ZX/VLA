"""Load-adaptive prune-rate controller (P2 method D core).

THE METHOD (0/37 papers do this): instead of a FIXED prune rate `r`, set `r`
dynamically from the serving engine's runtime load -- prune MORE under high
load / high KV-occupancy (where M2 showed the req/s speedup is largest:
r75 goes 1.26x->1.76x as concurrency rises 1->12 -- KV-pressure relief
amplifies with prune depth), prune LESS under light load (preserve accuracy
where throughput is not bottlenecked). This is the serving-aware contribution.

== Engine-load read (the integration crux, SOLVED in vLLM V0) ==
V0 runs the model in-process, so the scheduler is reachable:
    llm.llm_engine.scheduler            # list[Scheduler], one per TP rank
    llm.llm_engine.scheduler[0].running # deque[SequenceGroup] (num running seqs)
    llm.llm_engine.scheduler[0].block_manager.get_num_free_gpu_blocks()  # int
    llm.llm_engine.scheduler[0].block_manager.num_total_gpu_blocks       # int
KV-occupancy = 1 - free/total. num-running = len(running). Both read at
request-submission time (before llm.chat()), so r is decided per-request from
the live engine state.

== Controller policy ==
Piecewise-linear map load -> r in [r_min, r_max]. `load` is either KV-occupancy
(fraction of KV blocks used, default signal) or num-running-seqs (fallback /
supplementary). Low load (occupancy < occ_lo) -> r_min; high load
(occupancy > occ_hi) -> r_max; linear interp between. Thresholds + bounds are
CLI args so the policy is tunable.

== Varying-load profile (REQUIRED to show the adaptive win) ==
The adaptive benefit only appears under VARYING load (constant max -> just use
r_max; constant low -> r_min). serve_bench gains a `--load-profile` mode that
creates time-varying concurrency: bursts of many requests followed by idle
gaps, or a step profile. The adaptive controller then prunes aggressively
during the bursts (high KV-pressure) and lightly during the gaps.

CPU-testable (no vLLM import): the controller is pure math given a load value.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LoadReading:
    """A snapshot of the engine's runtime load at one instant.

    `kv_occupancy` in [0,1] = fraction of GPU KV-cache blocks in use
    (1 - free/total). `num_running` = number of sequences currently in the
    RUNNING queue. Either may be None if unreadable (the controller falls
    back to the other signal).
    """
    kv_occupancy: Optional[float] = None
    num_running: Optional[int] = None
    num_waiting: Optional[int] = None
    num_swapped: Optional[int] = None
    ts: float = 0.0           # time of reading (perf_counter), for logging


def read_engine_load(llm) -> LoadReading:
    """Read the V0 engine's current load. Returns a LoadReading.

    Path (V0, in-process): llm.llm_engine.scheduler[0] -> Scheduler, which
    has .running/.waiting/.swapped deques and a .block_manager exposing
    get_num_free_gpu_blocks() and num_total_gpu_blocks.

    For TP=1 (our case) scheduler is a 1-element list. Resilient: if any
    attribute is missing (V1 engine, or an unexpected shape), the corresponding
    field is None and the controller falls back to whatever signal IS
    available. Returns all-None if the scheduler is unreachable.
    """
    import time
    reading = LoadReading(ts=time.perf_counter())
    try:
        engine = llm.llm_engine
        sched_list = getattr(engine, "scheduler", None)
        if not sched_list:
            return reading
        sched = sched_list[0]
        running = getattr(sched, "running", None)
        waiting = getattr(sched, "waiting", None)
        swapped = getattr(sched, "swapped", None)
        if running is not None:
            reading.num_running = len(running)
        if waiting is not None:
            reading.num_waiting = len(waiting)
        if swapped is not None:
            reading.num_swapped = len(swapped)
        bm = getattr(sched, "block_manager", None)
        free = None
        total = None
        if bm is not None:
            try:
                free = bm.get_num_free_gpu_blocks()
            except Exception:
                free = None
            total = getattr(bm, "num_total_gpu_blocks", None)
        if free is not None and total is not None and total > 0:
            reading.kv_occupancy = 1.0 - (free / total)
    except Exception:
        # any structural surprise -> leave fields None (controller falls back)
        pass
    return reading


@dataclass
class LoadAdaptiveController:
    """Piecewise-linear map engine-load -> prune rate r in [r_min, r_max].

    Policy (KV-occupancy as the primary signal by default):
        occ < occ_lo            -> r_min            (light load: keep accuracy)
        occ > occ_hi            -> r_max            (heavy load: max KV relief)
        occ_lo <= occ <= occ_hi -> linear interp r_min -> r_max

    Fallback signal: if `kv_occupancy` is None at read time, fall back to
    num-running with its own thresholds (run_lo, run_hi). If both are None,
    return r_min (safe default -- never prune more than necessary when we
    can't see the load).

    All thresholds/bounds are dataclass fields so the policy is fully CLI-driven.
    `realized` accumulates the per-request (r, reading) pairs for事后 analysis
    of whether the controller actually adapted (the headline-validation signal).
    """
    r_min: float = 0.25
    r_max: float = 0.50
    # KV-occupancy thresholds (primary signal)
    occ_lo: float = 0.40
    occ_hi: float = 0.70
    # num-running thresholds (fallback signal)
    run_lo: int = 4
    run_hi: int = 8
    signal: str = "kv_occupancy"   # "kv_occupancy" | "num_running"
    realized: list = field(default_factory=list)   # [(r, LoadReading), ...]

    def __post_init__(self) -> None:
        if not 0.0 <= self.r_min <= self.r_max < 1.0:
            raise ValueError(
                f"need 0 <= r_min <= r_max < 1; got r_min={self.r_min} "
                f"r_max={self.r_max}")
        if not 0.0 <= self.occ_lo <= self.occ_hi <= 1.0:
            raise ValueError(
                f"need 0 <= occ_lo <= occ_hi <= 1; got occ_lo={self.occ_lo} "
                f"occ_hi={self.occ_hi}")
        if not 0 <= self.run_lo <= self.run_hi:
            raise ValueError(
                f"need run_lo <= run_hi; got run_lo={self.run_lo} "
                f"run_hi={self.run_hi}")

    @staticmethod
    def _interp(x: float, x_lo: float, x_hi: float, y_lo: float, y_hi: float) -> float:
        """Clamped linear interpolation: x in [x_lo,x_hi] -> y in [y_lo,y_hi]."""
        if x <= x_lo:
            return y_lo
        if x >= x_hi:
            return y_hi
        t = (x - x_lo) / (x_hi - x_lo)
        return y_lo + t * (y_hi - y_lo)

    def decide_r(self, reading: LoadReading) -> float:
        """Map a LoadReading to a prune rate r in [r_min, r_max]."""
        r: Optional[float] = None
        if self.signal == "kv_occupancy":
            occ = reading.kv_occupancy
            if occ is not None:
                r = self._interp(occ, self.occ_lo, self.occ_hi,
                                 self.r_min, self.r_max)
        if r is None:
            # fallback (or signal=num_running): use num-running if available
            nr = reading.num_running
            if nr is not None:
                r = self._interp(float(nr), float(self.run_lo), float(self.run_hi),
                                 self.r_min, self.r_max)
        if r is None:
            # both signals unreadable -> safe default (don't over-prune blind)
            r = self.r_min
        # clamp + record
        r = max(self.r_min, min(self.r_max, r))
        self.realized.append((r, reading))
        return r

    def realized_summary(self) -> dict:
        """Aggregate the per-request r distribution (the adaptation proof)."""
        rs = [x[0] for x in self.realized]
        occs = [x[1].kv_occupancy for x in self.realized
                if x[1].kv_occupancy is not None]
        nrs = [x[1].num_running for x in self.realized
               if x[1].num_running is not None]
        if not rs:
            return {"n": 0}
        out = {
            "n": len(rs),
            "r_mean": sum(rs) / len(rs),
            "r_min": min(rs),
            "r_max": max(rs),
            "r_first5": rs[:5],
            "r_last5": rs[-5:],
        }
        if occs:
            out["occ_mean"] = sum(occs) / len(occs)
            out["occ_min"] = min(occs)
            out["occ_max"] = max(occs)
        if nrs:
            out["num_running_mean"] = sum(nrs) / len(nrs)
            out["num_running_max"] = max(nrs)
        return out


# --------------------------------------------------------------------------- #
# Varying-load profile generators (REQUIRED to show the adaptive win)
# --------------------------------------------------------------------------- #
# The adaptive benefit only appears under VARYING load. serve_bench's
# --load-profile controls HOW requests are submitted over time so the engine
# load rises and falls and the controller has something to react to.
#
# Each generator yields (batch_of_samples, gap_seconds) tuples: submit the
# batch, then sleep gap_seconds before the next batch. This makes the engine
# concurrency swing between ~1 (during gaps) and ~max_num_seqs (during bursts),
# which is exactly the regime where M2 showed the req/s speedup grows with
# concurrency -- so the controller's r should track the swing.
#
# Profiles:
#   constant : one big batch, no gaps (= the M2 batch-submit "constant high"
#              case; the controller should sit at ~r_max the whole time).
#   bursty   : small bursts of `burst` requests separated by `gap` s idle
#              (concurrency swings 0 -> burst -> 0 -> burst; r should swing
#              r_min -> r_max -> r_min).
#   step     : a low-rate phase (1-at-a-time, short gap) then a high-rate
#              phase (big batch) then low again (a clean two-level staircase
#              that maps to the occ_lo/occ_hi thresholds).


def gen_constant(samples: list, max_num_seqs: int) -> list:
    """One batch containing everything (the M2 'constant high' baseline).

    Returns [(batch_list, gap_seconds)] -- a single (all_samples, 0.0) tuple.
    """
    return [(samples, 0.0)]


def gen_bursty(samples: list, max_num_seqs: int, burst: int = 4,
               gap: float = 0.3) -> list:
    """Submit `burst` requests at a time, then `gap`s idle, repeat.

    Concurrency swings between ~0 (after a long gap) and ~burst (during the
    burst, bounded by max_num_seqs). The gap is kept SHORT (< typical decode
    time of a burst) so the previous burst's requests are STILL in-flight when
    the next burst arrives -> the controller sees residual KV-occupancy rising
    across the first few bursts (the adaptation signal). With a long gap the
    engine fully drains between bursts and the controller sees low load every
    time (no adaptation) -- which is itself a valid (if uninteresting) result.

    Default burst=4, gap=0.3s: at c12, 16-token decode takes ~1-2s, so a 0.3s
    gap leaves the prior burst mid-flight -> occupancy at the next burst's
    decision point is non-zero and grows.
    """
    out = []
    for i in range(0, len(samples), burst):
        batch = samples[i:i + burst]
        # gap after every burst except the last
        out.append((batch, gap if i + burst < len(samples) else 0.0))
    return out


def gen_step(samples: list, max_num_seqs: int, n_low: int = 20,
             low_gap: float = 0.8, n_high: int = 40, high_gap: float = 2.0) -> list:
    """Low-rate phase (1-at-a-time) then high-rate phase (big batch) then low.

    A clean two-level staircase that exercises both controller thresholds:
      phase 1 (low ): submit n_low requests one-at-a-time with short gaps
                      -> concurrency ~1, occupancy low -> r ~ r_min
      phase 2 (high): submit n_high requests as ONE batch up to max_num_seqs
                      -> concurrency ~max_num_seqs, occupancy rises -> r ~ r_max
      phase 3 (low ): remaining requests one-at-a-time -> r back to r_min
    """
    out = []
    low = samples[:n_low]
    high = samples[n_low:n_low + n_high]
    tail = samples[n_low + n_high:]
    for s in low:
        out.append(([s], low_gap))
    if high:
        out.append((high, high_gap))
    for s in tail:
        out.append(([s], low_gap))
    return out


PROFILES = {
    "constant": gen_constant,
    "bursty": gen_bursty,
    "step": gen_step,
}


# --------------------------------------------------------------------------- #
# CPU self-test (run via `python -m src.load_controller`)
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    c = LoadAdaptiveController(r_min=0.25, r_max=0.50, occ_lo=0.4, occ_hi=0.7)

    # light load -> r_min
    r = c.decide_r(LoadReading(kv_occupancy=0.1, num_running=1))
    assert r == 0.25, f"light load should give r_min, got {r}"

    # heavy load -> r_max
    r = c.decide_r(LoadReading(kv_occupancy=0.9, num_running=20))
    assert r == 0.50, f"heavy load should give r_max, got {r}"

    # midpoint occupancy -> midpoint r
    r = c.decide_r(LoadReading(kv_occupancy=0.55, num_running=4))
    assert abs(r - 0.375) < 1e-6, f"midpoint occ should give midpoint r, got {r}"

    # kv_occupancy None -> fall back to num_running
    r = c.decide_r(LoadReading(kv_occupancy=None, num_running=20))
    assert r == 0.50, f"fallback num_running heavy should give r_max, got {r}"

    # both None -> safe default r_min
    r = c.decide_r(LoadReading(kv_occupancy=None, num_running=None))
    assert r == 0.25, f"blind should give r_min, got {r}"

    # num_running midpoint (with kv None)
    r = c.decide_r(LoadReading(kv_occupancy=None, num_running=6))
    assert abs(r - 0.375) < 1e-6, f"num_running midpoint should give midpoint r, got {r}"

    # realized summary
    summ = c.realized_summary()
    assert summ["n"] == 6
    assert summ["r_min"] == 0.25 and summ["r_max"] == 0.50
    assert summ["r_mean"] > 0.25 and summ["r_mean"] < 0.50

    # profile generators: shapes + coverage
    samp = list(range(50))
    cons = gen_constant(samp, max_num_seqs=12)
    assert len(cons) == 1 and len(cons[0][0]) == 50
    bur = gen_bursty(samp, max_num_seqs=12, burst=6, gap=1.0)
    assert len(bur) == 9  # 50 / 6 = 8 full + 1 partial
    assert len(bur[0][0]) == 6 and bur[0][1] == 1.0
    assert bur[-1][1] == 0.0  # last has no trailing gap
    # step: low(20) + high(1 batch of up to 40) + tail. With 50 samples the
    # high phase gets only 30 (50-20); tail is empty.
    st = gen_step(samp, max_num_seqs=12, n_low=20, n_high=40)
    assert len(st[:20]) == 20 and all(len(b) == 1 for b, _ in st[:20])
    assert len(st[20][0]) == 30  # high phase = remaining after n_low (capped)
    assert len(st) == 21  # 20 low + 1 high, no tail (samples exhausted)
    # step with more samples than n_low+n_high -> tail present
    samp2 = list(range(100))
    st2 = gen_step(samp2, max_num_seqs=12, n_low=20, n_high=40)
    assert len(st2[20][0]) == 40
    assert len(st2) == 20 + 1 + (100 - 20 - 40)  # 20 low + 1 high + 40 tail

    # boundary validation in __post_init__
    try:
        LoadAdaptiveController(r_min=0.6, r_max=0.5)
        raise AssertionError("should have rejected r_min > r_max")
    except ValueError:
        pass
    try:
        LoadAdaptiveController(occ_lo=0.8, occ_hi=0.7)
        raise AssertionError("should have rejected occ_lo > occ_hi")
    except ValueError:
        pass

    print(f"load_controller self-test OK: controller decides r in "
          f"[0.25,0.50] across light/mid/heavy/blind; realized_summary n={summ['n']}; "
          f"profiles constant/bursty/step shapes verified.")


if __name__ == "__main__":
    _self_test()
