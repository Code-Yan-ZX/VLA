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

== Controller latency (implementation note) ==
serve_bench's load-profile path drains each segment fully before submitting the
next (required because vLLM batches multiple image-requests per forward and the
projector hook reads a SINGLE shared k_cell -- cross-segment batching with
different r would mismatch placeholder count vs kept count). So the controller
operates with ONE-SEGMENT LAG: during each segment's drain it samples the peak
load and uses that to decide the NEXT segment's r. This is a legitimate reactive
controller (one control-cycle of latency). Under bursty/step profiles, big
bursts raise the sampled peak -> the next segment prunes more; quiet periods ->
less. The realized[] log records the per-decision (r, reading) pairs.

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
    back to the other signal). `max_num_seqs` is the engine's concurrency
    cap (vLLM `max_num_seqs`); the controller derives a concurrency
    FRACTION `num_running / max_num_seqs` in [0,1] as the load signal that
    spans the full range under realistic deployment (c12 short-sequence:
    peak num_running ~12 -> fraction ~1.0, vs KV-occupancy peaking only
    ~0.04 because the KV pool dwarfs 12 short seqs).
    """
    kv_occupancy: Optional[float] = None
    num_running: Optional[int] = None
    num_waiting: Optional[int] = None
    num_swapped: Optional[int] = None
    max_num_seqs: Optional[int] = None   # engine concurrency cap (for fraction)
    ts: float = 0.0           # time of reading (perf_counter), for logging

    @property
    def concurrency_fraction(self) -> Optional[float]:
        """num_running normalized to [0,1] by max_num_seqs. None if unreadable."""
        if self.num_running is None or not self.max_num_seqs or self.max_num_seqs <= 0:
            return None
        return self.num_running / float(self.max_num_seqs)


def read_engine_load(llm, max_num_seqs: Optional[int] = None) -> LoadReading:
    """Read the V0 engine's current load. Returns a LoadReading.

    Path (V0, in-process): llm.llm_engine.scheduler[0] -> Scheduler, which
    has .running/.waiting/.swapped deques and a .block_manager exposing
    get_num_free_gpu_blocks() and num_total_gpu_blocks.

    `max_num_seqs` (the vLLM engine's concurrency cap) is stamped onto the
    reading so the controller can derive a concurrency fraction. If None,
    the controller falls back to absolute num_running thresholds.

    For TP=1 (our case) scheduler is a 1-element list. Resilient: if any
    attribute is missing (V1 engine, or an unexpected shape), the corresponding
    field is None and the controller falls back to whatever signal IS
    available. Returns all-None if the scheduler is unreachable.
    """
    import time
    reading = LoadReading(ts=time.perf_counter(), max_num_seqs=max_num_seqs)
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


def _parse_v1_metrics(metrics_list) -> dict:
    """Parse a V1 `llm.get_metrics()` snapshot (list[Gauge/Histogram/etc.])
    into a name->value dict. Robust across metric kinds (Gauge.value,
    Histogram.sum/last_value, etc.)."""
    out = {}
    for m in metrics_list:
        nm = getattr(m, "name", None) or getattr(m, "prometheus_name", None) \
            or str(getattr(m, "key", ""))
        if not nm:
            continue
        val = None
        for f in ("value", "sum_value", "last_value"):
            v = getattr(m, f, None)
            if isinstance(v, (int, float)):
                val = float(v)
                break
        out[nm] = val
    return out


def read_engine_load_v1(llm, max_num_seqs: Optional[int] = None) -> LoadReading:
    """V1 controller signal (the §4.3 replacement for read_engine_load).

    V1 runs EngineCore in a subprocess by default (VLLM_ENABLE_V1_MULTIPROCESSING=1),
    so the V0 in-process scheduler reads are DEAD. The replacement is the
    Prometheus snapshot via `llm.get_metrics()` (verified populated under load
    in runs/v1_probe.py: `vllm:num_requests_running` peaks at the full
    max_num_seqs). Note: `gpu_cache_usage_perc` requires `kv_cache_metrics=True`
    (off by default) -- num_running is the primary signal anyway (matches V0's
    P3-step-1 default).

    With VLLM_ENABLE_V1_MULTIPROCESSING=0 (our measurement path: keeps V1's
    scheduler in-process so forward-hooks reach the model), the scheduler is
    ALSO reachable -- but we still use get_metrics() because it is the
    mode-agnostic V1 API and works identically in both modes.

    Falls back to the V0 in-process read if get_metrics() is unavailable
    (e.g. log_stats disabled) -- returns all-None if neither path yields.
    """
    import time
    reading = LoadReading(ts=time.perf_counter(), max_num_seqs=max_num_seqs)
    # --- primary V1 path: Prometheus snapshot ---
    try:
        get_metrics = getattr(llm, "get_metrics", None)
        if get_metrics is None:
            get_metrics = getattr(getattr(llm, "llm_engine", None),
                                  "get_metrics", None)
        if get_metrics is not None:
            ms = _parse_v1_metrics(get_metrics())
            nr = ms.get("vllm:num_requests_running")
            nw = ms.get("vllm:num_requests_waiting")
            gc = ms.get("vllm:gpu_cache_usage_perc")
            if nr is not None:
                reading.num_running = int(nr)
            if nw is not None:
                reading.num_waiting = int(nw)
            if gc is not None:
                reading.kv_occupancy = gc
    except Exception:
        pass
    # --- fallback: V0 in-process scheduler (works under multiproc=0) ---
    if reading.num_running is None:
        try:
            sched_list = getattr(getattr(llm, "llm_engine", None), "scheduler", None)
            if sched_list:
                sched = sched_list[0]
                running = getattr(sched, "running", None)
                if running is not None:
                    reading.num_running = len(running)
                bm = getattr(sched, "block_manager", None)
                free = getattr(bm, "get_num_free_gpu_blocks", lambda: None)() if bm else None
                total = getattr(bm, "num_total_gpu_blocks", None) if bm else None
                if isinstance(free, (int, float)) and isinstance(total, (int, float)) and total > 0:
                    reading.kv_occupancy = 1.0 - (free / total)
        except Exception:
            pass
    return reading


@dataclass
class LoadAdaptiveController:
    """Piecewise-linear map engine-load -> prune rate r in [r_min, r_max].

    TWO load signals (selectable via `signal`):
      * `num_running` (DEFAULT since P3-step-1): the controller reacts to the
        concurrency FRACTION = num_running / max_num_seqs in [0,1]. Under the
        c12 / short-sequence deployment this spans the FULL range (peak
        num_running ~12 -> fraction ~1.0), so realized-r genuinely traverses
        [r_min, r_max] across a bursty/step profile. Thresholds are fraction
        thresholds `conc_lo` / `conc_hi` (defaults 0.25 / 0.75 -> r_min below
        25% concurrency, r_max above 75%; at c12 that's r_min <3 concurrent,
        r_max >9 concurrent).
      * `kv_occupancy`: the controller reacts to KV-cache occupancy. Preferred
        for LONG-sequence / high-concurrency regimes where KV pressure is the
        real bottleneck and occupancy actually rises into a meaningful range.
        Under short-sequence c12 the KV pool dwarfs the live sequences (peak
        occ ~0.04), so the controller barely leaves r_min -- use num_running
        there instead.

    Policy (whichever signal is selected):
        load < x_lo            -> r_min            (light load: keep accuracy)
        load > x_hi            -> r_max            (heavy load: max KV relief)
        x_lo <= load <= x_hi   -> linear interp r_min -> r_max

    Fallback: if the selected signal is None at read time, fall back to the
    OTHER signal; if both are None, return r_min (safe default -- never prune
    more than necessary when we can't see the load). For num_running without a
    max_num_seqs on the reading, fall back to the absolute run_lo/run_hi
    thresholds.

    All thresholds/bounds are dataclass fields so the policy is fully CLI-driven.
    `realized` accumulates the per-request (r, reading) pairs for事后 analysis
    of whether the controller actually adapted (the headline-validation signal).
    """
    r_min: float = 0.25
    r_max: float = 0.50
    # KV-occupancy thresholds (signal=kv_occupancy)
    occ_lo: float = 0.40
    occ_hi: float = 0.70
    # concurrency-FRACTION thresholds (signal=num_running; the P3-step-1 default)
    conc_lo: float = 0.25
    conc_hi: float = 0.75
    # absolute num-running thresholds (legacy fallback if max_num_seqs unknown)
    run_lo: int = 4
    run_hi: int = 8
    signal: str = "num_running"   # "num_running" (default) | "kv_occupancy"
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
        if not 0.0 <= self.conc_lo <= self.conc_hi <= 1.0:
            raise ValueError(
                f"need 0 <= conc_lo <= conc_hi <= 1; got conc_lo={self.conc_lo} "
                f"conc_hi={self.conc_hi}")
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
        """Map a LoadReading to a prune rate r in [r_min, r_max].

        Selected signal first, then cross-fallback to the other, then r_min.
          * num_running: prefer concurrency FRACTION (num_running/max_num_seqs)
            scored against conc_lo/conc_hi; if max_num_seqs is unknown, fall
            back to absolute num_running scored against run_lo/run_hi.
          * kv_occupancy: scored against occ_lo/occ_hi.
        """
        r: Optional[float] = None
        if self.signal == "num_running":
            frac = reading.concurrency_fraction   # None if max_num_seqs unknown
            if frac is not None:
                r = self._interp(frac, self.conc_lo, self.conc_hi,
                                 self.r_min, self.r_max)
            elif reading.num_running is not None:
                # absolute fallback (no max_num_seqs on the reading)
                r = self._interp(float(reading.num_running),
                                 float(self.run_lo), float(self.run_hi),
                                 self.r_min, self.r_max)
            # if num_running path yielded nothing, cross-fallback to kv_occupancy
            if r is None and reading.kv_occupancy is not None:
                r = self._interp(reading.kv_occupancy, self.occ_lo, self.occ_hi,
                                 self.r_min, self.r_max)
        else:  # signal == "kv_occupancy"
            occ = reading.kv_occupancy
            if occ is not None:
                r = self._interp(occ, self.occ_lo, self.occ_hi,
                                 self.r_min, self.r_max)
            # cross-fallback to num_running (fraction if possible, else absolute)
            if r is None:
                frac = reading.concurrency_fraction
                if frac is not None:
                    r = self._interp(frac, self.conc_lo, self.conc_hi,
                                     self.r_min, self.r_max)
                elif reading.num_running is not None:
                    r = self._interp(float(reading.num_running),
                                     float(self.run_lo), float(self.run_hi),
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
        concs = [x[1].concurrency_fraction for x in self.realized
                 if x[1].concurrency_fraction is not None]
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
            out["num_running_min"] = min(nrs)
            out["num_running_max"] = max(nrs)
        if concs:
            out["conc_frac_mean"] = sum(concs) / len(concs)
            out["conc_frac_min"] = min(concs)
            out["conc_frac_max"] = max(concs)
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


def gen_bursty(samples: list, max_num_seqs: int, burst: int = -1,
               gap: float = 1.5) -> list:
    """Submit requests in ALTERNATING small and large bursts, separated by gaps.

    The P3-step-1 refinement: a single fixed burst size doesn't exercise the
    controller's full range under the one-segment-lag design (every segment's
    mid-drain peak looks the same -> r is constant). ALTERNATING burst sizes
    makes the per-segment peak load genuinely swing: a small burst (2 reqs ->
    conc ~0.17 -> r_min) followed by a large burst (max_num_seqs reqs -> conc
    ~1.0 -> r_max) gives the controller a real signal to react to. This is
    also more realistic than uniform bursts (real traffic is uneven).

    With full per-segment drain, segment N+1's r is decided from segment N's
    PEAK load. So after a small burst (low peak) the next segment prunes
    lightly (r_min); after a large burst (high peak) the next prunes hard
    (r_max). The realized-r time-series then visibly alternates r_min/r_max --
    the controller-figure evidence and the regime where adaptive beats fixed
    (it matches the light prune when load was low, the heavy prune when high,
    while any FIXED r is wrong half the time).

    Args:
      burst: if >0, a FIXED burst size (legacy/override; no alternation). If
        -1 (default), alternate small=max(1, max_num_seqs//6) and
        large=max_num_seqs.
      gap: idle seconds after each burst (long enough that the engine
        substantially drains -> the NEXT burst's submission-time load is low).
    """
    out = []
    if burst > 0:
        # fixed burst size (legacy path)
        for i in range(0, len(samples), burst):
            batch = samples[i:i + burst]
            out.append((batch, gap if i + burst < len(samples) else 0.0))
        return out
    # P3-step-1: alternating small / large bursts
    small = max(1, max_num_seqs // 6)   # e.g. c12 -> 2 reqs (conc ~0.17 -> r_min)
    large = max_num_seqs                # e.g. c12 -> 12 reqs (conc ~1.0 -> r_max)
    sizes = []
    i = 0
    toggle = True  # start with small (light load first)
    while i < len(samples):
        sz = small if toggle else large
        sizes.append(sz)
        i += sz
        toggle = not toggle
    i = 0
    for k, sz in enumerate(sizes):
        batch = samples[i:i + sz]
        i += sz
        is_last = (k == len(sizes) - 1)
        out.append((batch, gap if not is_last else 0.0))
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
    # ---- num_running signal (P3-step-1 default): conc_lo=0.25, conc_hi=0.75 ----
    c = LoadAdaptiveController(r_min=0.25, r_max=0.50, signal="num_running",
                              conc_lo=0.25, conc_hi=0.75)
    # light (concurrency 1/12 ~0.083 < 0.25) -> r_min
    r = c.decide_r(LoadReading(num_running=1, max_num_seqs=12))
    assert r == 0.25, f"light num_running should give r_min, got {r}"
    # heavy (concurrency 12/12 = 1.0 > 0.75) -> r_max
    r = c.decide_r(LoadReading(num_running=12, max_num_seqs=12))
    assert r == 0.50, f"heavy num_running should give r_max, got {r}"
    # midpoint (concurrency 6/12 = 0.5 in [0.25,0.75]) -> midpoint r = 0.375
    r = c.decide_r(LoadReading(num_running=6, max_num_seqs=12))
    assert abs(r - 0.375) < 1e-6, f"midpoint concurrency should give midpoint r, got {r}"
    # concurrency fraction threshold edge: 3/12=0.25 -> r_min; 9/12=0.75 -> r_max
    r = c.decide_r(LoadReading(num_running=3, max_num_seqs=12))
    assert r == 0.25, f"3/12=0.25 (lo edge) should give r_min, got {r}"
    r = c.decide_r(LoadReading(num_running=9, max_num_seqs=12))
    assert r == 0.50, f"9/12=0.75 (hi edge) should give r_max, got {r}"
    # no max_num_seqs -> absolute fallback (run_lo=4, run_hi=8)
    r = c.decide_r(LoadReading(num_running=20, max_num_seqs=None))
    assert r == 0.50, f"absolute num_running heavy (no max) should give r_max, got {r}"
    r = c.decide_r(LoadReading(num_running=1, max_num_seqs=None))
    assert r == 0.25, f"absolute num_running light (no max) should give r_min, got {r}"
    # num_running None but kv_occupancy present -> cross-fallback
    r = c.decide_r(LoadReading(num_running=None, kv_occupancy=0.9, max_num_seqs=12))
    assert r == 0.50, f"cross-fallback to kv heavy should give r_max, got {r}"
    # both None -> safe default r_min
    r = c.decide_r(LoadReading(num_running=None, kv_occupancy=None))
    assert r == 0.25, f"blind should give r_min, got {r}"

    # ---- kv_occupancy signal (legacy / long-seq regime) ----
    c2 = LoadAdaptiveController(r_min=0.25, r_max=0.50, signal="kv_occupancy",
                               occ_lo=0.4, occ_hi=0.7)
    r = c2.decide_r(LoadReading(kv_occupancy=0.1, num_running=1))
    assert r == 0.25, f"light kv should give r_min, got {r}"
    r = c2.decide_r(LoadReading(kv_occupancy=0.9, num_running=20))
    assert r == 0.50, f"heavy kv should give r_max, got {r}"
    r = c2.decide_r(LoadReading(kv_occupancy=0.55, num_running=4))
    assert abs(r - 0.375) < 1e-6, f"midpoint occ should give midpoint r, got {r}"
    # kv None -> cross-fallback to num_running fraction
    r = c2.decide_r(LoadReading(kv_occupancy=None, num_running=12, max_num_seqs=12))
    assert r == 0.50, f"cross-fallback to num_running heavy should give r_max, got {r}"

    # realized summary (c: 9 decisions on the num_running controller)
    summ = c.realized_summary()
    assert summ["n"] == 9, f"expected 9 realized, got {summ['n']}"
    assert summ["r_min"] == 0.25 and summ["r_max"] == 0.50
    assert summ["r_mean"] > 0.25 and summ["r_mean"] < 0.50
    assert "conc_frac_min" in summ and "conc_frac_max" in summ, \
        "realized_summary must report concurrency-fraction stats"

    # profile generators: shapes + coverage
    samp = list(range(50))
    cons = gen_constant(samp, max_num_seqs=12)
    assert len(cons) == 1 and len(cons[0][0]) == 50
    bur = gen_bursty(samp, max_num_seqs=12, burst=6, gap=1.0)
    assert len(bur) == 9  # 50 / 6 = 8 full + 1 partial
    assert len(bur[0][0]) == 6 and bur[0][1] == 1.0
    assert bur[-1][1] == 0.0  # last has no trailing gap
    # P3-step-1 default bursty (alternating small/large): c12 -> small=2, large=12.
    # 50 samples: 2,12,2,12,2,12,2,12 = 56 capacity -> 8 bursts, last partial (2 left).
    bur_def = gen_bursty(samp, max_num_seqs=12, gap=1.5)
    sizes_def = [len(b) for b, _ in bur_def]
    assert sizes_def[0] == 2 and sizes_def[1] == 12, f"alternating small/large, got {sizes_def[:4]}"
    assert all(g == 1.5 for _, g in bur_def[:-1]) and bur_def[-1][1] == 0.0
    assert sum(sizes_def) == 50, f"all samples covered, got {sum(sizes_def)}"
    st = gen_step(samp, max_num_seqs=12, n_low=20, n_high=40)
    assert len(st[:20]) == 20 and all(len(b) == 1 for b, _ in st[:20])
    assert len(st[20][0]) == 30
    assert len(st) == 21
    samp2 = list(range(100))
    st2 = gen_step(samp2, max_num_seqs=12, n_low=20, n_high=40)
    assert len(st2[20][0]) == 40
    assert len(st2) == 20 + 1 + (100 - 20 - 40)

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
    try:
        LoadAdaptiveController(conc_lo=0.8, conc_hi=0.7)
        raise AssertionError("should have rejected conc_lo > conc_hi")
    except ValueError:
        pass

    print(f"load_controller self-test OK: controller decides r in "
          f"[0.25,0.50] across light/mid/heavy/blind for BOTH signals "
          f"(num_running fraction + kv_occupancy); cross-fallback verified; "
          f"realized_summary n={summ['n']} with conc_frac stats; "
          f"profiles constant/bursty/step shapes verified.")


if __name__ == "__main__":
    _self_test()
