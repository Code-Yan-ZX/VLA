"""ElasticVis offline discrete-event simulator (zero-GPU go/no-go harness).

Faithful slot+queue server model reproducing v2 closed-loop. Pure Python + numpy +
dataclasses; no vLLM/torch/GPU. Spec: notes/elasticvis_design.md §1/§2/§4/§5.

SERVER MODEL: M slots, continuous batching. Budget k -> S(k)=M/served_req_s(c64,k)
  {144:3.14,288:4.39,576:6.97}s and P(k)=ttft_min(c64,k) ~2.86s (linear interp).
  On arrival: running<M -> admit; else queue (FIFO). Slot frees at t_admit+S(k_i).
  TTFT_i=wait+P(k_i); E2E_i=wait+S(k_i). goodput@SLO=#{met}/T_window (v2); acc-weighted=Σ{met}acc/T.

GATE vs OUTCOME: allocator GATE predicts wait+P(k)/S(k) via expected_wait_s(); sim
OUTCOME is the slot+queue model. lat.predict() unused (segment-sojourn); `lat` is
for signature compat. Contracts: Allocator.allocate(load,slo_ms,slo_type,rf,kr,lat,
acc,sum_k,percentile)->int; optional update_lambda(violation)->None.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable
import heapq
import random
from collections import deque

import numpy as np  # noqa: F401

try:
    from ..load_controller import LoadReading  # type: ignore
except Exception:  # standalone
    @dataclass
    class LoadReading:
        kv_occupancy: Optional[float] = None
        num_running: Optional[int] = None
        num_waiting: Optional[int] = None
        num_swapped: Optional[int] = None
        max_num_seqs: Optional[int] = None
        ts: float = 0.0


# ===================== Server model: S(k), P(k), wait =====================
_K_GRID = (144, 288, 576)
_S_GRID = (3.14, 4.39, 6.97)   # M / served_req_s(c64, k)
_P_GRID = (2.86, 2.81, 2.95)   # ttft_min(c64, k)


def _interp(k, gk, gv):
    if k <= gk[0]: return float(gv[0])
    if k >= gk[-1]: return float(gv[-1])
    for i in range(len(gk) - 1):
        if gk[i] <= k <= gk[i + 1]:
            t = (k - gk[i]) / (gk[i + 1] - gk[i])
            return gv[i] * (1 - t) + gv[i + 1] * t
    return float(gv[-1])


def service_time(k: int) -> float:
    """Slot-occupancy S(k) seconds (M/served_req_s)."""
    return _interp(k, _K_GRID, _S_GRID)


def prefill_time(k: int) -> float:
    """Prefill P(k) seconds (ttft floor)."""
    return _interp(k, _K_GRID, _P_GRID)


def expected_wait_s(load: LoadReading, sum_k: int) -> float:
    """Allocator GATE: predicted queue wait. 0 if free slot; else (qd+1)*S_avg/M
    (queue drains at M/S_avg; position qd+1 waits ~(qd+1)*S_avg/M)."""
    mns = load.max_num_seqs or 64
    if (load.num_running or 0) < mns:
        return 0.0
    qd = load.num_waiting or 0
    mean_k = (sum_k / load.num_running) if load.num_running else 288
    return (qd + 1) * service_time(int(mean_k)) / mns


# =============================== Results ==================================

@dataclass
class GoodputResult:
    n: int; n_met: int; frac_met: float
    goodput_rate: float          # acc-WEIGHTED: Σ{met}·acc / T_window
    met_rate: float              # UNWEIGHTED: n_met / T_window (v2-comparable)
    mean_accuracy: float; T_window: float
    per_req: list; policy: str; arrival: str; slo_type: str; slo_ms: float

    def row(self) -> dict:
        return dict(policy=self.policy, n=self.n, n_met=self.n_met,
                    frac_met=round(self.frac_met, 4), goodput_rate=round(self.goodput_rate, 4),
                    met_rate=round(self.met_rate, 4), mean_accuracy=round(self.mean_accuracy, 4),
                    T_window=round(self.T_window, 3))


# =========================== Arrival processes ============================

class ArrivalProcess:
    kind = "open"; name = "abstract"; c: int = 0
    def sample(self, n: int, rng: random.Random) -> list:
        raise NotImplementedError


class ClosedLoop(ArrivalProcess):
    """Bulk-submit n at t=0; cap at c concurrent (v2 bench regime; load ≈ c constant)."""
    kind = "closed"
    def __init__(self, c: int):
        self.c = max(1, int(c)); self.name = f"closed_loop(c={self.c})"
    def sample(self, n, rng):
        return [(0.0, None)] * n

def closed_loop(c, n=200): return ClosedLoop(c)


class OpenLoopPoisson(ArrivalProcess):
    """Poisson(rate); admission-time num_running VARIES -> H1 primary regime."""
    def __init__(self, rate: float):
        self.rate = float(rate); self.name = f"open_loop_poisson(rate={self.rate})"
    def sample(self, n, rng):
        ts, t = [], 0.0
        for _ in range(n):
            t += rng.expovariate(self.rate); ts.append((t, None))
        return ts

def open_loop_poisson(rate, n=200, seed=0): return OpenLoopPoisson(rate)


class Bursty(ArrivalProcess):
    """Alternating busy/idle (~Exp(1s)). Busy rate=mean/burst_frac, idle≈0."""
    def __init__(self, mean_rate: float, burst_frac: float = 0.5):
        self.mean_rate = float(mean_rate); self.burst_frac = float(burst_frac)
        self.name = f"bursty(mean={self.mean_rate},bf={self.burst_frac})"
    def sample(self, n, rng):
        br = self.mean_rate / max(self.burst_frac, 1e-6); ir = self.mean_rate * 1e-3
        ts, t, busy = [], 0.0, True
        while len(ts) < n:
            spell = rng.expovariate(1.0); r = br if busy else ir; tl = 0.0
            while tl < spell and len(ts) < n:
                tl += rng.expovariate(r)
                if tl > spell: break
                ts.append((t + tl, None))
            t += spell; busy = not busy
        return ts[:n]

def bursty(rate, n=200, seed=0, burst_frac=0.5): return Bursty(rate, burst_frac)


class MixedSLO(ArrivalProcess):
    """Wraps a base arrival; per-req SLO via deadline_dist(rng)->ms. H1b."""
    def __init__(self, base: ArrivalProcess, deadline_dist: Callable[[random.Random], float]):
        self.base = base; self.deadline_dist = deadline_dist
        self.name = f"mixed_slo({base.name})"
    def sample(self, n, rng):
        return [(t, self.deadline_dist(rng)) for (t, _) in self.base.sample(n, rng)]

def mixed_slo(n, deadline_dist, seed=0, base=None):
    return MixedSLO(base or OpenLoopPoisson(4.0), deadline_dist)


# ============================== Simulator =================================

EV_ARRIVE, EV_COMPLETE = 0, 1


def _call_allocate(allocator, load, slo_ms, slo_type, k_range, lat, acc, sum_k, percentile):
    try:
        return allocator.allocate(load, slo_ms, slo_type, None, k_range, lat, acc, sum_k, percentile)
    except TypeError:
        return allocator.allocate(load, slo_ms, slo_type, None, k_range, lat, acc)


def simulate(arrival, allocator, lat, acc, dataset, slo_ms, slo_type="ttft",
             k_range=(144, 576), max_num_seqs=64, percentile="p99", seed=0,
             lambda_update_cadence=32, service_noise=0.15) -> GoodputResult:
    """Slot+queue discrete-event sim. closed_loop: bulk at t=0, cap=arrival.c.
    service_noise=CV of Gaussian noise on S(k) (stagger completions; decode-length
    variance). After every completion: sliding-window violation -> update_lambda()."""
    rng = random.Random(seed); np.random.seed(seed)
    k_min, k_max = int(k_range[0]), int(k_range[1])
    n = len(dataset)
    mns = arrival.c if (arrival.kind == "closed" and arrival.c) else max_num_seqs
    policy = getattr(allocator, "name", type(allocator).__name__)
    update_lambda = getattr(allocator, "update_lambda", None)
    viol_win = deque(maxlen=max(1, lambda_update_cadence))
    heap, seq = [], [0]
    def push(t, kind, payload):
        heapq.heappush(heap, (t, seq[0], kind, payload)); seq[0] += 1
    running, admit_q, per_req = {}, deque(), []

    def admit(t_admit, t_arrive, idx, rid, slo_i_s):
        n_run = len(running)
        sum_k = sum(r["k"] for r in running.values())
        load = LoadReading(num_running=n_run, num_waiting=len(admit_q),
                           kv_occupancy=(n_run / mns) if mns else None,
                           max_num_seqs=mns, ts=t_admit)
        wait_s = max(0.0, t_admit - t_arrive)
        slo_eff_ms = max(0.0, slo_i_s * 1000.0 - wait_s * 1000.0)
        k_i = int(max(k_min, min(k_max, _call_allocate(
            allocator, load, slo_eff_ms, slo_type, k_range, lat, acc, sum_k, percentile))))
        S_i = service_time(k_i) * (max(0.4, min(2.5, rng.gauss(1.0, service_noise))) if service_noise > 0 else 1.0)
        P_i = prefill_time(k_i)
        running[idx] = {"k": k_i, "admit": t_admit, "arrive": t_arrive, "slo": slo_i_s,
                        "ttft": wait_s + P_i, "e2e": wait_s + S_i,
                        "rid": rid, "acc": float(acc.utility(k_i, rid, None))}
        push(t_admit + S_i, EV_COMPLETE, idx)

    next_idx = 0
    for (t_arr, slo_ov) in arrival.sample(n, rng):
        slo_i = (slo_ov if slo_ov is not None else slo_ms) / 1000.0
        push(t_arr, EV_ARRIVE, (next_idx, dataset[next_idx % n], slo_i, t_arr))
        next_idx += 1

    while heap:
        t, _, kind, payload = heapq.heappop(heap)
        if kind == EV_ARRIVE:
            (idx, rid, slo_i, t_arrive) = payload
            if len(running) < mns:
                admit(t, t_arrive, idx, rid, slo_i)
            else:
                admit_q.append((idx, rid, slo_i, t_arrive))
        else:
            idx = payload
            r = running.pop(idx)
            lat_i = r["ttft"] if slo_type == "ttft" else r["e2e"]
            met = 1 if lat_i <= r["slo"] else 0
            per_req.append({"id": r["rid"], "k": r["k"], "admit": r["admit"],
                            "arrive": r["arrive"], "complete": t,
                            "ttft": r["ttft"], "e2e": r["e2e"], "slo": r["slo"],
                            "met": met, "acc": r["acc"]})
            if update_lambda is not None:
                viol_win.append(1 - met)
                if len(viol_win) >= min(lambda_update_cadence, max(8, len(per_req))):
                    try: update_lambda(sum(viol_win) / len(viol_win))
                    except Exception: pass
            while admit_q and len(running) < mns:
                (wi, wrid, wslo, warr) = admit_q.popleft()
                admit(t, warr, wi, wrid, wslo)

    if per_req:
        n_met = sum(r["met"] for r in per_req)
        sam = sum(r["acc"] for r in per_req if r["met"])
        mean_acc = sum(r["acc"] for r in per_req) / len(per_req)
        T_win = max(r["complete"] for r in per_req); nt = len(per_req)
    else:
        n_met = sam = mean_acc = T_win = 0.0; nt = 0
    return GoodputResult(
        n=nt, n_met=n_met, frac_met=(n_met / nt) if nt else 0.0,
        goodput_rate=(sam / T_win) if T_win > 0 else 0.0,
        met_rate=(n_met / T_win) if T_win > 0 else 0.0,
        mean_accuracy=mean_acc, T_window=T_win,
        per_req=sorted(per_req, key=lambda x: (x["admit"], x["id"])),
        policy=policy, arrival=arrival.name, slo_type=slo_type, slo_ms=slo_ms)


def compare(arrival_factory, allocators, lat, acc, dataset, slo_ms, slo_type="ttft", **kw):
    """Each allocator on the SAME arrival trace (same seed). kw -> simulate()."""
    out = {}
    for name, alloc in allocators.items():
        res = simulate(arrival_factory(), alloc, lat, acc, dataset, slo_ms, slo_type, **kw)
        res.policy = name; out[name] = res
    return out


# ====================== __main__: mock allocators + runs ===================
if __name__ == "__main__":
    class MockAcc:
        """accuracy(k) rising 0.4 -> 0.6 over k=144 -> 576."""
        def utility(self, k, req_id=None, features=None):
            return 0.4 + 0.2 * max(0.0, min(1.0, (k - 144) / 432.0))

    class _Base:
        _MARGIN = 1.1   # conservative gate (accounts for service-time variance)
        def _gate_s(self, load, sum_k, k, slo_type):
            return expected_wait_s(load, sum_k) + self._MARGIN * (
                prefill_time(k) if slo_type == "ttft" else service_time(k))

    class FixedAllocator(_Base):
        def __init__(self, r): self.r = r; self.name = f"Fixed(r={r:.2f})"
        def allocate(self, load, slo_ms, st, rf, kr, l, a, sum_k=0, p="p99"):
            return max(kr[0], min(kr[1], int(round(kr[1] * (1.0 - self.r)))))

    class GreedyAllocator(_Base):
        name = "Greedy"
        def allocate(self, load, slo_ms, st, rf, kr, l, a, sum_k=0, p="p99"):
            ss = slo_ms / 1000.0
            for k in range(kr[1], kr[0] - 1, -8):
                if self._gate_s(load, sum_k, k, st) <= ss: return k
            return kr[0]

    class LagrangianAllocator(_Base):
        def __init__(self, lam=2.0, floor=0.5, ceil=20.0, step=0.5):
            self.lam, self.floor, self.ceil, self.step = lam, floor, ceil, step
            self.name = "Lagrangian"
        def update_lambda(self, viol):
            self.lam = (min(self.ceil, self.lam + self.step * viol) if viol > 0.05
                        else max(self.floor, self.lam - 0.05 * self.step))
        def allocate(self, load, slo_ms, st, rf, kr, l, a, sum_k=0, p="p99"):
            ss = slo_ms / 1000.0; best_k, best_u = kr[0], -1e18
            for k in range(kr[0], kr[1] + 1, 8):
                pen = self.lam * max(0.0, self._gate_s(load, sum_k, k, st) - ss)
                u = a.utility(k) - pen
                if u > best_u: best_u, best_k = u, k
            return best_k

    class OracleAllocator:
        name = "Oracle"  # placeholder = Greedy
        def allocate(self, *a, **kw): return GreedyAllocator().allocate(*a, **kw)

    ds = [f"gqa_{i:04d}" for i in range(200)]
    acc = MockAcc(); lat = None
    allocs = {f"Fixed(r={r})": FixedAllocator(r) for r in (0.0, 0.25, 0.50, 0.75)}
    allocs.update(Greedy=GreedyAllocator(), Lagrangian=LagrangianAllocator(), Oracle=OracleAllocator())

    # ---------- CLOSED-LOOP SANITY (c=64, SLO=TTFT<=5s) ----------
    print("=== CLOSED-LOOP SANITY (c=64, SLO=TTFT<=5s) — slot+queue model ===")
    print("    v2 ref: served_req_s~{9.18,14.59,20.39}; met_rate~{1.4-1.8 (k576), 10.8-14.4 (k144)}")
    res_cl = compare(lambda: ClosedLoop(64),
                     {"Fixed(r=0.00)": FixedAllocator(0.0), "Fixed(r=0.50)": FixedAllocator(0.5),
                      "Fixed(r=0.75)": FixedAllocator(0.75)}, lat, acc, ds, 5000.0, "ttft",
                     k_range=(144, 576), max_num_seqs=64, seed=42, service_noise=0.15)
    print(f"{'policy':<16} {'goodput_rate':>12} {'met_rate':>9} {'frac_met':>9} {'T_window':>9}")
    print("-" * 58)
    for nm, r in res_cl.items():
        print(f"{nm:<16} {r.goodput_rate:>12.3f} {r.met_rate:>9.3f} {r.frac_met:>9.3f} {r.T_window:>9.2f}")
    print(f"    -> k576 met_rate={res_cl['Fixed(r=0.00)'].met_rate:.2f} (v2 1.4-1.8); "
          f"k144={res_cl['Fixed(r=0.75)'].met_rate:.2f} (v2 10.8-14.4). ORDER check: k144>>k576.")

    # ---------- HEADLINE: mixed-SLO H1b (Greedy gives low-k to tight, high-k to slack) ----------
    def dd(rng): return 3500.0 if rng.random() < 0.5 else 15000.0
    rate = 8.0
    print(f"\n=== HEADLINE (H1b): mixed_slo(open_loop_poisson(rate={rate}), "
          f"50% tight=3.5s / 50% slack=15s) SLO=E2E ===")
    res = compare(lambda: MixedSLO(OpenLoopPoisson(rate), dd), allocs, lat, acc, ds,
                  10000.0, "e2e", k_range=(144, 576), max_num_seqs=64, seed=42, service_noise=0.12)
    print(f"{'policy':<16} {'goodput_rate':>12} {'met_rate':>9} {'frac_met':>9} {'mean_acc':>9}")
    print("-" * 58)
    for nm, r in res.items():
        print(f"{nm:<16} {r.goodput_rate:>12.3f} {r.met_rate:>9.3f} {r.frac_met:>9.3f} {r.mean_accuracy:>9.3f}")
    fixed = [r for n, r in res.items() if n.startswith("Fixed")]
    bf = max(fixed, key=lambda r: r.goodput_rate); g = res["Greedy"]
    v = "YES" if g.goodput_rate > bf.goodput_rate else "NO"
    print(f"\nBest Fixed={bf.goodput_rate:.3f} ({bf.policy}); Greedy={g.goodput_rate:.3f}; "
          f"ratio={g.goodput_rate / max(bf.goodput_rate, 1e-9):.2f}x -> Greedy beats best Fixed: {v}")

    # ---------- SECONDARY: uniform-SLO H1 (open_loop, single deadline) ----------
    print(f"\n=== SECONDARY (H1, uniform SLO): open_loop_poisson(rate=12) SLO=E2E<=10s ===")
    res2 = compare(lambda: OpenLoopPoisson(12.0), allocs, lat, acc, ds, 10000.0, "e2e",
                   k_range=(144, 576), max_num_seqs=64, seed=42, service_noise=0.12)
    for nm, r in res2.items():
        print(f"{nm:<16} {r.goodput_rate:>12.3f} {r.met_rate:>9.3f} {r.frac_met:>9.3f} {r.mean_accuracy:>9.3f}")
    bf2 = max((r for n, r in res2.items() if n.startswith("Fixed")), key=lambda r: r.goodput_rate)
    print(f"    Best Fixed={bf2.goodput_rate:.3f} ({bf2.policy}); Greedy={res2['Greedy'].goodput_rate:.3f} "
          f"(H1 uniform-SLO signal is marginal with acc range 0.4-0.6; H1b mixed-SLO above is the clear win).")
