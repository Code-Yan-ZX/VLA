"""ElasticVis predictors: ``LatencyPred`` (per-request latency) + ``AccuracyTerm``
(accuracy as a function of visual-token budget k).

Fit from existing v2 probe data (per-request raw). See ``notes/elasticvis_design.md``
sections §2 (data map) and §5 (interface contract).

Latency model (per spec §5):
    latency_ms = α + β·num_running + γ·own_k + δ·sum_k + ε·(num_running · sum_k)
fit by OLS on the per-request ``ttft_ms`` and ``e2e_ms`` series SEPARATELY.
p99 is derived from p50 via a fitted p99/p50 multiplier (median across fit cells
of ``agg.{metric}_p99 / agg.{metric}_p50``).

Fit data (12 cells, EV-0):
  * ``runs/v2_p2/serial_c1_r{0,50,75}.json`` — c=1, clean isolated-request compute
    baseline (no queueing).
  * ``runs/v2_p2/batch_c{4,16,64}_r{0,50,75}.json`` — closed-loop sojourn time at
    concurrency c∈{4,16,64}, k∈{576,288,144}.
Excluded:
  * ``runs/v2_p2/batch_c1_*`` — queue-degenerate (200 reqs submitted to a c=1
    engine → sojourn dominated by ~100x queue wait) AND feature-collides with
    serial_c1 (identical (num_running=1, sum_k=k) features but ~100x labels),
    which would corrupt the OLS fit. Kept as held-out validation.
  * ``runs/v2_p0/*`` — all ``ttft_ms`` are NaN and there is no ``e2e_ms`` field
    (older probe format, n=100). Unusable for latency.
  * ``runs/v2_p3/*`` — cross-compressor replicates at c64; used only as held-out
    cross-compressor validation of the c64 prediction.

CAVEAT (closed-loop confound, spec §2): in all batch cells every request shares
the same k, so ``sum_k ≈ num_running · own_k`` and the ``δ·sum_k`` /
``ε·num_running·sum_k`` terms are near-collinear with ``num_running²·own_k``. The
6-term augmented model is identifiable in (n,k) but individual coefs are NOT
interpretable (only the joint prediction is). ``kv_occupancy`` is NOT
independently varied in EV-0 (it scales with n·k, perfectly confounded) → its
coefficient is fixed at 0.0 here and reserved for EV-1 calibration.

PREDICT() ROUTING (transparent to callers — allocator gate and sim completion-
time both call predict() with no special-casing):
  1. num_running ≤ 2  →  lookup(1, own_k)  (serial_c1 COMPUTE baseline ~430ms;
     kills the ~7000ms closed-loop sojourn blowup so the allocator sees SLO=SAFE
     in open-loop light spells and gives high-k there). Off-grid own_k → step 3.
  2. exact (num_running, own_k) grid cell  →  measured p50/p99 (zero error at
     the 9 batch cells; tightens closed_c64 sanity reproduction).
  3. otherwise  →  augmented regression (cell-level LOO R²=0.996 on batch fit).

Only 3 k points exist per image (k∈{576,288,144}). The simulator MUST use only
these k values for per-image replay, or the interpolated AGGREGATE curve for
intermediate k.

Standalone: ``python -m src.elasticvis.predictors`` (or ``python src/elasticvis/
predictors.py``) fits both terms, writes
``runs/elasticvis_ev0/latpred_coeffs.json`` + ``accterms_data.json`` +
``predictors_fit_report.md``, and prints a summary.
"""
from __future__ import annotations

import glob
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from scipy import linalg as _scipy_linalg  # type: ignore
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

# ----------------------------------------------------------------------------
# Constants & paths
# ----------------------------------------------------------------------------
N_NATIVE = 576  # LLaVA-1.5-7B native visual-token count (proxy selector)
REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS = REPO_ROOT / "runs"
OUT_DIR = RUNS / "elasticvis_ev0"
COEFFS_PATH = OUT_DIR / "latpred_coeffs.json"
ACCURACY_PATH = OUT_DIR / "accuracy.json"
REPORT_PATH = OUT_DIR / "predictors_fit_report.md"

# r ∈ {0, 0.5, 0.75}  →  k = round(576·(1−r)) ∈ {576, 288, 144}
#
# Delivered (PRIMARY) feature expansion — spec §7 "pick by fit error". Adds a
# 1/num_running term to the spec's 5-feature candidate: closed-loop sojourn is
# ∝ (backlog / num_running) × compute-per-wave, i.e. non-monotonic in n, which
# the pure linear form cannot capture (cell-level LOO R²=0.50 on batch-9 with
# the spec form vs 0.996 with +1/n). Fit on batch c{4,16,64} only (sojourn
# regime); serial_c1 is a separate compute regime (see caveats).
FEATURE_NAMES = ["ones", "num_running", "own_k", "sum_k",
                 "num_running_x_sum_k", "inv_num_running"]
COEF_LABELS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
LATForm = ("latency_ms = alpha + beta*num_running + gamma*own_k + delta*sum_k"
           " + eps*(num_running*sum_k) + zeta/num_running")
LATForm_SPEC = ("latency_ms = alpha + beta*num_running + gamma*own_k"
                " + delta*sum_k + eps*(num_running*sum_k)   [spec §5 candidate]")


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def _r_from_name_or_field(d: dict, path: Path) -> float:
    pr = d.get("pruning_rate")
    if pr is not None:
        return round(float(pr), 3)
    m = re.search(r"_r(\d+(?:\.\d+)?)", path.name)
    return round(float(m.group(1)) / 100.0, 3) if m else 0.0


def _k_from_r(r: float) -> int:
    return int(round(N_NATIVE * (1.0 - r)))


def _parse_cell(path: Path) -> dict:
    d = json.load(open(path))
    r = _r_from_name_or_field(d, path)
    k = _k_from_r(r)
    n = int(d.get("max_num_seqs", 1))
    return {
        "path": str(path),
        "name": path.stem,
        "r": r,
        "k": k,
        "n": n,
        "agg": d.get("agg", {}),
        "raw": d.get("raw", []),
    }


def load_fit_cells() -> list[dict]:
    """All 12 measured cells (serial_c1 × 3 r + batch_c{4,16,64} × 3 r).

    Used for the measured (n,k)→p50/p99 lookup table.
    """
    cells: list[dict] = []
    for suf in ("0", "50", "75"):
        p = RUNS / "v2_p2" / f"serial_c1_r{suf}.json"
        if p.exists():
            cells.append(_parse_cell(p))
    for c in (4, 16, 64):
        for suf in ("0", "50", "75"):
            p = RUNS / "v2_p2" / f"batch_c{c}_r{suf}.json"
            if p.exists():
                cells.append(_parse_cell(p))
    return cells


def load_param_fit_cells() -> list[dict]:
    """9 closed-loop sojourn cells (batch c{4,16,64} × 3 r) used for the
    parametric OLS fit. serial_c1 is excluded (different regime — isolated-
    request compute, not closed-loop sojourn)."""
    cells: list[dict] = []
    for c in (4, 16, 64):
        for suf in ("0", "50", "75"):
            p = RUNS / "v2_p2" / f"batch_c{c}_r{suf}.json"
            if p.exists():
                cells.append(_parse_cell(p))
    return cells


def load_heldout_cells() -> dict[str, list[dict]]:
    """Held-out cells (not used in fit) for validation reporting."""
    out: dict[str, list[dict]] = {"batch_c1": [], "p3_c64": []}
    for suf in ("0", "50", "75"):
        p = RUNS / "v2_p2" / f"batch_c1_r{suf}.json"
        if p.exists():
            out["batch_c1"].append(_parse_cell(p))
    for comp in ("proxy", "true_cls", "tome_merge", "random"):
        for suf in ("0", "50", "75"):
            p = RUNS / "v2_p3" / f"{comp}_c64_r{suf}.json"
            if p.exists():
                out["p3_c64"].append(_parse_cell(p))
    return out


# ----------------------------------------------------------------------------
# Design matrix / OLS
# ----------------------------------------------------------------------------
def _features(own_k: float, sum_k: float, num_running: float) -> np.ndarray:
    """6-feature augmented form (primary). Caller passes sum_k explicitly so the
    notional heterogeneous-batch terms (delta·sum_k, eps·n·sum_k) remain in the
    contract; in EV-0 closed-loop data sum_k = num_running·own_k so these are
    collinear with n²·own_k (multicollinearity caveat — only the PREDICTION is
    identifiable, not individual coefs). The 1/num_running term is what captures
    the closed-loop sojourn non-monotonicity."""
    n = float(num_running)
    return np.array([1.0, n, float(own_k), float(sum_k), n * float(sum_k),
                     1.0 / n if n > 0 else 0.0])


def _features_spec5(own_k: float, sum_k: float, num_running: float) -> np.ndarray:
    """Spec §5 candidate 5-feature form (no 1/n term). Kept for comparison only."""
    n = float(num_running)
    return np.array([1.0, n, float(own_k), float(sum_k), n * float(sum_k)])


def _build_xy(cells: list[dict], metric: str,
              feats_fn=_features) -> tuple[np.ndarray, np.ndarray]:
    X: list[list[float]] = []
    y: list[float] = []
    for c in cells:
        n, k = c["n"], c["k"]
        sum_k = n * k
        feats = feats_fn(k, sum_k, n)
        for r in c["raw"]:
            v = r.get(metric)
            if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)) and v > 0:
                X.append(feats.tolist())
                y.append(float(v))
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


def _fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    if X.shape[0] < X.shape[1]:
        raise ValueError(f"too few rows ({X.shape[0]}) for {X.shape[1]} features")
    if _HAVE_SCIPY:
        coef, _res, _rnk, _sv = _scipy_linalg.lstsq(X, y)
    else:
        coef, _res, _rnk, _sv = np.linalg.lstsq(X, y, rcond=None)
    return coef


def _metrics(y: np.ndarray, yhat: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    yhat = np.clip(np.asarray(yhat, dtype=float), 1.0, None)  # latency ≥ ~1ms, no neg / div0
    if len(y) == 0:
        return {"r2": float("nan"), "mape": float("nan"), "n": 0,
                "mean_actual": float("nan"), "mean_pred": float("nan")}
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mape = float(np.mean(np.abs((yhat - y) / np.clip(y, 1.0, None))) * 100.0)
    return {"r2": float(r2), "mape": mape, "n": int(len(y)),
            "mean_actual": float(y.mean()), "mean_pred": float(yhat.mean())}


def _loo_cv(cells: list[dict], metric: str,
            feats_fn=_features) -> tuple[dict, dict]:
    """Leave-one-cell-out CV on per-request raw. Returns (overall, per_cell).

    NOTE: per-request R² is dominated by within-cell queue-position variance in
    the closed-loop batch cells (a request's ttft spans ~1-100x within one cell
    depending on its arrival position). The allocator-relevant signal is the
    cell-level (n,k)→p50 surface, scored by :func:`_loo_cv_celllevel`.
    """
    if len(cells) < 2:
        raise ValueError("need ≥2 cells for LOO CV")
    all_y: list[float] = []
    all_yhat: list[float] = []
    per_cell: dict[str, dict] = {}
    for i in range(len(cells)):
        train = [c for j, c in enumerate(cells) if j != i]
        test = cells[i]
        Xtr, ytr = _build_xy(train, metric, feats_fn)
        Xte, yte = _build_xy([test], metric, feats_fn)
        coef = _fit(Xtr, ytr)
        yhat = Xte @ coef
        all_y.extend(yte.tolist())
        all_yhat.extend(yhat.tolist())
        per_cell[test["name"]] = {**_metrics(yte, yhat),
                                  "k": test["k"], "n": test["n"]}
    overall = _metrics(np.asarray(all_y), np.asarray(all_yhat))
    return overall, per_cell


def _loo_cv_celllevel(cells: list[dict], metric: str, agg_key: str,
                      feats_fn=_features) -> tuple[dict, dict]:
    """Leave-one-cell-out CV at the CELL level: refit on n−1 cells, predict the
    held-out cell's (n,k) central value, compare to its agg p50 (or mean).

    This is the PRIMARY fit-quality metric for the allocator (which gates on
    predicted latency at given load, not on per-request scatter).
    """
    if len(cells) < 2:
        raise ValueError("need ≥2 cells for LOO CV")
    actuals: list[float] = []
    preds: list[float] = []
    per: dict[str, dict] = {}
    for i in range(len(cells)):
        train = [c for j, c in enumerate(cells) if j != i]
        test = cells[i]
        Xtr, ytr = _build_xy(train, metric, feats_fn)
        coef = _fit(Xtr, ytr)
        n, k = test["n"], test["k"]
        pred = max(0.0, float(feats_fn(k, n * k, n) @ coef))
        a = test["agg"]
        actual = a.get(agg_key)
        if not isinstance(actual, (int, float)) or actual <= 0:
            vals = [r.get(metric) for r in test["raw"]
                    if isinstance(r.get(metric), (int, float))
                    and not (isinstance(r.get(metric), float) and math.isnan(r.get(metric)))
                    and r.get(metric) > 0]
            actual = float(np.mean(vals)) if vals else float("nan")
        actuals.append(float(actual))
        preds.append(pred)
        per[test["name"]] = {"k": k, "n": n, "actual_p50": float(actual),
                             "pred": float(pred),
                             "err_pct": float(abs(pred - actual) / actual * 100.0)}
    overall = _metrics(np.asarray(actuals), np.asarray(preds))
    return overall, per


def cell_metrics_lookup(cells: list[dict]) -> dict:
    """Measured per-(n,k) central-tendency lookup from agg fields.

    The simulator should prefer these measured values at the 9 (n,k) grid
    points and use ``LatencyPred.predict`` only for interpolation/extrapolation
    to intermediate loads (the parametric model's per-request fit is weak — see
    report).
    """
    out: dict[str, dict] = {}
    for c in cells:
        a = c["agg"]
        key = f"n{c['n']}_k{c['k']}"
        entry = {"n": c["n"], "k": c["k"], "cell": c["name"]}
        for f in ("ttft_ms_p50", "ttft_ms_p99", "e2e_ms_p50", "e2e_ms_p99"):
            v = a.get(f)
            if isinstance(v, (int, float)):
                entry[f] = float(v)
        # mean fallback from raw
        for metric, key_mean in (("ttft_ms", "ttft_mean"), ("e2e_ms", "e2e_mean")):
            vals = [r.get(metric) for r in c["raw"]
                    if isinstance(r.get(metric), (int, float))
                    and not (isinstance(r.get(metric), float) and math.isnan(r.get(metric)))
                    and r.get(metric) > 0]
            entry[key_mean] = float(np.mean(vals)) if vals else None
        out[key] = entry
    return out


def _p99_multiplier(cells: list[dict], p50_key: str, p99_key: str) -> float:
    rats: list[float] = []
    for c in cells:
        a = c["agg"]
        p50, p99 = a.get(p50_key), a.get(p99_key)
        if isinstance(p50, (int, float)) and isinstance(p99, (int, float)) and p50 > 0:
            rats.append(p99 / p50)
    return float(np.median(rats)) if rats else 1.5


# ----------------------------------------------------------------------------
# Deliverable 1 — LatencyPred
# ----------------------------------------------------------------------------
@dataclass
class LatencyEstimate:
    """Predicted per-request latency (ms). p99 derived from p50 via fitted ratio."""
    ttft_p50: float
    ttft_p99: float
    e2e_p50: float
    e2e_p99: float


class LatencyPred:
    """Per-request latency predictor fit from v2 probe data.

    Args:
        fit_artifact: either a dict (from :func:`fit_latency`) or a path to the
            persisted ``latpred_coeffs.json``.
    """

    def __init__(self, fit_artifact):
        if isinstance(fit_artifact, (str, Path)):
            with open(fit_artifact) as fh:
                fit_artifact = json.load(fh)
        fa = fit_artifact
        self.ttft_coef = np.asarray(fa["ttft_coef"], dtype=float)
        self.e2e_coef = np.asarray(fa["e2e_coef"], dtype=float)
        self.ttft_p99_mult = float(fa["ttft_p99_mult"])
        self.e2e_p99_mult = float(fa["e2e_p99_mult"])
        self.feature_names = list(fa["feature_names"])
        self.kv_coef = float(fa.get("kv_occupancy_coef", 0.0))
        self.fit_form = fa.get("fit_form", LATForm)
        self.cv = {"ttft": fa.get("ttft_cv_perrequest"),
                   "e2e": fa.get("e2e_cv_perrequest")}
        self.cv_celllevel = {"ttft": fa.get("ttft_cv_celllevel"),
                             "e2e": fa.get("e2e_cv_celllevel")}
        # measured (n,k)→p50/p99 lookup; preferred by the sim at grid points.
        self.cell_lookup = fa.get("cell_lookup", {})

    def predict(self, own_k: int, sum_k: int, num_running: int,
                kv_occupancy: float = 0.0) -> LatencyEstimate:
        """Predict a request's latency (ms). Signature per spec §5; body routes
        through measured lookup first, regression last.

        Routing (transparent to callers — allocator gate + sim completion-time
        both call this and need no special-casing):

        1. LOW-LOAD (num_running ≤ 2): return ``lookup(1, own_k)`` if it exists.
           This is the serial_c1 COMPUTE baseline (~430ms), NOT the closed-loop
           sojourn. Without this, the regression (fit on batch sojourn) would
           return ~7000ms at c=1 and the allocator would wrongly prune in open-
           loop light spells, killing the H1 mechanism. Falls through to step 3
           if own_k is off the measured grid.
        2. EXACT GRID CELL: ``lookup(num_running, own_k)``; if it exists, return
           the measured p50/p99 (zero error at the 9 batch cells + tightens the
           closed_c64 sanity reproduction).
        3. OFF-GRID / FALLBACK: the augmented regression
           ``α + β·n + γ·own_k + δ·sum_k + ε·n·sum_k + ζ/n`` (cell-level LOO
           R²=0.996 on the batch fit cells).

        Args:
            own_k: this request's visual-token budget k_i.
            sum_k: Σ k_j over all currently-running requests (incl. this one if
                added). In closed-loop EV-0 cells sum_k ≈ num_running · own_k.
            num_running: concurrent request count the new request would join.
            kv_occupancy: reserved for EV-1 (coef=0 in EV-0; not independently
                varied — perfectly confounded with n·k in the probe data).
        """
        # Step 1 — low-load: use c=1 compute baseline (serial_c1).
        if num_running <= 2:
            est = self.lookup(1, own_k)
            if est is not None:
                return est
            # own_k off-grid at low load → fall through to regression (step 3).
        # Step 2 — exact measured (num_running, own_k) grid cell.
        est = self.lookup(num_running, own_k)
        if est is not None:
            return est
        # Step 3 — augmented regression fallback (off-grid n or k).
        x = _features(own_k, sum_k, num_running)
        ttft = max(0.0, float(x @ self.ttft_coef))
        e2e = max(0.0, float(x @ self.e2e_coef))
        return LatencyEstimate(
            ttft_p50=ttft,
            ttft_p99=ttft * self.ttft_p99_mult,
            e2e_p50=e2e,
            e2e_p99=e2e * self.e2e_p99_mult,
        )

    def lookup(self, num_running: int, k: int) -> LatencyEstimate | None:
        """Return the MEASURED p50/p99 at an exact (n,k) grid point if present,
        else None. The simulator should call this first (measured > parametric)
        and fall back to :meth:`predict` for off-grid (n,k).

        Grid points (EV-0): (n,k) ∈ {1,4,16,64} × {576,288,144}.
        """
        key = f"n{int(num_running)}_k{int(k)}"
        e = self.cell_lookup.get(key)
        if not e:
            return None
        return LatencyEstimate(
            ttft_p50=float(e.get("ttft_ms_p50", float("nan"))),
            ttft_p99=float(e.get("ttft_ms_p99", float("nan"))),
            e2e_p50=float(e.get("e2e_ms_p50", float("nan"))),
            e2e_p99=float(e.get("e2e_ms_p99", float("nan"))),
        )


def fit_latency(param_cells: list[dict] | None = None,
                lookup_cells: list[dict] | None = None) -> dict:
    """Fit ttft_ms and e2e_ms OLS models on batch c{4,16,64} (closed-loop
    sojourn regime) + LOO CV + p99 multipliers + measured (n,k) lookup.

    Args:
        param_cells: cells used for the parametric OLS fit (default: batch-9).
        lookup_cells: cells used for the measured (n,k)→p50/p99 lookup table
            (default: all 12, incl. serial_c1 for the c=1 compute baseline).
    """
    param_cells = param_cells if param_cells is not None else load_param_fit_cells()
    lookup_cells = lookup_cells if lookup_cells is not None else load_fit_cells()
    if not param_cells:
        raise RuntimeError("no param-fit cells found under runs/v2_p2/")
    X_t, y_t = _build_xy(param_cells, "ttft_ms", _features)
    X_e, y_e = _build_xy(param_cells, "e2e_ms", _features)
    ttft_coef = _fit(X_t, y_t)
    e2e_coef = _fit(X_e, y_e)
    ttft_full = _metrics(y_t, X_t @ ttft_coef)
    e2e_full = _metrics(y_e, X_e @ e2e_coef)
    # PRIMARY CV: cell-level with the augmented form on batch-9
    ttft_cv_pr_overall, _ = _loo_cv(param_cells, "ttft_ms", _features)
    e2e_cv_pr_overall, _ = _loo_cv(param_cells, "e2e_ms", _features)
    ttft_cv_cell_ov, ttft_cv_cell_per = _loo_cv_celllevel(param_cells, "ttft_ms",
                                                          "ttft_ms_p50", _features)
    e2e_cv_cell_ov, e2e_cv_cell_per = _loo_cv_celllevel(param_cells, "e2e_ms",
                                                        "e2e_ms_p50", _features)
    # COMPARISON: spec §5 5-feature form (no 1/n) on batch-9 — shows why we augment
    t_sp_ov, _ = _loo_cv_celllevel(param_cells, "ttft_ms", "ttft_ms_p50", _features_spec5)
    e_sp_ov, _ = _loo_cv_celllevel(param_cells, "e2e_ms", "e2e_ms_p50", _features_spec5)
    ttft_mult = _p99_multiplier(param_cells, "ttft_ms_p50", "ttft_ms_p99")
    e2e_mult = _p99_multiplier(param_cells, "e2e_ms_p50", "e2e_ms_p99")
    lookup = cell_metrics_lookup(lookup_cells)

    return {
        "feature_names": FEATURE_NAMES,
        "coef_labels": COEF_LABELS,
        "fit_form": LATForm,
        "fit_form_spec_candidate": LATForm_SPEC,
        "ttft_coef": ttft_coef.tolist(),
        "e2e_coef": e2e_coef.tolist(),
        "ttft_full_fit": ttft_full,
        "e2e_full_fit": e2e_full,
        "ttft_cv_perrequest": ttft_cv_pr_overall,
        "e2e_cv_perrequest": e2e_cv_pr_overall,
        "ttft_cv_celllevel": ttft_cv_cell_ov,
        "e2e_cv_celllevel": e2e_cv_cell_ov,
        "ttft_cv_celllevel_per_cell": ttft_cv_cell_per,
        "e2e_cv_celllevel_per_cell": e2e_cv_cell_per,
        # spec-form-5 comparison (batch-9)
        "spec5_ttft_cv_celllevel": t_sp_ov,
        "spec5_e2e_cv_celllevel": e_sp_ov,
        "ttft_p99_mult": ttft_mult,
        "e2e_p99_mult": e2e_mult,
        "cell_lookup": lookup,
        "kv_occupancy_coef": 0.0,
        "kv_note": "kv_occupancy not independently varied in EV-0 (confounded "
                   "with n·k); coef=0 reserved for EV-1 calibration.",
        "native_n": N_NATIVE,
        "param_fit_cells": [c["name"] for c in param_cells],
        "lookup_cells": [c["name"] for c in lookup_cells],
        "n_param_cells": len(param_cells),
        "n_lookup_cells": len(lookup_cells),
        "n_points_ttft": int(len(y_t)),
        "n_points_e2e": int(len(y_e)),
        "multicollinearity_note": "in EV-0 sum_k=num_running·own_k, so the "
            "delta/epsilon/own_k columns are near-collinear; individual coefs "
            "are NOT interpretable, only the joint prediction is.",
    }


def _eval_cells_against(fit_artifact: dict, cells: list[dict]) -> dict:
    """Predict every request in `cells` with the FULL-fit model and score."""
    if not cells:
        return {}
    pred = LatencyPred(fit_artifact)
    out: dict[str, dict] = {}
    for metric, coef_key in (("ttft_ms", "ttft_coef"), ("e2e_ms", "e2e_coef")):
        coef = np.asarray(fit_artifact[coef_key], dtype=float)
        ys: list[float] = []
        yhats: list[float] = []
        per: dict[str, dict] = {}
        for c in cells:
            n, k = c["n"], c["k"]
            feats = _features(k, n * k, n)
            yte, yhat = [], []
            for r in c["raw"]:
                v = r.get(metric)
                if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)) and v > 0:
                    yte.append(float(v))
                    yhat.append(float(feats @ coef))
            ys.extend(yte)
            yhats.extend(yhat)
            per[c["name"]] = {**_metrics(np.asarray(yte), np.asarray(yhat)),
                              "k": k, "n": n}
        out[metric] = {"overall": _metrics(np.asarray(ys), np.asarray(yhats)),
                       "per_cell": per}
    # silence unused-pred warning while keeping the convenience object
    _ = pred
    return out


# ----------------------------------------------------------------------------
# Deliverable 2 — AccuracyTerm
# ----------------------------------------------------------------------------
class AccuracyTerm:
    """Accuracy(k): aggregate curve (H1 system-signal allocator) + per-image
    lookup (simulator replay / oracle).

    Only 3 k points are measured per image (k∈{576,288,144}). For k not in that
    set, :meth:`utility` returns the linearly-interpolated AGGREGATE curve
    (NOT a per-image estimate — per-image interpolation is not supported).
    """

    def __init__(self, agg_curve: dict, per_image: dict):
        self._agg = {int(k): float(v) for k, v in agg_curve.items()}
        self._per = {
            iid: {int(kk): bool(vv) for kk, vv in kk_dict.items()}
            for iid, kk_dict in per_image.items()
        }

    def utility(self, k: int, req_id: str | None = None,
                features: dict | None = None) -> float:
        """Accuracy in [0,1] for budget k.

        - If ``req_id`` is in per_image AND k is one of the 3 measured k points
          for it → return float(measured correct).
        - Else → linear-interpolated aggregate curve at k (clamped to [k_min,
          k_max] endpoints; constant beyond).
        """
        k = int(k)
        if req_id is not None and req_id in self._per and k in self._per[req_id]:
            return float(self._per[req_id][k])
        return self._interp_agg(k)

    def aggregate_curve(self) -> dict[int, float]:
        return dict(self._agg)

    def per_image_curve(self, req_id: str) -> dict[int, bool]:
        return dict(self._per.get(req_id, {}))

    def _interp_agg(self, k: int) -> float:
        ks = sorted(self._agg)
        if not ks:
            return 0.0
        if k in self._agg:
            return self._agg[k]
        if k <= ks[0]:
            return self._agg[ks[0]]
        if k >= ks[-1]:
            return self._agg[ks[-1]]
        for a, b in zip(ks, ks[1:]):
            if a <= k <= b:
                t = (k - a) / (b - a)
                return self._agg[a] * (1.0 - t) + self._agg[b] * t
        return self._agg[ks[-1]]


def build_accuracy_term() -> tuple[AccuracyTerm, dict, dict]:
    """Pools all v2_p2 r-cells (serial_c1 + batch_c{1,4,16,64}) for a denoised
    aggregate accuracy(k) and a majority-vote per-image lookup.

    Accuracy is concurrency-independent in expectation (same model, same images);
    per-cell differences are generation sampling noise. Pooling across the 5
    c-cells × 3 r stabilizes both the aggregate means and the per-image labels.
    """
    files = sorted(glob.glob(str(RUNS / "v2_p2" / "serial_c1_r*.json")))
    files += sorted(glob.glob(str(RUNS / "v2_p2" / "batch_c*_r*.json")))
    if not files:
        raise RuntimeError("no v2_p2 r-cells found for accuracy term")
    by_k: dict[int, list[int]] = defaultdict(list)
    per_img_k: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for f in files:
        c = _parse_cell(Path(f))
        k = c["k"]
        for r in c["raw"]:
            cor = int(bool(r.get("correct")))
            by_k[k].append(cor)
            per_img_k[r["id"]][k].append(cor)
    agg_curve = {k: float(np.mean(v)) for k, v in by_k.items()}
    per_image: dict[str, dict[int, bool]] = {}
    for iid, kk in per_img_k.items():
        per_image[iid] = {k: (sum(vs) >= len(vs) / 2.0) for k, vs in kk.items()}
    return AccuracyTerm(agg_curve, per_image), agg_curve, per_image


def accuracy_stats(agg_curve: dict, per_image: dict) -> dict:
    ks = sorted(agg_curve)
    flips = sum(
        1 for iid, kk in per_image.items()
        if ks and ks[0] in kk and ks[-1] in kk and kk[ks[0]] != kk[ks[-1]]
    )
    measured = sum(1 for kk in per_image.values() for _ in kk)
    return {
        "agg_curve": {str(k): round(v, 4) for k, v in sorted(agg_curve.items())},
        "n_images": len(per_image),
        "k_points": ks,
        "flips_r0_to_r75": flips,
        "flip_rate": round(flips / len(per_image), 4) if per_image else None,
        "per_image_measurements": measured,
    }


# ----------------------------------------------------------------------------
# Persistence + report
# ----------------------------------------------------------------------------
def save_artifacts(lat_artifact: dict, acc_agg: dict, acc_per_image: dict) -> None:
    """Persist ``latpred_coeffs.json`` and ``accuracy.json``.

    accuracy.json schema (string keys, only the 3 measured k)::

        {"agg_curve": {"144": acc, "288": acc, "576": acc},
         "per_image": {id: {"144": bool, "288": bool, "576": bool}, ...},
         "native_n": 576, "k_points": [144, 288, 576]}
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(COEFFS_PATH, "w") as fh:
        json.dump(lat_artifact, fh, indent=2)
    ks = sorted(int(k) for k in acc_agg.keys())
    acc_dump = {
        "agg_curve": {str(int(k)): float(v) for k, v in acc_agg.items()},
        "per_image": {
            iid: {str(int(k)): bool(v) for k, v in kk.items()}
            for iid, kk in acc_per_image.items()
        },
        "native_n": N_NATIVE,
        "k_points": ks,
        "note": "per-image correct is majority vote across all v2_p2 r-cells "
                "(concurrency-independent); only 3 k points per image.",
    }
    with open(ACCURACY_PATH, "w") as fh:
        json.dump(acc_dump, fh, indent=2)


# ----------------------------------------------------------------------------
# Factory functions (for run_ev0.py / driver integration)
# ----------------------------------------------------------------------------
def load_latpred(coeffs_path=COEFFS_PATH) -> "LatencyPred":
    """Load a fitted ``LatencyPred`` from ``latpred_coeffs.json``.

    No GPU / vLLM / torch import — pure-json + numpy. Standalone-importable.
    """
    return LatencyPred(Path(coeffs_path))


def load_accuracy(path=ACCURACY_PATH) -> "AccuracyTerm":
    """Reconstruct an ``AccuracyTerm`` from ``accuracy.json`` (no refit).

    Reads the schema ``{"agg_curve": {str(k): acc}, "per_image": {id: {str(k):
    bool}}}`` and coerces keys back to int / values to float/bool.
    """
    with open(path) as fh:
        d = json.load(fh)
    agg = {int(k): float(v) for k, v in d["agg_curve"].items()}
    per_image = {
        iid: {int(k): bool(v) for k, v in kk.items()}
        for iid, kk in d["per_image"].items()
    }
    return AccuracyTerm(agg, per_image)


def write_report(lat_artifact: dict, acc_stats: dict, heldout: dict) -> str:
    lines: list[str] = []
    lines.append("# ElasticVis EV-0 — predictors fit report\n")
    lines.append(f"Generated by `src/elasticvis/predictors.py`. PRIMARY form:\n```\n{LATForm}\n```\n")

    lines.append("## 1. LatencyPred\n")
    lines.append(f"- parametric fit cells (n={lat_artifact['n_param_cells']}, "
                 f"closed-loop sojourn regime): "
                 + ", ".join(lat_artifact["param_fit_cells"]) + "\n")
    lines.append(f"- lookup cells (n={lat_artifact['n_lookup_cells']}, incl. "
                 f"serial_c1 for the c=1 compute baseline): "
                 + ", ".join(lat_artifact["lookup_cells"]) + "\n")
    lines.append(f"- per-request points in param fit: ttft={lat_artifact['n_points_ttft']}, "
                 f"e2e={lat_artifact['n_points_e2e']}\n")
    lines.append("### Fitted coefficients (augmented form, 6 features)\n")
    lines.append("> Multicollinearity: in EV-0 sum_k = num_running·own_k, so "
                 "β/γ/δ/ε are NOT individually interpretable — only the joint "
                 "prediction is meaningful.")
    lines.append("| term | ttft_ms | e2e_ms |")
    lines.append("|---|---|---|")
    for lbl, t, e in zip(lat_artifact["coef_labels"],
                         lat_artifact["ttft_coef"], lat_artifact["e2e_coef"]):
        lines.append(f"| {lbl} | {t:.4g} | {e:.4g} |")
    lines.append("\n### Fit quality (full data, in-sample, per-request raw)\n")
    for metric in ("ttft", "e2e"):
        m = lat_artifact[f"{metric}_full_fit"]
        lines.append(f"- **{metric}**: R²={m['r2']:.4f}  MAPE={m['mape']:.2f}%  "
                     f"mean_actual={m['mean_actual']:.1f}ms  mean_pred={m['mean_pred']:.1f}ms")

    lines.append("\n### Leave-one-cell-out CV — per-request raw (9 batch folds)\n")
    lines.append("> CAVEAT: per-request R² is low because within-cell queue-"
                 "position variance dominates (a request's ttft spans ~1-100x "
                 "within one closed-loop cell depending on arrival position). "
                 "This is NOT the allocator-relevant metric — see cell-level CV below.")
    for metric in ("ttft", "e2e"):
        cv = lat_artifact[f"{metric}_cv_perrequest"]
        lines.append(f"- **{metric}**: R²={cv['r2']:.4f}  MAPE={cv['mape']:.2f}%  "
                     f"(n={cv['n']} per-request points)")

    lines.append("\n### Leave-one-cell-out CV — CELL-LEVEL (PRIMARY metric)\n")
    lines.append("Refit on 8 batch cells, predict held-out cell's (n,k)→p50, "
                 "compare to its measured agg p50. This is the quality bar for "
                 "the allocator's gating decision.")
    for metric in ("ttft", "e2e"):
        cv = lat_artifact[f"{metric}_cv_celllevel"]
        lines.append(f"- **{metric}** (augmented 6-feat): R²={cv['r2']:.4f}  "
                     f"MAPE={cv['mape']:.2f}%  (n={cv['n']} cells)")
    lines.append("\nComparison — spec §5 5-feature form (no 1/n) on the same "
                 "batch-9 cells (justifies the 1/n augmentation, spec §7 "
                 "'pick by fit error'):")
    lines.append(f"- ttft spec-form: R²={lat_artifact['spec5_ttft_cv_celllevel']['r2']:.4f}  "
                 f"MAPE={lat_artifact['spec5_ttft_cv_celllevel']['mape']:.2f}%")
    lines.append(f"- e2e  spec-form: R²={lat_artifact['spec5_e2e_cv_celllevel']['r2']:.4f}  "
                 f"MAPE={lat_artifact['spec5_e2e_cv_celllevel']['mape']:.2f}%")
    lines.append("\n| cell | k | n | actual_p50 | pred | err% |")
    lines.append("|---|---|---|---|---|---|")
    for name, m in sorted(lat_artifact["ttft_cv_celllevel_per_cell"].items(),
                          key=lambda kv: (kv[1]["n"], kv[1]["k"])):
        lines.append(f"| {name} | {m['k']} | {m['n']} | {m['actual_p50']:.0f} | "
                     f"{m['pred']:.0f} | {m['err_pct']:.1f} |")
    lines.append("\n### p50 → p99 multipliers (median of agg across fit cells)\n")
    lines.append(f"- ttft p99/p50 = {lat_artifact['ttft_p99_mult']:.4f}")
    lines.append(f"- e2e  p99/p50 = {lat_artifact['e2e_p99_mult']:.4f}")
    lines.append("- Per-request raw supports a mean/p50 OLS fit; p99 is derived "
                 "via this multiplier (no per-request tail fit).\n")

    lines.append("## 2. Held-out validation (full-fit model applied to non-fit cells)\n")
    for tag, d in heldout.items():
        lines.append(f"### {tag}\n")
        for metric in ("ttft_ms", "e2e_ms"):
            mm = d.get(metric, {})
            ov = mm.get("overall", {})
            lines.append(f"- {metric}: R²={ov.get('r2', float('nan')):.4f}  "
                         f"MAPE={ov.get('mape', float('nan')):.2f}%  (n={ov.get('n', 0)})")

    lines.append("\n## 3. AccuracyTerm\n")
    lines.append(f"- aggregate accuracy(k): "
                 + ", ".join(f"k={k}→{v:.4f}" for k, v in sorted(acc_stats["agg_curve"].items(),
                                                                  key=lambda kv: int(kv[0]))))
    lines.append(f"- per-image: {acc_stats['n_images']} ids, "
                 f"3 k points each ({acc_stats['per_image_measurements']} measurements)")
    lines.append(f"- flips r0→r75 (majority-vote per-image): "
                 f"{acc_stats['flips_r0_to_r75']}/{acc_stats['n_images']} "
                 f"({acc_stats['flip_rate']:.2%})")
    lines.append("- raw iso-c flip counts (sanity, not majority vote): "
                 "serial_c1=46, batch_c1=46, batch_c4=44, batch_c16=46, batch_c64=47 "
                 "(spec §2 said 53; measured 44–47).")

    lines.append("\n## 4. Measured (n,k)→p50/p99 lookup (simulator-preferred)\n")
    lines.append("Prefer `LatPred.lookup(n,k)` at these grid points; use "
                 "`predict()` only for off-grid (n,k) interpolation/extrapolation.")
    lines.append("| key | n | k | ttft_p50 | ttft_p99 | e2e_p50 | e2e_p99 |")
    lines.append("|---|---|---|---|---|---|---|")
    for key, e in sorted(lat_artifact["cell_lookup"].items(),
                         key=lambda kv: (kv[1]["n"], kv[1]["k"])):
        lines.append(f"| {key} | {e['n']} | {e['k']} | "
                     f"{e.get('ttft_ms_p50', float('nan')):.0f} | "
                     f"{e.get('ttft_ms_p99', float('nan')):.0f} | "
                     f"{e.get('e2e_ms_p50', float('nan')):.0f} | "
                     f"{e.get('e2e_ms_p99', float('nan')):.0f} |")

    lines.append("\n## 5. Caveats / blockers for downstream\n")
    lines.append("- **Regime mismatch (serial_c1 vs batch) — HANDLED in predict():** "
                 "serial_c1 (c=1) measures isolated-request COMPUTE latency (~430ms); "
                 "batch cells measure closed-loop SOJOURN (queue+compute, ~10s). The "
                 "regression (fit on batch sojourn) alone would over-predict c=1 at "
                 "~7000ms. `predict()` AUTO-ROUTES: (1) num_running≤2 → `lookup(1,k)` "
                 "(serial_c1 compute baseline); (2) exact (n,k) grid cell → measured; "
                 "(3) off-grid → regression. So callers (allocator gate, sim "
                 "completion-time) need NO special-casing. EV-1 must still measure "
                 "open-loop compute latency at intermediate n (2, 8, 32) to de-confound.")
    lines.append("- **Closed-loop confound**: in batch cells sum_k ≈ num_running·own_k, "
                 "so δ·sum_k and ε·n·sum_k are near-collinear with n²·own_k. The model "
                 "is effectively latency(n, own_k) in EV-0; do NOT interpret β/γ/δ/ε "
                 "as independent physical effects. EV-1 must vary sum_k independently "
                 "(heterogeneous-k batches) to de-confound.")
    lines.append("- **kv_occupancy coef = 0**: not independently varied (confounded "
                 "with n·k). Reserved for EV-1.")
    lines.append("- **batch_c1 excluded from fit**: queue-degenerate (200 reqs at c=1 → "
                 "~42s sojourn) AND feature-collides with serial_c1. Reported as held-out "
                 "only; expected to be massively under-predicted (and it should be — an "
                 "allocator never creates that regime).")
    lines.append("- **v2_p0 unusable**: all ttft_ms are NaN, no e2e_ms (older probe, "
                 "n=100). c=12 latency is NOT measured → LatPred unverified at c=12 "
                 "(interpolates between c4 and c16).")
    lines.append("- **Only 3 k points/image (coarse)**: per-image interpolation is NOT "
                 "supported; intermediate k uses the interpolated AGGREGATE curve only. "
                 "Simulator MUST use k∈{576,288,144} for per-request replay.")
    lines.append("- **single engine/selector**: LLaVA-1.5-7B + proxy selector only. "
                 "Cross-arch (Qwen3-VL) and cross-compressor generalization unverified "
                 "(p3 held-out gives a first c64 cross-compressor sanity check).")
    lines.append(f"\nArtifacts: `{COEFFS_PATH.relative_to(REPO_ROOT)}`, "
                 f"`{ACCURACY_PATH.relative_to(REPO_ROOT)}`.\n")
    text = "\n".join(lines)
    REPORT_PATH.write_text(text)
    return text


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def _summary_lines(lat_artifact: dict, acc_stats: dict) -> list[str]:
    out = ["[ElasticVis predictors] fit complete."]
    out.append(f"  form: {LATForm}")
    out.append("  ttft: per-req R²={r2:.3f} (MAPE {mape:.0f}%) | CELL-LEVEL R²={cr2:.3f} (MAPE {cmape:.0f}%)".format(
        r2=lat_artifact["ttft_cv_perrequest"]["r2"], mape=lat_artifact["ttft_cv_perrequest"]["mape"],
        cr2=lat_artifact["ttft_cv_celllevel"]["r2"], cmape=lat_artifact["ttft_cv_celllevel"]["mape"]))
    out.append("  e2e : per-req R²={r2:.3f} (MAPE {mape:.0f}%) | CELL-LEVEL R²={cr2:.3f} (MAPE {cmape:.0f}%)".format(
        r2=lat_artifact["e2e_cv_perrequest"]["r2"], mape=lat_artifact["e2e_cv_perrequest"]["mape"],
        cr2=lat_artifact["e2e_cv_celllevel"]["r2"], cmape=lat_artifact["e2e_cv_celllevel"]["mape"]))
    out.append("  (spec 5-feat form, batch-9 cell-level: ttft R²={:.3f}, e2e R²={:.3f})".format(
        lat_artifact["spec5_ttft_cv_celllevel"]["r2"],
        lat_artifact["spec5_e2e_cv_celllevel"]["r2"]))
    out.append("  ttft_coef=[{t}]  e2e_coef=[{e}]".format(
        t=", ".join(f"{x:.4g}" for x in lat_artifact["ttft_coef"]),
        e=", ".join(f"{x:.4g}" for x in lat_artifact["e2e_coef"])))
    out.append("  p99/p50 multipliers: ttft={:.3f}  e2e={:.3f}".format(
        lat_artifact["ttft_p99_mult"], lat_artifact["e2e_p99_mult"]))
    out.append("  accuracy(k): " + ", ".join(
        f"{k}→{v:.4f}" for k, v in sorted(acc_stats["agg_curve"].items(), key=lambda kv: int(kv[0]))))
    out.append("  per-image flips r0→r75: {}/{} ({:.0%})".format(
        acc_stats["flips_r0_to_r75"], acc_stats["n_images"], acc_stats["flip_rate"]))
    out.append(f"  artifacts: {COEFFS_PATH}  +  {ACCURACY_PATH}  +  {REPORT_PATH}")
    return out


def main() -> None:
    lat = fit_latency()                       # batch-9 param + all-12 lookup
    heldout_raw = load_heldout_cells()
    heldout_eval = {
        tag: _eval_cells_against(lat, cs) for tag, cs in heldout_raw.items()
    }
    acc_term, agg_curve, per_image = build_accuracy_term()
    acc_stats = accuracy_stats(agg_curve, per_image)
    save_artifacts(lat, agg_curve, per_image)
    text = write_report(lat, acc_stats, heldout_eval)
    for line in _summary_lines(lat, acc_stats):
        print(line)
    print("---- report preview (head) ----")
    print("\n".join(text.splitlines()[:14]))


if __name__ == "__main__":
    main()
