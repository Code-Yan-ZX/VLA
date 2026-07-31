#!/usr/bin/env python3
"""smoke_test_cpu.py — CPU-only smoke test for the real-data L2 figure pipeline.

Builds a tiny SYNTHETIC dataset in /tmp/real_l2_smoke/ (NPZ matching the exact
capture schema + schema-compatible sidecar JSON + a PIL gradient image — NO
real input image and NO model), then exercises the pipeline stages on it:

  (a) validate_real_l2.py --skip-repeatability  -> all applicable gates PASS;
      the gate-7 re-capture gate is SKIPPED gracefully for the synthetic model.
      (run_all.sh NEVER passes --skip-repeatability; real runs are unaffected.)
  (b) gen_fig1_rbm.py measured mode on the synthetic NPZ -> renders.
  (c) render_compare.py on the synthetic data -> renders the 4-panel figure
      (verified: 7.09 in wide raster, all four quadrants carry content).
  (d) NEGATIVE tests: corrupted NPZ (wrong score length / wrong keep-mask
      count) must make validate exit nonzero.
  (e) runner formula assertions (pins the exact paper scoring formulas):
      _score_tokens(x, "l2") == per-row L2 norm;
      _score_units(f, "l2")  == mean over the 4 patch L2 norms per unit
      (imported from src/v3_premerger/v3_premerger_runner.py).

Runner dry checks: the repo was grepped (tests/, scripts/, src/) for existing
dry/unit checks of _score_units/_score_tokens — none exist, so block (e) below
is the canonical CPU pin of those formulas.

CPU-only: CUDA_VISIBLE_DEVICES is forced empty before any torch/vllm import
(inherited by subprocesses); the test asserts torch.cuda is unavailable.

Exit nonzero on ANY failure; prints a summary table. All outputs go to
/tmp/real_l2_smoke/.

CLI:  python drafts/figures/real_data_pipeline/scripts/smoke_test_cpu.py
"""
from __future__ import annotations

import os

# ---- CPU-only enforcement BEFORE any torch/vllm import (also inherited by
#      every subprocess this test spawns) ---- #
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import hashlib
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
ROOT = PIPELINE_DIR.parents[2]
VALIDATE = SCRIPT_DIR / "validate_real_l2.py"
COMPARE = SCRIPT_DIR / "render_compare.py"
GEN_FIG1 = ROOT / "drafts" / "figures" / "gen_fig1_rbm.py"

WORK = Path("/tmp/real_l2_smoke")
KEEP_RATIO = 0.25
PATCH = 16
MERGE = 2
UH, UW = 8, 12                      # unit grid -> N = 96
H, W = UH * MERGE, UW * MERGE       # patch grid 16 x 24
N = UH * UW
K = max(1, round(N * KEEP_RATIO))   # 24

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"[smoke] {'PASS' if ok else 'FAIL'}  {name}"
          + (f"  ({detail})" if detail else ""), flush=True)


def run_cmd(name: str, cmd: list[str], expect_rc: int = 0,
            log_name: str | None = None) -> subprocess.CompletedProcess:
    log_path = WORK / f"{log_name or name}.log"
    with open(log_path, "w") as lf:
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=lf,
                              stderr=subprocess.STDOUT)
    ok = proc.returncode == expect_rc
    check(name, ok,
          f"rc={proc.returncode} expected={expect_rc} log={log_path.name}")
    return proc


def deterministic_topk(scores: np.ndarray, k: int):
    """score DESC, ties -> ascending unit index (same rule as validator)."""
    n = scores.shape[0]
    order = np.lexsort((np.arange(n), -scores.astype(np.float64)))
    keep = np.zeros(n, dtype=bool)
    keep[order[:k]] = True
    rank = np.empty(n, dtype=np.int32)
    rank[order] = np.arange(1, n + 1, dtype=np.int32)
    return keep, rank


def build_synthetic_dataset() -> tuple[Path, Path]:
    """Returns (data_dir, inputs_dir) with synthetic NPZ + sidecar + image."""
    data_dir = WORK / "data"
    inputs_dir = WORK / "inputs"
    data_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    pre = np.round(rng.uniform(1.0, 10.0, N), 3).astype(np.float32)
    post = np.round(rng.uniform(1.0, 10.0, N), 3).astype(np.float32)
    # inject EXACT ties so the deterministic tie-break (score-desc/index-asc)
    # is genuinely exercised by gates 4/5
    pre[5] = pre[7] = np.float32(6.123)
    pre[40] = pre[41] = pre[42] = np.float32(2.5)
    post[10] = post[20] = np.float32(9.5)

    pre_keep, pre_rank = deterministic_topk(pre, K)
    post_keep, post_rank = deterministic_topk(post, K)

    sid = "synth_sample"
    np.savez(data_dir / f"{sid}.npz",
             pre_l2=pre, post_l2=post,
             pre_keep=pre_keep.astype(bool), post_keep=post_keep.astype(bool),
             pre_rank=pre_rank.astype(np.int32),
             post_rank=post_rank.astype(np.int32),
             grid_thw=np.asarray([1, H, W], dtype=np.int32),
             unit_grid_hw=np.asarray([UH, UW], dtype=np.int32))

    # tiny synthetic input image: RGB gradient (NOT any real input image)
    from PIL import Image
    iw, ih = W * PATCH // 4, H * PATCH // 4        # 96 x 64 px gradient
    gx = np.linspace(0, 255, iw, dtype=np.uint8)
    gy = np.linspace(0, 255, ih, dtype=np.uint8)
    img = np.zeros((ih, iw, 3), dtype=np.uint8)
    img[..., 0] = gx[None, :]
    img[..., 1] = gy[:, None]
    img[..., 2] = 128
    img_name = "synth_gradient.png"
    Image.fromarray(img).save(inputs_dir / img_name)

    dummy = WORK / "dummy.bin"
    dummy.write_bytes(b"synthetic-test-dummy-content")
    sha = hashlib.sha256(dummy.read_bytes()).hexdigest()

    sidecar = {
        "sample_id": sid,
        "original_filename": img_name,
        "sha256_original": sha,
        "model_family": "synthetic",
        "model_id": "synthetic-test",
        "resolved_revision": "0" * 40,
        "processor_class": "SyntheticProcessor",
        "model_class": "SyntheticModel",
        "visual_class": "SyntheticVisual",
        "package_versions": {"numpy": np.__version__},
        "repo_git_commit": "smoke-test",
        "dtype": "float32",
        "device": "cpu (synthetic)",
        "seed": 0,
        "max_pixels": 0,
        "keep_ratio": KEEP_RATIO,
        "k": K,
        "n_units": N,
        "patch_size": PATCH,
        "spatial_merge_size": MERGE,
        "processor_resized_image_size_px": {"h_px": H * PATCH, "w_px": W * PATCH},
        "grid_thw": [1, H, W],
        "unit_grid_hw": [UH, UW],
        "pre_hook_module": "synthetic (n/a)",
        "post_hook_definition": "synthetic (n/a)",
        "score_definitions": {
            "pre": "pre: mean over 4 patch-feature L2 norms per unit",
            "post": "post: L2 norm of concatenated row per unit"},
        "tie_break_rule": ("sort by score descending; ties broken by ascending "
                           "unit index (stable deterministic top-k); rank 1 = "
                           "largest score"),
        "scoring_helpers": "synthetic data built with the same deterministic "
                           "top-k rule validated by gates 4/5",
        "hooks_no_numeric_change": "synthetic (no model run)",
        "capture_prompt": "n/a (synthetic)",
        "max_generated_tokens": 0,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "capture_command": "synthetic-smoke (no model)",
    }
    with open(data_dir / f"{sid}.json", "w") as f:
        json.dump(sidecar, f, indent=2)
    return data_dir, inputs_dir


def build_corrupt_dirs() -> tuple[Path, Path]:
    """Two corrupted variants of the synthetic NPZ; validate must reject both."""
    good = np.load(WORK / "data" / "synth_sample.npz")
    sidecar = json.loads((WORK / "data" / "synth_sample.json").read_text())

    # (1) wrong score length: post_l2 truncated by 1
    d1 = WORK / "corrupt_len"
    d1.mkdir(parents=True, exist_ok=True)
    np.savez(d1 / "synth_len.npz",
             pre_l2=good["pre_l2"], post_l2=good["post_l2"][:-1],
             pre_keep=good["pre_keep"], post_keep=good["post_keep"],
             pre_rank=good["pre_rank"], post_rank=good["post_rank"],
             grid_thw=good["grid_thw"], unit_grid_hw=good["unit_grid_hw"])
    s1 = dict(sidecar, sample_id="synth_len")
    (d1 / "synth_len.json").write_text(json.dumps(s1))

    # (2) keep-mask count off: one extra True in pre_keep
    d2 = WORK / "corrupt_mask"
    d2.mkdir(parents=True, exist_ok=True)
    bad_keep = good["pre_keep"].copy()
    flip = int(np.flatnonzero(~bad_keep)[0])
    bad_keep[flip] = True                      # k+1 Trues now
    np.savez(d2 / "synth_mask.npz",
             pre_l2=good["pre_l2"], post_l2=good["post_l2"],
             pre_keep=bad_keep, post_keep=good["post_keep"],
             pre_rank=good["pre_rank"], post_rank=good["post_rank"],
             grid_thw=good["grid_thw"], unit_grid_hw=good["unit_grid_hw"])
    s2 = dict(sidecar, sample_id="synth_mask")
    (d2 / "synth_mask.json").write_text(json.dumps(s2))
    return d1, d2


def gate7_skipped_gracefully(report: dict) -> bool:
    rep = report.get("repeatability", {})
    return bool(rep.get("pass") and "skipped" in str(rep.get("detail", "")))


def four_panels_rendered(png_path: Path) -> tuple[bool, str]:
    """compare canvas is 1x4 panels: each vertical quadrant must carry content."""
    from PIL import Image
    arr = np.asarray(Image.open(png_path).convert("RGB")).astype(np.float32)
    h, w, _ = arr.shape
    ok_w = abs(w - 7.09 * 300) <= 3              # 250? no: compare dpi default 300
    ink = ~((arr > 248).all(axis=-1))
    quads = []
    for i in range(4):
        q = ink[:, i * w // 4:(i + 1) * w // 4]
        quads.append(float(q.mean()))
    ok_panels = all(f > 0.005 for f in quads)
    return (bool(ok_w and ok_panels),
            f"width_px={w} quad_ink_fracs={['%.4f' % f for f in quads]}")


def runner_formula_assertions() -> tuple[bool, str]:
    """Pin the exact paper scoring formulas on hand-computed tensors."""
    import torch
    sys.path.insert(0, str(ROOT / "src" / "v3_premerger"))
    from v3_premerger_runner import _score_tokens, _score_units

    # l2 tokens == per-row L2 norm
    tok = torch.tensor([[3.0, 4.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    got_t = _score_tokens(tok, "l2")
    exp_t = torch.tensor([5.0, 1.0, 0.0])
    ok_t = torch.allclose(got_t, exp_t, atol=1e-6)

    # l2 units == MEAN over the 4 patch-feature L2 norms per unit
    feats = torch.tensor([
        [[3.0, 4.0], [6.0, 8.0], [0.0, 0.0], [0.0, 0.0]],   # (5+10+0+0)/4
        [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],   # all sqrt(2)
    ])
    got_u = _score_units(feats, "l2")
    exp_u = torch.tensor([(5.0 + 10.0 + 0.0 + 0.0) / 4.0, math.sqrt(2.0)])
    ok_u = torch.allclose(got_u, exp_u, atol=1e-6)
    return (bool(ok_t and ok_u),
            f"tokens={got_t.tolist()} units={[round(x, 6) for x in got_u.tolist()]}")


def main() -> int:
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    py = sys.executable

    import torch
    check("cpu_only", not torch.cuda.is_available(),
          "CUDA_VISIBLE_DEVICES='' honored; torch.cuda unavailable")

    # ---- build synthetic dataset ---- #
    data_dir, inputs_dir = build_synthetic_dataset()
    check("synthetic_dataset_built",
          (data_dir / "synth_sample.npz").is_file()
          and (data_dir / "synth_sample.json").is_file()
          and (inputs_dir / "synth_gradient.png").is_file(),
          f"N={N} k={K} grid=({H},{W})")

    # ---- (a) validate (gate-7 skipped gracefully for synthetic model) ---- #
    proc = run_cmd("a_validate_synthetic",
                   [py, str(VALIDATE), "--data-dir", str(data_dir),
                    "--keep-ratio", str(KEEP_RATIO), "--skip-repeatability"])
    report = {}
    rep_path = data_dir / "validation_report.json"
    if rep_path.is_file():
        report = json.loads(rep_path.read_text())
    if report:
        s0 = report["samples"][0]
        gates = s0["gates"]
        check("a_all_gates_pass",
              bool(report.get("all_passed")) and s0.get("sample_passed")
              and all(g["pass"] for g in gates.values())
              and report.get("hook_check", {}).get("pass", False),
              f"gates={sum(1 for g in gates.values() if g['pass'])}/"
              f"{len(gates)} hook={report.get('hook_check', {}).get('pass')}")
        check("a_gate7_skipped_gracefully", gate7_skipped_gracefully(report),
              f"repeatability.detail={report.get('repeatability', {}).get('detail')}")
    else:
        check("a_all_gates_pass", False, "no validation_report.json written")
        check("a_gate7_skipped_gracefully", False, "no report")

    # ---- (b) gen_fig1 measured mode on synthetic NPZ ---- #
    out_fig1 = WORK / "out" / "fig1"
    run_cmd("b_gen_fig1_measured",
            [py, str(GEN_FIG1), "--scores-npz", str(data_dir / "synth_sample.npz"),
             "--metadata-json", str(data_dir / "synth_sample.json"),
             "--inputs-dir", str(inputs_dir), "--outdir", str(out_fig1),
             "--validation-report", str(data_dir / "validation_report.json")])
    check("b_fig1_outputs_exist",
          (out_fig1 / "fig1_synth_sample.pdf").is_file()
          and (out_fig1 / "fig1_synth_sample.png").is_file())

    # ---- (c) render_compare on synthetic data (4 panels) ---- #
    out_cmp = WORK / "out" / "compare"
    run_cmd("c_render_compare",
            [py, str(COMPARE), "--data-dir", str(data_dir),
             "--inputs-dir", str(inputs_dir), "--outdir", str(out_cmp)])
    cmp_png = out_cmp / "compare_synth_sample.png"
    check("c_compare_outputs_exist",
          cmp_png.is_file() and (out_cmp / "compare_synth_sample.pdf").is_file())
    if cmp_png.is_file():
        ok, detail = four_panels_rendered(cmp_png)
        check("c_four_panels_rendered", ok, detail)
    else:
        check("c_four_panels_rendered", False, "no png")
    csum = data_dir / "comparison_summary.json"
    if csum.is_file():
        cs = json.loads(csum.read_text())
        check("c_fastv_status",
              cs.get("fastv_status_global") == "skipped_missing_question"
              and cs.get("n_rendered") == 1 and cs.get("n_failed") == 0,
              f"n_rendered={cs.get('n_rendered')} fastv={cs.get('fastv_status_global')}")
    else:
        check("c_fastv_status", False, "no comparison_summary.json")

    # ---- (d) negative tests: corrupted NPZ must fail validation ---- #
    d_len, d_mask = build_corrupt_dirs()
    run_cmd("d_neg_wrong_length_rejected",
            [py, str(VALIDATE), "--data-dir", str(d_len),
             "--keep-ratio", str(KEEP_RATIO), "--skip-repeatability"],
            expect_rc=1)
    run_cmd("d_neg_mask_count_rejected",
            [py, str(VALIDATE), "--data-dir", str(d_mask),
             "--keep-ratio", str(KEEP_RATIO), "--skip-repeatability"],
            expect_rc=1)

    # ---- (e) runner scoring-formula assertions (exact paper formulas) ---- #
    try:
        ok, detail = runner_formula_assertions()
        check("e_runner_score_formulas", ok, detail)
    except Exception as exc:                    # import/assert failure
        check("e_runner_score_formulas", False, f"{type(exc).__name__}: {exc}")

    # ---- summary ---- #
    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    n_fail = len(RESULTS) - n_pass
    print("\n" + "=" * 62)
    print(f"SMOKE SUMMARY: {n_pass}/{len(RESULTS)} passed, {n_fail} failed")
    print(f"work dir: {WORK}")
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"  FAILED: {name}  ({detail})")
    print("RESULT:", "ALL PASS" if n_fail == 0 else "FAILURES PRESENT")
    print("=" * 62)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
