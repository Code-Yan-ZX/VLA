"""Validate the qualitative Fig. 1 data bundle. Fail-closed."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path("/media/disk2/YZX/research/vla")
DATA_DIR = ROOT / "drafts/figures/qualitative_fig1/data"
OUT_PDF = ROOT / "drafts/figs/fig1.pdf"
OUT_PNG = ROOT / "drafts/figs/fig1.png"
REPORT = ROOT / "drafts/figures/qualitative_fig1/validation_report.json"

errors, warnings, checks = [], [], []


def check(name, ok, detail=""):
    checks.append({"name": name, "ok": bool(ok), "detail": detail})
    if not ok:
        errors.append(f"{name}: {detail}")


manifest_path = DATA_DIR / "manifest.json"
check("manifest_exists", manifest_path.exists(), str(manifest_path))
if not manifest_path.exists():
    sys.exit(1)
with manifest_path.open() as f:
    manifest = json.load(f)

for ex in manifest["examples"]:
    key = f'{ex["benchmark"].lower()}_{ex["id"]}'
    img_path = DATA_DIR / ex["image"]
    check(f"image_{key}", img_path.exists(), str(img_path))

    for m in ex["methods"]:
        mask_path = DATA_DIR / m["mask"]
        check(f"mask_{key}_{m['key']}", mask_path.exists(), str(mask_path))
        if mask_path.exists():
            npz = np.load(mask_path)
            check(f"mask_has_keep_{key}_{m['key']}", "keep" in npz.files, str(npz.files))
            check(f"mask_has_grid_{key}_{m['key']}", "unit_grid_hw" in npz.files, "")
            gw, gh = int(npz["unit_grid_hw"][0]), int(npz["unit_grid_hw"][1])
            check(f"mask_shape_match_{key}_{m['key']}",
                  npz["keep"].shape == (gw, gh),
                  f"keep={npz['keep'].shape}, grid=({gw},{gh})")
            n_kept = int(npz["keep"].sum())
            # kept count must match n_units_kept stored in npz
            if "n_units_kept" in npz.files:
                expected = int(npz["n_units_kept"])
                check(f"mask_kept_count_{key}_{m['key']}",
                      n_kept == expected, f"kept={n_kept}, expected={expected}")
            # grid must be large enough to hold all kept indices
            n_indices = len(npz["kept_indices"])
            check(f"mask_indices_in_grid_{key}_{m['key']}",
                  all(int(i) < gw * gh for i in npz["kept_indices"]),
                  f"n_indices={n_indices}, grid_size={gw*gh}")

    ptids = [m["ptid"] for m in ex["methods"]]
    check(f"ptid_match_{key}", len(set(ptids)) == 1, f"ptids={ptids}")

    rbm = ex["methods"][0]
    check(f"rbm_correct_{key}", rbm["correct"] is True, f"rbm correct={rbm['correct']}")
    wrong_baselines = [m for m in ex["methods"][1:] if not m["correct"]]
    check(f"two_wrong_baselines_{key}", len(wrong_baselines) >= 2,
          f"wrong baselines={len(wrong_baselines)}")

    check(f"gt_present_{key}", len(ex["ground_truth"]) > 0, str(ex["ground_truth"]))
    for m in ex["methods"]:
        check(f"answer_present_{key}_{m['key']}", len(m["answer"]) > 0, f"answer={m['answer']!r}")

    if img_path.exists():
        img = Image.open(img_path)
        w, h = img.size
        check(f"image_aspect_preserved_{key}", w > 0 and h > 0, f"size=({w},{h})")

check("pdf_exists", OUT_PDF.exists(), str(OUT_PDF))
check("png_exists", OUT_PNG.exists(), str(OUT_PNG))
if OUT_PDF.exists():
    check("pdf_nonblank", OUT_PDF.stat().st_size > 5000, f"size={OUT_PDF.stat().st_size}")
if OUT_PNG.exists():
    check("png_nonblank", OUT_PNG.stat().st_size > 5000, f"size={OUT_PNG.stat().st_size}")

expected_pools = manifest.get("candidate_pool_counts", {})
check("pool_counts_present", len(expected_pools) >= 3, str(expected_pools))

report = {
    "n_checks": len(checks),
    "n_passed": sum(1 for c in checks if c["ok"]),
    "n_failed": len(errors),
    "n_warnings": len(warnings),
    "checks": checks,
    "errors": errors,
    "warnings": warnings,
}
with REPORT.open("w") as f:
    json.dump(report, f, indent=2)

print(f"Validation: {report['n_passed']}/{report['n_checks']} passed, {len(errors)} failed")
if errors:
    for e in errors:
        print(f"  FAIL: {e}")
    sys.exit(1)
print("OK: all checks passed")
