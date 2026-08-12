"""Validate the frontispiece Fig. 1 data bundle. Fail-closed."""
from __future__ import annotations
import json, sys, hashlib
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path("/media/disk2/YZX/research/vla")
DATA_DIR = ROOT / "drafts/figures/frontispiece_fig1/data"
FIG_PDF = ROOT / "drafts/figs/fig1.pdf"
FIG_PNG = ROOT / "drafts/figs/fig1.png"
REPORT = ROOT / "drafts/figures/frontispiece_fig1/validation_report.json"

errors, checks = [], []

def check(name, ok, detail=""):
    checks.append({"name": name, "ok": bool(ok), "detail": detail})
    if not ok:
        errors.append(f"{name}: {detail}")

mp = DATA_DIR / "manifest.json"
check("manifest_exists", mp.exists(), str(mp))
if not mp.exists(): sys.exit(1)
with mp.open() as f:
    M = json.load(f)

# 1. source image exists
img_path = DATA_DIR / M["image"]
check("source_image_exists", img_path.exists(), str(img_path))

# 2. mask exists
mask_path = DATA_DIR / "rbm_mask.npz"
check("mask_exists", mask_path.exists(), str(mask_path))
if mask_path.exists():
    npz = np.load(mask_path)
    check("mask_has_keep", "keep" in npz.files, str(npz.files))
    gw, gh = int(npz["unit_grid_hw"][0]), int(npz["unit_grid_hw"][1])
    side_dim = int(np.sqrt(npz["keep"].size))
    # keep is reshaped to (side, side) square grid for rendering
    check("mask_shape_match", npz["keep"].shape == (side_dim, side_dim),
          f"keep={npz['keep'].shape}, expected=({side_dim},{side_dim}), unit_grid=({gw},{gh})")
    n_kept = int(npz["keep"].sum())
    n_full = int(npz["n_units_full"])
    n_expected = round(n_full * M["keep_ratio"])
    check("mask_kept_count", n_kept == n_expected,
          f"kept={n_kept}, expected={n_expected} (full={n_full})")

# 3. all methods same ptid
ptids = [m["ptid"] for m in M["methods"]]
check("ptid_match", len(set(ptids)) == 1, f"ptids={ptids}")

# 4. RBM correct, both baselines wrong
rbm = M["methods"][0]
check("rbm_correct", rbm["correct"] is True, f"rbm correct={rbm['correct']}")
wrong = [m for m in M["methods"][1:] if not m["correct"]]
check("two_wrong_baselines", len(wrong) >= 2, f"wrong={len(wrong)}")

# 5. GT present
check("gt_present", len(M["ground_truth"]) > 0, str(M["ground_truth"]))

# 6. all answers present
for m in M["methods"]:
    check(f"answer_present_{m['key']}", len(m["answer"]) > 0, f"answer={m['answer'][:40]!r}")

# 7. evidence box inside image bounds
if img_path.exists():
    img = Image.open(img_path)
    w, h = img.size
    ev = M["evidence_bbox_source_px"]
    check("evidence_box_in_bounds",
          ev[0] >= 0 and ev[1] >= 0 and ev[2] <= w and ev[3] <= h,
          f"ev={ev}, image=({w},{h})")
    # shorter rendered side >= 80 px
    shorter = min(ev[2] - ev[0], ev[3] - ev[1])
    check("evidence_shorter_side_80px", shorter >= 80, f"shorter={shorter}")

# 8. PDF/PNG nonblank
check("pdf_exists", FIG_PDF.exists(), str(FIG_PDF))
check("png_exists", FIG_PNG.exists(), str(FIG_PNG))
if FIG_PDF.exists():
    check("pdf_nonblank", FIG_PDF.stat().st_size > 10000, f"size={FIG_PDF.stat().st_size}")
if FIG_PNG.exists():
    check("png_nonblank", FIG_PNG.stat().st_size > 10000, f"size={FIG_PNG.stat().st_size}")

# 9. provenance exists
check("provenance_exists", (DATA_DIR / "provenance.json").exists(), "")

report = {
    "n_checks": len(checks),
    "n_passed": sum(1 for c in checks if c["ok"]),
    "n_failed": len(errors),
    "checks": checks,
    "errors": errors,
}
with REPORT.open("w") as f:
    json.dump(report, f, indent=2)

print(f"Validation: {report['n_passed']}/{report['n_checks']} passed, {len(errors)} failed")
if errors:
    for e in errors: print(f"  FAIL: {e}")
    sys.exit(1)
print("OK: all checks passed")
