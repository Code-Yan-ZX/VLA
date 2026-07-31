#!/usr/bin/env bash
# run_all.sh — orchestrator for the real-data L2 figure pipeline.
#
# Stages (canonical order; GPU stages run serially — one A40):
#   capture       capture_real_l2.py with --skip-existing (RESUME: skips samples
#                 whose NPZ+JSON already exist in data/; if all exist the model
#                 is NEVER loaded — plotting-only reruns never touch the model)
#   validate      validate_real_l2.py — 8 gates incl. gate-7 GPU re-capture of
#                 the first sample. NEVER passed --skip-repeatability here.
#   render_fig1   gen_fig1_rbm.py --batch-dir (fig1_* x N + contact sheet) AND
#                 render_compare.py (compare_* x N); asserts every expected PDF
#                 exists (gen_fig1 batch exits 0 even on per-sample failures)
#   render_fig2   render_fig2.py  (deterministic; honors SOURCE_DATE_EPOCH)
#   render_fig3   render_fig3.py  (deterministic; honors SOURCE_DATE_EPOCH)
#   verify_pdf    run_visual_qa.py --all-final over ALL 23 final PDFs
#                 (fig1_* x10, contact sheet, compare_* x10, fig2, fig3):
#                 1 page, width 7.09in +/-0.01, blank + edge-clipping
#                 heuristics at 250 dpi; exits nonzero on any failure.
#
# Stage selection (run from the repo root):
#   ./run_all.sh                        run ALL stages in order
#   ./run_all.sh validate render_fig1   run ONLY the named stages (canonical
#                                       order is preserved; ordering
#                                       dependencies are the caller's problem)
#   ./run_all.sh --from validate        run validate and every stage after it
#   ./run_all.sh --help                 usage
#
# Determinism: SOURCE_DATE_EPOCH is exported (default 0) and recorded; all
# renderers pin it for PDF metadata. Every run appends/rewrites
# data/run_all_manifest.json (git HEAD, package versions, exact commands,
# per-stage status; relative paths only, no secrets).
set -euo pipefail

# ---- repo root (4 levels up from this script) + run everything from there -- #
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${ROOT}"

# ---- environment ----------------------------------------------------------- #
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate qwen3vl_clean

# ---- deterministic rendering (recorded in run_all_manifest.json) ----------- #
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-0}"

PIPE="drafts/figures/real_data_pipeline"
SCRIPTS="${PIPE}/scripts"
DATA="${PIPE}/data"
OUT="${PIPE}/outputs"
INPUTS="${PIPE}/inputs"

ALL_STAGES=(capture validate render_fig1 render_fig2 render_fig3 verify_pdf)

usage() {
  cat <<EOF
usage: ./run_all.sh [STAGE ...] | --from STAGE | --help

stages (canonical order): ${ALL_STAGES[*]}

  (no args)          run all stages
  STAGE ...          run only the named stages (canonical order preserved)
  --from STAGE       run STAGE and every stage after it
  --help             this text

examples:
  ./run_all.sh
  ./run_all.sh validate render_fig1
  ./run_all.sh --from render_fig2
EOF
}

# ---- arg parsing ----------------------------------------------------------- #
REQ=()
FROM=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --from)    FROM="${2:---from needs a stage name}"; shift 2 ;;
    --from=*)  FROM="${1#*=}"; shift ;;
    *)         REQ+=("$1"); shift ;;
  esac
done

is_stage() { local s; for s in "${ALL_STAGES[@]}"; do [[ "${s}" == "$1" ]] && return 0; done; return 1; }

SELECTED=()
if [[ -n "${FROM}" ]]; then
  is_stage "${FROM}" || { echo "run_all: unknown stage '${FROM}'" >&2; usage >&2; exit 2; }
  found=0
  for s in "${ALL_STAGES[@]}"; do
    [[ "${s}" == "${FROM}" ]] && found=1
    [[ ${found} -eq 1 ]] && SELECTED+=("${s}")
  done
elif [[ ${#REQ[@]} -gt 0 ]]; then
  for r in "${REQ[@]}"; do
    is_stage "${r}" || { echo "run_all: unknown stage '${r}'" >&2; usage >&2; exit 2; }
  done
  for s in "${ALL_STAGES[@]}"; do        # canonical order regardless of arg order
    for r in "${REQ[@]}"; do [[ "${s}" == "${r}" ]] && SELECTED+=("${s}"); done
  done
else
  SELECTED=("${ALL_STAGES[@]}")
fi

# ---- bookkeeping (consumed by write_manifest on exit) ---------------------- #
WORK="$(mktemp -d /tmp/run_all.XXXXXX)"
CMDF="${WORK}/commands.tsv"     # stage<TAB>command
STF="${WORK}/status.tsv"        # stage<TAB>status
: > "${CMDF}"; : > "${STF}"
CURRENT_STAGE=""
RC=0

run() {                           # record the exact command, then execute it
  # NOTE: errexit is suppressed inside `if run_stage ...` conditions, so every
  # call site must propagate rc explicitly (`run ... || return $?`).
  printf '%s\t%s\n' "${CURRENT_STAGE}" "$*" >> "${CMDF}"
  echo "+ $*"
  "$@"
  local rc=$?
  [[ ${rc} -ne 0 ]] && echo "[run_all] command FAILED (rc=${rc}): $*" >&2
  return ${rc}
}

run_stage() {
  CURRENT_STAGE="$1"
  local fn="stage_$1"
  echo
  echo "==================== stage: $1 ===================="
  if "${fn}"; then
    printf '%s\tok\n' "$1" >> "${STF}"
  else
    printf '%s\tfailed\n' "$1" >> "${STF}"
    return 1
  fi
}

write_manifest() {
  local git_head
  git_head="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  GIT_HEAD="${git_head}" SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH}" \
  STAGES_SELECTED="$(IFS=,; echo "${SELECTED[*]}")" \
  python - "${DATA}/run_all_manifest.json" "${CMDF}" "${STF}" <<'PY' || true
import json, os, subprocess, sys
from datetime import datetime, timezone
from importlib import metadata as im

out, cmdf, stf = sys.argv[1], sys.argv[2], sys.argv[3]
sel = [s for s in os.environ.get("STAGES_SELECTED", "").split(",") if s]

def ver(p):
    try:
        return im.version(p)
    except Exception:
        return None

cmds, sts = [], {}
with open(cmdf) as f:
    for line in f:
        if "\t" in line:
            stage, cmd = line.rstrip("\n").split("\t", 1)
            cmds.append({"stage": stage, "command": cmd})
with open(stf) as f:
    for line in f:
        if "\t" in line:
            stage, st = line.rstrip("\n").split("\t", 1)
            sts[stage] = st

try:
    pop = subprocess.run(["pdfinfo", "-v"], capture_output=True,
                         text=True).stderr.splitlines()[0].strip()
except Exception:
    pop = "unknown"

man = {
    "pipeline": "real_data_pipeline_run_all",
    "orchestrator": "drafts/figures/real_data_pipeline/scripts/run_all.sh",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "git_head": os.environ.get("GIT_HEAD", ""),
    "source_date_epoch": os.environ.get("SOURCE_DATE_EPOCH", ""),
    "conda_env": "qwen3vl_clean",
    "python": sys.version.split()[0],
    "package_versions": {p: ver(p) for p in
                         ("numpy", "matplotlib", "scipy", "torch", "pillow",
                          "vllm", "transformers")},
    "poppler": pop,
    "stages_available": ["capture", "validate", "render_fig1", "render_fig2",
                         "render_fig3", "verify_pdf"],
    "stages_selected": sel,
    "stage_status": sts,
    "commands": cmds,
    "notes": [
        "all commands executed from the repo root (relative paths only)",
        "capture runs with --skip-existing (resume; no model load if all exist)",
        "validate always runs the gate-7 GPU re-capture (never --skip-repeatability)",
        "render stages are deterministic for a fixed SOURCE_DATE_EPOCH",
        "verify_pdf = run_visual_qa.py --all-final (23 final PDFs)",
    ],
}
with open(out, "w") as f:
    json.dump(man, f, indent=2, ensure_ascii=False)
print(f"[run_all] manifest -> {out}")
PY
  rm -rf "${WORK}"
}
trap write_manifest EXIT

# ---- stages ---------------------------------------------------------------- #
stage_capture() {
  run python "${SCRIPTS}/capture_real_l2.py" \
      --input-dir "${INPUTS}" --output-dir "${DATA}" \
      --model-family qwen3vl --model-id Qwen/Qwen3-VL-8B-Instruct \
      --keep-ratio 0.25 --max-pixels 1500000 --seed 0 --skip-existing || return $?
}

stage_validate() {
  # gate-7 re-capture loads the model once (serial GPU use) — intentional;
  # this stage must NEVER pass --skip-repeatability.
  run python "${SCRIPTS}/validate_real_l2.py" --data-dir "${DATA}" \
      --keep-ratio 0.25 || return $?
}

stage_render_fig1() {
  run python drafts/figures/gen_fig1_rbm.py \
      --batch-dir "${DATA}" --inputs-dir "${INPUTS}" --outdir "${OUT}" || return $?
  run python "${SCRIPTS}/render_compare.py" \
      --data-dir "${DATA}" --inputs-dir "${INPUTS}" --outdir "${OUT}" \
      --validation-report "${DATA}/validation_report.json" || return $?
  # gen_fig1 batch exits 0 even on per-sample failure -> assert expected files
  local n=0 missing=0 sid f
  for f in "${DATA}"/*.npz; do
    sid="$(basename "${f}" .npz)"
    n=$((n + 1))
    [[ -f "${OUT}/fig1_${sid}.pdf" && -f "${OUT}/fig1_${sid}.png" ]] ||
      { echo "MISSING fig1 outputs for ${sid}"; missing=$((missing + 1)); }
    [[ -f "${OUT}/compare_${sid}.pdf" && -f "${OUT}/compare_${sid}.png" ]] ||
      { echo "MISSING compare outputs for ${sid}"; missing=$((missing + 1)); }
  done
  [[ -f "${OUT}/fig1_contact_sheet.pdf" && -f "${OUT}/fig1_contact_sheet.png" ]] ||
    { echo "MISSING fig1_contact_sheet"; missing=$((missing + 1)); }
  # render_compare's own summary must report zero failures / full render count
  local csum="${DATA}/comparison_summary.json"
  run python - "${csum}" "${n}" <<'PY' || return $?
import json, sys
cs = json.load(open(sys.argv[1]))
n = int(sys.argv[2])
assert cs.get("n_failed", 1) == 0, f"comparison_summary n_failed={cs.get('n_failed')}"
assert cs.get("n_rendered") == n, \
    f"comparison_summary n_rendered={cs.get('n_rendered')} != {n}"
print(f"[render_fig1] comparison_summary: n_rendered={cs['n_rendered']} n_failed=0")
PY
  echo "[render_fig1] ${n} samples, missing outputs: ${missing}"
  [[ ${missing} -eq 0 ]]
}

stage_render_fig2() {
  run python "${SCRIPTS}/render_fig2.py" \
      --values "${DATA}/fig2_values.json" --outdir "${OUT}" || return $?
}

stage_render_fig3() {
  run python "${SCRIPTS}/render_fig3.py" \
      --values "${DATA}/fig3_values.json" --outdir "${OUT}" || return $?
}

stage_verify_pdf() {
  run python "${SCRIPTS}/run_visual_qa.py" --all-final \
      --outdir "${OUT}" --scratch /tmp/qa/all_final || return $?
}

# ---- main ------------------------------------------------------------------ #
echo "[run_all] root=${ROOT}"
echo "[run_all] stages: ${SELECTED[*]}"
echo "[run_all] SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH}"

for s in "${SELECTED[@]}"; do
  if ! run_stage "${s}"; then
    RC=1
    echo "[run_all] stage '${s}' FAILED — aborting." >&2
    break
  fi
done

echo
if [[ ${RC} -eq 0 ]]; then
  echo "[run_all] ALL SELECTED STAGES OK (${#SELECTED[@]}): ${SELECTED[*]}"
else
  echo "[run_all] FAILURE — see stage output above." >&2
fi
exit ${RC}
