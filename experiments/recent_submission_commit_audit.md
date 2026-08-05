# Recent-commit audit: Overleaf submission vs experimental evidence

Audit date: 2026-08-05
Committed baseline: `HEAD = 41c4717` (`origin/main`)
Scope: the latest five commits, with detailed comparison of
`drafts/overleaf_submission/main.tex`, `supp.tex`, and the experimental/writing
artifacts introduced by those commits.

This report audits committed content. During the audit, unrelated concurrent
worktree edits appeared in `main.tex`, `supp.tex`, and `README_overleaf.md`;
those edits are not attributed to any of the five audited commits. In
particular, the worktree now contains an uncommitted S11 GLM appendix that
addresses several gaps below.

## 1. Commit inventory and relevance

| Commit | What changed | Overleaf status at committed HEAD |
|---|---|---|
| `41c4717` | GLM-4.1V fourth-family gate; mechanism-first rewrite; regime-map framing | `main.tex` and bibliography updated; `supp.tex` untouched |
| `f33b9ff` | Created self-contained Overleaf package | Main, S1--S10 supplement, bibliography, three paper figures, four qualitative panels present |
| `4a995af` | Committed measured figure outputs and QA reports | `f33b9ff` copied FIG1/2/3 and four selected comparison PDFs into Overleaf; the other six panels/contact sheet remain code-release assets by design |
| `db336b7` | Promoted author-selected measured FIG1 (`df7282e1`) | Present in Overleaf as `figs/fig1.pdf` |
| `c23759c` | Real-data L2 figure pipeline and audited values | Reflected through measured figures and S10; raw pipeline/QA metadata appropriately remains outside TeX |

The only new benchmark evidence in these five commits is the GLM gate in
`41c4717`. The preceding four commits are packaging/figure commits, not new
accuracy experiments.

## 2. Material already incorporated in committed `main.tex`

The `41c4717` writing brief is substantially implemented:

1. **Mechanism first:** the abstract, introduction, and contributions now lead
   with M1 rank reshuffle, M2 text-stroke demotion, and M3 ranking-swap plus
   kept-set identity. The stage law is contribution 2 rather than contribution
   1.
2. **FastV/RBM positioning:** Table 3 is reframed as a workload-conditioned
   regime map; FastV is called the query-conditioned strong baseline; RBM is a
   query-blind OCR-preserving robust default and explicitly not uniformly
   optimal.
3. **Fourth-family evidence:** the GLM-4.1V-9B-Thinking fallback, independent
   GLM-4+AIMv2 lineage, strided-convolution-plus-MLP merger, `n=200`, 25% keep,
   L2 selector, model-default effective cap (~4.82M px), greedy-vs-released
   sampling boundary, and exact iso-token statement are in the body.
4. **Headline GLM numbers:** none/pre/post are present as TextVQA
   `0.242/0.218/0.050`, DocVQA ANLS `0.104/0.130/0.031`, and GQA EM
   `0.150/0.160/0.115`; deltas are `+16.8/+9.8/+4.5 pp`.
5. **Claim downgrade:** the body states that the full-split generality claim
   remains the original three families, the fourth family is only an `n=200`
   gate, and GLM GQA is protocol-inconclusive. It also records the 4096-token
   non-convergence probe and released sampling settings.
6. **Bibliography:** `hong2025glm41v` was added to both bibliography copies.

The core GLM values therefore should not be duplicated in another main-paper
table. Additional audit data belongs in the supplement.

## 3. Evidence absent from committed TeX but suitable for the supplement

Committed `supp.tex` is byte-identical to `f33b9ff`; it contains no GLM entry.
The following evidence exists only in `experiments/glm4v_stage_gate.md` and is
worth adding as a compact fourth-family audit appendix:

| Missing evidence | Values/provenance | Why it matters |
|---|---|---|
| Full precision + zero skips | `n=200`, 0 skipped in all 9 cells; `0.2417/0.2183/0.0500`, `0.1039/0.1297/0.0313`, `0.1500/0.1600/0.1150` | Reproducibility and exact backing for rounded body table |
| Prompt-token audit | none/pre/post means: TextVQA `996.8/269.1/269.1`; DocVQA `4868.1/1238.8/1238.8`; GQA `373.7/113.8/113.8`; pre/post IDs equal per sample | Direct evidence for iso-token control |
| Final-answer convergence | boxed counts TextVQA `56/52/15`; DocVQA `33/30/11`; GQA `46/47/39` | Explains why thinking-model decoding is a configuration boundary |
| Pre-registered verdict | text-dense deltas exceed the locked 5 pp bar; GQA `post>=pre` criterion not met; converged common subset `n=25`, pre/post `0.68/0.68`, post hoc and underpowered | Prevents selective-success framing |
| 4096-token diagnostic | GQA none/pre/post official `0.165/0.165/0.115`, boxed `1/0/0`, mean generation `4011/3997/4057`; TextVQA none official/containment `0.2383/0.775`, boxed `2`, mean `3927` | Supports "protocol-inconclusive", but must remain diagnostic rather than headline |
| Reproducibility index | 9 v1 cells in `runs/glm4v_gate/`, 4 v2 cells in `runs/glm4v_gate_v2/`, digest and rescore scripts | S9 currently has no Table-6/GLM source row |

The concurrent uncommitted S11 patch now covers most of this table. Before it
is committed, add the v1/v2 run-path row to S9 or S11 and keep the explicit
"post hoc / underpowered / GQA inconclusive" qualifiers.

Resource accounting (5.5 GPU h total), weight-download history, OOM recovery,
and scheduler flags are useful internal provenance but do not need body space;
at most retain them in the code-release digest.

## 4. Material inconsistencies requiring resolution

### P0: supplement Table 3 backing values do not match the body

Committed `supp.tex` S9.2/S9.3 labels its cells "official", but it prints the
JSON `acc`/runner values, while body Table 3 and the authoritative
`experiments/r2b_fastv_k3.md` / `r2c_rbm_scope.md` use official rescoring.
Examples:

| Cell | Body / authoritative official | S9 committed value |
|---|---:|---:|
| Qwen3 FastV TextVQA | 0.7771 | 0.8494 |
| Qwen3 FastV DocVQA | 0.5863 | 0.4650 |
| Qwen2.5 FastV TextVQA | 0.7467 | 0.8150 |
| Qwen2.5 RBM TextVQA | 0.6683 | 0.7250 |
| Qwen3 RBM OCR-Bench | 0.5750 | 0.6354 |

All eight FastV rows and all five S9.3 RBM rows need one coherent metric
policy. Recommended: retain ptid/skip/engine/path fields, replace the score
column with the official-rescore values used by body Table 3, and name the JSON
field/rescore artifact precisely. Do not silently mix containment/raw `acc`
with official VQA-acc/ANLS/EM.

This inconsistency predates `41c4717`, but the latest commit elevates Table 3
to a headline regime map, so it is now submission-blocking.

### P1: FastV summary wording is broader than Table 3

Body Table 3 shows FastV wins all four Qwen3 reference-grounded cells except
OCR, plus Qwen2.5 TextVQA; RBM wins Qwen2.5 GQA and DocVQA as well as both OCR
cells. Therefore these sentences are overbroad:

- abstract: "FastV wins its reference-grounded home regime
  (TextVQA/GQA/DocVQA cells)";
- discussion: "FastV leads it on TextVQA/GQA/DocVQA".

Use a scope-exact formulation such as: "FastV leads the Qwen3
TextVQA/GQA/DocVQA cells and Qwen2.5 TextVQA, whereas RBM leads both OCR cells
and the Qwen2.5 GQA/DocVQA cells." This preserves the no-universal-winner
framing and does not imply a cross-model sweep.

### P1: fourth-family wording should name the replicated arm

After correctly declaring GLM GQA inconclusive, the body says the gate
"supports the generality of the workload-conditional stage law on a fourth
lineage." Under the locked redline, narrow this to "supports the text-dense arm
of the stage law on a fourth lineage." Do not claim the full workload pattern
replicated on GLM.

### P1: GLM protocol typo in the source digest

`experiments/glm4v_stage_gate.md` says `--max-tokens 512` in its Protocol
section, but the same digest, `DECISIONS.md`, and
`src/v3_premerger/glm4v_gate.sh` (`MAXTOK=1024`) show the primary gate used
1024; the v2 probe used 4096. Correct the digest typo before citing the protocol
in TeX. The concurrent S11 currently uses 1024, which matches script/decision
evidence.

### P2: source-copy synchronization is not literal

`STATE.md` says `drafts/latex/paper_acmmm.tex` and Overleaf `main.tex` are
equivalent, but they are not byte-identical. Their scientific prose from
`41c4717` is synchronized; the source copy still contains figure placeholders,
while Overleaf embeds measured FIG1/2/3 and expanded captions. Treat Overleaf
as the submission source or synchronize the canonical copy before future edits.

The top comments in Overleaf `main.tex` also retain stale phrases such as
"3 figure placeholders" even though all figures are measured and embedded.

## 5. Claim-redline check

- **Pass:** no SOTA / "beats existing methods" claim was introduced.
- **Pass:** full-split cross-family evidence remains limited to three families.
- **Pass:** GLM absolute values are not compared across models; pixel cap,
  sample size, budget, and decoding protocol are disclosed.
- **Pass:** GLM GQA is labeled inconclusive in the table, paragraph, abstract,
  and limitations; it is not used as a direction claim.
- **Needs tightening:** replace the broad fourth-lineage "stage law"
  generality phrase with "text-dense arm" and scope the FastV summary exactly.
- **Do not promote:** the `n=25` converged-subset tie and 4096 probe are
  diagnostics only, never positive headline evidence.

## 6. Recommended merge order

1. Resolve S9.2/S9.3 official-vs-runner score columns against the authoritative
   r2b/r2c digests; this is the only submission-blocking numerical conflict
   found.
2. Keep the new S11 GLM audit (or equivalent), add v1/v2 reproducibility paths,
   and correct the `512 -> 1024` typo in the experiment digest.
3. Tighten the two FastV summary sentences and the fourth-lineage sentence to
   match the row-level evidence and claim redline.
4. Compile `main.tex` and `supp.tex`, inspect page count/floats/cross-references,
   and update package comments/README counts after content stabilizes.
5. Commit/push only the intended TeX, bibliography, README, state/decision, and
   small digest changes; do not include generated `.aux/.log/.pdf` build files,
   runs, logs, data, or weights.

## 7. Resolution (2026-08-05)

The accompanying submission update resolves every P0/P1 item above: S9.2/S9.3
now use all 13 authoritative official-rescore values; S11 records the nine v1
cells, exact token audit, convergence counts, v1/v2 paths, and 4096-token
diagnostic; FastV and fourth-family claims are scope-exact; and the GLM digest
protocol typo is corrected to 1024. `main.tex` and `supp.tex` compile without
fatal or undefined-reference errors (12 and 10 pages respectively). Rendered
inspection of the final S11 page found no clipping, overlap, or blank trailing
page. Generated PDFs and LaTeX intermediates are verification-only and are not
committed.
