# Citation audit for `overleaf_submission/main.tex`

Date: 2026-08-12
Scope: citation-key integrity, claim-to-source fit, bibliography metadata, and
2026 novelty coverage. This audit does not edit `main.tex`, because that file
already has concurrent user changes.

Status after editorial pass: all high-priority findings below are resolved in
commit `c93d29b`. The final manuscript cites 23 works; compilation reports no
undefined citations. The findings are retained as an audit trail.

## Executive verdict

- BibTeX integrity passes: 19 unique cited keys, 0 missing keys, 19 rendered
  references. Two entries are uncited (`kwon2023vllm`, `hong2025glm41v`).
- The bibliography compiles, but it is not submission-clean: BibTeX reports 35
  metadata warnings, mainly missing pages/publisher/address; one journal entry
  (`zeng2025glimpseprune`) is now stale.
- The larger risk is claim-to-source fit. The blanket stage claim in Related
  Work is not supported by the cited set, two named 2026 papers have no citation,
  and several architecture/README/code-reading assertions need citations or
  explicitly scoped wording.
- At 19 references, the related-work coverage is thin for an ACM MM 2027 paper
  making a novelty claim about a crowded 2025--2026 token-compression area.

## Must fix before submission

### C1. Unsupported blanket stage claim (high)

Location: `main.tex:269--283`.

Current text ends with:

> All operate on tokens that have already passed the encoder's native merger;
> none treats the merger stage itself as a design variable.

This overstates what the citations establish. FastV/PyramidDrop/SparseVLM are
in-LLM methods, but FasterVLM/FitPrune/PruMerge are not all demonstrated here as
post-*native-merger* methods. PruMerge is evaluated on LLaVA-family pipelines,
for which "passed the native merger" is not even a well-defined shared property.
The second clause is the defensible novelty boundary; the first is unnecessary.

Recommended replacement:

> These methods prune or merge at different encoder-output or LLM-side hook
> points. None isolates selection immediately before versus after a model's
> native spatial merger under iso-model, iso-token, and iso-selector control.

This also aligns with the paper's own narrower novelty statement at lines
307--315 and avoids a novelty-reject vector based on one counterexample.

### C2. Named works without citations (high)

Locations: `main.tex:308` (QuietPrune), `main.tex:1220` (Hi-Lo Prune).

Both are CVPR 2026 papers and must be cited when named. More importantly, they
are close enough to the proposed design space that omitting them weakens the
novelty positioning. Official CVF metadata was verified on 2026-08-12.

Suggested body edits:

```tex
QuietPrune-style in-ViT pruning~\cite{gao2026quietprune} is a different,
earlier, trained stage;
```

```tex
... a same-model comparison to Hi-Lo Prune~\cite{sun2026hiloprune} if its code
is released.
```

Suggested BibTeX:

```bibtex
@inproceedings{gao2026quietprune,
  author    = {Tianxiao Gao and Shanwei Zhao and Shuo Fang and Shiai Zhu and Chenguang Ma},
  title     = {{QuietPrune}: Query-Guided Early Token Pruning for Vision-Language Models},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {3553--3562},
  year      = {2026},
  url       = {https://openaccess.thecvf.com/content/CVPR2026/html/Gao_QuietPrune_Query-Guided_Early_Token_Pruning_for_Vision-Language_Models_CVPR_2026_paper.html}
}

@inproceedings{sun2026hiloprune,
  author    = {Zixun Sun and Yubo Dong and Hehe Fan and Yi Yang},
  title     = {{Hi-Lo Prune}: Look at What You'll Lose before Pruning with Hierarchical Token Selection},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {31941--31951},
  year      = {2026},
  url       = {https://openaccess.thecvf.com/content/CVPR2026/html/Sun_Hi-Lo_Prune_Look_at_What_Youll_Lose_before_Pruning_with_CVPR_2026_paper.html}
}
```

### C3. Cite vLLM where the implementation is introduced (medium)

Location: `main.tex:329--334`, first occurrence of "vLLM 0.19 (V1)".

The correct entry already exists as `kwon2023vllm` but is uncited. Use:

```tex
... vLLM 0.19 (V1)~\cite{kwon2023vllm}, single A40 (46~GB), ...
```

The exact software version remains an experimental disclosure; the citation is
for the serving system, not evidence for the local version number.

### C4. Introduction's literature premise needs citations and softer wording (high)

Location: `main.tex:147--156`.

"The Qwen2.5-VL authors themselves note it already compresses" should cite
`bai2025qwen25vl`. "Merging is reported to degrade OCR" can cite the paper's
existing AdaptMerge/VisionZip sources if those sources actually contain the
claimed observation. The sentence "the working assumption ... is that the
native merger is lossless" is not a proposition any cited paper appears to
state; it reads as a straw-man claim.

Recommended wording:

> Prior work recognizes that Qwen's merger already reduces visual-token count
> ~\cite{bai2025qwen25vl}, and post-encoder merging can degrade OCR-sensitive
> performance~\cite{islam2025adaptmerge,yang2024visionzip}. Yet existing
> compression work generally treats the native merger as a fixed boundary and
> does not isolate whether selection should occur immediately before or after it.

This retains the motivation without attributing an explicit "lossless"
assumption to the entire literature.

### C5. Stale GlimpsePrune metadata (medium)

`zeng2025glimpseprune` is no longer merely a 2025 "accepted" item. Crossref now
returns DOI `10.1109/TCSVT.2026.3702147`, publication year 2026, in IEEE TCSVT.
Keep the citekey to avoid churn, but update at least:

```bibtex
  year = {2026},
  doi  = {10.1109/TCSVT.2026.3702147},
  pages = {1--1},
```

Remove the `[UNVERIFIED volume/pages]` note from the rendered bibliography.
Before camera-ready, re-query IEEE/Crossref for assigned volume, issue, and final
page range; `1--1` is the current early-access metadata.

## Strongly recommended coverage

The 2026 frontier audit already in this repository identifies two directly
relevant, official CVPR 2026 papers. Adding one compact sentence prevents the
Related Work section from appearing frozen at 2025:

```tex
Recent layerwise methods score token evolution rather than merger-stage
placement: TransPrune uses token-transition variation
~\cite{li2026transprune}, while V2Drop exploits variation across adjacent
layers~\cite{chen2026v2drop}. They are complementary to the controlled native-
merger stage axis studied here.
```

```bibtex
@inproceedings{li2026transprune,
  author    = {Ao Li and Yuxiang Duan and Jinghui Zhang and Congbo Ma and Yutong Xie and Gustavo Carneiro and Mohammad Yaqub and Hu Wang},
  title     = {{TransPrune}: Token Transition Pruning for Efficient Large Vision-Language Model},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {39529--39538},
  year      = {2026},
  eprint    = {2507.20630},
  archivePrefix = {arXiv},
  url       = {https://openaccess.thecvf.com/content/CVPR2026/html/Li_TransPrune_Token_Transition_Pruning_for_Efficient_Large_Vision-Language_Model_CVPR_2026_paper.html}
}

@inproceedings{chen2026v2drop,
  author    = {Junjie Chen and Xuyang Liu and Zichen Wen and Yiyu Wang and Siteng Huang and Honggang Chen},
  title     = {Variation-Aware Vision Token Dropping for Faster Large Vision-Language Models},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {3489--3499},
  year      = {2026},
  eprint    = {2509.01552},
  archivePrefix = {arXiv},
  url       = {https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Variation-aware_Vision_Token_Dropping_for_Faster_Large_Vision-Language_Models_CVPR_2026_paper.html}
}
```

These two are higher priority than adding a long generic VLM bibliography. AOT
is video-specific and the tested RBM-OT extension is absent from the submitted
method, so AOT is optional unless the negative OT study enters the paper.

## Claim-source wording checks

1. `main.tex:297--302`: the exact VisionZip numbers and quoted README phrase are
   sourced only indirectly through a citation to the paper. Either cite a stable
   paper/table location in prose, move the README-specific quote to a footnote
   with URL/access date, or paraphrase without quotation marks.
2. `main.tex:297--305`: "a code reading confirms" and "likewise select after"
   are implementation-audit claims, not necessarily claims in the cited papers.
   Prefer "in the released Qwen implementation" and provide repository/version
   metadata in the supplement, or narrow the sentence to the verified methods.
3. `main.tex:470`: GQA's official metric is normally accuracy, whereas
   "word-normalized exact match" describes this paper's evaluator. Attribute
   the dataset to `hudson2019gqa` and the implementation to
   `zhang2024lmmseval`; do not imply the GQA paper defines that exact scorer.
4. `main.tex:467--472`: split sizes and metrics are compactly cited and are
   adequate, provided the split names in the supplement match the downloaded
   lmms-eval task configurations.
5. `main.tex:329--345`: Qwen and InternVL architecture descriptions are properly
   adjacent to their model citations. Local implementation verification is
   acceptable as method documentation but should include exact package/model
   revisions in the reproducibility supplement.

## Bibliography hygiene

- BibTeX emits 35 warnings. Missing publisher/address warnings are partly an
  ACM-style artifact, but missing page ranges for published conference papers
  should be fixed when authoritative metadata is available.
- Remove internal audit prose from rendered `note` fields, especially
  "venue/authors verified 2026-07", "under review", and `[UNVERIFIED ...]`.
  Provenance belongs in `citekey_map.md`, not the submission bibliography.
- Prefer final publisher pages/DOIs over arXiv URLs for published work. Keeping
  `eprint` is acceptable, but `note = {arXiv:...}` duplicates it in the rendered
  references and consumes reference-page space.
- `hong2025glm41v` is uncited and belongs to a removed experiment. Delete it
  from the submission bibliography unless the GLM result returns.
- Use consistent official proceedings names; the current entries alternate
  between abbreviated venue names and full proceedings titles.

## Acceptance criterion after edits

Run `latexmk -pdf main.tex`, then require: no undefined citation/reference,
all named methods cited at first mention, no internal verification notes in the
rendered bibliography, and no unsupported universal stage claim. Metadata-only
BibTeX warnings for genuinely pageless venues such as ICLR may remain.
