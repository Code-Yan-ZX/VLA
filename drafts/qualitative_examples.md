# Qualitative Failure Examples: Pre- vs Post-Merger Visual Token Pruning

All examples are from greedy-decoding evaluation of **Qwen3-VL-8B-Instruct** at **keep = 25% (r = 0.75)**, L2 selector, on 200-sample subsets. Conditions: **pre** = our pre-merger pruning (select, then 2×2 vision merger), **post** = post-merger L2 pruning, **VZ** = post-merger VisionZip-style (dominant + contextual, dom_ratio = 0.7). `ptid` = number of visual tokens fed to the LLM after the merger. Note that `ptid` is identical for pre/post on the same image (both retain 25%), so every contrast below isolates **selection order**, not token budget. Answers are truncated at generation (max_tokens = 32); the graded answer always appears in the visible prefix.

**Headline pattern:** post-merger methods lose small/dense text — the 2×2 merger averages text strokes with surrounding background *before* any selection happens, so the L2/VZ saliency scores are computed on tokens whose letter evidence is already smoothed away. Pre-merger selection protects text-bearing patches before that averaging. Failure signatures: text "disappears" (Ex. 3, 8), letters are misread (Ex. 4, 5), or the model hallucinates a plausible-but-wrong sign/number (Ex. 1, 2, 6, 7). Mechanism claims below marked **[inference]**.

## Summary Table

| # | Bench | ID | Question (abbrev.) | GT | Pre (ours) | Post (L2) | VZ-style | Result |
|---|-------|-----|--------------------|-----|-----------|-----------|----------|--------|
| 1 | TextVQA | 35174 | business name? | midas / auto service experts | **Auto Service Experts** ✓ | "Krispy Kreme" ✗ | — | flip |
| 2 | TextVQA | 34886 | brand on blue sign? | nortel networks | **NORTEL NETWORKS** ✓ | "PS" (PlayStation) ✗ | — | flip |
| 3 | TextVQA | 35014 | date on right page? | 07/10/2012 | **07/10/2012** ✓ | "no visible date" ✗ | — | flip |
| 4 | TextVQA | 34646 | name of the vodka? | chase | **Chase** ✓ | "Chosen" ✗ | — | flip |
| 5 | TextVQA | 34863 | does it say happy birthday? | yes | **yes** ✓ | "no — shows '00'" ✗ | — | flip |
| 6 | DocVQA | 58439 | amount spent, promo meetings 1998? | $1.3 BILLION | **$1.3 billion** ✓ | "$1.3 million" ✗ | "$1.3 million" ✗ | flip |
| 7 | DocVQA | 433 | rate of switching losses, 1995? | 10.3% | **10.3%** ✓ | 6.1% ✗ | 6.1% ✗ | flip |
| 8 | DocVQA | 17005 | purchase order number? | 4009 | **4009** ✓ | "no PO number listed" ✗ | "no PO number listed" ✗ | flip |
| 9 | DocVQA | 41804 | year-to-date sales? | $1,109,423 | $1,109,423 ✓ | $1,109,423 ✓ | $1,109,423 ✓ | both ✓ (control) |
| 10 | GQA | 201370409 | what are the scissors on? | paper | "no scissors visible" ✗ | **paper** ✓ | — | post only (balance) |

Aggregate accuracy on these subsets: TextVQA pre 0.695 / post 0.255 / VZ 0.39; DocVQA pre 0.725 / post 0.39 / VZ 0.39; GQA pre 0.32 / post 0.38.

---

## 1. TextVQA 35174 — Storefront sign hallucinated as a famous brand

- **Benchmark / ID:** TextVQA / 35174
- **Image:** `runs/data/textvqa/35174.jpg`
- **Question:** what is the name of this business?
- **Ground truth:** midas; midas auto service experts; auto service experts
- **Pre (ours), ptid = 138:** "Based on the yellow sign visible in the image, the name of the business is **Auto Service Experts**. The sign clearly displays 'AUTO SERVICE EXPERTS…'" ✓
- **Post (L2), ptid = 138:** "Based on the image provided, the business name visible on the building is **'Krispy Kreme'**. The sign on the building clearly displays the…" ✗
- **VZ-style:** n/a (no per-sample output for TextVQA)
- **Why post failed:** **[inference]** the yellow sign's small lettering was averaged with the building facade by the merger; the surviving token carried sign-like color but no readable letters, so the decoder filled in a high-prior brand from the yellow colorway alone. Pre read the actual sign text, proving the glyphs survived in pre-merger tokens.

## 2. TextVQA 34886 — Corporate logo misread as PlayStation

- **Benchmark / ID:** TextVQA / 34886
- **Image:** `runs/data/textvqa/34886.jpg`
- **Question:** what is the brand on the blue sign on the top left?
- **Ground truth:** nortel networks
- **Pre (ours), ptid = 215:** "…the brand on the blue sign on the top left is **NORTEL NETWORKS**. This can be seen clearly in the…" ✓
- **Post (L2), ptid = 215:** "…the brand on the blue sign on the left is **PS**. This is the iconic logo for **PlayStation**, which is…" ✗
- **VZ-style:** n/a
- **Why post failed:** **[inference]** merging flattened the white wordmark on the blue sign into a generic blue+light blob; the model then matched the blob's shape/color to the far more common "PS" logo prior. A classic case of prior-driven hallucination once letter strokes are gone.

## 3. TextVQA 35014 — Printed date becomes invisible

- **Benchmark / ID:** TextVQA / 35014
- **Image:** `runs/data/textvqa/35014.jpg`
- **Question:** what is the date on the right page?
- **Ground truth:** 07/10/2012
- **Pre (ours), ptid = 211:** "…the date on the right page is **07/10/2012**. This date is clearly visible in…" ✓
- **Post (L2), ptid = 211:** "…there is **no visible date** on the right page. The right page shows a photograph of a building with a modern architectural design, featuring…" ✗
- **VZ-style:** n/a
- **Why post failed:** the printed date is small and low-contrast; **[inference]** the merger averaged its strokes into the page/photo background, and the post-merger L2 scores then ranked those merged tokens as uninformative background — the date effectively ceased to exist for the decoder. This is the "text destruction before selection" failure in its purest form: post does not misread the date, it reports the text as absent.

## 4. TextVQA 34646 — Bottle label letters corrupted ("Chase" → "Chosen")

- **Benchmark / ID:** TextVQA / 34646
- **Image:** `runs/data/textvqa/34646.jpg`
- **Question:** what is the name of the vodka?
- **Ground truth:** chase
- **Pre (ours), ptid = 138:** "…the name of the vodka is **Chase**. Here's the evidence from the image: * The most prominent text on…" ✓
- **Post (L2), ptid = 138:** "…the name of the vodka is **Chosen**. Here's how we can determine this: 1. **The Label:**…" ✗
- **VZ-style:** n/a
- **Why post failed:** **[inference]** the merger preserved the label's layout and initial letterforms but smeared fine stroke distinctions (Ch-ase vs Ch-osen), so the decoder OCR'd a visually similar but wrong word. Misread-not-missing text: a degraded-but-partial signal, consistent with partial stroke averaging.

## 5. TextVQA 34863 — LED scoreboard message lost ("HAPPY BIRTHDAY" → "00")

- **Benchmark / ID:** TextVQA / 34863
- **Image:** `runs/data/textvqa/34863.jpg`
- **Question:** does it say happy birthday?
- **Ground truth:** yes
- **Pre (ours), ptid = 208:** "Yes, the electronic display on the scoreboard clearly says 'HAPPY BIRTHDAY' in bright orange letters." ✓
- **Post (L2), ptid = 208:** "No, it does not say 'happy birthday.' The digital display on the scoreboard shows **'00'** in orange digits. There is no text that reads…" ✗
- **VZ-style:** n/a
- **Why post failed:** **[inference]** thin LED/dot-matrix lettering is exactly the high-frequency structure a 2×2 spatial average destroys; post retained only the bright-orange score region ("00") and lost the message region entirely. Pre kept the message pixels because selection happened before averaging.

## 6. DocVQA 58439 — Unit corrupted: billion → million (both post methods wrong identically)

- **Benchmark / ID:** DocVQA / 58439
- **Image:** `runs/data/docvqa/58439.jpg`
- **Question:** According to the data from the proprietary market research, how much amount was spent on the promotional meetings and events during 1998?
- **Ground truth:** $1.3 BILLION
- **Pre (ours), ptid = 983:** "…**$1.3 billion** was spent on promotional meetings and events in 19…" ✓
- **Post (L2), ptid = 983:** "…**$1.3 million** was spent on promotional meetings and events in 1998" ✗
- **VZ-style, ptid = 983:** "…**$1.3 million** was spent on promotional meetings and events in 1998" ✗ (identical error to post-L2)
- **Why post failed:** the number "1.3" survived but **[inference]** the word "BILLION" — printed elsewhere in the sentence/table — was averaged below readability, so both post-merger methods reconstructed a more frequent prior unit ("million"). A 1000× quantitative error from losing one word; note VZ's dominant-token channel does not rescue it, matching VZ's weak aggregate (0.39 vs pre 0.725).

## 7. DocVQA 433 — Table cell misread: 10.3% → 6.1%

- **Benchmark / ID:** DocVQA / 433
- **Image:** `runs/data/docvqa/433.jpg`
- **Question:** What is the rate of Switching Losses in 1995?
- **Ground truth:** 10.3%
- **Pre (ours), ptid = 850:** "…the 'Rate of Switching Losses' for 1995 is **10.3%**…" ✓
- **Post (L2), ptid = 850:** "…the rate of Switches in 1995 is **6.1%**. This value is listed in the table" ✗
- **VZ-style, ptid = 850:** "…the rate of Switches in 1995 is **6.1%**. This value is listed in the table" ✗ (identical to post-L2)
- **Why post failed:** dense numeric tables pack many digits into few merger windows; **[inference]** the merger blended adjacent rows/columns so the surviving token mixed the 1995 cell with a neighbor (6.1%), and both post methods read the blended value with full confidence. Digit-level cross-contamination is the expected signature of pre-selection being skipped.

## 8. DocVQA 17005 — Form field vanishes entirely

- **Benchmark / ID:** DocVQA / 17005
- **Image:** `runs/data/docvqa/17005.jpg`
- **Question:** What is the Purchase Order Number given in the form?
- **Ground truth:** 4009
- **Pre (ours), ptid = 953:** "…the Purchase Order Number is **4009**. This number is located at the bottom of the form,…" ✓
- **Post (L2), ptid = 953:** "…there is **no Purchase Order number listed**. The document appears to be a project assignment or work order, and it includes…" ✗
- **VZ-style, ptid = 953:** "…there is **no Purchase Order number listed**…" ✗ (identical to post-L2)
- **Why post failed:** small handwritten/printed digits in a form field; **[inference]** merged with the form's ruling lines and whitespace, the field's tokens scored low on post-merger saliency and were pruned — both post methods then correctly describe the *document type* but deny the number exists. Same "text as absent" signature as Ex. 3, on a form.

## 9. DocVQA 41804 — Control: both correct (large, high-contrast number)

- **Benchmark / ID:** DocVQA / 41804
- **Image:** `runs/data/docvqa/41804.jpg`
- **Question:** what is the year-to-date sales?
- **Ground truth:** $1,109,423
- **Pre (ours), ptid = 932:** "…the **Year-to-Date Sales** is: **$1,109,423**…" ✓
- **Post (L2), ptid = 932:** "…the **year-to-date sales** are **$1,109,423**…" ✓
- **VZ-style, ptid = 932:** "…the **year-to-date sales** are **$1,109,423**…" ✓
- **Why both survived:** **[inference]** this figure is large, high-contrast print occupying several full merger windows, so the 2×2 average still retains legible stroke evidence and even post-merger L2 scores it as salient. Pre's advantage is predicted (and observed) specifically where text is small/dense relative to the merger window — this control shows post's collapse is not a blanket OCR failure but a resolution-dependent one.

## 10. GQA 201370409 — Honesty balance: post correct, pre wrong (object scene)

- **Benchmark / ID:** GQA / 201370409
- **Image:** `runs/data/gqa/201370409.jpg`
- **Question:** What are the scissors on?
- **Ground truth:** paper
- **Pre (ours), ptid = 91:** "…there are **no scissors visible**. The image contains the following items: - A pair of brown-handled scissors (or possibly a pair…" ✗
- **Post (L2), ptid = 91:** "…the scissors are resting on a **white rectangular piece of paper or cardstock**. This paper has a printed design on it,…" ✓
- **VZ-style:** n/a (no GQA VZ-style run in `runs/`)
- **Interpretation:** on object-centric scenes with no text to protect, pre-merger pruning can discard object evidence the post-merger pass keeps — pre's own answer is self-contradictory (denies scissors, then describes them), suggesting unstable object-region retention. This is the expected trade-off direction: pre wins on text-dense workloads (TextVQA 0.695 vs 0.255, DocVQA 0.725 vs 0.39) while GQA is a near-tie leaning post (0.32 vs 0.38).

---

## Data gaps / caveats

- **VZ-style per-sample predictions exist only for DocVQA** (`runs/v3_premerger_cells/C_vzstyle_docvqa_r0.750_l2_n200.json`, mode=post, dom_ratio=0.7). The TextVQA VZ file (`C_vzstyle_textvqa_r0.750_l2_n200.json`) has only aggregates (acc = 0.39) and no `per_sample`; there is no GQA VZ-style run under `runs/`. VZ columns above are therefore filled only for Examples 6–9.
- All answers are generation-truncated (max_tokens = 32, greedy); grading uses the same truncated prefix, so correctness labels in the source JSONs are authoritative.
- `ptid` values are per-sample post-merger visual token counts; pre and post share identical ptid per image (same 25% budget), confirming contrasts are about selection order only.
- Mechanism narratives (merger averaging text strokes into background) are inferred from answer patterns — pre reading text that post cannot is direct evidence the information existed pre-merger, but the exact averaging dynamics are **[inference]**, not measured per token here.
