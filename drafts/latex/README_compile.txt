================================================================================
 README_compile.txt  --  "Rank Before You Merge" ACM MM'27 LaTeX build notes
================================================================================
This directory is COMPILATION-READY. It was produced on a host with NO LaTeX
toolchain, so it has only been statically checked (see "Static self-check" at
the bottom). Compile it in your own TeX environment.

FILES
-----
  paper_acmmm.tex    main submission (sigconf, review, anonymous; double-blind)
  supp_acmmm.tex     supplementary material S1-S9 (same class; double-blind)
  references.bib     bibliography, 20 entries (copied verbatim from
                     drafts/references.bib; keys match every \cite in the body)
  ../figs/           (empty) put your rendered figures here: fig1.pdf ... fig3.pdf

All three figures are currently PLACEHOLDERS (\fbox boxes). The paper compiles
as-is with the placeholders; you only need the figure step before submission.


--------------------------------------------------------------------------------
1. REQUIRED PACKAGES (what the .tex files actually use)
--------------------------------------------------------------------------------
Both files:
  acmart              the ACM document class (loads amsmath, graphicx, xcolor,
                      natbib, caption, hyperref, etc. itself)
  booktabs            \toprule \midrule \bottomrule   (also loaded by acmart)
  multirow            (paper only) Table 1 / Table 3 model-column row spans
  pifont              \ding{51}=check, \ding{55}=cross  (replaces emoji)
  graphicx            \includegraphics, \fbox
  url                 \path{...} for run/digest paths (acmart loads it via
                      hyperref; listed explicitly anyway)
Paper only:
  algorithm           Algorithm 1 float
  algorithmic         Algorithm 1 pseudocode (\REQUIRE \STATE \FOR ...)
Bibliography style:
  ACM-Reference-Format.bst   (ships with acmart / TeX Live; \bibliographystyle)

All of the above are standard in a current TeX Live / MiKTeX. No exotic
packages, no sudo install needed. The .tex files are pure ASCII (no Unicode),
so default pdflatex font encodings work without inputenc/fontenc tweaks.


--------------------------------------------------------------------------------
2. COMPILE COMMAND SEQUENCE
--------------------------------------------------------------------------------
From inside drafts/latex/ :

  # ---- main paper ----
  pdflatex -interaction=nonstopmode paper_acmmm.tex
  bibtex   paper_acmmm
  pdflatex -interaction=nonstopmode paper_acmmm.tex
  pdflatex -interaction=nonstopmode paper_acmmm.tex

  # ---- supplementary (no \cite in supp -> bibtex step is optional) ----
  pdflatex -interaction=nonstopmode supp_acmmm.tex
  pdflatex -interaction=nonstopmode supp_acmmm.tex

Output: paper_acmmm.pdf and supp_acmmm.pdf.
(The two pdflatex passes after bibtex resolve cross-references and the
bibliography; the `review` option prints line numbers + page folios, useful
for the page-budget check.)

Convenience (latexmk, if installed):
  latexmk -pdf paper_acmmm.tex
  latexmk -pdf supp_acmmm.tex


--------------------------------------------------------------------------------
3. HOW TO EMBED THE THREE FIGURES (3 steps each)
--------------------------------------------------------------------------------
The figures are hand-drawn by you; specs are in drafts/figs_spec_for_user.md
(numbers there are official/audited). Each placeholder in paper_acmmm.tex is a
\begin{figure} ... \fbox{...} ... \end{figure} block with a comment above it.

  Step 1: render the figure to PDF and save as
            drafts/figs/fig1.pdf   (pipeline schematic)
            drafts/figs/fig2.pdf   (three-family pre-post Delta bars, HEADLINE)
            drafts/figs/fig3.pdf   (retention-vs-gap curves)
          Keep model/repo/author info OUT of the figure art (double-blind);
          model names (Qwen3-VL etc.) are allowed.

  Step 2: in paper_acmmm.tex, find the block (search for "[FIG:1", "[FIG:2",
          "[FIG:3"). Replace the single line
            \fbox{\parbox{0.92\columnwidth}{ ... }}
          with
            \includegraphics[width=\columnwidth]{../figs/figN.pdf}
          (use \columnwidth for single-column figures; for a two-column-wide
           figure put the block in a figure* environment and use \textwidth.)

  Step 3: recompile (one pdflatex pass is enough for a figure swap). Check the
          caption still matches the figure and that nothing overflows the
          column. Captions are drafted already -- edit only the figure art.

  Note: fig2 (headline) is the one to get right; OCR-Bench is /1000 pts while
  the other three bars are pp -- annotate the units per bar (see its caption).


--------------------------------------------------------------------------------
4. PAGE BUDGET  (HARD CAP: <= 8 pp body + <= 2 pp references)
--------------------------------------------------------------------------------
Over-length is a desk-reject risk for ACM MM. The body is number-dense and all
numbers / red-line sentences were kept VERBATIM (that was a hard constraint),
so the file is delivered slightly conservatively. Estimated as-built: ~9-11 pp
body (see risk notes). If your compile exceeds 8 pp, compress in THIS order --
each lever is marked in paper_acmmm.tex with a % [COMPRESSED ...] or
% [COMPRESS-OPTIONAL ...] comment so you can find it:

  (already applied)
   [A] sec 3.3 "Configuration disclosures"  -> condensed to one sentence;
       full M-RoPE / merger-tap detail lives in Supp S8.       (search: COMPRESSED -> supp S8)

  (apply if needed, in priority order)
   [B] Table 3 notes (iii)-(iv)  -> move the harness + budget detail
       (native-vs-mimic, ~214 vs ~213 ptid) to a supplementary note; keep
       notes (i)-(ii) with the table. Numbers preserved in supp.
       (search: COMPRESS-OPTIONAL ... Table 3 notes)
   [C] sec 6(a) router detail  -> move the ptid-threshold-router block
       (0.655/0.634/0.452/0.702) and the 27%/73% oracle decomposition to a
       supplementary note; keep the always-pre/best-router/oracle triple
       (0.494/0.484/0.576) in the body.
       (search: COMPRESS-OPTIONAL ... (a) carries granular router)
   [D] Figure placeholders -> the real figures may be smaller than the
       placeholder boxes; a compact fig2/fig3 saves vertical space.
   [E] Tighten spacing: \setlist{nosep} on the itemize/enumerate blocks, or
       drop the \vspace{3pt} before table notes.

  Tables that span both columns already use table* (Table 1) -- do NOT shrink
  Table 1's font below \small or its 8 columns get unreadable.


--------------------------------------------------------------------------------
5. RISK NOTES (read before submitting)
--------------------------------------------------------------------------------
  * Table 1 width: 8 columns + 5 footnotes in a table*. It fits \textwidth at
    \small, but if your acmart build is the narrower single-column-check
    layout, verify it does not overfull (look for "Overfull \hbox" warnings on
    the Table 1 page).
  * CCS concepts: the \ccsdesc / CCSXML block uses plausible 2012 taxonomy IDs
    (metadata only). Regenerate via the ACM CCS generator if the IDs must be
    exact; it does not affect compilation.
  * Copyright/ACM-ref blocks are suppressed (\setcopyright{none},
    printacmref=false) for anonymous review. Re-enable + fill \acmConference /
    \acmISBN / \acmDOI at de-anonymization / camera-ready.
  * \author is the "Anonymous Author(s)" placeholder; fill real authors at
    de-anonymization (the [anonymous] option hides them meanwhile).
  * kwon2023vllm is in references.bib but not \cite'd in the body (retained as
    an optional engine citation; harmless if unused -- bibtex simply skips it).


--------------------------------------------------------------------------------
6. STATIC SELF-CHECK (what was verified WITHOUT a compiler)
--------------------------------------------------------------------------------
Both .tex files were checked by a Python script:
  [PASS] \begin{X} / \end{X} pairing          (paper: 11 envs / 23 pairs;
                                               supp: 5 envs / 30 pairs)
  [PASS] tabular column spec vs & count        (paper: 6 tabulars, 0 mismatch;
                                               supp: 13 tabulars, 0 mismatch)
  [PASS] every \cite key exists in references.bib (19 cited, all present)
  [PASS] math $ balanced                       (paper 660, supp 374 -> even)
  [PASS] braces { } balanced                   (final depth 0, never negative)
  [PASS] every \ref has a matching \label       (0 dangling -> no "??")
  [PASS] pure ASCII                            (0 non-ASCII bytes; pdflatex-safe)
  [PASS] double-blind: no author/repo/institution/internal-digest path in the
         body (only the required Anonymous placeholders + legitimate references
         to OTHER papers' authors)
Not verifiable here (do at compile time): actual page count, float placement,
overfull hboxes, and that the figures render.
================================================================================
