#!/usr/bin/env python3
"""Toy unit tests for the OFFICIAL benchmark scorers (CPU-only, no pytest).

Focus: the answer-NORMALIZATION branches of the GQA official scorer
(articles / punctuation / case / number-words) and the OCRBench per-category
matching + category roll-up. Run with:  python3 tests/test_official_scorers_toy.py

These are intentionally tiny hand-checkable cases (5 GQA normalization cases +
OCRBench rules), complementing the fuller self-test embedded in
src/v3_premerger/official_scorers.py (__main__).
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from v3_premerger.official_scorers import (  # noqa: E402
    score_gqa,
    score_ocrbench,
    ocrbench_category,
    vqa_normalize,
    OCRBENCH_HME,
)

FAILED = []


def check(name, got, want):
    ok = got == want
    print(f"[{'OK ' if ok else 'FAIL'}] {name}: got={got!r} want={want!r}")
    if not ok:
        FAILED.append(name)


# --- 5 GQA normalization-branch toy cases (the writing red-line metric) -----
# (1) CASE: capitalization must not break a match
check("gqa/case", score_gqa("No", "no"), 1.0)
# (2) ARTICLE: leading a/an/the dropped on both sides
check("gqa/article", score_gqa("the tree", "tree"), 1.0)
# (3) PUNCTUATION: trailing/inner punctuation stripped
check("gqa/punct", score_gqa("cat.", "cat"), 1.0)
# (4) NUMBER-WORD -> digit ("three" == "3")
check("gqa/numword", score_gqa("three cats", "3 cats"), 1.0)
# (5) NO containment leak: gt "no" must NOT match "I don't know"
#     (the old ad-hoc containment rule wrongly passed this)
check("gqa/no-leak", score_gqa("I don't know", "no"), 0.0)

# normalization is a true function of the string (sanity on vqa_normalize)
check("norm/lower+article", vqa_normalize("The Dog"), "dog")
check("norm/punct", vqa_normalize("red, blue;"), "red blue")

# --- OCRBench per-sample matching rules -------------------------------------
# non-HME: case-insensitive containment
check("ocr/contain", score_ocrbench("The answer is Paris.", "paris",
                                    "Doc-oriented VQA"), 1)
# non-HME: no match
check("ocr/nomatch", score_ocrbench("cat", "dog", "Scene Text-centric VQA"), 0)
# HME: space-insensitive
check("ocr/hme-nospace", score_ocrbench("x ^ 2 + 1", "x^2+1", OCRBENCH_HME), 1)
# HME: CASE-SENSITIVE (official keeps case for LaTeX) -> "X^2" != "x^2"
check("ocr/hme-case", score_ocrbench("X^2", "x^2", OCRBENCH_HME), 0)
# ';'-joined acceptable answers (OR)
check("ocr/or", score_ocrbench("2,112", "2112;2 112;2,112", "Doc-oriented VQA"), 1)

# --- OCRBench category roll-up (data-side TR/HTR/ST-VQA/DT-VQA/KIE) ---------
check("cat/nonsemantic->HTR",
      ocrbench_category("Non-Semantic Text Recognition"),
      "Handwriting Text Recognition")
check("cat/HME->HTR", ocrbench_category(OCRBENCH_HME),
      "Handwriting Text Recognition")
check("cat/regular->TR", ocrbench_category("Regular Text Recognition"),
      "Text Recognition")
check("cat/doc->DT-VQA", ocrbench_category("Doc-oriented VQA"),
      "Document Text-centric VQA")
check("cat/explicit-wins",
      ocrbench_category("Doc-oriented VQA", category="Key Information Extraction"),
      "Key Information Extraction")

print()
if FAILED:
    print(f"TOY TESTS FAILED: {FAILED}")
    raise SystemExit(1)
print("ALL TOY TESTS PASSED")
