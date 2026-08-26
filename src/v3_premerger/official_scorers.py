"""Official benchmark scorers for re-scoring saved predictions (offline, CPU-only).

We REPLACE the ad-hoc containment rule with the *official* per-sample metrics:

  * TextVQA  -> standard VQA accuracy
      acc = min( (# of the 10 annotator answers that match pred after VQA-eval
                  normalization) / 3 , 1.0 )
      Normalization = lowercase, strip, remove articles a/an/the, remove
      punctuation, number-word -> digit, expand contractions
      (ports processPunctuation + processDigitArticle from the official
      VQA evaluation script / lmms-eval textvqa utils).

  * DocVQA   -> ANLS (Average Normalized Levenshtein Similarity)
      ANLS = max over answer variants of ( s if s>=0.5 else 0 ),
      s = 1 - normalized_levenshtein(pred, variant),
      normalized_levenshtein = edit_dist(pred, variant) / max(len(pred), len(variant))
      (empty-empty => s = 1). Case-insensitive.

Both functions take the raw model generation `pred` and a `;`-joined gt string,
and return a per-sample score in [0, 1]. They are deliberately applied to the
RAW generated text (as stored under per_sample[].answer) -- no answer
extraction is performed -- so the numbers are an honest lower bound of what the
official metric yields for these (often verbose) generations.

No GPU, no network. Pure string work.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# VQA-eval normalization tables (ported verbatim from the official VQA
# evaluation script / lmms-eval textvqa_utils).
# ---------------------------------------------------------------------------

_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "couldn'tve": "couldn't've", "couldnt've": "couldn't've",
    "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
    "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hes": "he's",
    "howd": "how'd", "howll": "how'll", "hows": "how's", "Id": "I'd",
    "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itll": "it'll",
    "its": "it's", "mightve": "might've", "mightnt": "mightn't",
    "mustve": "must've", "mustnt": "mustn't", "neednt": "needn't",
    "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at",
    "shant": "shan't", "shed": "she'd", "shell": "she'll", "shes": "she's",
    "shouldve": "should've", "shouldnt": "shouldn't", "somebodyd": "somebody'd",
    "someoned": "someone'd", "somethin": "somethin'", "thats": "that's",
    "thered": "there'd", "theres": "there's", "theyd": "they'd",
    "theyll": "they'll", "theyre": "they're", "theyve": "they've",
    "wasnt": "wasn't", "wed": "we'd", "were": "we're", "weve": "we've",
    "werent": "weren't", "whatd": "what'd", "whatll": "what'll",
    "whats": "what's", "whend": "when'd", "whenll": "when'll",
    "whens": "when's", "whered": "where'd", "wherell": "where'll",
    "wheres": "where's", "whod": "who'd", "wholl": "who'll", "whos": "who's",
    "whyd": "why'd", "whyll": "why'll", "whys": "why's", "wontve": "won't've",
    "wouldve": "would've", "wouldnt": "wouldn't", "yall": "y'all",
    "youd": "you'd", "youll": "you'll", "youre": "you're", "youve": "you've",
}

# number word -> digit
_MANUAL_MAP = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
    "nine": "9", "ten": "10",
}

_ARTICLES = ["a", "an", "the"]

_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(\d)(\,)(\d)")

_PUNCT = [
    ";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-",
    ">", "<", "@", "`", ",", "?", "!",
]


def _process_punctuation(in_text: str) -> str:
    """Port of VQA-eval processPunctuation."""
    out_text = in_text
    for p in _PUNCT:
        if (p + " " in in_text or " " + p in in_text) or (
            re.search(_COMMA_STRIP, in_text) is not None
        ):
            out_text = out_text.replace(p, "")
        else:
            out_text = out_text.replace(p, " ")
    out_text = _PERIOD_STRIP.sub("", out_text, re.UNICODE)
    return out_text


def _process_digit_article(in_text: str) -> str:
    """Port of VQA-eval processDigitArticle (lowercase, drop articles,
    map number-words to digits, expand contractions)."""
    out_text = []
    temp_text = in_text.lower().split()
    for word in temp_text:
        word = _MANUAL_MAP.get(word, word)
        if word not in _ARTICLES:
            out_text.append(word)
    for i, word in enumerate(out_text):
        if word in _CONTRACTIONS:
            out_text[i] = _CONTRACTIONS[word]
    return " ".join(out_text)


def vqa_normalize(text: str) -> str:
    """Canonical VQA-eval answer normalization."""
    text = str(text).replace("\n", " ").strip()
    text = _process_punctuation(text)
    text = _process_digit_article(text)
    return text


def score_textvqa_vqaacc(pred: str, gt_str: str) -> float:
    """Standard VQA accuracy for a single sample.

    gt_str: `;`-joined string of the 10 annotator answers.
    Returns per-sample accuracy in [0, 1] = min(#matches/3, 1).
    """
    if pred is None:
        pred = ""
    pred_norm = vqa_normalize(pred)
    gt_answers = [g for g in str(gt_str).split(";")]
    match = 0
    for g in gt_answers:
        if vqa_normalize(g) == pred_norm:
            match += 1
    return min(match / 3.0, 1.0)


# ---------------------------------------------------------------------------
# ANLS (DocVQA) via normalized Levenshtein similarity.
# ---------------------------------------------------------------------------

def _levenshtein(a: str, b: str) -> int:
    """Classic dynamic-programming edit distance (no external deps)."""
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    # ensure b is the shorter to keep the row small
    if len(b) > len(a):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def normalized_levenshtein(a: str, b: str) -> float:
    """edit_dist(a,b) / max(len(a),len(b)); empty-empty => 0."""
    denom = max(len(a), len(b))
    if denom == 0:
        return 0.0
    return _levenshtein(a, b) / denom


def score_docvqa_anls(pred: str, gt_str: str) -> float:
    """DocVQA ANLS for a single sample.

    gt_str: `;`-joined answer variants. Case-insensitive.
    ANLS = max over variants of (s if s>=0.5 else 0), s = 1 - norm_levenshtein.
    Empty pred vs empty variant => s = 1.
    Returns per-sample ANLS in [0, 1].
    """
    if pred is None:
        pred = ""
    pred_l = str(pred).lower().strip()
    best = 0.0
    for v in str(gt_str).split(";"):
        v_l = v.lower().strip()
        s = 1.0 - normalized_levenshtein(pred_l, v_l)
        s = s if s >= 0.5 else 0.0
        if s > best:
            best = s
    return best


# ---------------------------------------------------------------------------
# GQA -- official accuracy = normalized EXACT match.
#
# The OFFICIAL GQA eval script (nlp.stanford.edu/data/gqa/eval.zip -> eval.py,
# header "Evaluation code for GQA", scoring loop L347-351) does
#       gold = question["answer"]; predicted = predictions[qid]
#       correct = (predicted == gold)            # eval.py L350
# i.e. a RAW string exact match: the reference script assumes the predictions
# file is already cleaned (the dataset gold answers are clean single tokens,
# e.g. "no", "traffic light"). For (often verbose) VLM generations we first
# canonicalize BOTH sides with the VQA-eval normalization (vqa_normalize above:
# lowercase, strip, remove punctuation, drop articles a/an/the, number-word ->
# digit, expand contractions). This is the standard VLM GQA evaluation
# (lmms-eval / LLaVA) and a superset of the "lowercase + de-punctuate +
# de-article" cleanup; on already-clean GQA outputs it is a NO-OP, so it
# reproduces the official raw-exact number while staying robust to verbose
# generations. Returns 1.0 if the normalized prediction equals ANY ';'-joined
# normalized gold variant, else 0.0.
# ---------------------------------------------------------------------------

def score_gqa(pred: str, gt_str: str) -> float:
    """GQA official per-sample accuracy (normalized exact match). 0.0/1.0.

    gt_str: gold answer (single token for GQA; ';'-joined variants tolerated).
    """
    if not str(gt_str).strip():
        return 0.0
    if pred is None:
        pred = ""
    pred_n = vqa_normalize(pred)
    for g in str(gt_str).split(";"):
        if vqa_normalize(g) == pred_n:
            return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# OCRBench -- official accuracy: per-sample containment keyed on question_type,
# rolled up into the 5 official skills (each /200, total /1000).
#
# Ported from the official OCRBench evaluation (Yuliang-Liu/OCRBench). The
# canonical implementation is mirrored VERBATIM in VLMEvalKit
# (vlmeval/dataset/utils/ocrbench.py:OCRBench_eval, L8-60), used as reference:
#   * For EACH sample the model output `predict` is compared against the list
#     of acceptable answers; the sample is correct if ANY answer is *contained*
#     in the normalized output (L33 / L40:  `if answer in predict`).
#   * Normalization depends on the (fine) question_type (L29-42):
#       - "Handwritten Mathematical Expression Recognition" (HME100k LaTeX):
#             strip + "\n"->" " + remove ALL spaces on BOTH sides, and NO
#             lowercasing (case-SENSITIVE).            (L31-32)
#       - every other type:
#             lower() + strip() + "\n"->" " on BOTH sides.   (L38-39)
#   * The 10 fine question_types roll up into 5 categories. We use the
#     DATA-SIDE grouping (eval/full_splits/ocrbench.jsonl `category`, 200 each):
#       Text Recognition (TR)  = Regular+Irregular+Artistic+Digit String
#       Handwriting Text Recognition (HTR) = Handwriting+HME+Non-Semantic
#       Scene Text-centric VQA (ST-VQA), Document Text-centric VQA (DT-VQA),
#       Key Information Extraction (KIE)            (each 1:1).
#     (VLMEvalKit instead bundles all 6 recognition types under "Text
#     Recognition" and keeps HMER separate; the per-sample MATCHING rule above
#     is identical either way -- only the reporting roll-up differs.)
#   * Final Score = sum of the 5 category counts = /1000 on the FULL 1000-sample
#     benchmark (200 per category);  Final Score Norm = Final Score / 10. (L55-60)
#
# NOTE: this is the SAME containment rule the runner already applies
# (serve_bench.py / v3_premerger_runner.score_ocrbench). The ONLY behavioural
# difference is that the official HME branch is case-SENSITIVE while the runner
# lowercases HME too -- empirically identical on our cells (0 disagreeing
# samples; see experiments/j0b_official_scorers.md).
# ---------------------------------------------------------------------------

# fine question_type that drives the per-sample MATCHING rule (nospace +
# case-sensitive branch). NOTE: a FINE type, NOT one of the 5 categories.
OCRBENCH_HME = "Handwritten Mathematical Expression Recognition"

# data-side official 5 categories (eval/full_splits/ocrbench.jsonl `category`;
# 200 samples each, total 1000). Abbrevs: TR / HTR / ST-VQA / DT-VQA / KIE.
# NB this grouping differs from VLMEvalKit's: Handwriting/HME/Non-Semantic are
# bundled into "Handwriting Text Recognition" (HTR) and there is no separate
# HMER category -- we follow the data-side `category` (authoritative here).
OCRBENCH_CATEGORIES = [
    "Text Recognition",
    "Handwriting Text Recognition",
    "Scene Text-centric VQA",
    "Document Text-centric VQA",
    "Key Information Extraction",
]

# deterministic fine question_type -> category (verified on ocrbench.jsonl:
# TR={Regular,Irregular,Artistic,Digit String}; HTR={Handwriting,HME,
# Non-Semantic}; the 3 VQA/KIE types map 1:1).
OCRBENCH_QT_TO_CAT = {
    "Regular Text Recognition": "Text Recognition",
    "Irregular Text Recognition": "Text Recognition",
    "Artistic Text Recognition": "Text Recognition",
    "Digit String Recognition": "Text Recognition",
    "Handwriting Recognition": "Handwriting Text Recognition",
    OCRBENCH_HME: "Handwriting Text Recognition",
    "Non-Semantic Text Recognition": "Handwriting Text Recognition",
    "Scene Text-centric VQA": "Scene Text-centric VQA",
    "Doc-oriented VQA": "Document Text-centric VQA",
    "Key Information Extraction": "Key Information Extraction",
}


def ocrbench_category(question_type: str = "", category: str = None) -> str:
    """Resolve a sample's official 5-category (for /1000 aggregation). Prefer
    the explicit data-side `category` field; else map the fine question_type."""
    if category in OCRBENCH_CATEGORIES:
        return category
    return OCRBENCH_QT_TO_CAT.get(question_type, "Unknown")


# back-compat alias (older callers used the VLMEvalKit 'skill' naming)
ocrbench_skill = ocrbench_category


def score_ocrbench(pred: str, gt_str: str, question_type: str = "",
                   nospace: bool = None) -> int:
    """OCRBench official per-sample score (0/1).

    gt_str: ';'-joined acceptable answers (OCRBench `answer` list).
    question_type: fine OCRBench question_type; selects the HME branch.
    nospace: explicit override for the space-insensitive HME rule; if None it
        is inferred as (question_type == HME). Lets callers that only carry the
        runner's choices=["__nospace__"] flag (not the question_type) still hit
        the official HME branch.
    """
    if not str(gt_str).strip():
        return 0
    if pred is None:
        pred = ""
    if nospace is None:
        nospace = (question_type == OCRBENCH_HME)
    answers = str(gt_str).split(";")
    if nospace:
        p = str(pred).strip().replace("\n", " ").replace(" ", "")
        for a in answers:
            a = a.strip().replace("\n", " ").replace(" ", "")
            if a and a in p:
                return 1
        return 0
    p = str(pred).lower().strip().replace("\n", " ")
    for a in answers:
        a = a.lower().strip().replace("\n", " ")
        if a and a in p:
            return 1
    return 0


def score_ocrbench_total(per_cat_counts: dict) -> dict:
    """Roll per-CATEGORY CORRECT COUNTS into the official 5-category table +
    Final Score. Input maps category -> #correct (each category = 200 on the
    full benchmark). Returns {<5 categories>: count, 'Final Score': int,
    'Final Score Norm': float}."""
    out = {c: per_cat_counts.get(c, 0) for c in OCRBENCH_CATEGORIES}
    final = sum(out.values())
    out["Final Score"] = final
    out["Final Score Norm"] = float(final) / 10.0
    return out


# ---------------------------------------------------------------------------
# Batch wrappers: pure functions, input pred/gt lists -> {acc, n, per_item, ...}
# (convenience for standalone rescoring; the rescore script may also loop the
# per-sample functions directly, as it does for textvqa/docvqa).
# ---------------------------------------------------------------------------

def score_gqa_batch(preds: list, gts: list) -> dict:
    """Batch GQA accuracy. preds[i] vs gts[i] (gts may be ';'-joined variants).
    Returns {'acc','n','per_item'}."""
    per = [score_gqa(p, g) for p, g in zip(preds, gts)]
    n = len(per)
    return {"acc": (sum(per) / n if n else 0.0), "n": n, "per_item": per}


def score_ocrbench_batch(items: list) -> dict:
    """Batch OCRBench accuracy + official 5-category breakdown.

    items: list of (pred, gt_str, question_type[, category]).
    Returns {'acc','n','per_item','categories':{cat:{correct,total,acc}},
             'final_score','final_score_norm','official_total_1000_extrap'}.
    `official_total_1000_extrap` scales each present category's accuracy to the
    official 200-samples-per-category basis -- the /1000-scale number a
    (possibly unbalanced) subset implies; on the full balanced benchmark it ==
    the raw Final Score."""
    per, cat_correct, cat_total = [], {}, {}
    for it in items:
        pred, gt = it[0], it[1]
        qt = it[2] if len(it) > 2 else ""
        category = it[3] if len(it) > 3 else None
        c = score_ocrbench(pred, gt, qt)
        per.append(c)
        cat = ocrbench_category(qt, category)
        cat_total[cat] = cat_total.get(cat, 0) + 1
        if c:
            cat_correct[cat] = cat_correct.get(cat, 0) + 1
    n = len(per)
    categories = {}
    extrap = 0.0
    for cat in OCRBENCH_CATEGORIES:
        tot = cat_total.get(cat, 0)
        cor = cat_correct.get(cat, 0)
        acc = (cor / tot) if tot else 0.0
        categories[cat] = {"correct": cor, "total": tot, "acc": acc}
        if tot:
            extrap += acc * 200.0  # official 200-per-category basis
    total = score_ocrbench_total(cat_correct)
    return {
        "acc": (sum(per) / n if n else 0.0),
        "n": n,
        "per_item": per,
        "categories": categories,
        "final_score": total["Final Score"],
        "final_score_norm": total["Final Score Norm"],
        "official_total_1000_extrap": extrap,
    }


# ---------------------------------------------------------------------------
# Self-test (hand-checked cases).
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import math

    def approx(a, b, tol=1e-9):
        return abs(a - b) <= tol

    failures = []

    def check(name, got, want, tol=1e-9):
        ok = (got is not None and want is not None and abs(got - want) <= tol)
        print(f"[{'OK ' if ok else 'FAIL'}] {name}: got={got:.6f} want={want:.6f}")
        if not ok:
            failures.append(name)

    # ---- TextVQA VQA-accuracy -------------------------------------------
    # 1) pred "12:34" vs gt: only the two bare "12:34" entries match (the
    #    "12:34 am"/"12:34am" entries differ -- 'am' is NOT an article).
    #    2/10 matches -> min(2/3,1) = 0.6667
    gt_tvqa = "12:34;12:34 am;12:34 am;12:34 am;12:34;12:34 am;12:34am;9:00;12:34 am;12:34 am"
    check("textvqa exact-digit match", score_textvqa_vqaacc("12:34", gt_tvqa), 2.0 / 3.0)

    # 1b) pred "12:34 am" matches the 6 "12:34 am" variants -> min(6/3,1)=1.0
    check("textvqa am-variant match", score_textvqa_vqaacc("12:34 am", gt_tvqa), 1.0)

    # 2) pred "9:00" matches exactly 1/10 -> min(1/3,1)=0.3333
    check("textvqa single match", score_textvqa_vqaacc("9:00", gt_tvqa), 1.0 / 3.0)

    # 3) number-word normalization: pred "three" vs gt "3;three" (2 matches)->0.6667
    check("textvqa digit-article norm", score_textvqa_vqaacc("three", "3;three;four"), 2.0 / 3.0)

    # 4) verbose raw generation should NOT match a short gt -> 0.0
    verbose = ("Based on the image provided, there are two times displayed: "
               "1. The time on the phone's clock widget")
    check("textvqa verbose no-match", score_textvqa_vqaacc(verbose, gt_tvqa), 0.0)

    # ---- DocVQA ANLS -----------------------------------------------------
    # 1) exact match -> s=1.0
    check("docvqa exact", score_docvqa_anls("Charles D. Nesbit", "charles d. nesbit;Charles D. Nesbit"), 1.0)

    # 2) empty pred vs non-empty gt -> nl=1 -> s=0 -> 0.0
    check("docvqa empty-vs-text", score_docvqa_anls("", "charles d. nesbit"), 0.0)

    # 3) empty pred vs empty gt -> s=1.0
    check("docvqa empty-empty", score_docvqa_anls("", ""), 1.0)

    # 4) close variant above 0.5 threshold: "charles nesbit" vs "charles d nesbit"
    #    lev between 'charles nesbit'(14) and 'charles d nesbit'(16): edit dist 2
    #    nl = 2/16 = 0.125 -> s = 0.875 >= 0.5 -> ANLS 0.875
    got = score_docvqa_anls("charles nesbit", "charles d nesbit")
    check("docvqa near-match", got, 0.875)

    # 5) dissimilar below threshold -> 0.0
    check("docvqa below-threshold", score_docvqa_anls("xyz", "charles d. nesbit"), 0.0)

    # ---- GQA normalized exact match ------------------------------------
    # (a) case-insensitive: "No" vs "no"
    check("gqa case-insens", score_gqa("No", "no"), 1.0)
    # (b) article removal: "the tree" vs "tree"
    check("gqa article-drop", score_gqa("the tree", "tree"), 1.0)
    # (c) trailing punctuation: "cat." vs "cat"
    check("gqa trailing-punct", score_gqa("cat.", "cat"), 1.0)
    # (d) number-word -> digit: "three cats" vs "3 cats"
    check("gqa digit-article", score_gqa("three cats", "3 cats"), 1.0)
    # (e) wrong answer -> 0 (containment would WRONGLY pass: "no" in "I don't know")
    check("gqa no-contain-leak", score_gqa("I don't know", "no"), 0.0)
    # (f) ';'-joined variants: matches second variant
    check("gqa variant-match", score_gqa("bus", "car;bus"), 1.0)
    # batch wrapper
    gb = score_gqa_batch(["No", "dog", "the tree"], ["no", "cat", "tree"])
    check("gqa batch-acc", gb["acc"], 2.0 / 3.0)

    # ---- OCRBench official per-sample rules ----------------------------
    # (1) non-HME containment (case-insensitive): answer inside verbose output
    check("ocr docvqa-contain", score_ocrbench("The city is Paris.", "paris",
                                               "Doc-oriented VQA"), 1.0)
    # (2) non-HME no-match
    check("ocr docvqa-nomatch", score_ocrbench("cat", "dog",
                                              "Scene Text-centric VQA"), 0.0)
    # (3) HME space-insensitive: spaces stripped on both sides
    check("ocr hme-nospace", score_ocrbench("x ^ 2 + 1", "x^2+1",
                                           "Handwritten Mathematical Expression Recognition"), 1.0)
    # (4) HME is CASE-SENSITIVE (official, no lowercase): "X^2" != "x^2"
    check("ocr hme-case-sensitive", score_ocrbench("X^2", "x^2",
                                                  "Handwritten Mathematical Expression Recognition"), 0.0)
    # (5) HME matching case -> 1
    check("ocr hme-exact", score_ocrbench("x^2", "x^2",
                                         "Handwritten Mathematical Expression Recognition"), 1.0)
    # (6) ';'-joined answers: OR over acceptable answers
    check("ocr or-answers", score_ocrbench("2,112", "2112;2 112;2,112",
                                          "Doc-oriented VQA"), 1.0)
    # (7) nospace override via flag (no question_type) -> space-insensitive
    check("ocr nospace-flag", score_ocrbench("a + b", "a+b", "", nospace=True), 1.0)

    # category roll-up + official /1000 aggregation.
    # Full balanced benchmark = 200 per CATEGORY (5 categories); all correct -> 1000.
    full = {c: 200 for c in OCRBENCH_CATEGORIES}
    tot = score_ocrbench_total(full)
    check("ocr total-1000", float(tot["Final Score"]), 1000.0)
    check("ocr total-norm", tot["Final Score Norm"], 100.0)

    # question_type -> category resolver (data-side grouping)
    assert ocrbench_category("Non-Semantic Text Recognition") == "Handwriting Text Recognition"
    assert ocrbench_category(OCRBENCH_HME) == "Handwriting Text Recognition"
    assert ocrbench_category("Doc-oriented VQA") == "Document Text-centric VQA"
    assert ocrbench_category("", category="Key Information Extraction") == "Key Information Extraction"
    print("[OK ] ocr category-resolver (TR/HTR/ST-VQA/DT-VQA/KIE mapping)")

    # batch on a tiny unbalanced subset -> acc + extrapolation sanity
    ob = score_ocrbench_batch([
        ("Paris", "paris", "Doc-oriented VQA"),          # correct (cat DT-VQA)
        ("x^2", "x^2", OCRBENCH_HME),                    # correct (cat HTR)
        ("dog", "cat", "Regular Text Recognition"),      # wrong  (cat TR)
    ])
    check("ocr batch-acc", ob["acc"], 2.0 / 3.0)
    # extrap: DT-VQA 1/1*200 + HTR 1/1*200 + TR 0/1*200 = 400
    check("ocr batch-extrap", ob["official_total_1000_extrap"], 400.0)
    check("ocr batch-cat-HTR", ob["categories"]["Handwriting Text Recognition"]["acc"], 1.0)

    print()
    if failures:
        print(f"SELF-TEST FAILED on: {failures}")
        raise SystemExit(1)
    print("ALL SELF-TESTS PASSED")
