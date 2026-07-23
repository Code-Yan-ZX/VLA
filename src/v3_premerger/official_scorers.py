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

    print()
    if failures:
        print(f"SELF-TEST FAILED on: {failures}")
        raise SystemExit(1)
    print("ALL SELF-TESTS PASSED")
