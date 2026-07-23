"""mechanism_token_survival.py — PRE vs POST merger L2 token-survival mechanism.

Three run modes:

  --mode legacy   (default) the original 2-image deep-dive figures on the known
                  pre-OK/post-wrong samples (DocVQA 58439, TextVQA 35174),
                  from cached GPU captures in runs/v3_attn_robust/_vizcap_*.pt
                  (runs the GPU capture only if the caches are missing).

  --mode capture  GPU. Captures COMPACT per-image mechanism stats for a
                  deterministic fixed-seed sample of n images per benchmark
                  (docvqa/textvqa/gqa, --bench all = all three in ONE model
                  load): per-unit PRE L2 score, POST L2 score, Sobel edge
                  energy. Saved compactly to runs/v3_merger_aware/survival_
                  capture/{bench}.npz (+ {bench}_meta.json) — NO giant feature
                  tensors. Capture-only forward (max_tokens=1), NO generation
                  metric, NO pruning: the model instance/processor settings are
                  identical to the benchmark cells so the captured features are
                  representative.

  --mode analyze  CPU. Loads the captures, runs the M1/M2 analysis over the
                  full sample split by bench (text-dense = docvqa & textvqa;
                  object = gqa) and writes drafts/figures/token_survival_
                  stats.json (merged with any legacy single-image stats) plus
                  token_survival_m1_rank_overlap.{png,pdf} and
                  token_survival_m2_edge_demotion.{png,pdf}.

FAITHFULNESS (replicates src/v3_premerger/v3_premerger_runner.py EXACTLY):
  * Same vLLM model instance + same processor settings (max_pixels=1500000,
    mm_processor_kwargs) as the benchmark cells; capture-only, NO pruning.
  * PRE  score = _score_units(feats, "l2") on the FIRST merger called =
    deepstack_merger_list[0] input = ViT block-8 hidden states
    (verified: runner diag mask_computed_at='deepstack_0'; Qwen3-VL calls
    deepstack mergers inside the block loop at layer_num in [8,16,24], main
    merger after the loop).
  * POST score = _score_tokens(split, "l2") on the visual-tower output row =
    cat([main, ds0, ds1, ds2] merger outputs, dim=1), [num_units, 4*4096]
    (qwen3_vl.py visual.forward L653-656).
  * BOTH stages use the L2 (norm) selector — consistent with the headline
    cells (--selector l2). (The runner's "attn" centroid proxy is NOT used.)
  * keep k = max(1, round(num_units*(1-r))) per image (runner contract);
    top-k by score, token i <-> unit i is 1:1 (mergers map 4 consecutive
    block-major tokens -> 1 output token).
  * Scoring functions are imported from the runner module itself.

GEOMETRY (Qwen3-VL-8B-Instruct config, verified): patch_size=16,
spatial_merge_size=2 -> one merge unit = 2x2 patches = 32x32 px in
processor-resized coordinates. Qwen3-VL vision has NO window permutation
(qwen3_vl.py visual.forward: patch_embed -> +pos_embeds -> 27 blocks in
row-major patch order -> mergers), so unit u of an image with grid (t=1,h,w)
covers resized pixels [32*ur : 32*(ur+1), 32*uc : 32*(uc+1)] with
ur = u // (w//2), uc = u % (w//2). Grid (h,w) in 16px patches ->
resized image = (16*h, 16*w); the display image is resized to exactly that.

MECHANISM METRICS (per image, aggregated over the sample):
  * M1 — full-ranking overlap: Spearman rho + Kendall tau between PRE and POST
    per-unit scores over ALL merge-units (low => the merger reshuffles rank),
    alongside the top-k Jaccard@keep. Claim test: rho well below 1, stronger
    on text-dense than object.
  * M2 — disagreement x text-density: per-unit rank_shift = post_rank - pre_rank
    (+ => merger DEMOTED the unit); Spearman of rank_shift vs Sobel edge energy;
    and, at keep-ratio, the mean Sobel of (a) pre-kept/post-dropped vs
    (b) post-kept/pre-dropped vs (c) agreement-kept, plus the above-median-edge
    fraction of (a) vs (b). Claim test: (a) is the highest-edge (text) group and
    rank_shift correlates with edge, strongest on text-dense.

Outputs: drafts/figures/token_survival_docvqa.{png,pdf},
         drafts/figures/token_survival_textvqa.{png,pdf}        (legacy)
         drafts/figures/token_survival_m1_rank_overlap.{png,pdf} (M1)
         drafts/figures/token_survival_m2_edge_demotion.{png,pdf}(M2)
         drafts/figures/token_survival_stats.json
"""
from __future__ import annotations
import os, sys, json, argparse, datetime
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"text.parse_math": False,   # '$' in answers is literal
                     "font.size": 10, "axes.titlesize": 11})
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from PIL import Image
from scipy.ndimage import sobel as _sobel
from scipy.stats import spearmanr, kendalltau

ROOT = "/media/disk2/YZX/research/vla"
sys.path.insert(0, os.path.join(ROOT, "src", "v3_premerger"))
from v3_premerger_runner import _score_tokens, _score_units   # EXACT selector code

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
R = 0.75                      # keep 25%
MAX_PIXELS = 1500000          # same as BIG-config cells
PATCH = 16                    # px per patch (verified: vision_config.patch_size)
MERGE = 2                     # 2x2 (verified: spatial_merge_size)
UNIT = PATCH * MERGE          # 32 px per merge-unit side
BENCHES = ["docvqa", "textvqa", "gqa"]

# Samples known pre-OK / post-wrong (drafts/qualitative_examples.md rows 1 & 6;
# image paths + GT from eval/subsets/{docvqa_200,textvqa_200}.jsonl).
SAMPLES = [
    dict(key="docvqa", id="58439",
         image=f"{ROOT}/runs/data/docvqa/58439.jpg",
         question="Amount spent on promotional meetings and events, 1998?",
         gt="$1.3 billion",
         documented=dict(pre='"$1.3 billion" (correct)',
                         post='"$1.3 million" (wrong: billion->million)'),
         acc_ctx="DocVQA-200, L2 selector, keep 25%: pre .725 / post .39"),
    dict(key="textvqa", id="35174",
         image=f"{ROOT}/runs/data/textvqa/35174.jpg",
         question="What is the name of this business?",
         gt="Midas / auto service experts",
         documented=dict(pre='"Auto Service Experts" (correct)',
                         post='"Krispy Kreme" (wrong: hallucinated brand)'),
         acc_ctx="TextVQA-200, L2 selector, keep 25%: pre .695 / post .255"),
]


# --------------------------------------------------------------------------- #
# Feature capture (GPU) — hooks the 4 mergers WITHOUT changing numerics.
# --------------------------------------------------------------------------- #
def wrap_capture(visual):
    """Capture merger inputs/outputs WITHOUT changing numerics. Returns dict."""
    cap = {"grid_thw": None, "ds_in": [], "out": {}}  # out: tag -> [num_units, 4096]

    def _vis_prehook(module, args, kwargs):
        g = kwargs.get("grid_thw")
        if g is None and len(args) >= 2:
            g = args[1]
        if g is not None:
            cap["grid_thw"] = g.detach().cpu() if torch.is_tensor(g) else torch.as_tensor(g)

    visual.register_forward_pre_hook(_vis_prehook, with_kwargs=True)

    def _make(tag, is_ds0):
        m = getattr(visual, tag) if isinstance(tag, str) else tag
        orig = m.forward
        def _wrapped(*args, _orig=orig, **kw):
            hs = args[0]
            if is_ds0:
                cap["ds_in"].append(hs.detach().float().cpu())
            out = _orig(*args, **kw)
            cap["out"][tag] = out.detach().float().cpu()
            return out
        m.forward = _wrapped
        if hasattr(m, "do_not_compile"):      # force eager branch if decorated
            m.do_not_compile = True

    _make("merger", False)
    for i in range(3):
        m = visual.deepstack_merger_list[i]
        orig = m.forward
        _is_ds0 = (i == 0)
        def _wrapped(*args, _orig=orig, _tag=f"ds{i}", _capture=_is_ds0, **kw):
            hs = args[0]
            if _capture:
                cap["ds_in"].append(hs.detach().float().cpu())
            out = _orig(*args, **kw)
            cap["out"][_tag] = out.detach().float().cpu()
            return out
        m.forward = _wrapped
        if hasattr(m, "do_not_compile"):
            m.do_not_compile = True
    return cap


def reset(cap):
    cap["grid_thw"] = None
    cap["ds_in"] = []
    cap["out"] = {}


# --------------------------------------------------------------------------- #
# Shared analysis core (CPU). Used by BOTH the legacy single-image path and
# the n-image sample path. Scoring is L2 at both stages (runner-consistent).
# --------------------------------------------------------------------------- #
def scores_from_cap(cap):
    """Replicate runner L2 scoring on captured features for ONE image.
    Returns (pre_scores, post_scores, h, w) as float64 numpy arrays."""
    g = cap["grid_thw"]
    assert g is not None and g.shape[0] == 1, \
        f"expected 1 image/pass, got {None if g is None else tuple(g.shape)}"
    t, h, w = (int(x) for x in g[0].tolist())
    assert t == 1
    num_units = (h * w) // (MERGE ** 2)

    # PRE: deepstack[0] merger input = block-8 features [N,1,1152] -> [n,4,1152]
    hs_ds0 = cap["ds_in"][0]                       # [N, 1, 1152]
    ctx = hs_ds0.shape[-1]
    feats = hs_ds0.reshape(num_units, MERGE ** 2, ctx)
    pre_scores = _score_units(feats, "l2").numpy().astype(np.float64)

    # POST: visual output row = cat(main, ds0, ds1, ds2) [num_units, 4*4096]
    main = cap["out"]["merger"]
    ds = [cap["out"][f"ds{i}"] for i in range(3)]
    post_feat = torch.cat([main] + ds, dim=1)      # [num_units, 16384]
    post_scores = _score_tokens(post_feat, "l2").numpy().astype(np.float64)
    assert pre_scores.shape[0] == post_scores.shape[0] == num_units
    return pre_scores, post_scores, h, w


def unit_edge_from_image(image_path: str, h: int, w: int) -> np.ndarray:
    """Objective text-stroke proxy: mean Sobel edge energy per 32px unit."""
    H, W = h * PATCH, w * PATCH
    gray = np.asarray(
        Image.open(image_path).convert("L").resize((W, H), Image.BICUBIC)
    ).astype(np.float32) / 255.0
    ex, ey = _sobel(gray, axis=1), _sobel(gray, axis=0)
    edge = np.hypot(ex, ey)
    num_units = (h * w) // (MERGE ** 2)
    unit_edge = np.zeros(num_units, dtype=np.float64)
    for u in range(num_units):
        ur, uc = divmod(u, w // MERGE)
        unit_edge[u] = edge[ur * UNIT:(ur + 1) * UNIT,
                            uc * UNIT:(uc + 1) * UNIT].mean()
    return unit_edge


def analyze_arrays(pre_scores, post_scores, unit_edge, h, w, r=R):
    """M1/M2 stats + keep/drop sets for ONE image from per-unit arrays."""
    num_units = pre_scores.shape[0]
    k = max(1, int(round(num_units * (1.0 - r))))   # runner contract
    pre_keep = set(np.argsort(-pre_scores)[:k].tolist())
    post_keep = set(np.argsort(-post_scores)[:k].tolist())

    jacc = len(pre_keep & post_keep) / len(pre_keep | post_keep)

    # ---- M1: ranking overlap over ALL merge-units (how much merger reshuffles)
    rho = float(spearmanr(pre_scores, post_scores)[0])
    tau = float(kendalltau(pre_scores, post_scores)[0])

    # ---- M2: disagreement x text-density ----
    # rank 1 = highest score; rank_shift = post_rank - pre_rank (positive => the
    # merger DEMOTED that unit relative to pre).
    pre_rank = np.empty(num_units, dtype=np.int64)
    pre_rank[np.argsort(-pre_scores)] = np.arange(1, num_units + 1)
    post_rank = np.empty(num_units, dtype=np.int64)
    post_rank[np.argsort(-post_scores)] = np.arange(1, num_units + 1)
    rank_shift = (post_rank - pre_rank).astype(np.float64)
    rho_shift_edge = float(spearmanr(rank_shift, unit_edge)[0])

    # partition units at the target keep-ratio into disagreement / agreement sets
    grp_a = pre_keep - post_keep     # (a) pre-kept & post-dropped  (mech: high text)
    grp_b = post_keep - pre_keep     # (b) post-kept & pre-dropped
    grp_c = pre_keep & post_keep     # (c) agreement-kept (both keep)
    edge_a = float(unit_edge[sorted(grp_a)].mean()) if grp_a else float("nan")
    edge_b = float(unit_edge[sorted(grp_b)].mean()) if grp_b else float("nan")
    edge_c = float(unit_edge[sorted(grp_c)].mean()) if grp_c else float("nan")
    med_edge = float(np.median(unit_edge))
    frac_a = float((unit_edge[sorted(grp_a)] > med_edge).mean()) if grp_a else float("nan")
    frac_b = float((unit_edge[sorted(grp_b)] > med_edge).mean()) if grp_b else float("nan")

    stats = dict(
        num_units=num_units, k=k, h=h, w=w, H=h * PATCH, W=w * PATCH,
        jaccard=float(jacc),
        # M1: full-ranking overlap
        spearman_pre_post=rho,
        kendall_pre_post=tau,
        # M2: disagreement x text-density
        rank_shift_vs_edge_spearman=rho_shift_edge,
        edge_a_pre_only=edge_a,
        edge_b_post_only=edge_b,
        edge_c_agree_keep=edge_c,
        n_grp_a=len(grp_a), n_grp_b=len(grp_b), n_grp_c=len(grp_c),
        frac_above_median_edge_a=frac_a,
        frac_above_median_edge_b=frac_b,
        median_edge=med_edge,
        pre_edge_keep=float(unit_edge[list(pre_keep)].mean()),
        pre_edge_drop=float(unit_edge[sorted(set(range(num_units)) - pre_keep)].mean()),
        post_edge_keep=float(unit_edge[list(post_keep)].mean()),
        post_edge_drop=float(unit_edge[sorted(set(range(num_units)) - post_keep)].mean()),
    )
    return dict(pre=pre_scores, post=post_scores, pre_keep=pre_keep,
                post_keep=post_keep, pre_rank=pre_rank, post_rank=post_rank,
                rank_shift=rank_shift, grp_a=grp_a, grp_b=grp_b, grp_c=grp_c,
                unit_edge=unit_edge, stats=stats)


ANALYZE_IMG = None  # legacy: set per-sample (analyze uses it for edge energy)


def analyze(cap):
    """Legacy entry: cap -> full result dict (scores from cap, edge from
    ANALYZE_IMG). Kept for the 2-image deep-dive figures."""
    pre, post, h, w = scores_from_cap(cap)
    edge = unit_edge_from_image(ANALYZE_IMG, h, w)
    return analyze_arrays(pre, post, edge, h, w)


# --------------------------------------------------------------------------- #
# Legacy 2-image figure (draw).
# --------------------------------------------------------------------------- #
def draw(s, res, out_base):
    st = res["stats"]
    H, W, h, w = st["H"], st["W"], st["h"], st["w"]
    img = np.asarray(
        Image.open(s["image"]).convert("RGB").resize((W, H), Image.BICUBIC)
    )
    n_ur, n_uc = h // MERGE, w // MERGE

    def overlay(ax, keep, scores, cmap, title):
        ax.imshow(img)
        dim = np.zeros((H, W, 4))                  # dim dropped units slightly
        for u in range(st["num_units"]):
            ur, uc = divmod(u, n_uc)
            if u not in keep:
                dim[ur * UNIT:(ur + 1) * UNIT, uc * UNIT:(uc + 1) * UNIT] = (0, 0, 0, 0.30)
        ax.imshow(dim)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1))
        ks = np.array([scores[u] for u in keep])           # color over kept set
        smin, smax = ks.min(), ks.max()
        for u in sorted(keep):
            ur, uc = divmod(u, n_uc)
            norm = (scores[u] - smin) / (smax - smin + 1e-12)
            c = cmap(norm)
            ax.add_patch(Rectangle((uc * UNIT, ur * UNIT), UNIT, UNIT,
                                   facecolor=c[:3], alpha=0.55,
                                   edgecolor="white", linewidth=0.4))
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        return sm

    portrait = H > W
    fig, axes = plt.subplots(1, 3, constrained_layout=True,
                             figsize=(11.0, 5.8) if portrait else (15.5, 5.6))
    sm1 = overlay(axes[0], res["pre_keep"], res["pre"], plt.cm.viridis,
                  f"PRE kept  ({st['k']}/{st['num_units']})")
    sm2 = overlay(axes[1], res["post_keep"], res["post"], plt.cm.viridis,
                  f"POST kept  ({st['k']}/{st['num_units']})")

    # panel 3: both outlines + disagreement annotation + objective stats
    grp_a = res.get("grp_a", set())
    axes[2].imshow(img)
    # group (a) = pre-kept & post-dropped: fill orange (mech predicts high-text)
    for u in sorted(grp_a):
        ur, uc = divmod(u, n_uc)
        axes[2].add_patch(Rectangle((uc * UNIT, ur * UNIT), UNIT, UNIT,
                                    facecolor="orange", alpha=0.55,
                                    edgecolor="orange", linewidth=0.3))
    for u in range(st["num_units"]):
        ur, uc = divmod(u, n_uc)
        in_pre, in_post = u in res["pre_keep"], u in res["post_keep"]
        if in_pre:
            axes[2].add_patch(Rectangle((uc * UNIT, ur * UNIT), UNIT, UNIT,
                                        facecolor="none", edgecolor="lime",
                                        linewidth=1.3))
        if in_post:
            axes[2].add_patch(Rectangle((uc * UNIT + 1.5, ur * UNIT + 1.5),
                                        UNIT - 3, UNIT - 3, facecolor="none",
                                        edgecolor="red", linewidth=1.1,
                                        linestyle=(0, (3, 1.5))))
    txt = (f"Jaccard={st['jaccard']:.3f}  Spearman(pre,post)={st['spearman_pre_post']:.4f}  "
           f"Kendall={st['kendall_pre_post']:.4f}\n"
           f"rho(rank_shift, edge)={st['rank_shift_vs_edge_spearman']:.4f}   "
           f"(+ => merger demotes high-edge/text)\n"
           f"mean Sobel | (a)pre-only {st['edge_a_pre_only']:.3f}  "
           f"(b)post-only {st['edge_b_post_only']:.3f}  "
           f"(c)both-keep {st['edge_c_agree_keep']:.3f}\n"
           f">median-edge frac | (a){st['frac_above_median_edge_a']:.2f}  "
           f"(b){st['frac_above_median_edge_b']:.2f}\n"
           f"kept/dropped Sobel | PRE {st['pre_edge_keep']:.3f}/{st['pre_edge_drop']:.3f}  "
           f"POST {st['post_edge_keep']:.3f}/{st['post_edge_drop']:.3f}")
    axes[2].text(0.008, 0.992, txt, transform=axes[2].transAxes, fontsize=8.5,
                 va="top", ha="left", color="white", family="monospace",
                 bbox=dict(fc="black", alpha=0.62, ec="none", pad=4))
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    axes[2].legend(handles=[Line2D([], [], color="lime", lw=2, label="PRE kept"),
                            Line2D([], [], color="red", lw=2,
                                   linestyle=(0, (3, 1.5)), label="POST kept"),
                            Patch(facecolor="orange", alpha=0.55,
                                  label="(a) pre-kept / post-DROPPED")],
                   loc="lower left", fontsize=8.5, framealpha=0.85)
    axes[2].set_title("Overlap (PRE green / POST red) + (a)=orange", fontsize=11)
    axes[2].set_xticks([]); axes[2].set_yticks([])

    fig.colorbar(sm1, ax=axes[0], label="PRE unit L2 (kept, min-max)")
    fig.colorbar(sm2, ax=axes[1], label="POST token L2 (kept, min-max)")

    doc = s["documented"]
    fig.suptitle(
        f"{s['key'].upper()} {s['id']} — keep 25% (r=0.75), L2 selector, "
        f"unit = {UNIT}×{UNIT}px (patch {PATCH}px, 2×2 merge); grid {h}×{w} patches\n"
        f"Q: \"{s['question']}\"   GT: {s['gt']}\n"
        f"documented answers — PRE {doc['pre']}   |   POST {doc['post']}\n"
        f"score source: PRE = block-8 (deepstack_0) unit L2 ; "
        f"POST = merged token (main+deepstack cat) L2    ({s['acc_ctx']})",
        fontsize=10.5)
    fig.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# --mode capture: compact per-image stats over a deterministic sample (GPU).
# --------------------------------------------------------------------------- #
def sample_indices(n_total: int, n: int, seed: int) -> list[int]:
    """Deterministic fixed-seed sample of line indices (sorted)."""
    rng = np.random.default_rng(seed)
    return sorted(int(i) for i in rng.choice(n_total, size=n, replace=False))


def capture_bench(llm, cap, sp, bench: str, n: int, seed: int, r: float,
                  out_dir: str, attempts: int = 2):
    """Capture one bench. RESUMABLE: if {bench}.npz already exists, images
    already captured (by id) are skipped and new captures are appended, so a
    re-run tops up images that transiently failed (vLLM occasionally completes
    a request without running the vision tower -> grid_thw None; a retry fixes
    it). The sample itself is deterministic (fixed seed)."""
    subset = f"{ROOT}/eval/subsets/{bench}_200.jsonl"
    with open(subset) as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    idx = sample_indices(len(lines), min(n, len(lines)), seed)
    rows = [json.loads(lines[i]) for i in idx]

    # resume: load previously captured arrays/ids
    npz_path = f"{out_dir}/{bench}.npz"
    prev_chunks, prev_counts = ([], [], []), []
    hs, ws, ids = [], [], []
    if os.path.exists(npz_path):
        z = np.load(npz_path)
        prev_chunks = ([z["pre"]], [z["post"]], [z["edge"]])
        prev_counts = [int(c) for c in np.diff(z["offsets"])]
        ids = z["ids"].tolist(); hs = z["h"].tolist(); ws = z["w"].tolist()
        print(f"[capture] {bench}: resuming ({len(ids)} images cached in npz)",
              flush=True)
    new_pre, new_post, new_edge = [], [], []
    done = set(ids)

    n_fail = 0
    for j, row in enumerate(rows):
        if str(row["id"]) in done:
            continue
        pre = post = edge = None
        for attempt in range(1, attempts + 1):
            reset(cap)
            msgs = [{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": "file://" + row["image"]}},
                {"type": "text", "text": row["question"]},
            ]}]
            try:
                out = llm.chat([msgs], sampling_params=sp)[0]
                pre, post, h, w = scores_from_cap(cap)
                edge = unit_edge_from_image(row["image"], h, w)
                break
            except Exception as e:
                fin = ""
                try:
                    fin = f" finish={out.outputs[0].finish_reason}"
                except Exception:
                    pass
                print(f"[capture] {bench} {row['id']}: attempt {attempt} "
                      f"failed ({type(e).__name__}: {str(e)[:100]}{fin})",
                      flush=True)
        if pre is None:
            n_fail += 1
            print(f"[capture] {bench} {row['id']}: SKIP after {attempts} "
                  f"attempts", flush=True)
            continue
        new_pre.append(pre.astype(np.float32))
        new_post.append(post.astype(np.float32))
        new_edge.append(edge.astype(np.float32))
        hs.append(h); ws.append(w); ids.append(str(row["id"]))
        if (j + 1) % 10 == 0 or j + 1 == len(rows):
            print(f"[capture] {bench}: {j + 1}/{len(rows)} "
                  f"(units this img={pre.shape[0]})", flush=True)

    # merge resumed + new chunks; per-image offsets from per-image counts
    pre_all = np.concatenate(prev_chunks[0] + new_pre) \
        if (prev_chunks[0] or new_pre) else np.zeros(0, np.float32)
    post_all = np.concatenate(prev_chunks[1] + new_post) \
        if (prev_chunks[1] or new_post) else np.zeros(0, np.float32)
    edge_all = np.concatenate(prev_chunks[2] + new_edge) \
        if (prev_chunks[2] or new_edge) else np.zeros(0, np.float32)
    counts = prev_counts + [int(c.shape[0]) for c in new_pre]
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    assert len(ids) == len(counts) == len(offsets) - 1
    assert int(offsets[-1]) == int(pre_all.shape[0]) == int(post_all.shape[0]) \
        == int(edge_all.shape[0])

    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(npz_path, pre=pre_all, post=post_all, edge=edge_all,
                        offsets=offsets,
                        h=np.asarray(hs, np.int32), w=np.asarray(ws, np.int32),
                        ids=np.asarray(ids, dtype=np.str_))
    meta = dict(bench=bench, n_ok=len(ids), n_fail=n_fail, n_requested=n,
                seed=seed, r=r, model=MODEL, max_pixels=MAX_PIXELS,
                subset=subset, sampled_line_indices=idx,
                scoring="L2 both stages (pre=_score_units ds0 input, "
                        "post=_score_tokens cat(main,ds0..2))",
                units_mean=float(np.mean(np.diff(offsets))) if len(ids) else 0.0,
                timestamp=datetime.datetime.now().isoformat(timespec="seconds"))
    with open(f"{out_dir}/{bench}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[capture] {bench}: saved {len(ids)} images -> {npz_path} "
          f"({os.path.getsize(npz_path) / 1e6:.2f} MB, {n_fail} failed, "
          f"{len(new_pre)} new)", flush=True)
    return meta


def run_capture(args):
    from vllm import LLM, SamplingParams
    benches = BENCHES if args.bench == "all" else [args.bench]
    llm = LLM(model=MODEL, dtype="bfloat16", tensor_parallel_size=1,
              gpu_memory_utilization=0.90, max_model_len=32768,
              trust_remote_code=False, enforce_eager=True,
              limit_mm_per_prompt={"image": 1},
              allowed_local_media_path=ROOT,
              max_num_seqs=4, enable_prefix_caching=False, seed=args.seed,
              max_num_batched_tokens=32768,
              mm_processor_kwargs={"max_pixels": MAX_PIXELS})
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    cap = wrap_capture(model.visual)
    sp = SamplingParams(max_tokens=1, temperature=0.0)   # capture-only forward
    metas = {}
    for bench in benches:
        metas[bench] = capture_bench(llm, cap, sp, bench, args.n, args.seed,
                                     args.r, args.out_dir)
    with open(f"{args.out_dir}/capture_meta.json", "w") as f:
        json.dump(metas, f, indent=2)
    print(f"[capture] all done -> {args.out_dir}", flush=True)


# --------------------------------------------------------------------------- #
# --mode analyze: aggregate M1/M2 over the sample, write stats + figures (CPU).
# --------------------------------------------------------------------------- #
def load_bench(out_dir: str, bench: str, r: float):
    z = np.load(f"{out_dir}/{bench}.npz")
    pre, post, edge = z["pre"], z["post"], z["edge"]
    offsets, hs, ws = z["offsets"], z["h"], z["w"]
    ids = z["ids"].tolist() if "ids" in z.files else [str(i) for i in range(len(offsets) - 1)]
    per_img = []
    for i in range(len(offsets) - 1):
        a, b = int(offsets[i]), int(offsets[i + 1])
        st = analyze_arrays(pre[a:b].astype(np.float64),
                            post[a:b].astype(np.float64),
                            edge[a:b].astype(np.float64),
                            int(hs[i]), int(ws[i]), r)["stats"]
        st["id"] = ids[i]
        per_img.append(st)
    return per_img


def aggregate(img_stats: list[dict]) -> dict:
    """mean +/- std (over images) of the per-image mechanism stats."""
    keys = ["spearman_pre_post", "kendall_pre_post", "jaccard",
            "rank_shift_vs_edge_spearman", "edge_a_pre_only",
            "edge_b_post_only", "edge_c_agree_keep",
            "frac_above_median_edge_a", "frac_above_median_edge_b",
            "num_units"]
    out = {"n": len(img_stats)}
    for key in keys:
        v = np.asarray([s[key] for s in img_stats], dtype=np.float64)
        out[key + "_mean"] = round(float(np.nanmean(v)), 4)
        out[key + "_std"] = round(float(np.nanstd(v, ddof=1)), 4) if len(v) > 1 else 0.0
    # per-image series (for transparency / later plotting)
    for key in ["spearman_pre_post", "kendall_pre_post", "jaccard",
                "rank_shift_vs_edge_spearman"]:
        out[key + "_per_image"] = [round(float(s[key]), 4) for s in img_stats]
    return out


def draw_m1(agg: dict[str, dict], out_base: str, r: float):
    keep_pct = int(round((1.0 - r) * 100))
    colors = {"docvqa": "#1f77b4", "textvqa": "#2ca02c", "gqa": "#ff7f0e"}
    metrics = [("spearman_pre_post", "Spearman rho (pre vs post, all units)"),
               ("kendall_pre_post", "Kendall tau (pre vs post, all units)"),
               ("jaccard", f"Top-{keep_pct}% kept-set Jaccard")]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), constrained_layout=True)
    rng = np.random.default_rng(0)
    for ax, (key, title) in zip(axes, metrics):
        data = [np.asarray(agg[b][key + "_per_image"]) for b in BENCHES]
        bp = ax.boxplot(data, widths=0.45, patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", lw=1.4))
        for patch, b in zip(bp["boxes"], BENCHES):
            patch.set_facecolor(colors[b]); patch.set_alpha(0.45)
        for i, (b, d) in enumerate(zip(BENCHES, data)):
            jit = rng.uniform(-0.09, 0.09, len(d))
            ax.scatter(np.full(len(d), i + 1) + jit, d, s=9, color=colors[b],
                       alpha=0.55, zorder=3,
                       label=("text-dense" if b != "gqa" else "object") if i < 3 else None)
        ax.set_title(title, fontsize=10.5)
        ax.set_xticks(range(1, 4)); ax.set_xticklabels(BENCHES)
        lo, hi = ax.get_ylim()
        pad = (hi - lo) * 0.07
        ax.set_ylim(lo - pad, hi + pad)
        for i, (b, d) in enumerate(zip(BENCHES, data)):
            ax.text(i + 1, lo - pad * 0.45,
                    f"{d.mean():.3f}±{d.std(ddof=1):.3f}",
                    ha="center", va="top", fontsize=8.5, color=colors[b])
        if key == "jaccard":
            ax.axhline(1.0, color="grey", lw=0.8, ls=":")
    axes[0].text(0, -0.30, "mean±std over images", transform=axes[0].transAxes,
                 fontsize=8.5, color="grey")
    fig.suptitle(
        f"M1 — merger reshuffles merge-unit saliency ranks "
        f"(PRE block-8 L2 vs POST merged-token L2; keep {keep_pct}%, "
        f"n=images/bench: " + ", ".join(f"{b}={agg[b]['n']}" for b in BENCHES) + ")",
        fontsize=11)
    fig.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    plt.close(fig)


def draw_m2(agg: dict[str, dict], out_base: str, r: float):
    keep_pct = int(round((1.0 - r) * 100))
    colors = {"docvqa": "#1f77b4", "textvqa": "#2ca02c", "gqa": "#ff7f0e"}
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.5), constrained_layout=True)
    x = np.arange(len(BENCHES))

    # panel 1: mean Sobel edge of disagreement groups a/b/c
    wbar = 0.26
    grp_keys = [("edge_a_pre_only", "(a) pre-kept / post-DROPPED", "#d62728"),
                ("edge_b_post_only", "(b) post-kept / pre-dropped", "#9467bd"),
                ("edge_c_agree_keep", "(c) kept by both", "#7f7f7f")]
    for gi, (key, lab, c) in enumerate(grp_keys):
        means = [agg[b][key + "_mean"] for b in BENCHES]
        stds = [agg[b][key + "_std"] for b in BENCHES]
        axes[0].bar(x + (gi - 1) * wbar, means, wbar, yerr=stds, color=c,
                    alpha=0.85, label=lab, capsize=3, error_kw=dict(lw=1))
    axes[0].set_xticks(x); axes[0].set_xticklabels(BENCHES)
    axes[0].set_ylabel("mean Sobel edge energy / unit")
    axes[0].set_title("Demoted-by-merger units carry the most text (edge)",
                      fontsize=10.5)
    axes[0].legend(fontsize=8.2, loc="upper right")

    # panel 2: Spearman(rank_shift, edge) per image, box + mean
    data = [np.asarray(agg[b]["rank_shift_vs_edge_spearman_per_image"])
            for b in BENCHES]
    bp = axes[1].boxplot(data, widths=0.45, patch_artist=True, showfliers=False,
                         medianprops=dict(color="black", lw=1.4))
    for patch, b in zip(bp["boxes"], BENCHES):
        patch.set_facecolor(colors[b]); patch.set_alpha(0.5)
    axes[1].axhline(0.0, color="grey", lw=0.8, ls="--")
    axes[1].set_xticks(range(1, 4)); axes[1].set_xticklabels(BENCHES)
    axes[1].set_ylabel("Spearman rho(rank_shift, edge)")
    axes[1].set_title("rank_shift vs edge per image (+ => merger demotes "
                      "high-edge/text)", fontsize=10.5)
    lo, hi = axes[1].get_ylim(); pad = (hi - lo) * 0.09
    axes[1].set_ylim(lo - pad, hi + pad)
    for i, d in enumerate(data):
        axes[1].text(i + 1, lo - pad * 0.4, f"{d.mean():.3f}±{d.std(ddof=1):.3f}",
                     ha="center", va="top", fontsize=8.5, color=colors[BENCHES[i]])

    # panel 3: above-median-edge fraction, (a) vs (b)
    wbar = 0.36
    fa = [agg[b]["frac_above_median_edge_a_mean"] for b in BENCHES]
    fb = [agg[b]["frac_above_median_edge_b_mean"] for b in BENCHES]
    fa_s = [agg[b]["frac_above_median_edge_a_std"] for b in BENCHES]
    fb_s = [agg[b]["frac_above_median_edge_b_std"] for b in BENCHES]
    axes[2].bar(x - wbar / 2, fa, wbar, yerr=fa_s, color="#d62728", alpha=0.85,
                label="(a) pre-kept / post-DROPPED", capsize=3, error_kw=dict(lw=1))
    axes[2].bar(x + wbar / 2, fb, wbar, yerr=fb_s, color="#9467bd", alpha=0.85,
                label="(b) post-kept / pre-dropped", capsize=3, error_kw=dict(lw=1))
    axes[2].axhline(0.5, color="grey", lw=0.8, ls="--")
    axes[2].set_xticks(x); axes[2].set_xticklabels(BENCHES)
    axes[2].set_ylabel("frac units > per-image median edge")
    axes[2].set_title("High-edge majority among demoted units", fontsize=10.5)
    axes[2].legend(fontsize=8.2, loc="upper right")

    fig.suptitle(
        f"M2 — the units POST drops (that PRE keeps) are high-edge/text units "
        f"(keep {keep_pct}%; n: "
        + ", ".join(f"{b}={agg[b]['n']}" for b in BENCHES) + ")",
        fontsize=11)
    fig.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    plt.close(fig)


def run_analyze(args):
    fig_dir = f"{ROOT}/drafts/figures"
    os.makedirs(fig_dir, exist_ok=True)
    per_bench_imgs: dict[str, list[dict]] = {}
    for bench in BENCHES:
        npz = f"{args.out_dir}/{bench}.npz"
        if not os.path.exists(npz):
            raise SystemExit(f"missing capture {npz}; run --mode capture first")
        per_bench_imgs[bench] = load_bench(args.out_dir, bench, args.r)
        print(f"[analyze] {bench}: {len(per_bench_imgs[bench])} images loaded",
              flush=True)

    agg = {b: aggregate(per_bench_imgs[b]) for b in BENCHES}
    agg["text_dense"] = aggregate(per_bench_imgs["docvqa"]
                                  + per_bench_imgs["textvqa"])
    agg["object"] = aggregate(per_bench_imgs["gqa"])

    for name, a in agg.items():
        print(f"[analyze] {name} (n={a['n']}): "
              f"spearman={a['spearman_pre_post_mean']}±{a['spearman_pre_post_std']} "
              f"kendall={a['kendall_pre_post_mean']}±{a['kendall_pre_post_std']} "
              f"jaccard={a['jaccard_mean']}±{a['jaccard_std']} | "
              f"rho(shift,edge)={a['rank_shift_vs_edge_spearman_mean']}±{a['rank_shift_vs_edge_spearman_std']} "
              f"edge a/b/c={a['edge_a_pre_only_mean']}/{a['edge_b_post_only_mean']}/{a['edge_c_agree_keep_mean']} "
              f"frac>med a/b={a['frac_above_median_edge_a_mean']}/{a['frac_above_median_edge_b_mean']}",
              flush=True)

    draw_m1(agg, f"{fig_dir}/token_survival_m1_rank_overlap", args.r)
    draw_m2(agg, f"{fig_dir}/token_survival_m2_edge_demotion", args.r)
    print(f"[analyze] wrote {fig_dir}/token_survival_m1_rank_overlap.png + "
          f"token_survival_m2_edge_demotion.png", flush=True)

    # merge into the stats json (preserve legacy single-image entries)
    stats_path = args.stats_json
    old = {}
    if os.path.exists(stats_path):
        try:
            old = json.load(open(stats_path))
        except json.JSONDecodeError:
            old = {}
    out = {k: v for k, v in old.items() if k not in ("sample", "meta")}
    out["meta"] = dict(
        generated=datetime.date.today().isoformat(),
        r=args.r, keep_pct=round((1 - args.r) * 100, 1),
        scoring="L2 both stages (PRE = _score_units on deepstack[0] input; "
                "POST = _score_tokens on cat(main,ds0..2)) — matches headline "
                "--selector l2 cells",
        capture_dir=args.out_dir,
        per_bench_n={b: agg[b]["n"] for b in BENCHES},
        note="top-level 'docvqa'/'textvqa' = legacy single-image deep dives; "
             "'sample' = M1/M2 over the deterministic n-image captures "
             "(text_dense = docvqa+textvqa pooled, object = gqa)")
    out["sample"] = agg
    with open(stats_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[analyze] wrote {stats_path}", flush=True)


# --------------------------------------------------------------------------- #
# --mode legacy: the original 2-image deep-dive (cached captures).
# --------------------------------------------------------------------------- #
def run_legacy(args):
    global ANALYZE_IMG
    os.makedirs(f"{ROOT}/drafts/figures", exist_ok=True)
    CAP_DIR = f"{ROOT}/runs/v3_attn_robust"
    os.makedirs(CAP_DIR, exist_ok=True)
    cap_path = lambda s: f"{CAP_DIR}/_vizcap_{s['key']}_{s['id']}.pt"

    # capture phase (GPU) only for samples lacking a cached capture
    need_gpu = [s for s in SAMPLES if not os.path.exists(cap_path(s))]
    if need_gpu:
        from vllm import LLM, SamplingParams
        llm = LLM(model=MODEL, dtype="bfloat16", tensor_parallel_size=1,
                  gpu_memory_utilization=0.90, max_model_len=32768,
                  trust_remote_code=False, enforce_eager=True,
                  limit_mm_per_prompt={"image": 1},
                  allowed_local_media_path=ROOT,
                  max_num_seqs=4, enable_prefix_caching=False, seed=0,
                  max_num_batched_tokens=32768,
                  mm_processor_kwargs={"max_pixels": MAX_PIXELS})
        model = llm.llm_engine.model_executor.driver_worker.model_runner.model
        cap = wrap_capture(model.visual)
        sp = SamplingParams(max_tokens=32, temperature=0.0)
        for s in need_gpu:
            reset(cap)
            msgs = [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "file://" + s["image"]}},
                {"type": "text", "text": s["question"]},
            ]}]
            out = llm.chat([msgs], sampling_params=sp)[0]
            torch.save({"grid_thw": cap["grid_thw"], "ds_in": cap["ds_in"],
                        "out": cap["out"], "answer": out.outputs[0].text.strip()},
                       cap_path(s))
            print(f"[viz] cached capture -> {cap_path(s)}", flush=True)
    else:
        print("[viz] all captures cached; skipping LLM load/GPU (reuse mode)", flush=True)

    summary = {}
    stats_path = args.stats_json
    old = {}
    if os.path.exists(stats_path):
        try:
            old = json.load(open(stats_path))
        except json.JSONDecodeError:
            old = {}
    for s in SAMPLES:
        c = torch.load(cap_path(s), weights_only=True, map_location="cpu")
        cap = {"grid_thw": c["grid_thw"], "ds_in": c["ds_in"], "out": c["out"]}
        ANALYZE_IMG = s["image"]
        res = analyze(cap)
        res["stats"]["capture_answer"] = c["answer"]
        st = res["stats"]
        print(f"[viz] {s['key']} {s['id']}: grid=({st['h']},{st['w']}) "
              f"units={st['num_units']} k={st['k']} "
              f"jaccard={st['jaccard']:.3f} "
              f"pre_edge={st['pre_edge_keep']:.3f}/{st['pre_edge_drop']:.3f} "
              f"post_edge={st['post_edge_keep']:.3f}/{st['post_edge_drop']:.3f} "
              f"answer={c['answer']!r}", flush=True)
        print(f"[viz]   M1: spearman(pre,post)={st['spearman_pre_post']:.4f} "
              f"kendall(pre,post)={st['kendall_pre_post']:.4f} | "
              f"M2: rho(rank_shift,edge)={st['rank_shift_vs_edge_spearman']:.4f} "
              f"edge a/b/c={st['edge_a_pre_only']:.3f}/{st['edge_b_post_only']:.3f}/{st['edge_c_agree_keep']:.3f} "
              f"frac>med a/b={st['frac_above_median_edge_a']:.2f}/{st['frac_above_median_edge_b']:.2f} "
              f"n a/b/c={st['n_grp_a']}/{st['n_grp_b']}/{st['n_grp_c']}", flush=True)
        base = f"{ROOT}/drafts/figures/token_survival_{s['key']}"
        draw(s, res, base)
        print(f"[viz] wrote {base}.png / .pdf", flush=True)
        summary[s["key"]] = res["stats"]
    merged = {**old, **summary}          # keep 'sample'/'meta' if present
    with open(stats_path, "w") as f:
        json.dump(merged, f, indent=2)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["legacy", "capture", "analyze"],
                    default="legacy",
                    help="legacy = 2-image deep-dive figures (default); "
                         "capture = GPU capture of compact per-image stats for "
                         "a deterministic n-image sample; analyze = CPU M1/M2 "
                         "aggregation + figures + stats json.")
    ap.add_argument("--bench", default="all",
                    choices=["docvqa", "textvqa", "gqa", "all"],
                    help="capture: which benchmark subset to sample (all = the "
                         "three in ONE model load).")
    ap.add_argument("--n", type=int, default=64,
                    help="capture: images per benchmark (deterministic sample).")
    ap.add_argument("--seed", type=int, default=0,
                    help="capture: sample + vLLM seed (fixed => same images).")
    ap.add_argument("--r", type=float, default=R,
                    help="keep = 1-r; target keep-ratio for k and Jaccard@k.")
    ap.add_argument("--out-dir",
                    default=f"{ROOT}/runs/v3_merger_aware/survival_capture",
                    help="capture output dir (npz + meta).")
    ap.add_argument("--stats-json",
                    default=f"{ROOT}/drafts/figures/token_survival_stats.json")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.mode == "capture":
        run_capture(args)
    elif args.mode == "analyze":
        run_analyze(args)
    else:
        run_legacy(args)


if __name__ == "__main__":
    main()
