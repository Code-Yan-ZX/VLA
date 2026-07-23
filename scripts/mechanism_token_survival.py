"""mechanism_token_survival.py — PRE vs POST merger L2 token-survival figures.

Visualizes which 2x2 merge-units survive top-25% L2 selection at each stage,
on two known pre-OK/post-wrong samples (DocVQA 58439, TextVQA 35174).

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
  * keep k = max(1, round(num_units*(1-0.75))) per image (runner contract);
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

HONESTY: the script computes objective per-unit Sobel edge energy (text-stroke
proxy) for kept vs dropped sets and the pre/post Jaccard overlap, prints them,
and puts them on the figure — no threshold tuning.

Outputs: drafts/figures/token_survival_docvqa.{png,pdf},
         drafts/figures/token_survival_textvqa.{png,pdf}
"""
from __future__ import annotations
import os, sys, json
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

ROOT = "/media/disk2/YZX/research/vla"
sys.path.insert(0, os.path.join(ROOT, "src", "v3_premerger"))
from v3_premerger_runner import _score_tokens, _score_units   # EXACT selector code

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
R = 0.75                      # keep 25%
MAX_PIXELS = 1500000          # same as BIG-config cells
PATCH = 16                    # px per patch (verified: vision_config.patch_size)
MERGE = 2                     # 2x2 (verified: spatial_merge_size)
UNIT = PATCH * MERGE          # 32 px per merge-unit side

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


def analyze(cap):
    """Replicate runner L2 selection on captured features for ONE image."""
    g = cap["grid_thw"]
    assert g.shape[0] == 1, f"expected 1 image/pass, got {g.shape}"
    t, h, w = (int(x) for x in g[0].tolist())
    assert t == 1
    num_units = (h * w) // (MERGE ** 2)

    # PRE: deepstack[0] merger input = block-8 features [N,1,1152] -> units [n,4,1152]
    hs_ds0 = cap["ds_in"][0]                       # [N, 1, 1152]
    ctx = hs_ds0.shape[-1]
    feats = hs_ds0.reshape(num_units, MERGE ** 2, ctx)
    pre_scores = _score_units(feats, "l2").numpy()

    # POST: visual output row = cat(main, ds0, ds1, ds2) [num_units, 4*4096]
    main = cap["out"]["merger"]
    ds = [cap["out"][f"ds{i}"] for i in range(3)]
    post_feat = torch.cat([main] + ds, dim=1)      # [num_units, 16384]
    post_scores = _score_tokens(post_feat, "l2").numpy()

    k = max(1, int(round(num_units * (1.0 - R))))  # runner contract
    pre_keep = set(np.argsort(-pre_scores)[:k].tolist())
    post_keep = set(np.argsort(-post_scores)[:k].tolist())

    # objective text-stroke proxy: Sobel edge energy per 32px unit
    H, W = h * PATCH, w * PATCH
    gray = np.asarray(
        Image.open(ANALYZE_IMG).convert("L").resize((W, H), Image.BICUBIC)
    ).astype(np.float32) / 255.0
    from scipy.ndimage import sobel
    ex, ey = sobel(gray, axis=1), sobel(gray, axis=0)
    edge = np.hypot(ex, ey)
    unit_edge = np.zeros(num_units)
    for u in range(num_units):
        ur, uc = divmod(u, w // MERGE)
        unit_edge[u] = edge[ur * UNIT:(ur + 1) * UNIT,
                            uc * UNIT:(uc + 1) * UNIT].mean()

    jacc = len(pre_keep & post_keep) / len(pre_keep | post_keep)
    stats = dict(
        num_units=num_units, k=k, h=h, w=w, H=H, W=W,
        jaccard=round(jacc, 3),
        pre_edge_keep=round(float(unit_edge[list(pre_keep)].mean()), 4),
        pre_edge_drop=round(float(unit_edge[sorted(set(range(num_units)) - pre_keep)].mean()), 4),
        post_edge_keep=round(float(unit_edge[list(post_keep)].mean()), 4),
        post_edge_drop=round(float(unit_edge[sorted(set(range(num_units)) - post_keep)].mean()), 4),
    )
    return dict(pre=pre_scores, post=post_scores, pre_keep=pre_keep,
                post_keep=post_keep, unit_edge=unit_edge, stats=stats)


ANALYZE_IMG = None  # set per-sample (analyze uses it for edge energy)


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

    # panel 3: both outlines + objective stats
    axes[2].imshow(img)
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
    txt = (f"Jaccard(pre,post)={st['jaccard']}\n"
           f"mean Sobel | kept vs dropped:\n"
           f"  PRE : {st['pre_edge_keep']:.3f} vs {st['pre_edge_drop']:.3f}\n"
           f"  POST: {st['post_edge_keep']:.3f} vs {st['post_edge_drop']:.3f}")
    axes[2].text(0.008, 0.992, txt, transform=axes[2].transAxes, fontsize=9,
                 va="top", ha="left", color="white", family="monospace",
                 bbox=dict(fc="black", alpha=0.62, ec="none", pad=4))
    from matplotlib.lines import Line2D
    axes[2].legend(handles=[Line2D([], [], color="lime", lw=2, label="PRE kept"),
                            Line2D([], [], color="red", lw=2,
                                   linestyle=(0, (3, 1.5)), label="POST kept")],
                   loc="lower left", fontsize=9, framealpha=0.85)
    axes[2].set_title("Overlap (PRE green / POST red)", fontsize=11)
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


def main():
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
    for s in SAMPLES:
        c = torch.load(cap_path(s), weights_only=True, map_location="cpu")
        cap = {"grid_thw": c["grid_thw"], "ds_in": c["ds_in"], "out": c["out"]}
        ANALYZE_IMG = s["image"]
        res = analyze(cap)
        res["stats"]["capture_answer"] = c["answer"]
        print(f"[viz] {s['key']} {s['id']}: grid=({res['stats']['h']},{res['stats']['w']}) "
              f"units={res['stats']['num_units']} k={res['stats']['k']} "
              f"jaccard={res['stats']['jaccard']} "
              f"pre_edge={res['stats']['pre_edge_keep']}/{res['stats']['pre_edge_drop']} "
              f"post_edge={res['stats']['post_edge_keep']}/{res['stats']['post_edge_drop']} "
              f"answer={c['answer']!r}", flush=True)
        base = f"{ROOT}/drafts/figures/token_survival_{s['key']}"
        draw(s, res, base)
        print(f"[viz] wrote {base}.png / .pdf", flush=True)
        summary[s["key"]] = res["stats"]
    with open(f"{ROOT}/drafts/figures/token_survival_stats.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
