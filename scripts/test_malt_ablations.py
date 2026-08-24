#!/usr/bin/env python3
"""Unit tests for the MALT causal ablations (task 2026-08-25).

Tiny random-init Qwen2.5-VL on CPU (no weights, no GPU).  Verifies:

  (A) pure functions
    A1 rb_rho1_keep_set == rankbridge_keep_indices(rho=1.0) keep set
    A2 _apply_ablate_mask blocks exactly (text|anchor query, transient key)
       and leaves every other entry untouched
  (B) tiny-model end-to-end (rankbridge rho=1.0, fastv-k 1, keep 25%):
    B1 keep set / kept positions / L_after identical across ALL arms
       (H1, H2 no_text_read, H3 no_anchor_read, H4 kv_only, no_both)
    B2 ablate="none" is bit-identical to no-ablate key (no behavior change)
    B3 H2/H3/H4 hidden differ from H1 (blocking actually changed something)
    B4 H4 semantics: cache-layer-1 transient K/V == projection of the RAW
       inputs_embeds (the revert) -- proves transients provide K/V only
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src",
                                "v3_premerger"))
from baselines_hf import (  # noqa: E402
    _tiny_model, prefill_pruned, rb_rho1_keep_set, _apply_ablate_mask,
    rankbridge_keep_indices, make_causal_mask, _cache_kv)

HID = 32
DEV = "cpu"


def _rb_cfg(ranks, units, keep_frac=0.25):
    return {"r": 0.75, "fastv_k": 1, "ratios": [1, .75, .5, .25],
            "rb": {"pre_ranks": ranks, "units_per_image": units,
                   "keep_frac": keep_frac, "fuse": "quota", "rho": 1.0,
                   "rrf_lambda": 1.0, "rrf_c": 60.0}}


def test_A1():
    img = torch.zeros(12, dtype=torch.bool); img[2:8] = True   # 6 image tokens
    pr = torch.tensor([4, 1, 3, 6, 2, 5])                      # within-image ranks
    k1 = rb_rho1_keep_set(img, pr, [6], 0.25)
    k2, d2 = rankbridge_keep_indices(
        torch.zeros(1, 1, 12, 12), img, pr, [6], 0.25, "quota", 1.0)
    assert torch.equal(k1, k2), (k1, k2)
    assert int(d2["n_protected"]) == 2  # keep_frac .25 of 6 -> 2 (round)
    print("[A1] OK rb_rho1_keep_set == rankbridge rho=1 (query-blind)")


def test_A2():
    L = 12
    img = torch.zeros(L, dtype=torch.bool); img[2:8] = True
    text = ~img
    # anchors = positions 2,3 (top-2 pre ranks 1,2 are local idx 1,0)
    am = torch.zeros(L, dtype=torch.bool); am[2] = am[3] = True
    tr = img & ~am
    base = make_causal_mask(L, torch.device("cpu"), torch.float32)
    # no_text_read: (text, transient) -> -inf, everything else unchanged
    m = _apply_ablate_mask(base, "no_text_read", text, am, tr)
    for q in text.nonzero(as_tuple=False).view(-1).tolist():
        for k in tr.nonzero(as_tuple=False).view(-1).tolist():
            assert m[0, 0, q, k] == float(torch.finfo(m.dtype).min), (q, k)
    # causality also -inf above diagonal is preserved; below-diag text reads ok
    for q in range(L):
        for k in range(L):
            if not (q in text.nonzero(as_tuple=False).view(-1).tolist()
                    and k in tr.nonzero(as_tuple=False).view(-1).tolist()):
                assert m[0, 0, q, k] == base[0, 0, q, k], (q, k, m[0, 0, q, k])
    # no_anchor_read: (anchor, transient) -> -inf
    m3 = _apply_ablate_mask(base, "no_anchor_read", text, am, tr)
    for q in am.nonzero(as_tuple=False).view(-1).tolist():
        for k in tr.nonzero(as_tuple=False).view(-1).tolist():
            assert m3[0, 0, q, k] == float(torch.finfo(m3.dtype).min), (q, k)
    print("[A2] OK ablate masks block exactly the intended entries")


def test_B():
    torch.manual_seed(7)
    m, cfg = _tiny_model()
    L = 12
    X = torch.randn(1, L, HID)
    P = torch.arange(L).view(1, 1, L).expand(3, 1, L).contiguous().clone()
    img = torch.zeros(L, dtype=torch.bool); img[2:8] = True   # 6 image tokens
    sc = X[0, 2:8].float().norm(dim=-1)
    order = sc.argsort(descending=True)
    ranks = torch.empty(6, dtype=torch.long)
    ranks[order] = torch.arange(1, 7)
    k_def = max(1, int(round(6 * 0.25)))                      # -> 2
    base = _rb_cfg(ranks, [6])
    # ---- H1 reference ----
    h1, pos1, _, _, d1 = prefill_pruned(
        m, X.clone(), P.clone(), img.clone(), "rankbridge", dict(base))
    assert d1["n_image_kept"] == k_def
    # ---- arms ----
    arms = {"none": "none", "H2": "no_text_read", "H3": "no_anchor_read",
            "H4": "kv_only", "no_both": "no_both"}
    hid = {}
    for name, ab in arms.items():
        c = dict(base)
        if ab != "none":
            c["ablate"] = ab
        h, pos, _, im, d = prefill_pruned(
            m, X.clone(), P.clone(), img.clone(), "rankbridge", c)
        hid[name] = h
        # B1: keep set identical everywhere
        assert d["n_image_kept"] == d1["n_image_kept"], (name, d)
        assert d["L_after"] == d1["L_after"], (name, d)
        assert torch.equal(pos, pos1), (name, "pos differs")
        assert d.get("ablate", "none") == ab, (name, d.get("ablate"))
    # B2: explicit none == no key
    assert torch.equal(hid["none"], h1), "ablate=none must be bit-identical"
    # B3: blocking changed survivor hidden vs H1 (nonzero diff; the tiny
    # random model has small attention-output scale so the magnitude is tiny
    # but nonzero -- the real scale question is answered on Qwen3-VL-8B)
    for name in ["H2", "H3", "H4", "no_both"]:
        d = (hid[name] - h1).abs().max().item()
        assert d > 1e-5, f"{name} survivor hidden must differ from H1 (got {d})"
        print(f"[B3] {name} vs H1 survivor hidden maxdiff={d:.2e}")
    # B4: H4 -> layer-1 transient K/V projected from RAW inputs_embeds.
    # (Values carry no RoPE, so they are directly comparable; keys are
    # rope-rotated in-cache for this family.)  We capture value_states at the
    # moment they are WRITTEN to the cache (DynamicCache.update) -- after
    # prefill the transient K/V are gone (cropped at the prune layer), so the
    # post-hoc cache can't be indexed by transient position.
    from transformers import DynamicCache
    captured = {}
    _orig_update = DynamicCache.update

    def _upd(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx == 1:
            captured["v1"] = value_states.detach().clone()
        return _orig_update(self, key_states, value_states, layer_idx,
                            cache_kwargs)

    DynamicCache.update = _upd
    try:
        prefill_pruned(m, X.clone(), P.clone(), img.clone(), "rankbridge",
                       {**base, "ablate": "kv_only"})
    finally:
        DynamicCache.update = _orig_update
    tr_pos = (img & ~torch.zeros(L, dtype=torch.bool)
              .scatter(0, rb_rho1_keep_set(img, ranks, [6], 0.25), True))
    tr_pos = tr_pos.nonzero(as_tuple=False).view(-1)
    layer1 = m.model.language_model.layers[1]
    ln1 = layer1.input_layernorm
    raw = ln1(X[0, tr_pos])                 # [n_tr, H] (rows, not cols)
    v_proj = layer1.self_attn.v_proj(raw)   # [n_tr, kv_heads*head_dim]
    bsz, n_tr = 1, v_proj.shape[0]
    n_head = layer1.self_attn.num_key_value_heads
    hdim = layer1.self_attn.head_dim
    v_exp = v_proj.view(bsz, n_tr, n_head, hdim).transpose(1, 2)
    v_got = captured["v1"][:, :, tr_pos, :]
    assert torch.allclose(v_got, v_exp, atol=1e-4), \
        f"kv_only layer-1 transient V != raw proj (maxdiff=" \
        f"{(v_got-v_exp).abs().max().item():.2e})"
    print("[B4] OK kv_only: layer-1 transient V == projection of raw "
          "inputs_embeds (transients provide K/V only, no self-update)")
    print("[B]  ALL tiny-model ablation checks passed")


if __name__ == "__main__":
    import torch
    test_A1()
    test_A2()
    test_B()
