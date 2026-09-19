#!/usr/bin/env python3
"""PSAR 유닛 — 노션 "Physical-Support Additive Readout (PSAR)" 식·불변식 + C1 visual provenance.

usage: python tests_hj_test_psar.py   (CPU)
"""
import os
import sys
from copy import deepcopy
from types import SimpleNamespace

import torch

TREE = "/home/users/whddn12316/wsi_latent_0915_decode_hj"
sys.path.insert(0, TREE)
from memory import psar  # noqa: E402

PASS = FAIL = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {extra}")


print("[config / groups]")
check("window 24-30", psar.layer_window("24-30", 36) == (24, 30))
check("window clipped", psar.layer_window("30-99", 36) == (30, 35))
check("window single", psar.layer_window("27", 36) == (27, 27))
# obs 0,1 roots (5x); obs 2,4 children of 0; obs 3 child of 1. C1 emptied obs 2 except one column,
# columns arrive in cache order but crop membership is read per column (no contiguity assumed)
parents = (-1, -1, 0, 1, 0)
obs_ids = [0, 0, 2, 4, 1, 3, 3]
g = psar.support_groups(obs_ids, parents, "branch")
check("branch groups from per-column obs", g == [[0, 1, 2, 3], [4, 5, 6]], g)
check("obs groups", psar.support_groups(obs_ids, parents, "obs") == [[0, 1], [4], [2], [5, 6], [3]])
check("self-parent -> own support", psar.support_groups([0, 1, 1], (-1, 1), "branch") == [[0], [1, 2]])
gs = psar.support_groups(obs_ids, parents, "shuffled", seed=3)
check("shuffled keeps sizes, covers all", sorted(len(x) for x in gs) == [3, 4]
      and sorted(j for x in gs for j in x) == list(range(7)))
try:
    psar.support_groups(obs_ids, parents, "grid")
    check("bad support rejected", False)
except ValueError:
    check("bad support rejected", True)

print("[pure algebra]")
torch.manual_seed(0)
m = torch.randn(2, 3, 5)
ref = torch.randn(2, 3, 5)
check("MatchScale norm = ref norm", torch.allclose(psar.match_scale(m, ref).norm(dim=-1), ref.norm(dim=-1), atol=1e-6))
o_n, o_v = torch.randn(4, 8), torch.randn(4, 8)
check("eta=0 visual = native", torch.allclose(psar.combine(o_n, o_v, m.new_ones(4, 8), 0.0, "visual"), o_n + o_v))
check("eta=0 full = native", torch.allclose(psar.combine(o_n, o_v, m.new_ones(4, 8), 0.0, "full"), o_n + o_v))
# within-support multiplicity: duplicating every token of support A leaves m_slide unchanged
H, T, D = 2, 3, 6
sc = torch.randn(1, H, T, 5)
Vv = torch.randn(1, H, 5, D)
groups = [[0, 1, 2], [3, 4]]
base = psar.support_message(sc, Vv, groups)
sc_dup = torch.cat([sc, sc[..., [0, 1, 2]]], dim=-1)
V_dup = torch.cat([Vv, Vv[:, :, [0, 1, 2]]], dim=-2)
dup = psar.support_message(sc_dup, V_dup, [[0, 1, 2, 5, 6, 7], [3, 4]])
check("within support: duplicating a support's tokens leaves m_slide unchanged", torch.allclose(base, dup, atol=1e-6))
# the native single-softmax read DOES move under the same duplication (what PSAR removes)
nat = torch.matmul(torch.softmax(sc, -1), Vv)
nat_dup = torch.matmul(torch.softmax(sc_dup, -1), V_dup)
check("native read moves under the same duplication (contrast)", not torch.allclose(nat, nat_dup, atol=1e-3))
# across supports: equal weight regardless of logit scale of a support
sc_boost = sc.clone(); sc_boost[..., [3, 4]] += 50.0
check("across supports: a support's logit offset does not change its weight",
      torch.allclose(psar.support_message(sc_boost, Vv, groups), base, atol=1e-5))
one = psar.support_message(sc, Vv, [[0, 1, 2, 3, 4]])
check("single support = renormalized native visual read", torch.allclose(one, nat, atol=1e-6))


print("[attention layer]")


def tiny(n_layers=3):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention, Qwen3VLTextRotaryEmbedding
    cfg = Qwen3VLTextConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                            intermediate_size=128, num_hidden_layers=n_layers, vocab_size=50,
                            rope_scaling={"rope_type": "default", "mrope_section": [3, 3, 2]})
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    attns = [Qwen3VLTextAttention(cfg, layer_idx=i).eval() for i in range(n_layers)]
    rope = Qwen3VLTextRotaryEmbedding(cfg)
    bb = SimpleNamespace(lm=SimpleNamespace(layers=[SimpleNamespace(self_attn=a) for a in attns]),
                         device=torch.device("cpu"))
    return bb, attns, rope


def causal(past, T):
    mk = torch.full((1, 1, T, past + T), float("-inf"))
    for r in range(T):
        mk[0, 0, r, : past + r + 1] = 0.0
    return mk


def run(attn, rope, hidden, past, cache):
    T = hidden.shape[1]
    pos = torch.arange(past, past + T)[None, None, :].expand(3, 1, T)
    out = attn(hidden_states=hidden, position_embeddings=rope(hidden, pos),
               attention_mask=causal(past, T), past_key_values=cache)
    return out[0] if isinstance(out, tuple) else out


try:
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, repeat_kv

    bb, attns, rope = tiny()
    attn = attns[0]
    torch.manual_seed(1)
    ctx, rows = torch.randn(1, 14, 64), torch.randn(1, 4, 64)
    vis = torch.arange(2, 10)                        # 8 visual columns
    base_cache = DynamicCache()
    with torch.no_grad():
        run(attn, rope, ctx, 0, base_cache)
        native = run(attn, rope, rows, 14, deepcopy(base_cache))
    groups2 = [[0, 1, 2, 3, 4], [5, 6, 7]]

    def go(**kw):
        with torch.no_grad(), psar.install_psar(bb, vis, kw.pop("groups", groups2), layers=(0, 0), **kw) as st:
            out = run(attn, rope, rows, 14, deepcopy(base_cache))
        return out, st

    ident, st = go(identity=True, rows="all")
    check("identity rows=all == native eager", torch.allclose(ident, native, atol=1e-5), float((ident - native).abs().max()))
    check("identity maxdiff recorded small", st["maxdiff_identity"] < 1e-4)
    out_gen, _ = go(rows="gen")
    check("rows=gen: prompt rows before the last untouched (exact)", torch.equal(out_gen[:, :3], native[:, :3]))
    check("rows=gen: last row changed", not torch.allclose(out_gen[:, 3], native[:, 3], atol=1e-4))
    out_eta0, _ = go(rows="all", eta=0.0)
    check("eta=0 == native", torch.allclose(out_eta0, native, atol=1e-5))
    out_one, _ = go(rows="all", groups=[list(range(8))])
    check("single support, target=visual, eta=1 == native (invariant)", torch.allclose(out_one, native, atol=1e-5),
          float((out_one - native).abs().max()))

    def manual(target, eta, grp):
        c = deepcopy(base_cache)
        with torch.no_grad():
            run(attn, rope, rows, 14, c)
            K = repeat_kv(c.layers[0].keys, 2); Vf = repeat_kv(c.layers[0].values, 2)
            pos = torch.arange(14, 18)[None, None, :].expand(3, 1, 4)
            cos, sin = rope(rows, pos)
            q = attn.q_norm(attn.q_proj(rows).view(1, 4, -1, 16)).transpose(1, 2)
            q, _ = apply_rotary_pos_emb(q, torch.zeros_like(q), cos, sin)
            s = q @ K.transpose(-2, -1) * attn.scaling + causal(14, 4)
            a = torch.softmax(s, -1)
            mask = torch.zeros(18, dtype=torch.bool); mask[vis] = True
            oV = (a * mask.float()) @ Vf; oN = (a * (~mask).float()) @ Vf
            ms = []
            for gg in grp:
                cols_g = vis[gg]
                ms.append(torch.softmax(s[..., cols_g], -1) @ Vf[:, :, cols_g, :])
            msl = torch.stack(ms).mean(0)
            refv = oV if target == "visual" else oV + oN
            scaled = msl * refv.norm(dim=-1, keepdim=True) / msl.norm(dim=-1, keepdim=True)
            new = (oN + (1 - eta) * oV + eta * scaled) if target == "visual" else ((1 - eta) * (oN + oV) + eta * scaled)
            return attn.o_proj(new.transpose(1, 2).reshape(1, 4, -1))

    for target, eta in (("visual", 1.0), ("visual", 0.3), ("full", 1.0), ("full", 0.5)):
        got, _ = go(rows="all", target=target, eta=eta)
        want = manual(target, eta, groups2)
        check(f"formula target={target} eta={eta}", torch.allclose(got, want, atol=1e-5), float((got - want).abs().max()))

    # hooks only inside the window
    bb3, attns3, rope3 = tiny(3)
    with psar.install_psar(bb3, vis, groups2, layers=(1, 1)):
        hooked = [len(a._forward_hooks) for a in attns3]
    check("hooks only on window layers, removed after", hooked == [0, 1, 0] and all(len(a._forward_hooks) == 0 for a in attns3))
except Exception as exc:  # noqa: BLE001
    import traceback
    traceback.print_exc()
    check("attention block", False, repr(exc))

print("[engine_groups / provenance guard]")
meta = {"abs_cols": torch.tensor([30, 31, 40, 41, 50]), "obs_id": torch.tensor([0, 0, 1, 2, 2]),
        "parents": (-1, 0, -1), "c1": True, "kept_counts": (2, 1, 2)}
eng = SimpleNamespace(_backbone=SimpleNamespace(_visual_meta=meta), _rpath_visual_cols=torch.tensor([30, 31, 40, 41, 50]),
                      _rpath_case_index=0)
r = psar.engine_groups(eng, "branch")
check("groups from provenance", r["groups"] == [[0, 1, 2], [3, 4]], r)
eng.__dict__["_rpath_visual_cols"] = torch.tensor([30, 31, 40, 41, 51])
r = psar.engine_groups(eng, "branch")
check("column mismatch -> no guessed mapping", r["groups"] is None and "!=" in r["note"])
eng2 = SimpleNamespace(_backbone=SimpleNamespace(_visual_meta=None, _visual_meta_note="token/grid mismatch"),
                       _rpath_visual_cols=torch.tensor([1, 2]))
check("missing provenance -> skip with reason", psar.engine_groups(eng2, "branch")["groups"] is None)

print("[backbone _record_visual_meta]")
try:
    from backbone.qwen3vl import Qwen3VLBackbone
    fake = SimpleNamespace(
        _prune_morphology_enabled=True,
        model=SimpleNamespace(config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2))),
        _prune_image_magnifications=(5, 20, 20), _prune_image_parent_indices=(-1, 0, 2),
        _prune_image_boxes=((0, 0, 400, 400), (100, 100, 100, 100), (0, 200, 100, 100)),
        _prune_image_patch_ids=("R1", "R1-P1", "R1-P2"))
    grids = torch.tensor([[1, 4, 4], [1, 4, 4], [1, 2, 4]])      # merged 2x2, 2x2, 1x2 -> 4, 4, 2 tokens
    keep = torch.tensor([1, 0, 0, 1, 0, 1, 1, 0, 1, 1], dtype=torch.bool)
    abs_cols = torch.tensor([7, 10, 11, 12, 14, 15])
    Qwen3VLBackbone._record_visual_meta(fake, grids, keep, abs_cols)
    vm = fake._visual_meta
    check("meta written", vm is not None, getattr(fake, "_visual_meta_note", ""))
    check("orig_index = keep.nonzero()", vm["orig_index"].tolist() == [0, 3, 5, 6, 8, 9])
    check("obs_id per retained column", vm["obs_id"].tolist() == [0, 0, 1, 1, 2, 2])
    check("cell in crop", vm["tok_in_crop"].tolist() == [0, 3, 1, 2, 0, 1])
    check("scale per column", vm["scale"].tolist() == [5, 5, 20, 20, 20, 20])
    check("kept_counts", vm["kept_counts"] == (2, 2, 2))
    check("xy = crop cell centre", vm["xy"] is not None and vm["xy"][0].tolist() == [100.0, 100.0]
          and vm["xy"][1].tolist() == [300.0, 300.0], None if vm["xy"] is None else vm["xy"].tolist())
    check("patch ids + abs cols carried", vm["patch_id"] == ("R1", "R1-P1", "R1-P2") and vm["abs_cols"].tolist() == abs_cols.tolist())
    Qwen3VLBackbone._record_visual_meta(fake, grids, None, torch.arange(10))
    check("C1 off: all columns, c1=False", fake._visual_meta["orig_index"].tolist() == list(range(10)) and not fake._visual_meta["c1"])
    Qwen3VLBackbone._record_visual_meta(fake, grids, keep, torch.tensor([1, 2]))
    check("kept/column count mismatch -> None + note", fake._visual_meta is None and "kept" in fake._visual_meta_note)
    Qwen3VLBackbone._record_visual_meta(fake, grids, None, torch.arange(9))
    check("grid mismatch -> None + note", fake._visual_meta is None and "mismatch" in fake._visual_meta_note)
    fake._prune_morphology_enabled = False
    Qwen3VLBackbone._record_visual_meta(fake, grids, None, torch.arange(10))
    check("non-reasoner stage -> None", fake._visual_meta is None)
except Exception as exc:  # noqa: BLE001
    import traceback
    traceback.print_exc()
    check("backbone provenance block", False, repr(exc))

print(f"\n{PASS}/{PASS + FAIL} passed")
sys.exit(1 if FAIL else 0)
