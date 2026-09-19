#!/usr/bin/env python3
"""SRVW 유닛 — 노션 11.9 "Support-Resolved Visual Write Handoff" §4~§7 식·보존 불변식 + hook 경로.

usage: python tests_hj_test_srvw.py   (CPU)
"""
import os
import sys
from copy import deepcopy
from types import SimpleNamespace

import torch

TREE = "/home/users/whddn12316/wsi_latent_0915_decode_hj/tmp_hj/srvw_test"
sys.path.insert(0, TREE)
from memory import srvw  # noqa: E402

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
check("layers all", srvw.layer_window("all", 36) == (0, 35))
check("layers band", srvw.layer_window("24-30", 36) == (24, 30))
check("obs groups ordered by crop id", srvw.obs_groups([0, 0, 2, 4, 1, 3, 3]) == [[0, 1], [4], [2], [5, 6], [3]])
check("group ids", srvw.group_ids([[0, 1], [4], [2], [5, 6], [3]], 7).tolist() == [0, 0, 2, 4, 1, 3, 3])
os.environ["VLMAS_SRVW_BOUNDARY"] = "middle"
try:
    srvw.config()
    check("bad boundary rejected", False)
except ValueError:
    check("bad boundary rejected", True)
os.environ.pop("VLMAS_SRVW_BOUNDARY")
check("default boundary last, off", srvw.config()["boundary"] == "last" and not srvw.enabled())

print("[pure algebra: page 4-7]")
torch.manual_seed(0)
H, n_vis, D, P, d = 3, 9, 5, 3, 7
groups = [[0, 1, 2, 3], [4, 5], [6, 7, 8]]
gid = srvw.group_ids(groups, n_vis)
a_row = torch.rand(H, n_vis) * 0.05
V_row = torch.randn(H, n_vis, D)
msg = srvw.support_messages(a_row, V_row, gid, P)
man = torch.stack([torch.stack([(a_row[h, cols, None] * V_row[h, cols]).sum(0) for cols in groups]) for h in range(H)])
check("support message = sum_{j in G_g} alpha_j V_j", torch.allclose(msg, man, atol=1e-6))
check("linearity: sum_g message = visual head message", torch.allclose(msg.sum(1), (a_row[..., None] * V_row).sum(1), atol=1e-6))
W = torch.randn(d, H * D)
Mo = srvw.project_out(msg, W)
man_o = torch.stack([W @ msg[:, g, :].reshape(-1) for g in range(P)])
check("W_O [concat_h] per support", torch.allclose(Mo, man_o, atol=1e-5))
M = torch.randn(P, d); C = torch.randn(P, d)
a = srvw.nnls_coeff(M, C)
grid = torch.linspace(0, 5, 50001, dtype=torch.float64)
brute = torch.stack([grid[((grid[:, None] * M[g].double() - C[g].double()) ** 2).sum(-1).argmin()] for g in range(P)])
check("nnls closed form = brute-force argmin_{a>=0}", torch.allclose(a, brute, atol=2e-4), (a, brute))
check("negative projection -> 0", float(srvw.nnls_coeff(torch.tensor([[1.0, 0.0]]), torch.tensor([[-2.0, 1.0]]))) == 0.0)
check("zero M -> 0", float(srvw.nnls_coeff(torch.zeros(1, 3), torch.ones(1, 3))) == 0.0)
check("scale: C = 2.5 M -> a = 2.5", abs(float(srvw.nnls_coeff(M[:1].double(), 2.5 * M[:1].double())) - 2.5) < 1e-9)

a_vis = torch.rand(1, H, 2, n_vis) * 0.02
coeff = torch.tensor([0.3, 2.0, 1.1], dtype=torch.float64)
new = srvw.reweight(a_vis, coeff, gid)
check("total visual mass conserved exactly", torch.allclose(new.sum(-1), a_vis.sum(-1), atol=1e-7))
for g, cols in enumerate(groups):
    c = torch.as_tensor(cols)
    check(f"within-support ratios preserved (g={g})",
          torch.allclose(new[..., c] / new[..., c].sum(-1, keepdim=True), a_vis[..., c] / a_vis[..., c].sum(-1, keepdim=True), atol=1e-6))
mass = torch.stack([a_vis[..., torch.as_tensor(cols)].sum(-1) for cols in groups], -1)
want = a_vis.sum(-1, keepdim=True) * coeff.float() * mass / (mass * coeff.float()).sum(-1, keepdim=True)
got = torch.stack([new[..., torch.as_tensor(cols)].sum(-1) for cols in groups], -1)
check("support mass = rho a_g m_g / sum_r a_r m_r", torch.allclose(got, want, atol=1e-7))
check("equal a -> identity", torch.allclose(srvw.reweight(a_vis, torch.full((P,), 0.7, dtype=torch.float64), gid), a_vis, atol=1e-7))
check("all a = 0 -> native", torch.equal(srvw.reweight(a_vis, torch.zeros(P, dtype=torch.float64), gid), a_vis))
a0 = a_vis.clone(); a0[..., 4:6] = 0.0
new0 = srvw.reweight(a0, coeff, gid)
check("support with zero native mass stays 0, mass conserved",
      bool((new0[..., 4:6] == 0).all()) and torch.allclose(new0.sum(-1), a0.sum(-1), atol=1e-7))
only = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
new1 = srvw.reweight(a_vis, only, gid)
check("a only on support 1 -> all visual mass on support 1",
      torch.allclose(new1[..., 4:6].sum(-1), a_vis.sum(-1), atol=1e-7) and float(new1[..., :4].abs().max()) == 0.0)

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


def native_alpha(a, rope, x, past, cache_after, li):
    """alpha [1,H,T,L], V [1,H,L,D] of rows x (already appended in cache_after)."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, repeat_kv
    T = x.shape[1]
    K = repeat_kv(cache_after.layers[li].keys, 2); Vf = repeat_kv(cache_after.layers[li].values, 2)
    pos = torch.arange(past, past + T)[None, None, :].expand(3, 1, T)
    cos, sin = rope(x, pos)
    q = a.q_norm(a.q_proj(x).view(1, T, -1, 16)).transpose(1, 2)
    q, _ = apply_rotary_pos_emb(q, torch.zeros_like(q), cos, sin)
    al = torch.softmax(q @ K.transpose(-2, -1) * a.scaling + causal(past, T), -1)
    return al, Vf


try:
    from transformers.cache_utils import DynamicCache

    bb, attns, rope = tiny(3)
    vis = torch.arange(2, 10)
    bb._visual_meta = {"abs_cols": vis.clone(), "obs_id": torch.tensor([0, 0, 0, 1, 1, 2, 2, 2]), "parents": (-1, -1, -1)}
    torch.manual_seed(1)
    ctx = torch.randn(1, 14, 64)
    steps = [torch.randn(1, 1, 64) for _ in range(3)]
    rows = torch.randn(1, 4, 64)
    caches = [DynamicCache() for _ in attns]

    os.environ["VLMAS_SRVW"] = "1"
    with torch.no_grad(), srvw.reasoner_probe(bb, m=3, stage="reasoner") as probe:
        for li, a in enumerate(attns):
            run(a, rope, ctx, 0, caches[li])
        for t, stp in enumerate(steps):
            for li, a in enumerate(attns):
                run(a, rope, stp, 14 + t, caches[li])
        for li, a in enumerate(attns):                       # a 4th single-row call beyond m must be ignored
            run(a, rope, torch.randn(1, 1, 64), 17, deepcopy(caches[li]))
    st = bb._srvw
    check("probe: C shape [L=3, P=3, d=64]", st is not None and tuple(st["C"].shape) == (3, 3, 64), None if st is None else st["C"].shape)
    check("probe: steps = m", st["steps"] == 3)
    check("probe: groups obs", st["groups"] == [[0, 1, 2], [3, 4], [5, 6, 7]])
    with torch.no_grad():
        manC = torch.zeros(3, 3, 64)
        for li, a in enumerate(attns):
            c = DynamicCache(); run(a, rope, ctx, 0, c)
            acc = torch.zeros(4, 3, 16)
            for t, stp in enumerate(steps):
                run(a, rope, stp, 14 + t, c)
                al, Vf = native_alpha(a, rope, stp, 14 + t, c, li)
                for g, cols in enumerate(st["groups"]):
                    j = vis[cols]
                    acc[:, g, :] += (al[0, :, 0, j, None] * Vf[0, :, j, :]).sum(1)
            for g in range(3):
                manC[li, g] = a.o_proj.weight @ acc[:, g, :].reshape(-1)
    check("probe: C == W_O [concat_h sum_t sum_{j in G_g} alpha V] (all layers)", torch.allclose(st["C"], manC, atol=1e-4),
          float((st["C"] - manC).abs().max()))
    with srvw.reasoner_probe(bb, m=3, stage="answerer") as p2:
        check("probe: other stage -> None", p2 is None)
    os.environ.pop("VLMAS_SRVW")
    with srvw.reasoner_probe(bb, m=3, stage="reasoner") as p3:
        check("probe: env off -> None, no hooks", p3 is None and not attns[0]._forward_hooks)
    os.environ["VLMAS_SRVW"] = "1"

    engine = SimpleNamespace(_backbone=bb, _rpath_visual_cols=vis.clone(), _rpath_case_index=0)
    state = srvw.engine_state(engine)
    check("engine_state ok", state["ok"], state.get("note"))
    bad = SimpleNamespace(_backbone=bb, _rpath_visual_cols=torch.arange(1, 9), _rpath_case_index=0)
    check("engine_state: moved columns -> not ok", not srvw.engine_state(bad)["ok"])

    base = deepcopy(caches[0])
    with torch.no_grad():
        native = run(attns[0], rope, rows, 17, deepcopy(base))

    def go(**kw):
        with torch.no_grad(), srvw.install_srvw(bb, state["C"], state["vis"], state["gid"], state["sizes"],
                                                 layers=(0, 0), **kw) as stt:
            c = deepcopy(base)
            out = run(attns[0], rope, rows, 17, c)
            dec = torch.randn(1, 1, 64, generator=torch.Generator().manual_seed(5))
            out2 = run(attns[0], rope, dec, 21, c)
        return out, out2, stt, dec

    ident, ident2, stt_i, dec = go(identity=True, boundary="first")
    with torch.no_grad():
        cn = deepcopy(base); run(attns[0], rope, rows, 17, cn); native2 = run(attns[0], rope, dec, 21, cn)
    check("identity (boundary first) == native eager, prefill rows", torch.allclose(ident, native, atol=1e-5), float((ident - native).abs().max()))
    check("identity == native, next decode row", torch.allclose(ident2, native2, atol=1e-5))
    check("identity maxdiff recorded small", stt_i["maxdiff_identity"] < 1e-4)

    out_last, out_last2, stt_l, _ = go(boundary="last")
    check("boundary=last: prompt rows before s0 untouched (exact)", torch.equal(out_last[:, :3], native[:, :3]))
    check("boundary=last: s0 row changed", not torch.allclose(out_last[:, 3], native[:, 3], atol=1e-5))
    check("boundary=last: decode row changed", not torch.allclose(out_last2, native2, atol=1e-5))
    check("stats: boundary_T = 4, one layer", stt_l["boundary_T"] == 4 and stt_l["n_layers"] == 1)

    out_first, out_first2, stt_f, _ = go(boundary="first")
    with torch.no_grad():
        c = deepcopy(base); run(attns[0], rope, rows, 17, c)
        al, Vf = native_alpha(attns[0], rope, rows, 17, c, 0)         # [1,H,4,21]
        Mh = torch.zeros(4, 3, 16)
        for g, cols in enumerate(state["groups"]):
            j = vis[cols]
            Mh[:, g, :] = (al[0, :, 0, j, None] * Vf[0, :, j, :]).sum(1)
        Mg = torch.stack([attns[0].o_proj.weight @ Mh[:, g, :].reshape(-1) for g in range(3)])
        Cg = state["C"][0]
        a_man = ((Mg * Cg).sum(-1) / (Mg * Mg).sum(-1)).clamp_min(0)
        a_vis = al[..., vis]
        mass = torch.stack([a_vis[..., torch.as_tensor(cols)].sum(-1) for cols in state["groups"]], -1)
        rho = mass.sum(-1, keepdim=True)
        den = (mass * a_man).sum(-1, keepdim=True)
        new_a = torch.zeros_like(a_vis)
        for g, cols in enumerate(state["groups"]):
            cc = torch.as_tensor(cols)
            new_a[..., cc] = rho * a_man[g] * a_vis[..., cc] / den
        o_new = al @ Vf + (new_a - a_vis) @ Vf[:, :, vis, :]
        want = attns[0].o_proj(o_new.transpose(1, 2).reshape(1, 4, -1))
    check("a == manual NNLS at s0 (first)", torch.allclose(stt_f["a"][0].float(), a_man, atol=1e-4), (stt_f["a"][0], a_man))
    check("formula page 5-7 (boundary=first, all prompt rows)", torch.allclose(out_first, want, atol=1e-5), float((out_first - want).abs().max()))
    check("boundary first != last on s0 row (coefficient read at different rows)",
          not torch.allclose(out_first[:, 3], out_last[:, 3], atol=1e-6))
    print("   ", srvw.summary(stt_f, state))

    # -- light implementation (no kv repeat, delta on native output) must match recompute
    print("[light impl]")
    os.environ["VLMAS_SRVW_IMPL"] = "light"
    bbL, attnsL, ropeL = tiny(3)
    bbL._visual_meta = bb._visual_meta
    cachesL = [DynamicCache() for _ in attnsL]
    with torch.no_grad(), srvw.reasoner_probe(bbL, m=3, stage="reasoner"):
        for li, a in enumerate(attnsL):
            run(a, ropeL, ctx, 0, cachesL[li])
        for t, stp in enumerate(steps):
            for li, a in enumerate(attnsL):
                run(a, ropeL, stp, 14 + t, cachesL[li])
    check("light probe: C == recompute C", torch.allclose(bbL._srvw["C"], st["C"], atol=1e-4),
          float((bbL._srvw["C"] - st["C"]).abs().max()))
    check("light config picked up", srvw.config()["impl"] == "light")
    for bnd in ("last", "first"):
        with torch.no_grad(), srvw.install_srvw(bb, state["C"], state["vis"], state["gid"], state["sizes"],
                                                 layers=(0, 0), boundary=bnd) as sl:
            c = deepcopy(base); outL = run(attns[0], rope, rows, 17, c)
            dec = torch.randn(1, 1, 64, generator=torch.Generator().manual_seed(5)); outL2 = run(attns[0], rope, dec, 21, c)
        os.environ["VLMAS_SRVW_IMPL"] = "recompute"
        ref, ref2, sr, _ = go(boundary=bnd)
        os.environ["VLMAS_SRVW_IMPL"] = "light"
        check(f"light == recompute output (boundary={bnd}, prefill rows)", torch.allclose(outL, ref, atol=1e-5), float((outL - ref).abs().max()))
        check(f"light == recompute output (boundary={bnd}, decode row)", torch.allclose(outL2, ref2, atol=1e-5))
        check(f"light a == recompute a (boundary={bnd})", torch.allclose(sl["a"][0], sr["a"][0], atol=1e-6))
        check(f"light stats impl tag", sl["impl"] == "light" and sr["impl"] == "recompute")
    with torch.no_grad(), srvw.install_srvw(bb, state["C"], state["vis"], state["gid"], state["sizes"],
                                             layers=(0, 0), boundary="first", identity=True) as si:
        c = deepcopy(base); idL = run(attns[0], rope, rows, 17, c); idL2 = run(attns[0], rope, dec, 21, c)
    check("light identity: output == native bit-exact", torch.equal(idL, native) and torch.equal(idL2, native2))
    check("light identity: measured delta tiny", si["maxdiff_identity"] < 1e-5, si["maxdiff_identity"])
    # GQA helpers vs explicit repeat
    from transformers.models.qwen3_vl.modeling_qwen3_vl import repeat_kv
    Vkv = torch.randn(1, 2, 8, 16); aV = torch.rand(4, 8)
    gq = srvw._gqa_support_messages(aV, Vkv[0], state["gid"], 3)
    ex = srvw.support_messages(aV, repeat_kv(Vkv, 2)[0], state["gid"], 3)
    check("gqa support messages == repeat_kv messages", torch.allclose(gq, ex, atol=1e-6))
    dl = torch.randn(1, 4, 2, 8)
    W = attns[0].o_proj.weight
    exd = W @ (dl @ repeat_kv(Vkv, 2)).transpose(1, 2).reshape(1, 2, -1).transpose(-1, -2)
    check("gqa delta out == repeat_kv delta", torch.allclose(srvw._gqa_delta_out(dl, Vkv[0], W), exd.transpose(-1, -2), atol=1e-5))
    os.environ.pop("VLMAS_SRVW_IMPL")

    bb3, attns3, rope3 = tiny(3)
    c3 = []
    with torch.no_grad():
        for li, a in enumerate(attns3):
            c = DynamicCache(); run(a, rope3, ctx, 0, c)
            for t, stp in enumerate(steps):
                run(a, rope3, stp, 14 + t, c)
            c3.append(c)
    with torch.no_grad(), srvw.install_srvw(bb3, state["C"], state["vis"], state["gid"], state["sizes"],
                                             layers=(1, 1), boundary="last") as st3:
        for li, a in enumerate(attns3):
            run(a, rope3, rows, 17, c3[li])
    check("hooks only in window 1-1", st3["calls"] == 1 and st3["n_layers"] == 1, st3)
    check("hooks removed after the call", not any(a._forward_hooks for a in attns3))
except ImportError as exc:
    print("  skip attention-layer tests:", exc)
finally:
    os.environ.pop("VLMAS_SRVW", None)

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
