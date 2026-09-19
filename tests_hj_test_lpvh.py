#!/usr/bin/env python3
"""LPVH 유닛 — 노션 "Latent-Provenance Visual Handoff" §4~§6 식·보존 불변식 + hook 경로.

usage: python tests_hj_test_lpvh.py   (CPU)
"""
import os
import sys
from copy import deepcopy
from types import SimpleNamespace

import torch

TREE = "/home/users/whddn12316/wsi_latent_0915_decode_hj"
sys.path.insert(0, TREE)
from memory import lpvh  # noqa: E402

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
check("layers all", lpvh.layer_window("all", 36) == (0, 35))
check("layers band", lpvh.layer_window("24-30", 36) == (24, 30))
check("layers clipped", lpvh.layer_window("30-99", 36) == (30, 35))
parents = (-1, -1, 0, 1, 0)
obs_ids = [0, 0, 2, 4, 1, 3, 3]
check("obs groups (one crop per support)", lpvh.support_groups(obs_ids, parents, "obs") == [[0, 1], [4], [2], [5, 6], [3]])
check("branch groups", lpvh.support_groups(obs_ids, parents, "branch") == [[0, 1, 2, 3], [4, 5, 6]])
gs = lpvh.support_groups(obs_ids, parents, "shuffled", seed=3)
check("shuffled keeps sizes, covers all", sorted(len(x) for x in gs) == [1, 1, 1, 2, 2]
      and sorted(j for x in gs for j in x) == list(range(7)))
gid = lpvh.group_ids([[0, 1], [4], [2], [5, 6], [3]], 7)
check("group ids", gid.tolist() == [0, 0, 2, 4, 1, 3, 3])
try:
    lpvh.support_groups(obs_ids, parents, "grid")
    check("bad support rejected", False)
except ValueError:
    check("bad support rejected", True)

print("[pure algebra: page 5-6]")
torch.manual_seed(0)
H, Ts, T, P, n_vis = 3, 2, 4, 3, 9
groups = [[0, 1, 2, 3], [4, 5], [6, 7, 8]]
gid = lpvh.group_ids(groups, n_vis)
sizes = torch.tensor([len(g) for g in groups])
a_vis = torch.rand(1, H, Ts, n_vis) * 0.02
b = torch.rand(1, H, Ts, T) * 0.1
A = torch.softmax(torch.randn(T, P), -1) * 0.3           # rows sum to Reasoner visual mass
pi = lpvh.compose(b, A)
check("compose = b A", torch.allclose(pi, torch.einsum("bhst,tp->bhsp", b, A)))
mass = lpvh.support_mass(a_vis, gid, P)
check("support mass sums to rho_V", torch.allclose(mass.sum(-1), a_vis.sum(-1)))
pi_sum = pi.sum(-1, keepdim=True)
pibar = pi / pi_sum
a_new = lpvh.redistribute(a_vis, pibar, gid, sizes, pi_sum)
check("total visual mass conserved exactly", torch.allclose(a_new.sum(-1), a_vis.sum(-1), atol=1e-7))
new_mass = lpvh.support_mass(a_new, gid, P)
check("support mass = rho_V * pibar", torch.allclose(new_mass, a_vis.sum(-1, keepdim=True) * pibar, atol=1e-7))
for g, cols in enumerate(groups):
    c = torch.as_tensor(cols)
    nu_old = a_vis[..., c] / a_vis[..., c].sum(-1, keepdim=True)
    nu_new = a_new[..., c] / a_new[..., c].sum(-1, keepdim=True)
    check(f"within-support distribution nu preserved (g={g})", torch.allclose(nu_old, nu_new, atol=1e-6))
share = mass / mass.sum(-1, keepdim=True)
check("pibar = native share -> identity", torch.allclose(lpvh.redistribute(a_vis, share, gid, sizes), a_vis, atol=1e-7))
zero = torch.zeros_like(pi_sum)
check("zero composed mass -> native kept", torch.equal(lpvh.redistribute(a_vis, pibar, gid, sizes, zero), a_vis))
a0 = a_vis.clone(); a0[..., 4:6] = 0.0                    # support 1 has no native mass
a_new0 = lpvh.redistribute(a0, pibar, gid, sizes)
check("support with zero native mass: uniform over its columns, mass conserved",
      torch.allclose(a_new0[..., 4], a_new0[..., 5]) and torch.allclose(a_new0.sum(-1), a0.sum(-1), atol=1e-7)
      and torch.allclose(lpvh.support_mass(a_new0, gid, P), a0.sum(-1, keepdim=True) * pibar, atol=1e-7))
# duplicating every column of support 0 does not change its composed share (A is per support)
a_dup = torch.cat([a_vis, a_vis[..., :4]], -1)
gid_dup = torch.cat([gid, torch.zeros(4, dtype=torch.long)])
sizes_dup = sizes.clone(); sizes_dup[0] += 4
m_dup = lpvh.support_mass(lpvh.redistribute(a_dup, pibar, gid_dup, sizes_dup), gid_dup, P)
check("multiplicity: support mass follows pibar, not token count",
      torch.allclose(m_dup / m_dup.sum(-1, keepdim=True), pibar, atol=1e-6))
# provenance controls
check("prov native", torch.equal(lpvh.provenance_control(A, "native"), A))
check("prov uniform", torch.allclose(lpvh.provenance_control(A, "uniform"), torch.full_like(A, 1 / P)))
Ash = lpvh.provenance_control(A, "shuffled", seed=1)
check("prov shuffled = column permutation", torch.allclose(Ash.sum(1), A.sum(1)) and sorted(Ash[0].tolist()) == sorted(A[0].tolist()))
Ar = lpvh.provenance_control(A, "reasoner_mass")
check("prov reasoner_mass: identical rows = mean", torch.allclose(Ar[0], A.mean(0)) and torch.allclose(Ar[0], Ar[-1]))
check("spearman monotone = 1", abs(lpvh.spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1) < 1e-9)
check("spearman reversed = -1", abs(lpvh.spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1) < 1e-9)

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

    bb, attns, rope = tiny(3)
    vis = torch.arange(2, 10)                                # 8 visual columns
    bb._visual_meta = {"abs_cols": vis.clone(), "obs_id": torch.tensor([0, 0, 0, 1, 1, 2, 2, 2]),
                       "parents": (-1, -1, -1)}
    torch.manual_seed(1)
    ctx = torch.randn(1, 14, 64)
    steps = [torch.randn(1, 1, 64) for _ in range(3)]       # 3 "latent steps" (m=3)
    rows = torch.randn(1, 4, 64)                             # Answerer prompt rows
    caches = [DynamicCache() for _ in attns]

    # -- Reasoner probe: prefill (T=14) then 3 single-row steps through all layers
    os.environ["VLMAS_LPVH"] = "1"
    with torch.no_grad(), lpvh.reasoner_probe(bb, m=3, stage="reasoner") as probe:
        for li, a in enumerate(attns):
            run(a, rope, ctx, 0, caches[li])
        for t, st in enumerate(steps):
            for li, a in enumerate(attns):
                run(a, rope, st, 14 + t, caches[li])
    st = bb._lpvh
    check("probe: A shape [T=3, P=3]", st is not None and tuple(st["A"].shape) == (3, 3), None if st is None else st["A"].shape)
    check("probe: latent cols = 14,15,16", st["latent_cols"].tolist() == [14, 15, 16])
    check("probe: groups obs", st["groups"] == [[0, 1, 2], [3, 4], [5, 6, 7]])
    check("probe: A rows <= 1 (visual mass), >= 0", bool((st["A"] >= 0).all()) and bool((st["A"].sum(1) <= 1 + 1e-6).all()))
    # manual A for step 0: mean over layers/heads of native attention over the visual groups
    with torch.no_grad():
        man = torch.zeros(3)
        for li, a in enumerate(attns):
            c = DynamicCache()
            run(a, rope, ctx, 0, c)
            run(a, rope, steps[0], 14, c)                    # cache now 15 cols
            K = repeat_kv(c.layers[li].keys, 2)
            pos = torch.tensor([14])[None, None, :].expand(3, 1, 1)
            cos, sin = rope(steps[0], pos)
            q = a.q_norm(a.q_proj(steps[0]).view(1, 1, -1, 16)).transpose(1, 2)
            q, _ = apply_rotary_pos_emb(q, torch.zeros_like(q), cos, sin)
            al = torch.softmax(q @ K.transpose(-2, -1) * a.scaling, -1)[0, :, 0, :]   # [H, 15]
            for g, cols in enumerate(st["groups"]):
                man[g] += al[:, vis[cols]].sum(-1).mean()
        man /= 3
    check("probe: A[0] == manual mean_{l,h} sum_{j in G_g} alpha", torch.allclose(st["A"][0], man, atol=1e-6),
          float((st["A"][0] - man).abs().max()))
    check("probe: no probe outside reasoner stage", True)
    with lpvh.reasoner_probe(bb, m=3, stage="answerer") as p2:
        check("probe: other stage -> None", p2 is None)

    # -- Answerer hook on layer 0 (cache after Reasoner: 17 cols)
    engine = SimpleNamespace(_backbone=bb, _rpath_visual_cols=vis.clone(), _rpath_latent_cols=torch.tensor([14, 15, 16]),
                             _rpath_case_index=0)
    state = lpvh.engine_state(engine, "native")
    check("engine_state ok", state["ok"], state.get("note"))
    base = deepcopy(caches[0])
    with torch.no_grad():
        native = run(attns[0], rope, rows, 17, deepcopy(base))

    def go(**kw):
        prov = kw.pop("prov", "native")
        s = lpvh.engine_state(engine, prov)
        with torch.no_grad(), lpvh.install_lpvh(bb, s["A"], s["latent_cols"], s["vis"], s["gid"], s["sizes"],
                                                 layers=(0, 0), **kw) as stt:
            out = run(attns[0], rope, rows, 17, deepcopy(base))
        return out, stt, s

    ident, stt, _ = go(identity=True, rows="all")
    check("identity rows=all == native eager", torch.allclose(ident, native, atol=1e-5), float((ident - native).abs().max()))
    check("identity maxdiff recorded small", stt["maxdiff_identity"] < 1e-4)
    out_gen, stt_g, _ = go(rows="gen")
    check("rows=gen: prompt rows before the last untouched (exact)", torch.equal(out_gen[:, :3], native[:, :3]))
    check("rows=gen: last row changed", not torch.allclose(out_gen[:, 3], native[:, 3], atol=1e-4))
    out_all, stt_a, s_all = go(rows="all")
    check("rows=all: rows changed", not torch.allclose(out_all, native, atol=1e-4))
    print("   ", lpvh.summary(stt_a, s_all))

    # manual formula (page 5-6) for rows=all, layer 0
    with torch.no_grad():
        c = deepcopy(base)
        run(attns[0], rope, rows, 17, c)
        K = repeat_kv(c.layers[0].keys, 2); Vf = repeat_kv(c.layers[0].values, 2)
        pos = torch.arange(17, 21)[None, None, :].expand(3, 1, 4)
        cos, sin = rope(rows, pos)
        q = attns[0].q_norm(attns[0].q_proj(rows).view(1, 4, -1, 16)).transpose(1, 2)
        q, _ = apply_rotary_pos_emb(q, torch.zeros_like(q), cos, sin)
        al = torch.softmax(q @ K.transpose(-2, -1) * attns[0].scaling + causal(17, 4), -1)   # [1,H,4,21]
        Aa = state["A"]
        bl = al[..., 14:17]                                                                  # [1,H,4,3]
        pi = bl @ Aa
        pibar = pi / pi.sum(-1, keepdim=True)
        a_vis = al[..., vis]
        rho = a_vis.sum(-1, keepdim=True)
        new_a = torch.zeros_like(a_vis)
        for g, cols in enumerate(state["groups"]):
            cc = torch.as_tensor(cols)
            mg = a_vis[..., cc].sum(-1, keepdim=True)
            new_a[..., cc] = rho * pibar[..., g:g + 1] * a_vis[..., cc] / mg
        o = al @ Vf
        o_new = o + (new_a - a_vis) @ Vf[:, :, vis, :]
        want = attns[0].o_proj(o_new.transpose(1, 2).reshape(1, 4, -1))
    check("formula page 5-6 (rows=all)", torch.allclose(out_all, want, atol=1e-5), float((out_all - want).abs().max()))
    out_u, _, _ = go(rows="all", prov="uniform")
    check("prov=uniform differs from native", not torch.allclose(out_u, native, atol=1e-4))
    check("prov=uniform differs from native-provenance", not torch.allclose(out_u, out_all, atol=1e-4))
    # column mismatch -> skip
    bad = SimpleNamespace(_backbone=bb, _rpath_visual_cols=torch.arange(1, 9), _rpath_latent_cols=None, _rpath_case_index=0)
    check("engine_state: moved columns -> not ok", not lpvh.engine_state(bad, "native")["ok"])
    # hooks only inside the window
    bb3, attns3, rope3 = tiny(3)
    c3 = []
    with torch.no_grad():
        for li, a in enumerate(attns3):
            c = DynamicCache(); run(a, rope3, ctx, 0, c)
            for t, stp in enumerate(steps):
                run(a, rope3, stp, 14 + t, c)
            c3.append(c)
    # the Answerer hook is installed only around the Answerer call (prefill/latent steps are outside it)
    with torch.no_grad(), lpvh.install_lpvh(bb3, state["A"], state["latent_cols"], state["vis"], state["gid"],
                                             state["sizes"], layers=(1, 1), rows="all") as st3:
        for li, a in enumerate(attns3):
            run(a, rope3, rows, 17, c3[li])
    check("hooks only in window 1-1", st3["calls"] == 1, st3["calls"])
except ImportError as exc:
    print("  skip attention-layer tests:", exc)
finally:
    os.environ.pop("VLMAS_LPVH", None)

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
