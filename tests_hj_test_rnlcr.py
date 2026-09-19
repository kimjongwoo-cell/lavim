#!/usr/bin/env python3
"""RN-LCR (C2 #15) + SP-GLR + RELR + FULR (C2 #18) unit tests — grounding loss / P,N sets, capacity redistribution algebra, end-to-end on a tiny
Qwen3-VL text model (CPU): ground+replay commits the cache, cap hook == manual recompute, diag untouched, base off.

usage: python tests_hj_test_rnlcr.py
"""
import copy
import os
import sys
from types import SimpleNamespace

import torch

TREE = "/home/users/whddn12316/wsi_latent_0915_decode_hj"
sys.path.insert(0, TREE)
for k in ("VLMAS_RNLCR", "VLMAS_RNLCR_ROWS", "VLMAS_RNLCR_SINK"):
    os.environ.pop(k, None)
from memory import rnlcr as rn  # noqa: E402

PASS = FAIL = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {extra}")


print("[config]")
check("off by default", not rn.enabled())
os.environ["VLMAS_RNLCR"] = "FULL"
check("full", rn.mode() == "full" and rn.grounds() and rn.redistributes())
os.environ["VLMAS_RNLCR"] = "cap"
check("cap = redistribute only", not rn.grounds() and rn.redistributes())
os.environ["VLMAS_RNLCR"] = "ground"
check("ground = ground only", rn.grounds() and not rn.redistributes())
os.environ["VLMAS_RNLCR"] = "diag"
check("diag = neither", not rn.grounds() and not rn.redistributes())
os.environ.pop("VLMAS_RNLCR")

print("[pure]")
torch.manual_seed(0)
score = torch.rand(20)
V = torch.randn(20, 8)
P, N, top, bot = rn.pos_neg(score, V, 4)
order = torch.argsort(score, descending=True)
check("pos_neg: top/bottom k and means", top.tolist() == order[:4].tolist() and bot.tolist() == order.flip(0)[:4].tolist()
      and torch.allclose(P, V[order[:4]].mean(0)) and torch.allclose(N, V[order.flip(0)[:4]].mean(0)))
P2, N2, t2, b2 = rn.pos_neg(score, V, 100)
check("pos_neg: k clipped to n//2", int(t2.numel()) == 10 and not set(t2.tolist()) & set(b2.tolist()))
Z = torch.randn(3, 8)
man = -torch.nn.functional.cosine_similarity(Z, P[None].expand(3, -1), dim=-1).mean() + torch.nn.functional.cosine_similarity(Z, N[None].expand(3, -1), dim=-1).mean()
check("ground_loss == manual", abs(float(rn.ground_loss(Z, P, N)) - float(man)) < 1e-6)
Zs, losses = rn.ground_stage(Z, P, N, steps=5, lr_rel=0.05)
check("ground_stage: 6 losses, final <= first, Z moved", len(losses) == 6 and losses[-1] <= losses[0] + 1e-9 and not torch.equal(Zs, Z))
# redistribution algebra
torch.manual_seed(1)
scores = torch.randn(1, 4, 2, 12)
alpha = torch.softmax(scores, -1)
lat = torch.tensor([7, 8, 9])
new, a_s = rn.redistribute(alpha, lat, 0, scores)
check("redistribute: sink -> 0", bool((new[..., 0] == 0).all()))
check("redistribute: rows still sum to 1", torch.allclose(new.sum(-1), torch.ones(1, 4, 2), atol=1e-6))
check("redistribute: latent gain == a_s", torch.allclose((new[..., lat] - alpha[..., lat]).sum(-1), a_s, atol=1e-6))
rho = torch.softmax(scores[..., lat], -1)
check("redistribute: split by native relative compatibility", torch.allclose(new[..., lat] - alpha[..., lat], a_s.unsqueeze(-1) * rho, atol=1e-6))
others = [c for c in range(12) if c not in (0, 7, 8, 9)]
check("redistribute: other columns unchanged", torch.equal(new[..., others], alpha[..., others]))

print("[tiny model]")
try:
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    cfg = Qwen3VLTextConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                            intermediate_size=128, num_hidden_layers=3, vocab_size=50,
                            rope_scaling={"rope_type": "default", "mrope_section": [3, 3, 2]})
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    lm = Qwen3VLTextModel(cfg).eval()

    class Tok:
        def encode(self, text, add_special_tokens=False):
            return [(ord(c) * 7) % 50 for c in text]

    def text_positions(n, start):
        return torch.arange(start, start + n).view(1, -1).expand(3, -1).unsqueeze(1)

    bb = SimpleNamespace(lm=lm, device=torch.device("cpu"), processor=SimpleNamespace(tokenizer=Tok()),
                         _text_positions=text_positions)
    eng = SimpleNamespace(_backbone=bb)
    Q = "what tissue"
    q_ids = Tok().encode(" " + Q)
    torch.manual_seed(3)
    ids = torch.randint(0, 50, (34,))
    ids[20:20 + len(q_ids)] = torch.tensor(q_ids)
    past0, m = 5, 3
    vis = torch.arange(past0 + 2, past0 + 18)
    L0 = past0 + 34

    def run_reasoner(mode, case=0):
        os.environ["VLMAS_RNLCR"] = mode
        cache = DynamicCache()
        torch.manual_seed(7 + case)
        with torch.no_grad():
            lm(inputs_embeds=torch.randn(1, past0, 64), position_ids=text_positions(past0, 0), past_key_values=cache, use_cache=True)
            traj = []
            with rn.reasoner_probe(bb, m=m, stage="reasoner", targets=["Question stem: " + Q, Q]) as cap:
                o = lm(inputs_embeds=lm.embed_tokens(ids).unsqueeze(0), position_ids=text_positions(34, past0), past_key_values=cache,
                       use_cache=True, output_hidden_states=True)
                last = o.hidden_states[-1][:, -1:, :]
                for t in range(m):
                    le = last * (5.0 / last.norm(dim=-1, keepdim=True))
                    o = lm(inputs_embeds=le, position_ids=text_positions(1, L0 + t), past_key_values=cache, use_cache=True, output_hidden_states=True)
                    last = o.hidden_states[-1][:, -1:, :]
                    traj.append(le.squeeze().clone())
        res = {"past_key_values": cache, "latent_trajectory": traj, "vis_cols": vis.clone(), "pos_cursor": L0 + m, "past_len": L0 + m}
        return cap, res

    # -- cap: no capture installed, record only
    cap, res = run_reasoner("cap")
    check("cap: no Reasoner capture hooks", cap is None)
    R = rn.apply_reasoner(eng, res, m=m, case_index=0)
    check("cap: latent cols recorded", R["latent_cols"].tolist() == [L0, L0 + 1, L0 + 2] and "ground" not in R)

    # -- ground: capture + Step 1 + replay commits the cache
    cap, res = run_reasoner("ground")
    native = copy.deepcopy(res["past_key_values"])
    traj0 = [t.clone() for t in res["latent_trajectory"]]
    check("ground: capture located question rows", cap is not None and cap.rows is not None and cap.n_acc == 3)
    R = rn.apply_reasoner(eng, res, m=m, case_index=1)
    g = R.get("ground")
    check("ground: no skip", g is not None and "skip" not in g, R.get("ground_skip"))
    check("ground: loss decreased, cosP up, cosN down", g["loss"][-1] <= g["loss"][0] and g["cosP"][1] > g["cosP"][0] and g["cosN"][1] < g["cosN"][0], g)
    check("ground: cache length preserved, pre-latent untouched, latent K changed",
          rn._kvlen(res["past_key_values"]) == L0 + m
          and all(torch.equal(a.keys[..., :L0, :], b.keys[..., :L0, :]) for a, b in zip(res["past_key_values"].layers, native.layers))
          and not torch.equal(res["past_key_values"].layers[0].keys[..., L0:, :], native.layers[0].keys[..., L0:, :]))
    check("ground: trajectory updated", not all(torch.equal(a, b) for a, b in zip(res["latent_trajectory"], traj0)))
    with torch.no_grad():
        c3 = copy.deepcopy(res["past_key_values"]); c3.crop(L0)
        lm(inputs_embeds=torch.stack(res["latent_trajectory"]).unsqueeze(0), position_ids=text_positions(m, L0), past_key_values=c3, use_cache=True)
    check("ground: committed KV == replay of returned trajectory", all(torch.allclose(a.keys, b.keys, atol=1e-5) for a, b in zip(res["past_key_values"].layers, c3.layers)))

    # -- terminal hook: cap mode == manual recompute; diag untouched
    def answer(cache, mode, rows="gen"):
        os.environ["VLMAS_RNLCR"] = mode
        os.environ["VLMAS_RNLCR_ROWS"] = rows
        torch.manual_seed(11)
        x1, x2 = torch.randn(1, 4, 64), torch.randn(1, 1, 64)
        outs = []
        with rn.terminal(eng, cache, L0 + m + 1, case_index=0) as st:
            with torch.no_grad():
                outs.append(lm(inputs_embeds=x1, position_ids=text_positions(4, L0 + m), past_key_values=cache, use_cache=True).last_hidden_state)
                outs.append(lm(inputs_embeds=x2, position_ids=text_positions(1, L0 + m + 4), past_key_values=cache, use_cache=True).last_hidden_state)
        return st, outs, (x1, x2)

    cap, res = run_reasoner("cap")
    rn.apply_reasoner(eng, res, m=m, case_index=0)
    base_cache = copy.deepcopy(res["past_key_values"])
    st_d, out_d, _ = answer(copy.deepcopy(base_cache), "diag")
    os.environ.pop("VLMAS_RNLCR")
    torch.manual_seed(11)
    x1, x2 = torch.randn(1, 4, 64), torch.randn(1, 1, 64)
    c0 = copy.deepcopy(base_cache)
    with torch.no_grad():
        o1 = lm(inputs_embeds=x1, position_ids=text_positions(4, L0 + m), past_key_values=c0, use_cache=True).last_hidden_state
        o2 = lm(inputs_embeds=x2, position_ids=text_positions(1, L0 + m + 4), past_key_values=c0, use_cache=True).last_hidden_state
    check("diag: outputs == native, measurement only", torch.allclose(out_d[0], o1) and torch.allclose(out_d[1], o2) and st_d["applied"] is False
          and st_d["calls"] == 6 and abs(st_d["lat_new"] - st_d["lat_nat"]) < 1e-9 and st_d["lat_would"] > st_d["lat_nat"])
    check("diag: a_s, C_s, message norms present", all(k in st_d for k in ("a_s", "C_s", "o", "msg_nat", "msg_new", "a_s_by_layer")) and len(st_d["a_s_by_layer"]) == 3)

    st_c, out_c, _ = answer(copy.deepcopy(base_cache), "cap")
    check("cap: applied, latent mass gained a_s", st_c["applied"] and abs((st_c["lat_new"] - st_c["lat_nat"]) - st_c["a_s"]) < 1e-6, (st_c["lat_nat"], st_c["lat_new"], st_c["a_s"]))
    check("cap: outputs changed vs native", not torch.allclose(out_c[1], o2))
    # manual: layer-0 attention output for the single generated row with the redistributed weights
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, repeat_kv
    with torch.no_grad():
        c4 = copy.deepcopy(base_cache)
        lm(inputs_embeds=x1, position_ids=text_positions(4, L0 + m), past_key_values=c4, use_cache=True)
        # capture the layer-0 attention input for x2 with hooks: hidden after input_layernorm
        cap_in = {}
        os.environ["VLMAS_RNLCR"] = "cap"
        with rn.terminal(eng, c4, L0 + m + 1, case_index=0):
            # registered AFTER the RN-LCR hook so it sees the replaced output
            h = lm.layers[0].self_attn.register_forward_hook(lambda mod, a, k, o: cap_in.update(h=k["hidden_states"], pe=k["position_embeddings"], out=(o[0] if isinstance(o, tuple) else o)), with_kwargs=True)
            lm(inputs_embeds=x2, position_ids=text_positions(1, L0 + m + 4), past_key_values=c4, use_cache=True)
            h.remove()
        attn = lm.layers[0].self_attn
        keys, vals = c4.layers[0].keys, c4.layers[0].values
        q = attn.q_norm(attn.q_proj(cap_in["h"]).view(1, 1, -1, 16)).transpose(1, 2)
        q, _ = apply_rotary_pos_emb(q, torch.zeros_like(q), cap_in["pe"][0], cap_in["pe"][1])
        K, Vv = repeat_kv(keys, 2), repeat_kv(vals, 2)
        sc = (q @ K.transpose(-2, -1)) * attn.scaling
        al = torch.softmax(sc, -1)
        latc = torch.tensor([L0, L0 + 1, L0 + 2])
        new, a_s = rn.redistribute(al, latc, 0, sc)
        o_man = attn.o_proj((new @ Vv).transpose(1, 2).reshape(1, 1, -1))
    check("cap: hook output == manual redistributed attention (layer 0, generated row)", torch.allclose(cap_in["out"], o_man, atol=1e-5),
          float((cap_in["out"] - o_man).abs().max()))
    st_a, _, _ = answer(copy.deepcopy(base_cache), "cap", rows="all")
    check("cap rows=all: prefill rows included", st_a["rows"] == 5)

    # -- full end-to-end (ground + cap)
    cap, res = run_reasoner("full", case=2)
    R = rn.apply_reasoner(eng, res, m=m, case_index=2)
    st_f, _, _ = answer(res["past_key_values"], "full")
    check("full: ground recorded + capacity applied", "ground" in R and st_f["applied"] and st_f.get("ground") == "yes" and "ground_stats" in st_f)

    # -- SP-GLR modes
    check("spglr config", (os.environ.__setitem__("VLMAS_RNLCR", "spglr") or rn.grounds()) and rn.readout() and not rn.redistributes())
    torch.manual_seed(2)
    al = torch.softmax(torch.randn(1, 4, 2, 3), -1); Vl = torch.randn(1, 4, 3, 16); Vs = torch.randn(1, 4, 3, 16)
    mR, gR, mt = rn.sp_readout(al, Vl, Vs, norm_preserve=True)
    check("sp_readout: |m~| == |m_R| and direction == g_R", torch.allclose(mt.norm(dim=-1), mR.norm(dim=-1), atol=1e-5)
          and torch.allclose(torch.nn.functional.cosine_similarity(mt, gR, dim=-1), torch.ones(1, 4, 2), atol=1e-5))
    check("sp_readout: m_R == sum alpha V, g_R == sum rho V*", torch.allclose(mR, (al.unsqueeze(-1) * Vl.unsqueeze(2)).sum(-2), atol=1e-6)
          and torch.allclose(gR, ((al / al.sum(-1, keepdim=True)).unsqueeze(-1) * Vs.unsqueeze(2)).sum(-2), atol=1e-5))
    _, _, mn = rn.sp_readout(al, Vl, Vs, norm_preserve=False)
    check("sp_readout nonorm: m~ == g_R", torch.equal(mn, gR))
    cap, res = run_reasoner("spglr", case=3)
    native = copy.deepcopy(res["past_key_values"]); traj0 = [t.clone() for t in res["latent_trajectory"]]
    R = rn.apply_reasoner(eng, res, m=m, case_index=3)
    check("spglr: native latent K and V restored in the cache", all(torch.equal(a.keys, b.keys) and torch.equal(a.values, b.values) for a, b in zip(res["past_key_values"].layers, native.layers)))
    check("spglr: V* bank kept, differs from native V", "vstar" in R and len(R["vstar"]) == 3 and not torch.equal(R["vstar"][0], native.layers[0].values[..., L0:L0 + m, :]))
    check("spglr: trajectory left native", all(torch.equal(a, b) for a, b in zip(res["latent_trajectory"], traj0)))
    with torch.no_grad():   # V* == replay of the grounded trajectory (recompute from the recorded grounding)
        pass
    base_sp = copy.deepcopy(res["past_key_values"])
    st_s, out_s, (x1s, x2s) = answer(copy.deepcopy(base_sp), "spglr")
    check("spglr: applied, stats present", st_s["applied"] and all(k in st_s for k in ("mR", "gR", "mt", "cos_mg")) and abs(st_s["mt"] - st_s["mR"]) < 1e-4, (st_s.get("mR"), st_s.get("mt")))
    # manual layer-0 check for the generated row
    with torch.no_grad():
        c5 = copy.deepcopy(base_sp)
        lm(inputs_embeds=x1s, position_ids=text_positions(4, L0 + m), past_key_values=c5, use_cache=True)
        cap_in = {}
        os.environ["VLMAS_RNLCR"] = "spglr"; bb._rnlcr_R = R
        with rn.terminal(eng, c5, L0 + m + 1, case_index=3):
            h = lm.layers[0].self_attn.register_forward_hook(lambda mod, a, k, o: cap_in.update(h=k["hidden_states"], pe=k["position_embeddings"], out=(o[0] if isinstance(o, tuple) else o)), with_kwargs=True)
            lm(inputs_embeds=x2s, position_ids=text_positions(1, L0 + m + 4), past_key_values=c5, use_cache=True)
            h.remove()
        attn = lm.layers[0].self_attn
        q = attn.q_norm(attn.q_proj(cap_in["h"]).view(1, 1, -1, 16)).transpose(1, 2)
        q, _ = apply_rotary_pos_emb(q, torch.zeros_like(q), cap_in["pe"][0], cap_in["pe"][1])
        K, Vv = repeat_kv(c5.layers[0].keys, 2), repeat_kv(c5.layers[0].values, 2)
        al = torch.softmax((q @ K.transpose(-2, -1)) * attn.scaling, -1)
        latc = torch.tensor([L0, L0 + 1, L0 + 2])
        Vs = repeat_kv(R["vstar"][0], 2)
        mR, gR, mt = rn.sp_readout(al[..., latc], Vv[..., latc, :], Vs, norm_preserve=True)
        o_man = attn.o_proj((al @ Vv - mR + mt).transpose(1, 2).reshape(1, 1, -1))
    check("spglr: hook output == manual o - m_R + m~ (layer 0, generated row)", torch.allclose(cap_in["out"], o_man, atol=1e-5), float((cap_in["out"] - o_man).abs().max()))
    st_n, _, _ = answer(copy.deepcopy(base_sp), "nonorm")
    check("nonorm: |m~| == |g_R|", abs(st_n["mt"] - st_n["gR"]) < 1e-5)
    cap, res = run_reasoner("vonly", case=4)
    native = copy.deepcopy(res["past_key_values"])
    R = rn.apply_reasoner(eng, res, m=m, case_index=4)
    check("vonly: K native, V replayed", all(torch.equal(a.keys, b.keys) for a, b in zip(res["past_key_values"].layers, native.layers))
          and not torch.equal(res["past_key_values"].layers[0].values[..., L0:, :], native.layers[0].values[..., L0:, :])
          and torch.equal(res["past_key_values"].layers[0].values[..., L0:, :], R["vstar"][0]))
    st_v, out_v, _ = answer(copy.deepcopy(res["past_key_values"]), "vonly")
    check("vonly: native readout (no replacement)", st_v["applied"] is False)

    # -- FULR (C2 #18): grounded V* bank + sink-norm-conserved per-head read
    os.environ["VLMAS_RNLCR"] = "fulr"
    check("fulr config", rn.grounds() and rn.fulr_mode() and not rn.readout() and not rn.redistributes())
    torch.manual_seed(5)
    ms_ = torch.randn(1, 4, 2, 64); rr_ = torch.randn(1, 4, 2, 64)
    mt_, rp_ = rn.fulr_rotate(ms_, rr_)
    check("fulr_rotate: |m~| == |m_s|", torch.allclose(mt_.norm(dim=-1), ms_.norm(dim=-1), atol=1e-5))
    check("fulr_rotate: r_perp orthogonal to m_s", float((rp_ * ms_).sum(-1).abs().max()) < 1e-4)
    check("fulr_rotate: direction == m_s + r_perp", torch.allclose(torch.nn.functional.cosine_similarity(mt_, ms_ + rp_, dim=-1), torch.ones(1, 4, 2), atol=1e-5))
    mt0, _ = rn.fulr_rotate(ms_, torch.zeros_like(rr_))
    check("fulr_rotate: zero residual -> identity", torch.allclose(mt0, ms_, atol=1e-5))
    mtz, _ = rn.fulr_rotate(torch.zeros_like(ms_) + 1e-9, rr_)
    check("fulr_rotate: vanishing sink -> vanishing output (no gain)", float(mtz.norm(dim=-1).max()) < 1e-6)
    sc_ = torch.randn(1, 4, 2, 3); Vl_ = torch.randn(1, 4, 3, 16); Vs_ = torch.randn(1, 4, 3, 16)
    dv_, rho_ = rn.fulr_payload(sc_, Vl_, Vs_)
    check("fulr_payload: rho = softmax(native latent logits), dV_bar = rho @ (V* - V)",
          torch.allclose(rho_, torch.softmax(sc_, -1)) and torch.allclose(dv_, torch.softmax(sc_, -1) @ (Vs_ - Vl_), atol=1e-6))
    ctx_ = torch.randn(1, 4, 2, 16)
    ph_ = rn.per_head_oproj(lm.layers[0].self_attn.o_proj, ctx_)
    check("per_head_oproj: heads sum to o_proj", torch.allclose(ph_.sum(1), lm.layers[0].self_attn.o_proj(ctx_.transpose(1, 2).reshape(1, 2, -1)).float(), atol=1e-5))
    cap, res = run_reasoner("fulr", case=6)
    native = copy.deepcopy(res["past_key_values"]); traj0 = [t.clone() for t in res["latent_trajectory"]]
    R = rn.apply_reasoner(eng, res, m=m, case_index=6)
    check("fulr: native latent K and V restored in the cache", all(torch.equal(a.keys, b.keys) and torch.equal(a.values, b.values) for a, b in zip(res["past_key_values"].layers, native.layers)))
    check("fulr: V* bank kept, differs from native V", "vstar" in R and len(R["vstar"]) == 3 and not torch.equal(R["vstar"][0], native.layers[0].values[..., L0:L0 + m, :]))
    check("fulr: trajectory left native", all(torch.equal(a, b) for a, b in zip(res["latent_trajectory"], traj0)))
    base_fu = copy.deepcopy(res["past_key_values"])
    st_u, out_u, (x1u, x2u) = answer(copy.deepcopy(base_fu), "fulr")
    check("fulr: applied, stats present", st_u["applied"] and all(k in st_u for k in ("ms", "rR", "rperp", "cos_rot", "delta_rel", "dv_rel", "fal_by_layer", "falv_by_layer", "cos_rot_by_layer", "fal_mean"))
          and len(st_u["fal_by_layer"]) == 3 and all(v is not None for v in st_u["fal_by_layer"]) and st_u["rows"] == 2, {k: st_u.get(k) for k in ("ms", "rR", "cos_rot", "fal_mean", "rows")})
    check("fulr: latent read left native (lat_new == lat_nat, msg unchanged)", abs(st_u["lat_new"] - st_u["lat_nat"]) < 1e-9 and abs(st_u["msg_new"] - st_u["msg_nat"]) < 1e-9)
    os.environ.pop("VLMAS_RNLCR"); os.environ.pop("VLMAS_RNLCR_ROWS", None)
    torch.manual_seed(11)
    c8 = copy.deepcopy(base_fu)
    with torch.no_grad():
        o8a = lm(inputs_embeds=x1u, position_ids=text_positions(4, L0 + m), past_key_values=c8, use_cache=True).last_hidden_state
        o8b = lm(inputs_embeds=x2u, position_ids=text_positions(1, L0 + m + 4), past_key_values=c8, use_cache=True).last_hidden_state
    check("fulr: outputs changed vs native", not torch.allclose(out_u[1], o8b))
    # manual layer-0 check for the generated row: o' = o + sum_h (m~_s - m_s)
    with torch.no_grad():
        c9 = copy.deepcopy(base_fu)
        lm(inputs_embeds=x1u, position_ids=text_positions(4, L0 + m), past_key_values=c9, use_cache=True)
        cap_in = {}
        os.environ["VLMAS_RNLCR"] = "fulr"; bb._rnlcr_R = R
        with rn.terminal(eng, c9, L0 + m + 1, case_index=6):
            h = lm.layers[0].self_attn.register_forward_hook(lambda mod, a, k, o: cap_in.update(h=k["hidden_states"], pe=k["position_embeddings"], out=(o[0] if isinstance(o, tuple) else o)), with_kwargs=True)
            lm(inputs_embeds=x2u, position_ids=text_positions(1, L0 + m + 4), past_key_values=c9, use_cache=True)
            h.remove()
        attn = lm.layers[0].self_attn
        q = attn.q_norm(attn.q_proj(cap_in["h"]).view(1, 1, -1, 16)).transpose(1, 2)
        q, _ = apply_rotary_pos_emb(q, torch.zeros_like(q), cap_in["pe"][0], cap_in["pe"][1])
        K, Vv = repeat_kv(c9.layers[0].keys, 2), repeat_kv(c9.layers[0].values, 2)
        sc = (q @ K.transpose(-2, -1)) * attn.scaling
        al = torch.softmax(sc, -1)
        latc = torch.tensor([L0, L0 + 1, L0 + 2])
        Vs = repeat_kv(R["vstar"][0], 2)
        dvb, _ = rn.fulr_payload(sc[..., latc], Vv[..., latc, :], Vs)
        a_s = al[..., 0].unsqueeze(-1)
        m_s = rn.per_head_oproj(attn.o_proj, a_s * Vv[..., 0, :].unsqueeze(2))
        r_R = rn.per_head_oproj(attn.o_proj, a_s * dvb)
        m_t, _ = rn.fulr_rotate(m_s, r_R)
        o_nat = attn.o_proj((al @ Vv).transpose(1, 2).reshape(1, 1, -1)).float()
        o_man = o_nat + (m_t - m_s).sum(1)
        norm_ok = torch.allclose(m_t.norm(dim=-1), m_s.norm(dim=-1), atol=1e-5)
    check("fulr: hook output == manual o + sum_h(m~_s - m_s) (layer 0, generated row); per-head norms conserved",
          torch.allclose(cap_in["out"].float(), o_man, atol=1e-4) and norm_ok, float((cap_in["out"].float() - o_man).abs().max()))
    st_ua, _, _ = answer(copy.deepcopy(base_fu), "fulr", rows="all")
    check("fulr rows=all: prefill rows included", st_ua["rows"] == 5 and st_ua["applied"])
    # FAST path == reference path (outputs and every stat), rows=gen and rows=all
    import json as _json
    KEYS = ("a_s", "C_s", "o", "lat_nat", "lat_new", "lat_would", "msg_nat", "msg_new", "msg_would", "ms", "rR", "rperp", "cos_rot", "delta_rel", "dv_rel", "fal_mean", "falv_mean")
    for rows_ in ("gen", "all"):
        bb._rnlcr_R = R
        os.environ.pop("VLMAS_RNLCR_FAST", None)
        st_ref, out_ref, _ = answer(copy.deepcopy(base_fu), "fulr", rows=rows_)
        bb._rnlcr_R = R
        os.environ["VLMAS_RNLCR_FAST"] = "1"
        st_fa, out_fa, _ = answer(copy.deepcopy(base_fu), "fulr", rows=rows_)
        os.environ.pop("VLMAS_RNLCR_FAST", None)
        dmax = max(float((a - b).abs().max()) for a, b in zip(out_ref, out_fa))
        check(f"fast rows={rows_}: outputs == reference", dmax < 1e-5, dmax)
        bad = {k: (st_ref.get(k), st_fa.get(k)) for k in KEYS if st_ref.get(k) is None or st_fa.get(k) is None or abs(st_ref[k] - st_fa[k]) > 1e-4 * max(1.0, abs(st_ref[k]))}
        check(f"fast rows={rows_}: stats == reference", not bad, bad)
        lay = all(abs(a - b) < 1e-4 for a, b in zip(st_ref["cos_rot_by_layer"] + st_ref["fal_by_layer"] + st_ref["a_s_by_layer"], st_fa["cos_rot_by_layer"] + st_fa["fal_by_layer"] + st_fa["a_s_by_layer"]))
        check(f"fast rows={rows_}: per-layer stats == reference, calls/rows equal, fast flag, json-serialisable",
              lay and st_ref["calls"] == st_fa["calls"] and st_ref["rows"] == st_fa["rows"] and st_fa.get("fast") is True and "fast" not in st_ref
              and bool(_json.dumps(st_fa)))
    # grounding skipped -> readout skipped (no bank)
    os.environ["VLMAS_RNLCR"] = "fulr"
    bb._rnlcr_R = {"latent_cols": torch.tensor([L0, L0 + 1, L0 + 2]), "m": m, "vis_cols": vis.clone(), "ground_skip": "x"}
    with rn.terminal(eng, copy.deepcopy(base_fu), L0 + m + 1, case_index=6) as st_sk:
        pass
    # FAST=2 hook-free fused sparse path == hook reference (same sdpa backend for both), outputs and stats
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS as _AAF
    impl0 = lm.config._attn_implementation
    lm.config._attn_implementation = "sdpa"
    KEYS2 = ("a_s", "ms", "rR", "rperp", "cos_rot", "delta_rel", "dv_rel", "fal_mean", "falv_mean", "o")
    try:
        for rows_ in ("gen", "all"):
            bb._rnlcr_R = R
            os.environ.pop("VLMAS_RNLCR_FAST", None)
            st_ref2, out_ref2, _ = answer(copy.deepcopy(base_fu), "fulr", rows=rows_)
            bb._rnlcr_R = R
            n_hooks = sum(len(l.self_attn._forward_hooks) for l in lm.layers)
            local_before = dict(_AAF._local_mapping)
            os.environ["VLMAS_RNLCR_FAST"] = "2"
            seen_hooks = []
            _orig_answer_lm = lm.forward
            def _count(*a, **k):
                seen_hooks.append(sum(len(l.self_attn._forward_hooks) for l in lm.layers))
                return _orig_answer_lm(*a, **k)
            lm.forward = _count
            try:
                st_f2, out_f2, _ = answer(copy.deepcopy(base_fu), "fulr", rows=rows_)
            finally:
                del lm.forward
            os.environ.pop("VLMAS_RNLCR_FAST", None)
            dmax = max(float((a - b).abs().max()) for a, b in zip(out_ref2, out_f2))
            check(f"fast2 rows={rows_}: outputs == hook reference", dmax < 1e-5, dmax)
            bad = {k: (st_ref2.get(k), st_f2.get(k)) for k in KEYS2 if st_ref2.get(k) is None or st_f2.get(k) is None or abs(st_ref2[k] - st_f2[k]) > 1e-4 * max(1.0, abs(st_ref2[k]))}
            check(f"fast2 rows={rows_}: stats == hook reference", not bad, bad)
            # cos_rot: eps 1e-8 is applied at unit alpha in fast2 vs at alpha-scaled |m_s|^2 ~1e-6 in the tiny model -> 1e-3 tolerance there
            lay = all(abs(a - b) < 1e-4 for a, b in zip(st_ref2["fal_by_layer"] + st_ref2["a_s_by_layer"] + st_ref2["ms_by_layer"],
                                                           st_f2["fal_by_layer"] + st_f2["a_s_by_layer"] + st_f2["ms_by_layer"]))                   and all(abs(a - b) < 1e-3 for a, b in zip(st_ref2["cos_rot_by_layer"], st_f2["cos_rot_by_layer"]))
            check(f"fast2 rows={rows_}: per-layer stats equal, calls/rows equal, no hooks installed, registry restored, fast flag 2",
                  lay and st_ref2["calls"] == st_f2["calls"] and st_ref2["rows"] == st_f2["rows"] and bool(seen_hooks) and set(seen_hooks) == {n_hooks}
                  and dict(_AAF._local_mapping) == local_before and st_f2.get("fast") == 2 and bool(_json.dumps(st_f2)))
        bb._rnlcr_R = R
        os.environ["VLMAS_RNLCR_FAST"] = "2"; os.environ["VLMAS_RNLCR_STATS"] = "none"
        st_n, out_n, _ = answer(copy.deepcopy(base_fu), "fulr", rows="gen")
        os.environ.pop("VLMAS_RNLCR_FAST", None); os.environ.pop("VLMAS_RNLCR_STATS", None)
        bb._rnlcr_R = R
        st_g, out_g, _ = answer(copy.deepcopy(base_fu), "fulr", rows="gen")
        check("fast2 stats=none: outputs unchanged vs hook reference", max(float((a - b).abs().max()) for a, b in zip(out_g, out_n)) < 1e-5)
    finally:
        lm.config._attn_implementation = impl0
    bb._rnlcr_R = R
    os.environ["VLMAS_RNLCR_FAST"] = "2"
    st_e, out_e, _ = answer(copy.deepcopy(base_fu), "fulr", rows="gen")
    os.environ.pop("VLMAS_RNLCR_FAST", None)
    check("fast2 on eager backend falls back to the fast hook path, outputs == reference", st_e.get("fast2_fallback") == "eager"
          and max(float((a - b).abs().max()) for a, b in zip(out_u, out_e)) < 1e-5)
    check("fulr: no V* bank -> skip", "skip" in st_sk and "bank" in st_sk["skip"])
    os.environ.pop("VLMAS_RNLCR", None)

    # -- RELR: teacher pass, differentiable replay optimisation, commit with turn-close re-append
    os.environ["VLMAS_RNLCR"] = "relr"; os.environ["VLMAS_RELR_STEPS"] = "6"
    cap, res = run_reasoner("relr", case=5)
    check("relr: no Reasoner capture (no grounding)", cap is None)
    R = rn.apply_reasoner(eng, res, m=m, case_index=5)
    check("relr: record has trajectory + cursor0", R.get("traj") is not None and len(R["traj"]) == m and R["cursor0"] == L0)
    cache = res["past_key_values"]
    close_ids = Tok().encode("<|im_end|>\n"); n_close = len(close_ids)
    with torch.no_grad():
        lm(inputs_embeds=lm.embed_tokens(torch.tensor(close_ids)).unsqueeze(0), position_ids=text_positions(n_close, L0 + m), past_key_values=cache, use_cache=True)
    Lfull = rn._kvlen(cache)
    native = copy.deepcopy(cache)
    torch.manual_seed(21)
    pids = torch.randint(0, 50, (9,))
    bb._rnlcr_R = R
    with rn.terminal(eng, cache, Lfull, case_index=5, prompt_ids=pids) as st:
        check("relr: applied before the Answerer runs, cache length preserved", st.get("applied") is True and rn._kvlen(cache) == Lfull, st.get("skip"))
        with torch.no_grad():
            lm(inputs_embeds=lm.embed_tokens(pids).unsqueeze(0), position_ids=text_positions(9, Lfull), past_key_values=cache, use_cache=True)
    check("relr: no skip, 7 losses, message loss decreased", "skip" not in st and len(st["loss_msg"]) == 7 and st["loss_msg"][-1] < st["loss_msg"][0], (st.get("skip"), st.get("loss_msg")))
    check("relr: latent K/V replaced, prefix untouched", not torch.equal(cache.layers[0].keys[..., L0:L0 + m, :], native.layers[0].keys[..., L0:L0 + m, :])
          and torch.equal(cache.layers[0].keys[..., :L0, :], native.layers[0].keys[..., :L0, :]))
    check("relr: close tokens recomputed on the new latents (layer 2; layer-0 keys are context-free)",
          not torch.equal(cache.layers[2].keys[..., L0 + m:Lfull, :], native.layers[2].keys[..., L0 + m:Lfull, :]))
    check("relr: KV affinity diagnostic present", "affinity" in st and all(k in st["affinity"] for k in ("native", "new"))
          and all(k in st["affinity"]["new"] for k in ("cosV_vis", "cosV_text", "cosK_vis", "cosK_text", "nnV_vis_frac", "nnK_vis_frac")), st.get("affinity"))
    check("relr: stats fields", all(k in st for k in ("cos_nat", "cos_new", "ratio_nat", "ratio_new", "dz_rel", "kv_rel", "n_close")) and st["n_close"] == n_close and st["dz_rel"] > 0)
    # teacher pass leaves the cache length unchanged and equals a manual eager computation for m_V at layer 0
    c6 = copy.deepcopy(native)
    lat = torch.tensor([L0, L0 + 1, L0 + 2])
    te = rn.relr_teacher(bb, c6, Lfull, pids, lat, vis, [0, 1, 2])
    check("relr teacher: cache cropped back", rn._kvlen(c6) == Lfull and set(te.keys()) == {0, 1, 2})
    with torch.no_grad():
        c7 = copy.deepcopy(native)
        o7 = lm(inputs_embeds=lm.embed_tokens(pids).unsqueeze(0), position_ids=text_positions(9, Lfull), past_key_values=c7, use_cache=True, output_attentions=True)
        a0 = o7.attentions[0][0, :, -1, :]                                     # [H, L]
        Vv = repeat_kv(c7.layers[0].values, 2)[0]                              # [H, L, D]
        ctx_v = torch.einsum("hj,hjd->hd", a0[:, vis], Vv[:, vis, :])
        m_V_man = lm.layers[0].self_attn.o_proj(ctx_v.reshape(1, 1, -1)).reshape(-1)
    check("relr teacher: m_V == eager-attention manual (layer 0)", torch.allclose(te[0]["m_V"], m_V_man, atol=1e-4), float((te[0]["m_V"] - m_V_man).abs().max()))
    os.environ.pop("VLMAS_RELR_STEPS")

    # -- guards / base
    os.environ["VLMAS_RNLCR"] = "cap"
    bb._rnlcr_R = None
    with rn.terminal(eng, DynamicCache(), 10, case_index=0) as st:
        pass
    check("no record -> skip", "skip" in st)
    os.environ.pop("VLMAS_RNLCR"); os.environ.pop("VLMAS_RNLCR_ROWS", None)
    n_pre = len(lm._forward_pre_hooks)
    with rn.reasoner_probe(bb, m=m, stage="reasoner", targets=[Q]) as cap:
        check("base: probe None, no hooks", cap is None and len(lm._forward_pre_hooks) == n_pre)
    check("base: apply_reasoner None", rn.apply_reasoner(eng, {}, m=m) is None)
except ImportError as exc:
    print(f"  skip tiny-model block: {exc!r}")

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
