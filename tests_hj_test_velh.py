#!/usr/bin/env python3
"""VELH (C2 #14) unit tests — RoPE canonicalisation / rephase, question read, projection, end-to-end on a tiny
Qwen3-VL text model (CPU): capture, in-place arms, mass hook vs eager attentions, base off.

usage: python tests_hj_test_velh.py
"""
import copy
import os
import sys
from types import SimpleNamespace

import torch

TREE = "/home/users/whddn12316/wsi_latent_0915_decode_hj"
sys.path.insert(0, TREE)
os.environ.pop("VLMAS_VELH", None)
from memory import velh, lu  # noqa: E402

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
check("off by default", not velh.enabled())
os.environ["VLMAS_VELH"] = "FULL"
check("full (case-insensitive)", velh.mode() == "full")
os.environ["VLMAS_VELH"] = "nope"
check("unknown = off", not velh.enabled())
os.environ.pop("VLMAS_VELH")

print("[pure: projection / read]")
torch.manual_seed(0)
V = torch.randn(2, 8)
m = torch.randn(2, 8)
Vt, gap, proj, mn = velh.bind(V, m)
u = m / m.norm(dim=-1, keepdim=True)
check("bind: constraint satisfied u.V~ >= |m|", bool(((u * Vt).sum(-1) >= mn - 1e-5).all()))
check("bind: orthogonal content preserved", torch.allclose(Vt - (u * Vt).sum(-1, keepdim=True) * u, V - (u * V).sum(-1, keepdim=True) * u, atol=1e-5))
check("bind: change = [gap]_+ u exactly", torch.allclose(Vt - V, torch.clamp(gap, min=0)[:, None] * u, atol=1e-6))
Vbig = V + 100 * u
Vt2, gap2, _, _ = velh.bind(Vbig, m)
check("bind: identity when already satisfied", torch.equal(Vt2, Vbig.float()) and bool((gap2 < 0).all()))
q = torch.randn(4, 8)                      # Hq=4
Kb = torch.randn(2, 5, 8)                  # Hk=2, n=5
Vv = torch.randn(2, 5, 8)
mk, beta = velh.question_read(q, Kb, Vv, scaling=0.5, groups=2)
man = torch.zeros(2, 8)
for h in range(4):
    k = h // 2
    b = torch.softmax(q[h] @ Kb[k].T * 0.5, -1)
    man[k] += (b[:, None] * Vv[k]).sum(0) / 2
check("question_read: beta rows sum to 1", torch.allclose(beta.sum(-1), torch.ones(4)))
check("question_read: group-mean of sum beta V", torch.allclose(mk, man, atol=1e-5))

print("[tiny model]")
try:
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel, apply_rotary_pos_emb

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
    vis = torch.arange(past0 + 2, past0 + 18)          # 16 "visual" columns (absolute); 3-D grid positions below
    L0 = past0 + 34
    anchor_rel = 20 + len(q_ids) - 1

    def grid_positions(T, start):
        """text positions except the visual rows, which get a 2x8 (h,w) MRoPE grid like real image tokens."""
        pos = text_positions(T, start).clone()
        for i, c in enumerate((vis - past0).tolist()):
            pos[1, 0, c] = start + 2 + i // 8       # h axis
            pos[2, 0, c] = start + 2 + i % 8        # w axis
            pos[0, 0, c] = start + 2                # t axis
        return pos

    def run_reasoner(mode, case):
        os.environ["VLMAS_VELH"] = mode
        cache = DynamicCache()
        torch.manual_seed(7 + case)
        pre = torch.randn(1, past0, 64)
        with torch.no_grad():
            lm(inputs_embeds=pre, position_ids=text_positions(past0, 0), past_key_values=cache, use_cache=True)
            emb = lm.embed_tokens(ids).unsqueeze(0)
            with velh.reasoner_probe(bb, m=m, stage="reasoner", targets=["Question stem: " + Q, Q]) as cap:
                o = lm(inputs_embeds=emb, position_ids=grid_positions(34, past0), past_key_values=cache,
                       use_cache=True, output_hidden_states=True)
                last = o.hidden_states[-1][:, -1:, :]
                for t in range(m):
                    le = last * (5.0 / last.norm(dim=-1, keepdim=True))
                    o = lm(inputs_embeds=le, position_ids=text_positions(1, L0 + t), past_key_values=cache,
                           use_cache=True, output_hidden_states=True)
                    last = o.hidden_states[-1][:, -1:, :]
            hs_prefill = o_pre = None
        res = {"past_key_values": cache, "vis_cols": vis.clone(), "pos_cursor": L0 + m, "past_len": L0 + m}
        bb._velh_cap = cap
        R = velh.record_reasoner(eng, res, m=m, case_index=case)
        # close the turn with 2 text tokens (like <|im_end|>\n) -> p_A^- = L0+m+1, Answerer starts at L0+m+2
        with torch.no_grad():
            lm(inputs_embeds=torch.randn(1, 2, 64), position_ids=text_positions(2, L0 + m), past_key_values=cache, use_cache=True)
        return cap, R, cache

    cap, R, cache = run_reasoner("diag", 0)
    check("capture: anchor = last question row", cap.anchor == anchor_rel and R is not None and R["anchor_col"] == past0 + anchor_rel)
    check("record: endpoint = last latent column / position", R["ep_col"] == L0 + m - 1 and R["ep_pos"] == L0 + m - 1)
    # manual pre-RoPE anchor query per layer from a separate forward with hidden states
    with torch.no_grad():
        c2 = DynamicCache()
        torch.manual_seed(7)
        lm(inputs_embeds=torch.randn(1, past0, 64), position_ids=text_positions(past0, 0), past_key_values=c2, use_cache=True)
        o2 = lm(inputs_embeds=lm.embed_tokens(ids).unsqueeze(0), position_ids=grid_positions(34, past0), past_key_values=c2,
                use_cache=True, output_hidden_states=True, output_attentions=True)
        ok = True
        for li, layer in enumerate(lm.layers):
            h = layer.input_layernorm(o2.hidden_states[li])[:, anchor_rel:anchor_rel + 1]
            qm = layer.self_attn.q_norm(layer.self_attn.q_proj(h).view(1, 1, -1, 16))[0, 0]
            ok &= torch.allclose(qm, R["q_pre"][li], atol=1e-5)
        check("capture: anchor pre-RoPE query == manual (all layers)", ok)
        # canonicalisation: derotate(cached K at visual rows) == pre-RoPE k
        l0 = lm.layers[0]
        h0 = l0.input_layernorm(o2.hidden_states[0])[0]                                   # [34, 64]
        k_pre = l0.self_attn.k_norm(l0.self_attn.k_proj(h0).view(34, -1, 16)).transpose(0, 1)   # [Hk, 34, D]
        Kc = c2.layers[0].keys[0, :, vis, :]
        Kbar = velh.derotate(Kc, R["vis_cos"], R["vis_sin"])
        check("derotate: R(p_j)^-1 K_j == pre-RoPE key (3-D grid positions)", torch.allclose(Kbar, k_pre[:, vis - past0, :], atol=1e-4),
              float((Kbar - k_pre[:, vis - past0, :]).abs().max()))
        # rephase: moving the anchor key from p_Q to p' == rotating the pre-RoPE key at p'
        pA = L0 + m + 1
        cosA, sinA = velh.text_cos_sin(bb, pA)
        Kq = c2.layers[0].keys[0, :, past0 + anchor_rel, :]
        Kd = velh.rephase(Kq, R["anchor_cos"], R["anchor_sin"], cosA, sinA)
        kp = k_pre[:, anchor_rel, :][None, :, None, :]                                     # [1,Hk,1,D]
        cosA2, sinA2 = lm.rotary_emb(kp, text_positions(1, pA))
        _, k_at = apply_rotary_pos_emb(torch.zeros_like(kp), kp, cosA2, sinA2)
        check("rephase: R(p_A)R(p_Q)^-1 K_Q == RoPE(k_pre, p_A)", torch.allclose(Kd, k_at[0, :, 0, :], atol=1e-4))

    # -- arms (in place on the cache the "Answerer" consumes), plus the mass hook vs eager attentions
    def answerer(cache, mode, case=0):
        os.environ["VLMAS_VELH"] = mode
        pA = L0 + m + 1
        with velh.terminal(eng, cache, pA + 1, case_index=case) as st:
            with torch.no_grad():
                lm(inputs_embeds=torch.randn(1, 4, 64), position_ids=text_positions(4, pA + 1), past_key_values=cache, use_cache=True)
                lm(inputs_embeds=torch.randn(1, 1, 64), position_ids=text_positions(1, pA + 5), past_key_values=cache, use_cache=True)
        return st

    cap, R, cache = run_reasoner("diag", 0)
    native = copy.deepcopy(cache)
    st = answerer(cache, "diag")
    def prefix_eq(c, nat):
        n = nat.layers[0].keys.shape[-2]
        return all(torch.equal(a.keys[..., :n, :], b.keys) and torch.equal(a.values[..., :n, :], b.values)
                   for a, b in zip(c.layers, nat.layers))
    check("diag: cache untouched (native prefix)", prefix_eq(cache, native))
    check("diag: measurement fields", st["applied"] is False and st["calls"] == 6 and st["rows"] == 2 and 0 <= st["mass_ep"] <= 1
          and "act_rate" in st and st["shift"] == 2, st)
    # mass hook check against eager attentions of the same two Answerer calls
    with torch.no_grad():
        c3 = copy.deepcopy(native)
        torch.manual_seed(11)
        x1, x2 = torch.randn(1, 4, 64), torch.randn(1, 1, 64)
        os.environ["VLMAS_VELH"] = "diag"
        torch.manual_seed(11)
        with velh.terminal(eng, c3, L0 + m + 2, case_index=0) as st3:
            lm(inputs_embeds=x1, position_ids=text_positions(4, L0 + m + 2), past_key_values=c3, use_cache=True)
            lm(inputs_embeds=x2, position_ids=text_positions(1, L0 + m + 6), past_key_values=c3, use_cache=True)
        c4 = copy.deepcopy(native)
        a1 = lm(inputs_embeds=x1, position_ids=text_positions(4, L0 + m + 2), past_key_values=c4, use_cache=True, output_attentions=True).attentions
        a2 = lm(inputs_embeds=x2, position_ids=text_positions(1, L0 + m + 6), past_key_values=c4, use_cache=True, output_attentions=True).attentions
        ep = R["ep_col"]
        man_ep = (sum(a[0, :, -1, ep].mean() for a in a1) + sum(a[0, :, -1, ep].mean() for a in a2)) / 6
        man_vis = (sum(a[0, :, -1, vis].sum(-1).mean() for a in a1) + sum(a[0, :, -1, vis].sum(-1).mean() for a in a2)) / 6
        check("mass hook == eager attention (endpoint, visual)", abs(st3["mass_ep"] - float(man_ep)) < 1e-4 and abs(st3["mass_vis"] - float(man_vis)) < 1e-4,
              (st3["mass_ep"], float(man_ep), st3["mass_vis"], float(man_vis)))

    cap, R, cache = run_reasoner("bind", 0)
    native = copy.deepcopy(cache)
    st = answerer(cache, "bind")
    ep = R["ep_col"]
    changed_v = [not torch.equal(a.values[0, :, ep, :], b.values[0, :, ep, :]) for a, b in zip(cache.layers, native.layers)]
    nL = native.layers[0].keys.shape[-2]
    same_k = all(torch.equal(a.keys[..., :nL, :], b.keys) for a, b in zip(cache.layers, native.layers))
    others = all(torch.equal(torch.cat([a.values[..., :ep, :], a.values[..., ep + 1:nL, :]], -2),
                             torch.cat([b.values[..., :ep, :], b.values[..., ep + 1:, :]], -2)) for a, b in zip(cache.layers, native.layers))
    check("bind: only endpoint V changes (where the constraint is active), keys untouched", same_k and others and (any(changed_v) == (st["act_rate"] > 0)),
          (changed_v, st["act_rate"]))
    check("bind: dv_rel > 0 iff active", (st["dv_rel"] > 0) == (st["act_rate"] > 0))

    cap, R, cache = run_reasoner("handoff", 0)
    native = copy.deepcopy(cache)
    st = answerer(cache, "handoff")
    with torch.no_grad():
        cosA, sinA = velh.text_cos_sin(bb, L0 + m + 1)
        cosR, sinR = velh.text_cos_sin(bb, R["ep_pos"])
        ok = all(torch.allclose(a.keys[0, :, ep, :], velh.rephase(b.keys[0, :, ep, :], cosR, sinR, cosA, sinA), atol=1e-5)
                 for a, b in zip(cache.layers, native.layers))
        nL = native.layers[0].keys.shape[-2]
        same_v = all(torch.equal(a.values[..., :nL, :], b.values) for a, b in zip(cache.layers, native.layers))
    check("handoff: endpoint K == rephase(native), values untouched", ok and same_v)

    cap, R, cache = run_reasoner("direct", 0)
    native = copy.deepcopy(cache)
    st = answerer(cache, "direct")
    with torch.no_grad():
        ok_k = ok_v = True
        for li, (a, b) in enumerate(zip(cache.layers, native.layers)):
            attn = lm.layers[li].self_attn
            Kd = velh.rephase(b.keys[0, :, R["anchor_col"], :], R["anchor_cos"], R["anchor_sin"], cosA, sinA)
            ok_k &= torch.allclose(a.keys[0, :, ep, :], Kd, atol=1e-5)
            Kbar = velh.derotate(b.keys[0, :, vis, :], R["vis_cos"], R["vis_sin"])
            mk, _ = velh.question_read(R["q_pre"][li], Kbar, b.values[0, :, vis, :], attn.scaling, attn.num_key_value_groups)
            ok_v &= torch.allclose(a.values[0, :, ep, :], mk, atol=1e-5)
    check("direct: endpoint K == rephased anchor key, V == m_Q", ok_k and ok_v)

    cap, R, cache = run_reasoner("full", 0)
    st = answerer(cache, "full")
    check("full: applied, both stats present", st["applied"] and "mass_ep" in st and st["shift"] == 2)

    # shuffle: first case has no donor -> skip; second case uses case 0's anchor query
    bb._velh_R = None; bb._velh_prev = None
    cap, R0, cache0 = run_reasoner("shuffle", 0)
    st0 = answerer(cache0, "shuffle", case=0)
    check("shuffle: no donor on the first case -> skip", "skip" in st0 and "donor" in st0["skip"], st0.get("skip"))
    cap, R1, cache1 = run_reasoner("shuffle", 1)
    native1 = copy.deepcopy(cache1)
    st1 = answerer(cache1, "shuffle", case=1)
    with torch.no_grad():
        attn = lm.layers[0].self_attn
        Kbar = velh.derotate(native1.layers[0].keys[0, :, vis, :], R1["vis_cos"], R1["vis_sin"])
        mk_donor, _ = velh.question_read(R0["q_pre"][0], Kbar, native1.layers[0].values[0, :, vis, :], attn.scaling, attn.num_key_value_groups)
        Vt_donor, _, _, _ = velh.bind(native1.layers[0].values[0, :, R1["ep_col"], :], mk_donor)
    check("shuffle: donor = previous case, Step 1 uses its anchor query", st1.get("donor_case") == 0
          and torch.allclose(cache1.layers[0].values[0, :, R1["ep_col"], :], Vt_donor, atol=1e-5))

    # base: nothing installed
    os.environ.pop("VLMAS_VELH", None)
    n_pre = len(lm._forward_pre_hooks)
    with velh.reasoner_probe(bb, m=m, stage="reasoner", targets=[Q]) as cap:
        check("base: probe yields None, no hooks", cap is None and len(lm._forward_pre_hooks) == n_pre)
    check("base: record no-op", velh.record_reasoner(eng, {}, m=m) is None)
except ImportError as exc:
    print(f"  skip tiny-model block: {exc!r}")

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
