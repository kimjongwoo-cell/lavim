"""Add FULR tests to tests_hj_test_rnlcr.py (md5-guarded)."""
import hashlib, sys
p = sys.argv[1] if len(sys.argv) > 1 else "tests_hj_test_rnlcr.py"
src = open(p).read()
md5 = hashlib.md5(src.encode()).hexdigest()[:8]
assert md5 == "4b218411", f"tests md5 {md5}"

old = '''    # -- RELR: teacher pass, differentiable replay optimisation, commit with turn-close re-append
'''
new = '''    # -- FULR (C2 #18): grounded V* bank + sink-norm-conserved per-head read
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
    # grounding skipped -> readout skipped (no bank)
    os.environ["VLMAS_RNLCR"] = "fulr"
    bb._rnlcr_R = {"latent_cols": torch.tensor([L0, L0 + 1, L0 + 2]), "m": m, "vis_cols": vis.clone(), "ground_skip": "x"}
    with rn.terminal(eng, copy.deepcopy(base_fu), L0 + m + 1, case_index=6) as st_sk:
        pass
    check("fulr: no V* bank -> skip", "skip" in st_sk and "bank" in st_sk["skip"])
    os.environ.pop("VLMAS_RNLCR", None)

    # -- RELR: teacher pass, differentiable replay optimisation, commit with turn-close re-append
'''
assert src.count(old) == 1
src = src.replace(old, new)
src = src.replace('"""RN-LCR (C2 #15) unit tests', '"""RN-LCR (C2 #15) + SP-GLR + RELR + FULR (C2 #18) unit tests', 1)
open(p, "w").write(src)
print("patched", p, "->", hashlib.md5(src.encode()).hexdigest()[:8])
