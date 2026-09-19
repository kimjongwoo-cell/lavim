"""VLMAS_RNLCR_FAST=2: hook-free fused sparse FULR (attention-function wrapper, Gram-precomputed head-wise rotation in pre-o_proj space)."""
import hashlib, shutil
P = "/home/users/whddn12316/wsi_latent_0915_decode_hj/memory/rnlcr.py"
src = open(P).read()
md5 = hashlib.md5(src.encode()).hexdigest()
assert md5.startswith("5b673ace"), md5
shutil.copy(P, P + ".bak_0917_fast2")

def rep(old, new):
    global src
    assert src.count(old) == 1, (old[:90], src.count(old))
    src = src.replace(old, new)

rep('''              rho = softmax(z_lat) without the full alpha tensor, stats kept on GPU and synced once at the end of the case.
''', '''              rho = softmax(z_lat) without the full alpha tensor, stats kept on GPU and synced once at the end of the case.
  VLMAS_RNLCR_FAST=2 (fulr only): hook-free fused sparse read. The model's attention function is wrapped for the Answerer call;
              after the native attention, per layer/head (Gram terms precomputed once per case from the sink / latent columns):
              S=W_O^h V_s, B_k=W_O^h dV_k; <S,S>, <S,B_k>, <B_k,B_j>; per token alpha_s=exp(z_s-LSE) (fp32 key mirror),
              rho=softmax(z_lat), p=rho.<S,B>, |r|^2=rho'<B,B>rho, c=p/|S|^2, |v|^2=(1-c)^2|S|^2+2(1-c)p+|r|^2,
              lambda=|S|/|v|, and the pre-o_proj head update alpha_s*(lambda*((1-c)V_s+sum_k rho_k dV_k)-V_s) is added to
              the attention output, which the native o_proj maps to exactly sum_h(m~_s-m_s). VLMAS_RNLCR_STATS=lite|none.
''')

rep('''    idx_sl = torch.cat([torch.tensor([s], device=bb.device, dtype=lat.dtype), lat]) if fast else None
''', '''    idx_sl = torch.cat([torch.tensor([s], device=bb.device, dtype=lat.dtype), lat]) if fu else None
    fast2 = fu and os.environ.get("VLMAS_RNLCR_FAST", "").strip() == "2"
    stats_level = os.environ.get("VLMAS_RNLCR_STATS", "lite").strip() or "lite"
    impl_name = getattr(bb.lm.config, "_attn_implementation", None) or "eager"
    if fast2 and impl_name == "eager":
        fast2, fast = False, True
        stats["fast"] = True
        stats["fast2_fallback"] = "eager"
    if fast2:
        stats["fast"] = 2
''')

rep('''    handles = [layer.self_attn.register_forward_hook((make_post_fast if fast else make_post)(li), with_kwargs=True) for li, layer in enumerate(bb.lm.layers)]
''', '''    pre2 = {}

    def _fulr2_layer(module, key, value):
        li = int(module.layer_idx)
        Pl = pre2.get(li)
        L_ = int(key.shape[2])
        if Pl is None:
            KVH = int(key.shape[1]); G = int(module.num_key_value_groups); H = KVH * G; D = int(value.shape[-1])
            W = module.o_proj.weight.float().view(-1, H, D)
            toHv = lambda x: x.unsqueeze(1).expand(x.shape[0], G, *x.shape[1:]).reshape(H, *x.shape[1:])
            Vs = toHv(value[0, :, s, :].float())                                         # [H,D]
            Vl = value[0, :, lat, :].float()                                             # [KVH,m,D]
            dV = toHv(vstar[li].to(value.device).float()[0] - Vl)                       # [H,m,D]
            S = torch.einsum("ehd,hd->he", W, Vs)
            Bm = torch.einsum("ehd,hmd->hme", W, dV)
            Pl = {"H": H, "G": G, "KVH": KVH, "D": D, "Vs": Vs, "dV": dV, "SS": (S * S).sum(-1),
                  "SB": torch.einsum("he,hme->hm", S, Bm), "BB": torch.einsum("hme,hne->hmn", Bm, Bm),
                  "Vl_norm": toHv(Vl.norm(dim=-1).mean(-1, keepdim=True)).squeeze(-1), "Kbuf": None, "Klen": 0}
            del W, S, Bm
            pre2[li] = Pl
        buf = Pl["Kbuf"]
        if buf is None or buf.shape[2] < L_ or Pl["Klen"] > L_:
            nb = torch.empty(1, Pl["KVH"], L_ + 1024, int(key.shape[-1]), dtype=torch.float32, device=key.device)
            nb[:, :, :L_] = key[:1].float()
            Pl["Kbuf"], Pl["Klen"] = nb, L_
        elif Pl["Klen"] < L_:
            buf[:, :, Pl["Klen"]:L_] = key[:1, :, Pl["Klen"]:L_].float()
            Pl["Klen"] = L_
        return Pl

    def make_attn2(native):
        def attn(module, query, key, value, attention_mask, *args, **kwargs):
            out, w = native(module, query, key, value, attention_mask, *args, **kwargs)
            if getattr(module, "layer_idx", None) is None:
                return out, w
            li = int(module.layer_idx)
            T, Lc = int(query.shape[2]), int(key.shape[2])
            sel = torch.arange(T, device=query.device) if cfg["rows"] == "all" else torch.tensor([T - 1], device=query.device)
            Ts = int(sel.numel())
            with torch.no_grad():
                Pl = _fulr2_layer(module, key, value)
                H, G, KVH = Pl["H"], Pl["G"], Pl["KVH"]
                sc = kwargs.get("scaling", None)
                sc = module.scaling if sc is None else sc
                qg = query[:1].index_select(2, sel).float().reshape(1, KVH, G, Ts, -1)
                z = torch.matmul(qg, Pl["Kbuf"][:, :, :Lc].unsqueeze(2).transpose(-2, -1)) * sc   # [1,KVH,G,Ts,Lc]
                if T > 1:
                    past = Lc - T
                    col = torch.arange(Lc, device=query.device)[None, None, None, None, :]
                    row = (past + sel)[None, None, None, :, None]
                    z = z.masked_fill(col > row, float("-inf"))
                lse = torch.logsumexp(z, dim=-1)
                z_sl = z[..., idx_sl]
                toH = lambda x: x.reshape(1, H, *x.shape[3:])
                a_s = toH(torch.exp(z_sl[..., 0] - lse))                                  # [1,H,Ts]
                rho = toH(torch.softmax(z_sl[..., 1:], dim=-1))                           # [1,H,Ts,m]
                SS = Pl["SS"][None, :, None]
                p = torch.einsum("bhtm,hm->bht", rho, Pl["SB"])
                rr = torch.einsum("bhtm,hmn,bhtn->bht", rho, Pl["BB"], rho)
                c = p / (SS + 1e-8)
                one_c = 1.0 - c
                vv = (one_c * one_c * SS + 2.0 * one_c * p + rr).clamp_min(0.0)
                vn = vv.sqrt()
                lam = SS.sqrt() / (vn + 1e-8)
                dvh = torch.einsum("bhtm,hmd->bhtd", rho, Pl["dV"])                     # [1,H,Ts,D]
                Vs = Pl["Vs"][None, :, None, :]
                delta = a_s.unsqueeze(-1) * (lam.unsqueeze(-1) * (one_c.unsqueeze(-1) * Vs + dvh) - Vs)
                out = out.clone()
                rows_nat = out[:1].index_select(1, sel)                                   # [1,Ts,H,D]
                out[:1, sel] = (rows_nat.float() + delta.transpose(1, 2)).to(out.dtype)
                if stats_level != "none":
                    Sn = SS.sqrt()
                    ms_n = a_s * Sn
                    rperp2 = (rr - 2.0 * c * p + c * c * SS).clamp_min(0.0)
                    cos_rot = torch.where(ms_n > 1e-6, (one_c * SS + p) / (Sn * vn + 1e-12), torch.ones_like(ms_n))
                    wd = module.o_proj.weight.dtype
                    o_nat_f = module.o_proj(rows_nat.reshape(1, Ts, -1).to(wd)).float()
                    d_out = module.o_proj(delta.transpose(1, 2).reshape(1, Ts, -1).to(wd)).float()
                    acc["ms"] = acc["ms"] + ms_n.mean(); acc["rR"] = acc["rR"] + (a_s * rr.sqrt()).mean(); acc["rperp"] = acc["rperp"] + (a_s * rperp2.sqrt()).mean()
                    acc["cos_rot"] = acc["cos_rot"] + cos_rot.mean(); acc["delta_rel"] = acc["delta_rel"] + (d_out.norm(dim=-1) / (o_nat_f.norm(dim=-1) + 1e-8)).mean()
                    acc["dv_rel"] = acc["dv_rel"] + (dvh.norm(dim=-1) / (Pl["Vl_norm"][None, :, None] + 1e-8)).mean()
                    acc["a_s"] = acc["a_s"] + a_s.mean(); acc["o"] = acc["o"] + o_nat_f.norm(dim=-1).mean()
                    rot_layer[li] = rot_layer[li] + cos_rot.mean(); ms_layer[li] = ms_layer[li] + ms_n.mean()
                    a_s_layer[li] = a_s_layer[li] + a_s.mean()
                    if fal_layer[li] is None:
                        Lc_ = int(value.shape[2]); D = Pl["D"]
                        vis_i = vis_d[vis_d < Lc_]
                        tmask = torch.ones(Lc_, dtype=torch.bool, device=value.device)
                        tmask[lat] = False; tmask[vis_i] = False; tmask[s] = False
                        toHD = lambda x: x.unsqueeze(2).expand(1, KVH, G, D).reshape(1, H, D)
                        mu_vis = toHD(value[:1, :, vis_i, :].float().mean(2))
                        mu_text = toHD(value[:1, :, tmask.nonzero(as_tuple=True)[0], :].float().mean(2))
                        d0 = dvh[:, :, 0, :]
                        fal_layer[li] = torch.nn.functional.cosine_similarity(d0, mu_vis - mu_text, dim=-1).mean()
                        falv_layer[li] = torch.nn.functional.cosine_similarity(d0, mu_vis, dim=-1).mean()
                acc["n"] += 1
                n_layer[li] += 1
            stats["calls"] += 1
            stats["rows"] += Ts
            return out, w
        return attn

    restore = None
    if fast2:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        local = ALL_ATTENTION_FUNCTIONS._local_mapping
        had, prev = impl_name in local, local.get(impl_name)
        native_fn = ALL_ATTENTION_FUNCTIONS[impl_name]
        ALL_ATTENTION_FUNCTIONS[impl_name] = make_attn2(native_fn)

        def restore():
            if had:
                local[impl_name] = prev
            else:
                local.pop(impl_name, None)
        handles = []
    else:
        handles = [layer.self_attn.register_forward_hook((make_post_fast if fast else make_post)(li), with_kwargs=True) for li, layer in enumerate(bb.lm.layers)]
''')

rep('''        for h in handles:
            h.remove()
        n = max(acc["n"], 1)''', '''        for h in handles:
            h.remove()
        if restore is not None:
            restore()
        n = max(acc["n"], 1)''')

rep('''            if fast:
                fal_layer[:] = [None if v is None else _g(v) for v in fal_layer]''', '''            if fast or fast2:
                fal_layer[:] = [None if v is None else _g(v) for v in fal_layer]''')

open(P, "w").write(src)
print("patched", hashlib.md5(src.encode()).hexdigest()[:8])
