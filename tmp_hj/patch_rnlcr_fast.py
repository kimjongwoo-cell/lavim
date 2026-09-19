"""Add VLMAS_RNLCR_FAST=1: same FULR read, computed without full-cache repeat_kv/fp32 copies, full alpha clones or per-stat GPU syncs."""
import hashlib, shutil, sys
P = "/home/users/whddn12316/wsi_latent_0915_decode_hj/memory/rnlcr.py"
src = open(P).read()
md5 = hashlib.md5(src.encode()).hexdigest()
assert md5.startswith("eebb1d73"), md5
shutil.copy(P, P + ".bak_0917_fast")

def rep(old, new, count=1):
    global src
    assert src.count(old) == count, (old[:80], src.count(old))
    src = src.replace(old, new)

rep('''  Stats       |m_s|, |r_R|, |r_perp|, cos(m_s, m~_s), |o'-o|/|o|, latent mass (native), a_s, C_s.
''', '''  Stats       |m_s|, |r_R|, |r_perp|, cos(m_s, m~_s), |o'-o|/|o|, latent mass (native), a_s, C_s.
  VLMAS_RNLCR_FAST=1 (fulr only): identical read, computed with the query folded into KV groups (no repeat_kv of the cache),
              logits cast per KV head, values gathered only at the sink/latent columns, alpha_s = exp(z_s - logsumexp z) and
              rho = softmax(z_lat) without the full alpha tensor, stats kept on GPU and synced once at the end of the case.
''')

rep('''    vis_d = vis_cols.to(bb.device) if fu else None
''', '''    vis_d = vis_cols.to(bb.device) if fu else None
    fast = fu and os.environ.get("VLMAS_RNLCR_FAST", "").strip() == "1"
    if fast:
        stats["fast"] = True
    idx_sl = torch.cat([torch.tensor([s], device=bb.device, dtype=lat.dtype), lat]) if fast else None

    def make_post_fast(li):
        def post(module, args, kwargs, output):
            hidden, pe, c = _call_parts(args, kwargs)
            if hidden is None or pe is None or c is None:
                return output
            attn_out = output[0] if isinstance(output, tuple) else output
            keys, values = c.layers[li].keys, c.layers[li].values
            Lc, T = int(keys.shape[2]), int(hidden.shape[1])
            sel = torch.arange(T, device=hidden.device) if cfg["rows"] == "all" else torch.tensor([T - 1], device=hidden.device)
            Ts = int(sel.numel())
            Hd, G = module.head_dim, module.num_key_value_groups
            KVH = int(keys.shape[1]); H = KVH * G
            cos, sin = pe
            h_sel = hidden.index_select(1, sel)
            q = module.q_norm(module.q_proj(h_sel).view(1, Ts, -1, Hd)).transpose(1, 2)
            q, _ = apply_rotary_pos_emb(q, torch.zeros_like(q), cos.index_select(1, sel), sin.index_select(1, sel))
            with torch.no_grad():
                qg = q.float().reshape(1, KVH, G, Ts, Hd)                                     # query folded into KV groups
                z = torch.matmul(qg, keys.float().unsqueeze(2).transpose(-2, -1)) * module.scaling   # [1,KVH,G,Ts,Lc]
                if T > 1:
                    past = Lc - T
                    col = torch.arange(Lc, device=keys.device)[None, None, None, None, :]
                    row = (past + sel)[None, None, None, :, None]
                    z = z.masked_fill(col > row, float("-inf"))
                lse = torch.logsumexp(z, dim=-1, keepdim=True)
                z_sl = z[..., idx_sl]                                                         # [1,KVH,G,Ts,1+m]
                w_sl = torch.exp(z_sl - lse)
                toH = lambda x: x.reshape(1, H, *x.shape[3:])
                a_s = toH(w_sl[..., 0])                                                       # [1,H,Ts]
                a_lat = toH(w_sl[..., 1:])                                                    # [1,H,Ts,m]
                rho = torch.softmax(z_sl[..., 1:], dim=-1)                                    # [1,KVH,G,Ts,m]
                V_sl = values[:, :, idx_sl, :].float()                                        # [1,KVH,1+m,D]
                V_lat_kv = V_sl[:, :, 1:, :]
                Vs_kv = vstar[li].to(values.device).float()                                   # [1,KVH,m,D]
                dvbar = toH(torch.matmul(rho, (Vs_kv - V_lat_kv).unsqueeze(2)))              # [1,H,Ts,D]
                D = int(values.shape[-1]); m_ = int(V_lat_kv.shape[2])
                V_lat = V_lat_kv.unsqueeze(2).expand(1, KVH, G, m_, D).reshape(1, H, m_, D)
                V_sH = V_sl[:, :, 0, :].unsqueeze(2).expand(1, KVH, G, D).reshape(1, H, D)
                flat = lambda x: x.transpose(1, 2).reshape(1, Ts, -1)
                wd = module.o_proj.weight.dtype
                a_col = a_s.unsqueeze(-1)
                sink_ctx = a_col * V_sH.unsqueeze(2)                                          # [1,H,Ts,D]
                m_s = per_head_oproj(module.o_proj, sink_ctx)
                r_R = per_head_oproj(module.o_proj, a_col * dvbar)
                m_t, r_perp = fulr_rotate(m_s, r_R)
                delta = (m_t - m_s).sum(1)
                o_nat_f = attn_out.index_select(1, sel).float()
                o_new = o_nat_f + delta
                ms_n = m_s.norm(dim=-1)
                cos_rot = torch.where(ms_n > 1e-6, torch.nn.functional.cosine_similarity(m_s, m_t, dim=-1), torch.ones_like(ms_n))
                acc["ms"] = acc["ms"] + ms_n.mean(); acc["rR"] = acc["rR"] + r_R.norm(dim=-1).mean(); acc["rperp"] = acc["rperp"] + r_perp.norm(dim=-1).mean()
                acc["cos_rot"] = acc["cos_rot"] + cos_rot.mean(); acc["delta_rel"] = acc["delta_rel"] + (delta.norm(dim=-1) / (o_nat_f.norm(dim=-1) + 1e-8)).mean()
                acc["dv_rel"] = acc["dv_rel"] + (dvbar.norm(dim=-1) / (V_lat.norm(dim=-1).mean(-1, keepdim=True) + 1e-8)).mean()
                rot_layer[li] = rot_layer[li] + cos_rot.mean(); ms_layer[li] = ms_layer[li] + ms_n.mean()
                if fal_layer[li] is None:
                    Lc_ = int(values.shape[2])
                    vis_i = vis_d[vis_d < Lc_]
                    tmask = torch.ones(Lc_, dtype=torch.bool, device=values.device)
                    tmask[lat] = False; tmask[vis_i] = False; tmask[s] = False
                    toHD = lambda x: x.unsqueeze(2).expand(1, KVH, G, D).reshape(1, H, D)
                    mu_vis = toHD(values[:, :, vis_i, :].float().mean(2))
                    mu_text = toHD(values[:, :, tmask.nonzero(as_tuple=True)[0], :].float().mean(2))
                    d0 = dvbar[:, :, 0, :]
                    fal_layer[li] = torch.nn.functional.cosine_similarity(d0, mu_vis - mu_text, dim=-1).mean()
                    falv_layer[li] = torch.nn.functional.cosine_similarity(d0, mu_vis, dim=-1).mean()
                # diagnostics kept equal to the reference path (fulr leaves the latent read native)
                lat_ctx_nat = torch.matmul(a_lat, V_lat)
                msg_nat = module.o_proj(flat(lat_ctx_nat).to(wd)).float().norm(dim=-1).mean()
                C_s = module.o_proj(flat(sink_ctx).to(wd)).float().norm(dim=-1).mean()
                lat_nat_v = a_lat.sum(-1).mean()
                lat_would_v = (a_lat.sum(-1) + a_s).mean()
                acc["a_s"] = acc["a_s"] + a_s.mean(); acc["C_s"] = acc["C_s"] + C_s; acc["o"] = acc["o"] + o_nat_f.norm(dim=-1).mean()
                acc["lat_nat"] = acc["lat_nat"] + lat_nat_v; acc["lat_would"] = acc["lat_would"] + lat_would_v; acc["lat_new"] = acc["lat_new"] + lat_nat_v
                acc["msg_nat"] = acc["msg_nat"] + msg_nat; acc["msg_would"] = acc["msg_would"] + msg_nat; acc["msg_new"] = acc["msg_new"] + msg_nat
                acc["n"] += 1
                a_s_layer[li] = a_s_layer[li] + a_s.mean(); n_layer[li] += 1
            stats["calls"] += 1
            stats["rows"] += Ts
            out = attn_out.clone()
            out[:, sel, :] = o_new.to(attn_out.dtype)
            return (out, *output[1:]) if isinstance(output, tuple) else out
        return post
''')

rep('''    handles = [layer.self_attn.register_forward_hook(make_post(li), with_kwargs=True) for li, layer in enumerate(bb.lm.layers)]
''', '''    handles = [layer.self_attn.register_forward_hook((make_post_fast if fast else make_post)(li), with_kwargs=True) for li, layer in enumerate(bb.lm.layers)]
''')

rep('''        if fu:
            stats["fal_by_layer"] = fal_layer
''', '''        if fu:
            if fast:
                fal_layer[:] = [None if v is None else _g(v) for v in fal_layer]
                falv_layer[:] = [None if v is None else _g(v) for v in falv_layer]
            stats["fal_by_layer"] = fal_layer
''')

open(P, "w").write(src)
print("patched", hashlib.md5(src.encode()).hexdigest()[:8])
