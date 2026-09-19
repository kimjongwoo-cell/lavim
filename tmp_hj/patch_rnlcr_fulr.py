"""0917 FULR (C2 #18, Notion 3ddd771b-c9e2-8159, canonical §0) as mode `fulr` of memory/rnlcr.py. md5-guarded.

Formation  = Step 1 grounding + replay (as spglr): V*_R kept as a side bank, native latent K and V restored in the cache.
Utilization= every layer / head, rows = gen (VLMAS_RNLCR_ROWS): K_s, q_A, alpha_s and the softmax denominator native;
             rho_k = softmax_k(q_A K_{R,k} / sqrt d) on the NATIVE latent keys,  dV_bar = sum_k rho_k (V*_k - V_k)
             m_s = W_O^h(alpha_s V_s),  r_R = W_O^h(alpha_s dV_bar)   (per head, o_proj has no bias)
             r_perp = r_R - <r_R,m_s>/(|m_s|^2+eps) m_s,  m~_s = |m_s| (m_s + r_perp)/(|m_s + r_perp|+eps)
             o' = o - sum_h m_s^h + sum_h m~_s^h      (|m~_s| = |m_s| per head: no gain, no threshold)
Falsifier  = per layer (first hooked row): cos(dV_bar, mu_vis - mu_text) and cos(dV_bar, mu_vis) over the cached values.
"""
import hashlib, sys
p = sys.argv[1] if len(sys.argv) > 1 else "memory/rnlcr.py"
src = open(p).read()
md5 = hashlib.md5(src.encode()).hexdigest()[:8]
assert md5 == "ed300d77", f"rnlcr md5 {md5} (expected ed300d77)"


def rep(old, new, count=1):
    global src
    assert src.count(old) == count, (old[:60], src.count(old))
    src = src.replace(old, new)


rep('''  Env VLMAS_RELR_STEPS=10  VLMAS_RELR_LR=0.05 (x rms Z)  VLMAS_RELR_LAMBDA=1.0  VLMAS_RELR_LAYERS=all|a-b
"""''', '''  Env VLMAS_RELR_STEPS=10  VLMAS_RELR_LR=0.05 (x rms Z)  VLMAS_RELR_LAMBDA=1.0  VLMAS_RELR_LAYERS=all|a-b

FULR (C2 #18, Notion 3ddd771b-c9e2-8159, canonical §0.1-0.3) -- mode fulr:
  Formation   Step 1 grounding + replay exactly as spglr: V*_R is kept as a side bank, the cache keeps the NATIVE
              latent K and V (so K_s, q_A, alpha_s and the softmax denominator stay native).
  Payload     rho_k = softmax_k(q_A K_{R,k}/sqrt d) over the native latent keys (stop-grad),
              dV_bar = sum_k rho_k (V*_{R,k} - V_{R,k})                       (grounding-induced latent residual)
  Read        per layer / head, rows = gen: m_s = W_O^h(alpha_s V_s), r_R = W_O^h(alpha_s dV_bar)   (o_proj has no bias)
              r_perp = r_R - <r_R,m_s>/(|m_s|^2+eps) m_s,  m~_s = |m_s| (m_s + r_perp) / (|m_s + r_perp| + eps)
              o' = o - sum_h m_s^h + sum_h m~_s^h   -> |m~_s| = |m_s| per head: no gain, no threshold, sink K untouched.
  Falsifier   per layer at the first hooked row: cos(dV_bar, mu_vis - mu_text), cos(dV_bar, mu_vis) (cached values, per head).
  Stats       |m_s|, |r_R|, |r_perp|, cos(m_s, m~_s), |o'-o|/|o|, latent mass (native), a_s, C_s.
"""''')

rep('''MODES = ("diag", "cap", "ground", "full", "spglr", "vonly", "nonorm", "relr")''',
    '''MODES = ("diag", "cap", "ground", "full", "spglr", "vonly", "nonorm", "relr", "fulr")''')

rep('''def grounds() -> bool:
    return mode() in ("ground", "full", "spglr", "vonly", "nonorm")''',
    '''def grounds() -> bool:
    return mode() in ("ground", "full", "spglr", "vonly", "nonorm", "fulr")''')

rep('''def sp_readout(alpha_lat, V_lat, Vstar_lat, norm_preserve: bool, eps: float = 1e-8):''',
    '''def fulr_mode() -> bool:
    return mode() == "fulr"


def fulr_payload(scores_lat, V_lat, Vstar_lat):
    """scores_lat [..., Ts, m] native latent logits, V_lat / Vstar_lat [..., m, D] -> (dV_bar [..., Ts, D], rho [..., Ts, m])."""
    rho = torch.softmax(scores_lat, dim=-1)
    return torch.matmul(rho, Vstar_lat - V_lat), rho


def fulr_rotate(m_s, r_R, eps: float = 1e-8):
    """Sink-norm-conserved direction update (page §0.3): returns (m~_s, r_perp), same shape as m_s [..., d]."""
    proj = (r_R * m_s).sum(-1, keepdim=True) / (m_s.pow(2).sum(-1, keepdim=True) + eps)
    r_perp = r_R - proj * m_s
    v = m_s + r_perp
    m_t = m_s.norm(dim=-1, keepdim=True) * v / (v.norm(dim=-1, keepdim=True) + eps)
    return m_t, r_perp


def per_head_oproj(o_proj, ctx):
    """ctx [1,H,Ts,D] -> per-head o_proj contributions [1,H,Ts,d_model] (fp32); requires a bias-free o_proj."""
    W = o_proj.weight
    H, D = int(ctx.shape[1]), int(ctx.shape[-1])
    return torch.einsum("bhtd,ehd->bhte", ctx.float(), W.view(W.shape[0], H, D).float())


def sp_readout(alpha_lat, V_lat, Vstar_lat, norm_preserve: bool, eps: float = 1e-8):''')

rep('''    if md in SP_MODES or md == "vonly":
        # SP-GLR: the replayed KEY is never used by the receiver (native latent addressing is kept).
        # spglr/nonorm: keep V*_R aside and restore native K and V; vonly: restore native K, keep V*_R in place.''',
    '''    if md in SP_MODES or md in ("vonly", "fulr"):
        # SP-GLR / FULR: the replayed KEY is never used by the receiver (native latent addressing is kept).
        # spglr/nonorm/fulr: keep V*_R aside and restore native K and V; vonly: restore native K, keep V*_R in place.''')

rep('''    stats = {"case": int(case_index), "mode": md, "applied": redistributes() or readout(), "calls": 0, "rows": 0}''',
    '''    fu = fulr_mode()
    stats = {"case": int(case_index), "mode": md, "applied": redistributes() or readout() or fu, "calls": 0, "rows": 0}''')

rep('''    vstar = R.get("vstar")
    if readout() and vstar is None:
        stats["skip"] = "SP-GLR readout needs the grounded value bank (grounding skipped?)"
        yield stats
        _write(cfg, stats, t0)
        return''',
    '''    vstar = R.get("vstar")
    if (readout() or fu) and vstar is None:
        stats["skip"] = "readout needs the grounded value bank (grounding skipped?)"
        yield stats
        _write(cfg, stats, t0)
        return
    vis_cols = R.get("vis_cols")
    if fu and (vis_cols is None or int(vis_cols.numel()) == 0):
        stats["skip"] = "FULR falsifier needs the visual columns"
        yield stats
        _write(cfg, stats, t0)
        return
    if fu and getattr(bb.lm.layers[0].self_attn.o_proj, "bias", None) is not None:
        stats["skip"] = "FULR per-head read needs a bias-free o_proj"
        yield stats
        _write(cfg, stats, t0)
        return''')

rep('''           "mR": 0.0, "gR": 0.0, "mt": 0.0, "cos_mg": 0.0, "n": 0}
    applied = redistributes()
    sp = readout()
    a_s_layer = [0.0] * n_layers
    n_layer = [0] * n_layers''',
    '''           "mR": 0.0, "gR": 0.0, "mt": 0.0, "cos_mg": 0.0, "n": 0,
           "ms": 0.0, "rR": 0.0, "rperp": 0.0, "cos_rot": 0.0, "delta_rel": 0.0, "dv_rel": 0.0}
    applied = redistributes()
    sp = readout()
    a_s_layer = [0.0] * n_layers
    n_layer = [0] * n_layers
    fal_layer = [None] * n_layers      # cos(dV_bar, mu_vis - mu_text) at the first hooked row
    falv_layer = [None] * n_layers     # cos(dV_bar, mu_vis)
    rot_layer = [0.0] * n_layers       # cos(m_s, m~_s) accumulated per layer
    ms_layer = [0.0] * n_layers        # |m_s| per head, accumulated per layer
    vis_d = vis_cols.to(bb.device) if fu else None''')

rep('''                new, a_s = redistribute(alpha, lat, s, scores)
                flat = lambda x: x.transpose(1, 2).reshape(1, Ts, -1)
                wd = module.o_proj.weight.dtype
                if sp:''',
    '''                new, a_s = redistribute(alpha, lat, s, scores)
                flat = lambda x: x.transpose(1, 2).reshape(1, Ts, -1)
                wd = module.o_proj.weight.dtype
                if fu:
                    Vs = repeat_kv(vstar[li].to(values.device), G).float()                   # [1,H,m,D] grounded values
                    V_lat = Vv[..., lat, :]
                    dvbar, _rho = fulr_payload(scores[..., lat], V_lat, Vs)                  # [1,H,Ts,D]
                    a_col = a_s.unsqueeze(-1)                                                # [1,H,Ts,1]
                    sink_ctx = a_col * Vv[..., s, :].unsqueeze(2)                            # [1,H,Ts,D]
                    m_s = per_head_oproj(module.o_proj, sink_ctx)                            # [1,H,Ts,d]
                    r_R = per_head_oproj(module.o_proj, a_col * dvbar)
                    m_t, r_perp = fulr_rotate(m_s, r_R)
                    delta = (m_t - m_s).sum(1)                                               # [1,Ts,d]
                    o_nat_f = attn_out.index_select(1, sel).float()
                    o_new = o_nat_f + delta
                    ms_n = m_s.norm(dim=-1)                                                  # [1,H,Ts]
                    cos_rot = torch.where(ms_n > 1e-6, torch.nn.functional.cosine_similarity(m_s, m_t, dim=-1), torch.ones_like(ms_n))
                    acc["ms"] += float(ms_n.mean()); acc["rR"] += float(r_R.norm(dim=-1).mean()); acc["rperp"] += float(r_perp.norm(dim=-1).mean())
                    acc["cos_rot"] += float(cos_rot.mean()); acc["delta_rel"] += float((delta.norm(dim=-1) / (o_nat_f.norm(dim=-1) + 1e-8)).mean())
                    acc["dv_rel"] += float((dvbar.norm(dim=-1) / (V_lat.norm(dim=-1).mean(-1, keepdim=True) + 1e-8)).mean())
                    rot_layer[li] += float(cos_rot.mean()); ms_layer[li] += float(ms_n.mean())
                    if fal_layer[li] is None:
                        Lc_ = int(Vv.shape[2])
                        tmask = torch.ones(Lc_, dtype=torch.bool, device=Vv.device)
                        tmask[lat] = False; tmask[vis_d[vis_d < Lc_]] = False; tmask[s] = False
                        mu_vis = Vv[..., vis_d[vis_d < Lc_], :].mean(2)                         # [1,H,D]
                        mu_text = Vv[..., tmask.nonzero(as_tuple=True)[0], :].mean(2)
                        d0 = dvbar[:, :, 0, :]
                        fal_layer[li] = _g(torch.nn.functional.cosine_similarity(d0, mu_vis - mu_text, dim=-1).mean())
                        falv_layer[li] = _g(torch.nn.functional.cosine_similarity(d0, mu_vis, dim=-1).mean())
                    ctx_new = None
                elif sp:''')

rep('''                else:
                    ctx_new = torch.matmul(new, Vv)                                          # [1,H,Ts,D]
                o_new = module.o_proj(flat(ctx_new).to(wd)).float()
                o_nat = attn_out.index_select(1, sel).float()
                lat_ctx_nat = torch.matmul(alpha[..., lat], Vv[..., lat, :])
                lat_ctx_new = m_t if sp else torch.matmul(new[..., lat], Vv[..., lat, :])''',
    '''                else:
                    ctx_new = torch.matmul(new, Vv)                                          # [1,H,Ts,D]
                if ctx_new is not None:
                    o_new = module.o_proj(flat(ctx_new).to(wd)).float()
                o_nat = attn_out.index_select(1, sel).float()
                lat_ctx_nat = torch.matmul(alpha[..., lat], Vv[..., lat, :])
                lat_ctx_new = lat_ctx_nat if fu else (m_t if sp else torch.matmul(new[..., lat], Vv[..., lat, :]))''')

rep('''            if not (applied or sp):
                return output
            out = attn_out.clone()''',
    '''            if not (applied or sp or fu):
                return output
            out = attn_out.clone()''')

rep('''        stats["a_s_by_layer"] = [_g(a / max(c, 1)) for a, c in zip(a_s_layer, n_layer)]
        stats["rows"] = stats["rows"] // max(n_layers, 1)''',
    '''        stats["a_s_by_layer"] = [_g(a / max(c, 1)) for a, c in zip(a_s_layer, n_layer)]
        if fu:
            stats["fal_by_layer"] = fal_layer
            stats["falv_by_layer"] = falv_layer
            stats["cos_rot_by_layer"] = [_g(a / max(c, 1)) for a, c in zip(rot_layer, n_layer)]
            stats["ms_by_layer"] = [_g(a / max(c, 1)) for a, c in zip(ms_layer, n_layer)]
            vals = [v for v in fal_layer if v is not None]
            stats["fal_mean"] = _g(sum(vals) / len(vals)) if vals else None
            valsv = [v for v in falv_layer if v is not None]
            stats["falv_mean"] = _g(sum(valsv) / len(valsv)) if valsv else None
        stats["rows"] = stats["rows"] // max(n_layers, 1)''')

rep('''    if stats["mode"] in SP_MODES:
        return (f"mode={stats['mode']} applied={stats['applied']} ground={stats.get('ground')} latent={stats.get('latent_cols')} "''',
    '''    if stats["mode"] == "fulr":
        fl = stats.get("fal_by_layer") or []
        rl = stats.get("cos_rot_by_layer") or []
        pick = lambda xs, i: (xs[i] if len(xs) > i and xs[i] is not None else float("nan"))
        return (f"mode=fulr applied={stats['applied']} ground={stats.get('ground')} latent={stats.get('latent_cols')} sink={stats.get('sink')} "
                f"a_s={stats['a_s']:.4f} C_s/|o|={stats['C_s']:.3g}/{stats['o']:.3g} |m_s|h={stats['ms']:.3g} |r_R|={stats['rR']:.3g} |r_perp|={stats['rperp']:.3g} "
                f"cos(m_s,m~)={stats['cos_rot']:.4f} (L27 {pick(rl, 27):.4f}, L35 {pick(rl, 35):.4f}) |o'-o|/|o|={stats['delta_rel']:.3g} dV_rel={stats['dv_rel']:.3g} "
                f"latent mass {stats['lat_nat']:.2e} | falsifier cos(dV,mu_v-mu_t) mean {stats.get('fal_mean')} (L27 {pick(fl, 27):.3f}, L35 {pick(fl, 35):.3f}) "
                f"cos(dV,mu_v) {stats.get('falv_mean')} rows={stats['rows']} sec={stats['sec']}")
    if stats["mode"] in SP_MODES:
        return (f"mode={stats['mode']} applied={stats['applied']} ground={stats.get('ground')} latent={stats.get('latent_cols')} "''')

open(p, "w").write(src)
print("patched", p, "->", hashlib.md5(src.encode()).hexdigest()[:8])
