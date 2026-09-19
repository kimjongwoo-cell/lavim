"""GLVR Final -- Latent-Consensus Visual Re-Read (LCVR), Notion C2 #28.

Terminal Answerer-prompt row only, every decoder layer, per query head h:

    g_k      = alpha(b -> R*_k) / sum_r alpha(b -> R*_r)          Answerer-native latent selector
    q_con    = sum_k g_k q*_{R,k}                                  grounded Reasoner queries (replay, post-rope)
    u_con    = softmax_V(q_con K_V^T / sqrt d) V_V                 one re-read of the persistent visual KV
    B        = H_C minus the grounded latent columns (inherited non-visual history before the Answerer prompt,
               without sink / visual / latent)
    o'_b     = o_b + lambda * (rho_B u_con - m_B)

Generated rows are left native. Measured alongside (never applied): the old post-read message
u_post = sum_k g_k u_{V,k} (Order 88 A arm), cos(u_con, u_post), query diversity of Q*_R, and
cos(q_con, q_A) for the native Answerer query.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import time

import torch


def _env(k, d=""):
    v = os.environ.get(k, "")
    return v if v.strip() else d


def _set_stats(X: torch.Tensor) -> dict:
    """X [m, D] -> pairwise cosine mean, effective rank (entropy of normalised singular values),
    top principal energy share."""
    m = int(X.shape[0])
    xn = torch.nn.functional.normalize(X, dim=-1)
    pc = float(((xn @ xn.T).sum() - m) / (m * (m - 1)))
    sv = torch.linalg.svdvals(X)
    p = (sv / sv.sum().clamp_min(1e-12)).clamp_min(1e-12)
    er = float(torch.exp(-(p * p.log()).sum()))
    e = sv.pow(2)
    return {"pair_cos": pc, "eff_rank": er, "top_energy": float(e[0] / e.sum().clamp_min(1e-12))}


def repr_stats(Qr, Kv, Vv, g, P_A, scaling) -> dict:
    """Order 88 §13 representation diagnostic at one layer (per head, then head-mean vectors).

    raw   u_k      = P_k V                      (P_k from the grounded Reasoner query q_k)
    A     d_k      = (P_k - P_A) V              native-referenced differential read
    B0    u_k_perp = u_k - c_hat c_hat^T u_k    c_hat = unit mean of the visual values (rank-1)
    """
    P = torch.softmax(torch.einsum("hmd,hnd->hmn", Qr, Kv) * scaling, -1)       # [H, m, Nv]
    U = torch.einsum("hmn,hnd->hmd", P, Vv)                                       # [H, m, D]
    Dd = torch.einsum("hmn,hnd->hmd", P - P_A[:, None, :], Vv)
    c = Vv.mean(1)                                                               # [H, D]
    ch = torch.nn.functional.normalize(c, dim=-1)
    Up = U - ch[:, None, :] * (U * ch[:, None, :]).sum(-1, keepdim=True)
    out = {}
    for name, X in (("raw", U), ("A", Dd), ("B0", Up)):
        st = _set_stats(X.mean(0))                                                # head-mean vectors
        per_head = [_set_stats(X[h]) for h in range(X.shape[0])]
        out.update({f"{name}_{k}": v for k, v in st.items()})
        out[f"{name}_pair_cos_perhead"] = sum(d["pair_cos"] for d in per_head) / len(per_head)
        out[f"{name}_eff_rank_perhead"] = sum(d["eff_rank"] for d in per_head) / len(per_head)
    uR = torch.einsum("hm,hmd->hd", g, U)
    dR = torch.einsum("hm,hmd->hd", g, Dd)
    out["A_dR_over_uR"] = float((dR.norm(dim=-1) / uR.norm(dim=-1).clamp_min(1e-9)).mean())
    out["B0_norm_kept"] = float((Up.norm(dim=-1) / U.norm(dim=-1).clamp_min(1e-9)).mean())
    out["B0_energy_removed"] = float(1 - (Up.pow(2).sum(-1) / U.pow(2).sum(-1).clamp_min(1e-12)).mean())
    return out


class LCVR:
    def __init__(self, bb, vis, lat, Q, U, lam, n_prompt, base_len, sink=0):
        self.bb, self.lam, self.n_prompt, self.base_len, self.sink = bb, float(lam), int(n_prompt), int(base_len), int(sink)
        self.vis, self.lat, self.Q, self.U = vis, lat, Q, U
        self.stats, self.done, self.note = {}, set(), None
        self.num, self.den = [], []

    def attn_post(self, li):
        def post(module, args, kwargs, output):
            from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb
            hs = kwargs.get("hidden_states", args[0] if args else None)
            cache, pe = kwargs.get("past_key_values"), kwargs.get("position_embeddings")
            if hs is None or cache is None or pe is None or li in self.done or li >= len(self.Q):
                return output
            T = int(hs.shape[1])
            if not (self.n_prompt and T >= self.n_prompt):        # only the Answerer prompt prefill call
                return output
            self.done.add(li)
            try:
                with torch.no_grad():
                    b = self.n_prompt - 1                           # terminal Answerer-prompt row
                    keys, vals = cache.layers[li].keys, cache.layers[li].values
                    K = (keys[0] if keys.dim() == 4 else keys).float(); V = (vals[0] if vals.dim() == 4 else vals).float()
                    L = int(K.shape[1]); Hd, G = module.head_dim, module.num_key_value_groups
                    cos, sin = pe
                    row = torch.tensor([b], device=hs.device)
                    q = module.q_norm(module.q_proj(hs.index_select(1, row)).view(1, 1, -1, Hd)).transpose(1, 2)
                    q, _ = apply_rotary_pos_emb(q, torch.zeros_like(q), cos.index_select(1, row), sin.index_select(1, row))
                    qA = q[0, :, 0].float()                                                   # [H, D]
                    H = int(qA.shape[0])
                    Kh, Vh = K.repeat_interleave(G, 0), V.repeat_interleave(G, 0)             # [H, L, D]
                    past = L - T
                    sc = torch.einsum("hd,hld->hl", qA, Kh) * module.scaling
                    sc[:, past + b + 1:] = float("-inf")
                    A = torch.softmax(sc, -1)                                                 # [H, L]
                    vis = self.vis.to(sc.device); lat = self.lat.to(sc.device)
                    al = A[:, lat]
                    g = al / al.sum(-1, keepdim=True).clamp_min(1e-12)                        # [H, m]
                    Qr = self.Q[li].to(sc.device).float()                                     # [H, m, D]
                    q_con = torch.einsum("hm,hmd->hd", g, Qr)
                    Kv, Vv = Kh[:, vis], Vh[:, vis]
                    P_con = torch.softmax(torch.einsum("hd,hnd->hn", q_con, Kv) * module.scaling, -1)
                    u_con = torch.einsum("hn,hnd->hd", P_con, Vv)
                    dmask = torch.zeros(L, dtype=torch.bool, device=sc.device)
                    dmask[:self.base_len] = True
                    dmask[vis] = False; dmask[lat] = False; dmask[self.sink] = False
                    Ab = A * dmask
                    rho_B = Ab.sum(-1, keepdim=True)
                    m_B = torch.einsum("hl,hld->hd", Ab, Vh)
                    corr = rho_B * u_con - m_B                                                 # [H, D]
                    out = output[0] if isinstance(output, tuple) else output
                    add = module.o_proj(corr.reshape(1, 1, -1).to(out.dtype))[0, 0]
                    cs = torch.nn.functional.cosine_similarity
                    st = {"rho_B": float(rho_B.mean()), "rho_lat": float(al.sum(-1).mean()),
                          "g_entropy": float((-(g.clamp_min(1e-12).log() * g).sum(-1)).mean()),
                          "cos_qcon_qA": float(cs(q_con, qA, dim=-1).mean()),
                          "dO_rel_lam1": float(add.float().norm() / out[0, b].float().norm().clamp_min(1e-9))}
                    qn = torch.nn.functional.normalize(Qr, dim=-1)
                    m = int(Qr.shape[1])
                    pair = (qn @ qn.transpose(1, 2)).sum((1, 2)) - m
                    st["q_pair_cos"] = float((pair / (m * (m - 1))).mean())
                    sv = torch.linalg.svdvals(Qr.mean(0))
                    p = (sv / sv.sum().clamp_min(1e-12)).clamp_min(1e-12)
                    st["q_eff_rank"] = float(torch.exp(-(p * p.log()).sum()))
                    u_post = None
                    if self.U is not None and li < len(self.U) and self.U[li] is not None:
                        Uk = self.U[li].to(sc.device).float()
                        if Uk.shape[0] != H:
                            Uk = Uk.repeat_interleave(H // Uk.shape[0], 0)
                        u_post = torch.einsum("hm,hmd->hd", g, Uk)
                        st["cos_upre_upost"] = float(cs(u_con, u_post, dim=-1).mean())
                        st["upre_upost_rel"] = float(((u_con - u_post).norm(dim=-1) / u_post.norm(dim=-1).clamp_min(1e-9)).mean())
                    P_A = torch.softmax(torch.einsum("hd,hnd->hn", qA, Kv) * module.scaling, -1)
                    mm = 0.5 * (P_con + P_A)
                    st["jsd_con_vs_answerer"] = float((0.5 * ((P_con * (P_con.clamp_min(1e-12) / mm.clamp_min(1e-12)).log()).sum(-1)
                                                         + (P_A * (P_A.clamp_min(1e-12) / mm.clamp_min(1e-12)).log()).sum(-1))).mean())
                    if _env("VLMAS_LCVR_REPR", "1") == "1":
                        rs = repr_stats(Qr, Kv, Vv, g, P_A, module.scaling)
                        c_hat = torch.nn.functional.normalize(Vv.mean(1), dim=-1)
                        if "cos_upre_upost" in st:
                            up = u_con - c_hat * (u_con * c_hat).sum(-1, keepdim=True)
                            uq = u_post - c_hat * (u_post * c_hat).sum(-1, keepdim=True)
                            rs["B0_cos_upre_upost"] = float(cs(up, uq, dim=-1).mean())
                        st.update(rs)
                    self.stats[li] = st
                    if self.lam:
                        a = add * self.lam
                        self.num.append(a.float().norm()); self.den.append(out[0, b].float().norm())
                        out[0, b] = out[0, b] + a
                        output = (out,) + tuple(output[1:]) if isinstance(output, tuple) else out
            except Exception as exc:  # noqa: BLE001
                self.note = f"{type(exc).__name__}: {exc}"
            return output
        return post


@contextlib.contextmanager
def terminal(engine, cache, *, system_prompt, user_prompt, json_prefix, case_index=-1):
    """VLMAS_LCVR=<jsonl>, VLMAS_LCVR_LAMBDA (default 0.75)."""
    t0 = time.time()
    lam = float(_env("VLMAS_LCVR_LAMBDA", "0.75"))
    rec = {"case": int(case_index), "lambda": lam}
    st, handles = None, []
    try:
        from vision_text_mas.latent_terminal import _assistant_prompt
        bb = engine._backbone
        gl = getattr(bb, "_glvr", None) or {}
        if gl.get("Q") is None or gl.get("vis") is None:
            rec["skip"] = f"no grounded queries ({gl.get('note')})"
        else:
            tok = bb.processor.tokenizer
            prompt = _assistant_prompt(bb.processor, system_prompt=system_prompt, user_prompt=user_prompt, json_prefix=json_prefix)
            n_prompt = int(tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1])
            base_len = int(bb._kv_len(cache))
            st = LCVR(bb, gl["vis"], gl["lat"], gl["Q"], gl.get("U"), lam, n_prompt, base_len)
            handles = [layer.self_attn.register_forward_hook(st.attn_post(li), with_kwargs=True)
                       for li, layer in enumerate(bb.lm.layers)]
            rec.update({"n_prompt": n_prompt, "base_len": base_len, "n_vis": int(gl["vis"].numel()),
                        "n_lat": int(gl["lat"].numel())})
    except Exception as exc:  # noqa: BLE001
        rec["skip"] = f"setup {type(exc).__name__}: {exc}"
    try:
        yield rec
    finally:
        for h in handles:
            h.remove()
    if st is not None:
        ls = sorted(st.stats)
        rec["layers_applied"] = len(ls)
        if ls:
            keys = st.stats[ls[0]].keys()
            rec["mean"] = {k: round(sum(st.stats[l][k] for l in ls if k in st.stats[l]) / max(1, sum(1 for l in ls if k in st.stats[l])), 5) for k in keys}
            rec["per_layer"] = {str(l): {k: round(v, 5) for k, v in st.stats[l].items()} for l in ls if l in (4, 12, 18, 27, 35)}
        rec["dO_rel_applied"] = round(float((torch.stack(st.num) / torch.stack(st.den).clamp_min(1e-9)).mean()), 5) if st.num else None
        rec["note"] = st.note
    rec["sec"] = round(time.time() - t0, 2)
    path = _env("VLMAS_LCVR")
    if path:
        with open(path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    m = rec.get("mean", {})
    print(f"[LCVR] case {case_index} lam={lam} layers={rec.get('layers_applied')} dO={rec.get('dO_rel_applied')} "
          f"cos(u_pre,u_post)={m.get('cos_upre_upost')} q_pair_cos={m.get('q_pair_cos')} cos(q_con,q_A)={m.get('cos_qcon_qA')} "
          f"skip={rec.get('skip')} note={rec.get('note')}", flush=True)
