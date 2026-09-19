"""Receiver-Conditioned current-slide Visual Re-read (Order 88 §12), kill-test realisation.

At every Answerer layer, for the rows being relayed (decision row + answer span):

    alpha      native attention of the row over the whole cache (recomputed; SDPA hides it)
    m_C        sum_{c in C} alpha_c V_c per head, C = every column except the sink and the
               current-slide visual bank V (question, inherited roles, Answerer prompt)
    r_C        o_proj(concat_h m_C)                          -- context contribution in residual space
    h_hat      h + r_C                                       -- h = the layer's pre-norm residual input
    q_hat      rope(q_norm(q_proj(input_layernorm(h_hat))))  -- pretrained projections only
    u_nat      softmax_V(q     K_V) V_V                      -- visual-only read, native query
    u_ctx      softmax_V(q_hat K_V) V_V                      -- visual-only read, context query
    o'         o + lambda * rho_V * (u_ctx - u_nat)          -- rho_V = native visual mass of the row

Controls (VLMAS_RCVR_MODE):
    ctx       main candidate (R2)
    identity  q_hat := q -> the correction is exactly 0 (R1 sanity arm)
    wrong     u_ctx is read from the PREVIOUS case's visual K/V with q_hat (R4 wrong slide)

Measurement (every layer, first relayed row): 1 - cos(q, q_hat), JSD of the within-visual
distributions, top-32 support overlap, |du|/|u_nat|, cos(u_nat, u_ctx), |dO|/|o| at lambda = 1.
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


_PREV_VIS = None          # (layer -> (K_V, V_V)) of the previous case, for the wrong-slide arm


class RCVR:
    def __init__(self, bb, vis: torch.Tensor, lam: float, n_prompt: int, gen_limit: int | None,
                 mode: str = "ctx", sink: int = 0, inject: str = "delta", base_len: int = 0):
        self.bb = bb
        # inject = "delta": o + lam*rho_V*(u_ctx - u_nat)            (Order 88 §12.4)
        # inject = "hc":    o + lam*(rho_B*u_fill - m_B), B = H_C     (GLVR receiver slot, §12.11)
        #   B = inherited columns before the Answerer prompt, minus the sink and the visual bank;
        #   u_fill = the re-read (identity -> u_nat, ctx -> u_ctx, wrong -> u_ctx on the previous slide)
        self.inject = inject
        self.base_len = int(base_len)
        self.vis = vis
        self.lam = float(lam)
        self.n_prompt = int(n_prompt)
        self.gen_limit = gen_limit
        self.gen_seen = 0
        self.mode = mode
        self.sink = int(sink)
        self.resid = {}                 # li -> pre-norm residual input of the current call
        self.stats = {}                 # li -> metrics at the first relayed row (decision row)
        self.num, self.den = [], []
        self.kv_vis = {}                # li -> (K_V, V_V) of THIS case (for the next case's wrong arm)
        self.note = None

    def _row_start(self, T):
        return max(0, self.n_prompt - 1) if (self.n_prompt and T >= self.n_prompt) else 0

    def layer_pre(self, li):
        def hook(module, args, kwargs):
            h = kwargs.get("hidden_states", args[0] if args else None)
            if h is not None:
                self.resid[li] = h
            return None
        return hook

    def attn_post(self, li):
        def post(module, args, kwargs, output):
            from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb
            hs = kwargs.get("hidden_states", args[0] if args else None)
            cache = kwargs.get("past_key_values")
            pe = kwargs.get("position_embeddings")
            h_res = self.resid.get(li)
            if hs is None or cache is None or pe is None or h_res is None:
                return output
            T = int(hs.shape[1])
            if self.gen_limit is not None and T == 1:
                if li == 0:
                    self.gen_seen += 1
                if self.gen_seen > self.gen_limit:
                    return output
            try:
                with torch.no_grad():
                    r0 = self._row_start(T)
                    rows = torch.arange(r0, T, device=hs.device)
                    R = int(rows.numel())
                    keys, vals = cache.layers[li].keys, cache.layers[li].values
                    K = (keys[0] if keys.dim() == 4 else keys).float()      # [Hkv, L, D]
                    V = (vals[0] if vals.dim() == 4 else vals).float()
                    L = int(K.shape[1])
                    Hd, G = module.head_dim, module.num_key_value_groups
                    cos, sin = pe
                    cs, sn = cos.index_select(1, rows), sin.index_select(1, rows)

                    def query(x_normed):
                        q = module.q_norm(module.q_proj(x_normed).view(1, R, -1, Hd)).transpose(1, 2)
                        q, _ = apply_rotary_pos_emb(q, torch.zeros_like(q), cs, sn)
                        return q[0].float()                                          # [H, R, D]

                    q = query(hs.index_select(1, rows))
                    H = int(q.shape[0]); Hkv = H // G
                    Kh = K.repeat_interleave(G, dim=0); Vh = V.repeat_interleave(G, dim=0)  # [H, L, D]
                    s = torch.einsum("hrd,hld->hrl", q, Kh) * module.scaling
                    past = L - T
                    col = torch.arange(L, device=s.device)[None, None, :]
                    s = s.masked_fill(col > (past + rows)[None, :, None], float("-inf"))
                    A = torch.softmax(s, dim=-1)                                       # [H, R, L]
                    vis = self.vis.to(s.device)
                    vis = vis[vis < L]
                    ctx_mask = torch.ones(L, dtype=torch.bool, device=s.device)
                    ctx_mask[vis] = False
                    ctx_mask[self.sink] = False
                    m_C = torch.einsum("hrl,hld->hrd", A * ctx_mask, Vh)              # [H, R, D]
                    out = output[0] if isinstance(output, tuple) else output
                    r_C = module.o_proj(m_C.permute(1, 0, 2).reshape(1, R, -1).to(out.dtype))  # [1, R, d]
                    layer = self.bb.lm.layers[li]
                    h_hat = h_res.index_select(1, rows) + r_C
                    q_hat = q if self.mode == "identity" else query(layer.input_layernorm(h_hat))
                    Kv, Vv = Kh[:, vis], Vh[:, vis]                                   # [H, Nv, D]
                    if li not in self.kv_vis:
                        self.kv_vis[li] = (K[:, vis].cpu(), V[:, vis].cpu())
                    p_nat = torch.softmax(torch.einsum("hrd,hnd->hrn", q, Kv) * module.scaling, -1)
                    u_nat = torch.einsum("hrn,hnd->hrd", p_nat, Vv)
                    if self.mode == "wrong" and _PREV_VIS is not None and li in _PREV_VIS:
                        Kw, Vw = _PREV_VIS[li]
                        Kw = Kw.to(s.device).repeat_interleave(G, 0); Vw = Vw.to(s.device).repeat_interleave(G, 0)
                        p_ctx = torch.softmax(torch.einsum("hrd,hnd->hrn", q_hat, Kw) * module.scaling, -1)
                        u_ctx = torch.einsum("hrn,hnd->hrd", p_ctx, Vw)
                    else:
                        p_ctx = torch.softmax(torch.einsum("hrd,hnd->hrn", q_hat, Kv) * module.scaling, -1)
                        u_ctx = torch.einsum("hrn,hnd->hrd", p_ctx, Vv)
                    rho_V = A[..., vis].sum(-1, keepdim=True)                          # [H, R, 1]
                    du = u_ctx - u_nat
                    if self.inject == "hc":
                        dmask = ctx_mask & (torch.arange(L, device=s.device) < self.base_len)
                        Ab = A * dmask
                        rho_B = Ab.sum(-1, keepdim=True)
                        m_B = torch.einsum("hrl,hld->hrd", Ab, Vh)
                        corr = rho_B * u_ctx - m_B
                    else:
                        corr = rho_V * du
                    add1 = module.o_proj(corr.permute(1, 0, 2).reshape(1, R, -1).to(out.dtype))
                    if li not in self.stats:                                           # decision row
                        c = torch.nn.functional.cosine_similarity
                        pn, pc = p_nat[:, 0], p_ctx[:, 0]
                        m = 0.5 * (pn + pc)
                        jsd = 0.5 * ((pn * (pn.clamp_min(1e-12) / m.clamp_min(1e-12)).log()).sum(-1)
                                     + (pc * (pc.clamp_min(1e-12) / m.clamp_min(1e-12)).log()).sum(-1))
                        k = min(32, int(pn.shape[-1]))
                        tn, tc = pn.topk(k, -1).indices, pc.topk(k, -1).indices
                        ov = sum(len(set(tn[i].tolist()) & set(tc[i].tolist())) / len(set(tn[i].tolist()) | set(tc[i].tolist()))
                                 for i in range(H)) / H
                        self.stats[li] = {
                            "q_change": float((1 - c(q[:, 0], q_hat[:, 0], dim=-1)).mean()),
                            "jsd": float(jsd.mean()), "top32_overlap": round(ov, 4),
                            "du_rel": float((du[:, 0].norm(dim=-1) / u_nat[:, 0].norm(dim=-1).clamp_min(1e-9)).mean()),
                            "u_cos": float(c(u_nat[:, 0], u_ctx[:, 0], dim=-1).mean()),
                            "rho_V": float(rho_V[:, 0].mean()),
                            "rho_C": float((A[:, 0] * ctx_mask).sum(-1).mean()),
                            "dO_rel_lam1": float(add1[0, 0].float().norm() / out[0, r0].float().norm().clamp_min(1e-9)),
                        }
                    if self.lam:
                        add = add1 * self.lam
                        self.num.append(add.float().norm()); self.den.append(out[:, r0:].float().norm())
                        out[:, r0:, :] = out[:, r0:, :] + add
                        output = (out,) + tuple(output[1:]) if isinstance(output, tuple) else out
            except Exception as exc:  # noqa: BLE001 - never break the case
                self.note = f"{type(exc).__name__}: {exc}"
            return output
        return post


@contextlib.contextmanager
def install(bb, vis, lam, n_prompt, gen_limit, mode, inject="delta", base_len=0):
    st = RCVR(bb, vis, lam, n_prompt, gen_limit, mode, inject=inject, base_len=base_len)
    hs = []
    for li, layer in enumerate(bb.lm.layers):
        hs.append(layer.register_forward_pre_hook(st.layer_pre(li), with_kwargs=True))
        hs.append(layer.self_attn.register_forward_hook(st.attn_post(li), with_kwargs=True))
    try:
        yield st
    finally:
        for h in hs:
            h.remove()


@contextlib.contextmanager
def terminal(engine, cache, *, system_prompt, user_prompt, json_prefix, case_index=-1):
    """Wrap the Answerer call. VLMAS_RCVR=<jsonl>, VLMAS_RCVR_LAMBDA, VLMAS_RCVR_MODE."""
    global _PREV_VIS
    t0 = time.time()
    rec = {"case": int(case_index), "mode": _env("VLMAS_RCVR_MODE", "ctx")}
    lam = float(_env("VLMAS_RCVR_LAMBDA", "0") or 0)
    rec["lambda"] = lam
    ctx, st = contextlib.nullcontext(), None
    try:
        from vision_text_mas.latent_terminal import _assistant_prompt
        bb = engine._backbone
        gl = getattr(bb, "_glvr", None) or {}
        vis = gl.get("vis")
        if vis is None or not vis.numel():
            rec["skip"] = "no visual columns recorded"
        else:
            tok = bb.processor.tokenizer
            prompt = _assistant_prompt(bb.processor, system_prompt=system_prompt, user_prompt=user_prompt,
                                       json_prefix=json_prefix)
            n_prompt = int(tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1])
            cands = [str(c).strip() for c in (getattr(engine, "_ver_candidates", None) or ()) if str(c).strip()]
            gen_limit = (max(len(tok(c, add_special_tokens=False)["input_ids"]) for c in cands) + 2) if cands else 16
            inject = _env("VLMAS_RCVR_INJECT", "delta")
            base_len = int(bb._kv_len(cache))
            rec.update({"n_vis": int(vis.numel()), "n_prompt": n_prompt, "gen_limit": gen_limit,
                        "wrong_ready": _PREV_VIS is not None, "inject": inject, "base_len": base_len})
            ctx = install(bb, vis, lam, n_prompt, gen_limit, rec["mode"], inject=inject, base_len=base_len)
    except Exception as exc:  # noqa: BLE001
        rec["skip"] = f"setup {type(exc).__name__}: {exc}"
    with ctx as st:
        yield rec
    if st is not None:
        layers = sorted(st.stats)
        rec["per_layer"] = {str(li): {k: round(v, 5) for k, v in st.stats[li].items()} for li in layers}
        if layers:
            rec["mean"] = {k: round(sum(st.stats[li][k] for li in layers) / len(layers), 5) for k in st.stats[layers[0]]}
        rec["dO_rel_applied"] = (round(float((torch.stack(st.num) / torch.stack(st.den).clamp_min(1e-9)).mean()), 5)
                                 if st.num else None)
        rec["note"] = st.note
        _PREV_VIS = st.kv_vis or _PREV_VIS
    rec["sec"] = round(time.time() - t0, 2)
    path = _env("VLMAS_RCVR")
    if path:
        with open(path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    m = rec.get("mean", {})
    print(f"[RCVR] case {case_index} mode={rec['mode']} lam={lam} q_change={m.get('q_change')} jsd={m.get('jsd')} "
          f"du_rel={m.get('du_rel')} dO@1={m.get('dO_rel_lam1')} applied={rec.get('dO_rel_applied')} "
          f"skip={rec.get('skip')} note={rec.get('note')}", flush=True)
