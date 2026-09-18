"""Receiver-Native Latent Capacity Redistribution (RN-LCR), C2 #15 -- Notion 3ddd771b-c9e2-81b7 (Candidate).

Opt-in.  VLMAS_RNLCR unset = no hook, no cache change, byte-identical pipeline.

Step 1 -- question-guided visual grounding of the FULL latent trajectory (page 3)
  r_j  = mean over the original question rows and heads (and all layers) of the native attention from the
         question rows of the Reasoner prompt to visual column j (same capture as memory/lu.py).
  P_Q  = mean of the projected visual input embeddings of the top-K visual states by r_j,
  N_Q  = mean of the bottom-K.                                        (K = VLMAS_RNLCR_TOPK, choice)
  L_ground = -(1/m) sum_k cos(z_k, P_Q) + (1/m) sum_k cos(z_k, N_Q)   optimised over the m native latent
  INPUT embeddings z_k (Adam, VLMAS_RNLCR_S1_STEPS, lr = VLMAS_RNLCR_S1_LR x rms(Z)), backbone frozen.
Step 2 -- latent KV realisation (page 4): crop the cache to the pre-latent length and replay the m
  grounded rows once at their original positions (same as memory/lu.py); no new column.
Carrier (page 5): the receiver-native null carrier s is FIXED per configuration -- the sequence-initial
  cache column (VLMAS_RNLCR_SINK=0; the page's audit found it there); its addressability a_s and its
  value-mediated contribution C_s = |o_proj(alpha_s V_s)| are measured, not searched, per case.
Step 3 -- receiver-native capacity redistribution (page 6), every decoder layer, every head, rows =
  last prompt row + generated rows (VLMAS_RNLCR_ROWS=gen; all = every Answerer row):
  alpha   = native softmax over ALL cached columns (fp32 recompute from q/K)
  a_s     = alpha_s ;  rho_k = softmax over the m latent columns only of q K*_k / sqrt(d)
  alpha'_{R,k} = alpha_{R,k} + a_s rho_k ;  alpha'_s = 0 ;  every other column unchanged
  o' = sum_j alpha'_j V_j (native values) -> o_proj  replaces the attention output of those rows.
  sum_k delta_k = a_s exactly, so the receiver's total read capacity is preserved.

Modes  VLMAS_RNLCR=diag|cap|ground|full
  diag    measure only (a_s, C_s, latent mass, latent message norm)
  cap     Step 3 on the NATIVE latent trajectory          (page 10 "Capacity -> Native Latent")
  ground  Step 1 + 2 only (grounded latent, native reads)
  full    Step 1 + 2 + 3                                   (page 10 "Capacity -> Grounded Latent")
Env  VLMAS_RNLCR_TOPK=32  VLMAS_RNLCR_S1_STEPS=5  VLMAS_RNLCR_S1_LR=0.05  VLMAS_RNLCR_SINK=0
     VLMAS_RNLCR_ROWS=gen|all  VLMAS_RNLCR_LOG=<jsonl>
Measurements (page 9): per case, mean over layers/rows: a_s, C_s vs |o|, latent mass native -> new,
latent value message |sum_k alpha_k W_O V_k| native -> new (o_proj of the latent-only context).

SP-GLR (C2 #16, Notion 3ddd771b-c9e2-810e) -- same hook points, modes spglr | vonly | nonorm:
  Step 1 grounding as above; Step 2 replay is used ONLY to obtain the grounded values V*_R: after the
  replay the native latent K and V are restored in the cache and V*_R is kept as a side bank
  (spglr / nonorm), or only the native K is restored and V*_R stays in the cache (vonly, native readout).
  Step 3 readout (spglr), every layer / head, rows = gen: alpha native (unchanged, sink untouched),
      m_R = sum_k alpha_{R,k} V_{R,k}          (native latent-group message)
      rho_k = alpha_{R,k} / (sum_u alpha_{R,u} + eps),  g_R = sum_k rho_k V*_{R,k}
      m~_R = |m_R| g_R / (|g_R| + eps)          (direction replaced, magnitude preserved)
      o' = o - m_R + m~_R  per head -> o_proj.   nonorm: m~_R = g_R (mechanism control).
  Measurements: |m_R|, |g_R|, cos(m_R, g_R), latent mass (native), a_s / C_s for reference.

RELR (C2 #17, Notion 3ddd771b-c9e2-81ef) -- mode relr, everything at the terminal (needs the Answerer prompt):
  teacher   one native prefill of the Answerer prompt (+ json prefix) on the cache; at the boundary row of
            every layer: q_A (post-RoPE, stop-grad), m_V = o_proj(sum_{j in V} alpha_j V_j) with alpha from the
            FULL native softmax, lse_other = logsumexp of the non-latent logits, native m_R.
  variable  Z' = the m native latent INPUT embeddings; each step replays Z' through the frozen model on the
            cache prefix [:L0] (differentiable, params untouched) -> K'_R, V'_R per layer;
            alpha'_{R,k} = exp(q_A K'_k / sqrt d) / (exp(lse_other) + sum_u exp(q_A K'_u / sqrt d))  (full-cache
            normalisation with the other columns fixed), m_R(Z') = o_proj(sum_k alpha'_k V'_k);
  loss      sum_l |m_R^l(Z') - sg[m_V^l]|^2 / (|m_V^l|^2 + eps) + lambda |Z'-Z|_F^2 / |Z|_F^2    (Adam, N steps)
  commit    crop the cache to L0, replay Z* once, re-append the turn-close tokens; native Answerer follows.
  gate      |m_R*| / |m_R|, cos(m_R*, m_V) vs cos(m_R, m_V), attention-output norm, EMPTY / repetition (launcher).
  Env VLMAS_RELR_STEPS=10  VLMAS_RELR_LR=0.05 (x rms Z)  VLMAS_RELR_LAMBDA=1.0  VLMAS_RELR_LAYERS=all|a-b

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
  VLMAS_RNLCR_FAST=1 (fulr only): identical read, computed with the query folded into KV groups (no repeat_kv of the cache),
              logits cast per KV head, values gathered only at the sink/latent columns, alpha_s = exp(z_s - logsumexp z) and
              rho = softmax(z_lat) without the full alpha tensor, stats kept on GPU and synced once at the end of the case.
  VLMAS_RNLCR_FAST=2 (fulr only): hook-free fused sparse read. The model's attention function is wrapped for the Answerer call;
              after the native attention, per layer/head (Gram terms precomputed once per case from the sink / latent columns):
              S=W_O^h V_s, B_k=W_O^h dV_k; <S,S>, <S,B_k>, <B_k,B_j>; per token alpha_s=exp(z_s-LSE) (fp32 key mirror),
              rho=softmax(z_lat), p=rho.<S,B>, |r|^2=rho'<B,B>rho, c=p/|S|^2, |v|^2=(1-c)^2|S|^2+2(1-c)p+|r|^2,
              lambda=|S|/|v|, and the pre-o_proj head update alpha_s*(lambda*((1-c)V_s+sum_k rho_k dV_k)-V_s) is added to
              the attention output, which the native o_proj maps to exactly sum_h(m~_s-m_s). VLMAS_RNLCR_STATS=lite|none.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import time

import torch


def _relay4_gate_rho(rho: torch.Tensor, backbone: object, layer_index: int) -> torch.Tensor:
    """Restrict FULR redistribution to the Relay4 K-K winning head/rows."""
    masks = getattr(backbone, "_relay4_visual_mask", None)
    mask = masks.get(int(layer_index)) if isinstance(masks, dict) else masks
    if not isinstance(mask, torch.Tensor) or mask.ndim != 2 or mask.shape[-1] != rho.shape[-1]:
        return rho
    mask = mask.to(device=rho.device, dtype=rho.dtype)
    if rho.ndim == 4:
        # Eager FULR has 32 query heads. Relay4's physical-KV-head mask
        # applies identically to every query head in the matching GQA group.
        if rho.shape[1] % mask.shape[0] != 0:
            return rho
        groups = rho.shape[1] // mask.shape[0]
        mask = mask.repeat_interleave(groups, dim=0).reshape(
            1, rho.shape[1], 1, rho.shape[-1]
        )
    elif rho.ndim == 5:
        # FAST FULR folds the query heads into [KV-head, GQA-group].
        kv_heads, groups = rho.shape[1], rho.shape[2]
        if mask.shape[0] == kv_heads:
            mask = mask.reshape(1, kv_heads, 1, 1, rho.shape[-1])
        else:
            return rho
    else:
        return rho
    gated = rho * mask
    denom = gated.sum(dim=-1, keepdim=True)
    return torch.where(denom > 1e-12, gated / denom.clamp_min(1e-12), torch.zeros_like(gated))

from memory.lu import ReasonerCapture, _call_parts, _kvlen, cosine_sim

MODES = ("diag", "cap", "ground", "full", "spglr", "vonly", "nonorm", "relr", "fulr")
SP_MODES = ("spglr", "nonorm")


def mode() -> str:
    v = os.environ.get("VLMAS_RNLCR", "").strip().lower()
    return v if v in MODES else ""


def enabled() -> bool:
    return mode() != ""


def config() -> dict:
    e = os.environ.get
    return {"mode": mode(), "topk": int(e("VLMAS_RNLCR_TOPK", "32")),
            "s1_steps": int(e("VLMAS_RNLCR_S1_STEPS", "5")), "s1_lr": float(e("VLMAS_RNLCR_S1_LR", "0.05")),
            "sink": int(e("VLMAS_RNLCR_SINK", "0")), "rows": (e("VLMAS_RNLCR_ROWS", "gen").strip() or "gen"),
            "relr_steps": int(e("VLMAS_RELR_STEPS", "10")), "relr_lr": float(e("VLMAS_RELR_LR", "0.05")),
            "relr_lambda": float(e("VLMAS_RELR_LAMBDA", "1.0")), "relr_layers": (e("VLMAS_RELR_LAYERS", "all").strip() or "all"),
            "log": e("VLMAS_RNLCR_LOG", "").strip()}


def grounds() -> bool:
    return mode() in ("ground", "full", "spglr", "vonly", "nonorm", "fulr")


def redistributes() -> bool:
    return mode() in ("cap", "full")


def readout() -> bool:
    return mode() in SP_MODES


def fulr_mode() -> bool:
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


def sp_readout(alpha_lat, V_lat, Vstar_lat, norm_preserve: bool, eps: float = 1e-8):
    """alpha_lat [..., Ts, m], V_lat / Vstar_lat [..., m, D] -> (m_R, g_R, m~_R) each [..., Ts, D]."""
    m_R = torch.matmul(alpha_lat, V_lat)                       # [..., Ts, m] @ [..., m, D] -> [..., Ts, D]
    rho = alpha_lat / (alpha_lat.sum(-1, keepdim=True) + eps)
    g_R = torch.matmul(rho, Vstar_lat)
    if not norm_preserve:
        return m_R, g_R, g_R
    mt = m_R.norm(dim=-1, keepdim=True) * g_R / (g_R.norm(dim=-1, keepdim=True) + eps)
    return m_R, g_R, mt


# --------------------------------------------------------------------------- pure


def pos_neg(score: torch.Tensor, V: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """score [n], V [n,d] -> (P_Q [d], N_Q [d], top idx [k], bottom idx [k]) : means of the top / bottom k states."""
    n = int(score.numel())
    k = max(1, min(int(k), n // 2))
    order = torch.argsort(score.float(), descending=True, stable=True)
    top, bot = order[:k], order.flip(0)[:k]
    return V[top].float().mean(0), V[bot].float().mean(0), top, bot


def ground_loss(Z: torch.Tensor, P: torch.Tensor, N: torch.Tensor) -> torch.Tensor:
    """Page Eq.: -(1/m) sum cos(z_k, P) + (1/m) sum cos(z_k, N)."""
    S = cosine_sim(Z, torch.stack([P, N]))                     # [m, 2]
    return -S[:, 0].mean() + S[:, 1].mean()


def ground_stage(Z0, P, N, steps: int, lr_rel: float):
    with torch.enable_grad():
        Z = Z0.detach().float().clone().requires_grad_(True)
        rms = float(Z0.float().pow(2).mean().sqrt())
        opt = torch.optim.Adam([Z], lr=lr_rel * rms)
        losses = []
        for _ in range(int(steps)):
            opt.zero_grad()
            loss = ground_loss(Z, P, N)
            loss.backward()
            opt.step()
            losses.append(float(loss))
        losses.append(float(ground_loss(Z, P, N)))
    return Z.detach(), losses


def redistribute(alpha: torch.Tensor, lat: torch.Tensor, s: int, scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """alpha [..., L] native weights, scores [..., L] native logits, lat [m] latent columns, s sink column.

    Returns (alpha' [..., L], a_s [...]) with alpha'_s = 0, alpha'_lat += a_s * softmax(scores[lat]), rest unchanged.
    """
    a_s = alpha[..., s].clone()
    rho = torch.softmax(scores[..., lat], dim=-1)
    new = alpha.clone()
    new[..., s] = 0.0
    new[..., lat] = alpha[..., lat] + a_s.unsqueeze(-1) * rho
    return new, a_s


# --------------------------------------------------------------------------- Reasoner side


@contextlib.contextmanager
def reasoner_probe(bb, m: int, stage: str, targets=None):
    """Grounding needs the LU capture (question rows -> visual attention, prefill embeddings); cap/diag do not."""
    if not enabled() or not grounds() or stage != "reasoner" or int(m) <= 0:
        yield None
        return
    cap = ReasonerCapture(bb, m, targets)
    bb._rnlcr_cap = None
    handles = [bb.lm.register_forward_pre_hook(cap.lm_pre, with_kwargs=True)]
    handles += [layer.self_attn.register_forward_hook(cap.attn_post(li), with_kwargs=True)
                for li, layer in enumerate(bb.lm.layers)]
    try:
        yield cap
    finally:
        for h in handles:
            h.remove()
        bb._rnlcr_cap = cap


def _g(x) -> float:
    return float(f"{float(x):.5g}")


def _write(cfg, rec, t0):
    rec["sec"] = round(time.time() - t0, 2)
    if cfg["log"]:
        with open(cfg["log"], "a") as fh:
            fh.write(json.dumps(rec) + "\n")


@torch.no_grad()
def apply_reasoner(engine, result: dict, *, m: int, case_index: int = -1):
    """After the Reasoner append: record the latent columns; in ground/full modes run Step 1 + Step 2 in place."""
    cfg = config()
    if not cfg["mode"]:
        return None
    bb = engine._backbone
    t0 = time.time()
    m = int(m)
    L1 = int(result["past_len"])
    R = {"case": int(case_index), "latent_cols": torch.arange(L1 - m, L1, dtype=torch.long), "m": m,
         "cursor0": int(result["pos_cursor"]) - m,
         "vis_cols": (result.get("vis_cols").detach().cpu().long() if isinstance(result.get("vis_cols"), torch.Tensor) else None)}
    if cfg["mode"] == "relr":
        traj = result.get("latent_trajectory") or []
        R["traj"] = [z.detach().cpu().clone() for z in traj] if len(traj) == m else None
    bb._rnlcr_R = R
    tag = f"[RNLCR:R] case {case_index}"
    if not grounds():
        return R
    rec = {"case": int(case_index), "stage": "ground", "m": m}
    cap = getattr(bb, "_rnlcr_cap", None)
    bb._rnlcr_cap = None

    def skip(why):
        rec["skip"] = why
        R["ground_skip"] = why
        _write(cfg, rec, t0)
        print(f"{tag}: ground SKIP ({why})", flush=True)
        return R

    if cap is None or cap.rows is None:
        return skip((cap.note if cap is not None else "no capture") or "question rows not located")
    if len(cap.lat_len) != m or cap.lat_len != list(range(cap.lat_len[0], cap.lat_len[0] + m)):
        return skip(f"latent bookkeeping {cap.lat_len}")
    L0 = int(cap.lat_len[0])
    c0 = int(cap.lat_pos[0])
    if c0 < 0 or cap.lat_pos != list(range(c0, c0 + m)) or L0 + m != L1 or int(result["pos_cursor"]) != c0 + m:
        return skip("latent positions / lengths disagree with the result")
    cache = result["past_key_values"]
    if _kvlen(cache) != L1:
        return skip(f"cache length {_kvlen(cache)} != {L1}")
    vis = R["vis_cols"]
    if vis is None or int(vis.numel()) == 0 or int(vis.min()) < cap.past0 or int(vis.max()) >= L0:
        return skip("visual columns missing / outside the prefill")
    if cap.n_acc != cap.n_layers:
        return skip(f"attention captured on {cap.n_acc}/{cap.n_layers} layers")
    traj = result.get("latent_trajectory")
    if not traj or len(traj) != m:
        return skip("latent_trajectory missing")
    dev = bb.device
    zdtype = traj[0].dtype
    Z0 = torch.stack([z.to(dev) for z in traj]).float()
    vis_dev = vis.to(dev)
    V = cap.embeds0[vis_dev - cap.past0]
    s = (cap.acc / cap.n_acc)[:, vis_dev].mean(0)
    P, N, top, bot = pos_neg(s, V, cfg["topk"])
    S0 = cosine_sim(Z0, torch.stack([P, N]))
    Z, losses = ground_stage(Z0, P, N, cfg["s1_steps"], cfg["s1_lr"])
    S1 = cosine_sim(Z, torch.stack([P, N]))
    nat = [(l.keys[..., L0:L1, :].detach().clone(), l.values[..., L0:L1, :].detach().clone()) for l in cache.layers]
    positions = bb._text_positions(m, start=c0)
    cache.crop(L0)
    bb.lm(inputs_embeds=Z.to(zdtype).unsqueeze(0), position_ids=positions, past_key_values=cache, use_cache=True)
    if _kvlen(cache) != L1:
        raise RuntimeError(f"RN-LCR replay left the cache at {_kvlen(cache)} != {L1}")
    knum = kden = vnum = vden = 0.0
    for (k0, v0), l in zip(nat, cache.layers):
        k1, v1 = l.keys[..., L0:L1, :], l.values[..., L0:L1, :]
        knum += float((k1.float() - k0.float()).pow(2).sum()); kden += float(k0.float().pow(2).sum())
        vnum += float((v1.float() - v0.float()).pow(2).sum()); vden += float(v0.float().pow(2).sum())
    result["latent_trajectory"] = [z.to(zdtype).cpu() for z in Z]
    if isinstance(vis, torch.Tensor):
        rec["affinity"] = {"native": kv_affinity(cache, R["latent_cols"], vis, lat_kv=nat), "grounded": kv_affinity(cache, R["latent_cols"], vis)}
    md = cfg["mode"]
    if md in SP_MODES or md in ("vonly", "fulr"):
        # SP-GLR / FULR: the replayed KEY is never used by the receiver (native latent addressing is kept).
        # spglr/nonorm/fulr: keep V*_R aside and restore native K and V; vonly: restore native K, keep V*_R in place.
        R["vstar"] = [l.values[..., L0:L1, :].detach().clone() for l in cache.layers]
        for (k0, v0), l in zip(nat, cache.layers):
            l.keys[..., L0:L1, :] = k0
            if md != "vonly":
                l.values[..., L0:L1, :] = v0
        rec["sp"] = md
        if md != "vonly":
            result["latent_trajectory"] = list(traj)     # the cache carries the native latent rows again
    rec.update({"n_vis": int(vis.numel()), "topk": int(top.numel()), "q_rows": int(cap.rows[1] - cap.rows[0]),
                "rel_top": [_g(v) for v in s[top][:5].tolist()], "rel_bottom": [_g(v) for v in s[bot][:5].tolist()],
                "loss": [_g(v) for v in losses], "cosP": [_g(S0[:, 0].mean()), _g(S1[:, 0].mean())],
                "cosN": [_g(S0[:, 1].mean()), _g(S1[:, 1].mean())],
                "dz_rel": _g((Z - Z0).norm() / Z0.norm().clamp_min(1e-12)),
                "kv_rel": {"k": _g(math.sqrt(knum / max(kden, 1e-30))), "v": _g(math.sqrt(vnum / max(vden, 1e-30)))}})
    R["ground"] = rec
    _write(cfg, rec, t0)
    aff = rec.get("affinity")
    afftxt = (f" | latent V cos(vis/text) nat {aff['native']['cosV_vis']:.3f}/{aff['native']['cosV_text']:.3f} -> grounded "
              f"{aff['grounded']['cosV_vis']:.3f}/{aff['grounded']['cosV_text']:.3f}; K nat {aff['native']['cosK_vis']:.3f}/{aff['native']['cosK_text']:.3f}"
              f" -> {aff['grounded']['cosK_vis']:.3f}/{aff['grounded']['cosK_text']:.3f}; nnV_vis {aff['native']['nnV_vis_frac']:.2f}->{aff['grounded']['nnV_vis_frac']:.2f}") if aff else ""
    print(f"{tag} ground n_vis={rec['n_vis']} K={rec['topk']} loss {rec['loss'][0]:.4g}->{rec['loss'][-1]:.4g} "
          f"cosP {rec['cosP'][0]:.3f}->{rec['cosP'][1]:.3f} cosN {rec['cosN'][0]:.3f}->{rec['cosN'][1]:.3f} "
          f"dz {rec['dz_rel']:.3g} kv_rel k {rec['kv_rel']['k']:.3g} v {rec['kv_rel']['v']:.3g} sec={rec['sec']}{afftxt}", flush=True)
    return R


# --------------------------------------------------------------------------- terminal


@contextlib.contextmanager
def terminal(engine, cache, position_cursor: int, *, case_index: int = -1, system_prompt=None, user_prompt=None,
             json_prefix=None, prompt_ids=None):
    """Step 3 (cap/full) as attention hooks during the Answerer call; every mode measures. Yields stats.

    relr: everything happens before the yield (teacher pass, optimisation, commit); the Answerer then runs natively.
    """
    cfg = config()
    md = cfg["mode"]
    bb = engine._backbone
    t0 = time.time()
    fu = fulr_mode()
    stats = {"case": int(case_index), "mode": md, "applied": redistributes() or readout() or fu, "calls": 0, "rows": 0}
    R = getattr(bb, "_rnlcr_R", None)
    if R is None or R.get("latent_cols") is None:
        stats["skip"] = "no Reasoner record"
        yield stats
        _write(cfg, stats, t0)
        return
    if md == "relr":
        try:
            relr_run(bb, cache, int(position_cursor), R, cfg, stats, system_prompt=system_prompt, user_prompt=user_prompt,
                     json_prefix=json_prefix, prompt_ids=prompt_ids)
        except Exception as exc:  # never break the case: leave the cache native and report
            stats["skip"] = f"relr error: {exc!r}"
            print(f"[RNLCR] case {case_index}: relr FAILED {exc!r}", flush=True)
        yield stats
        _write(cfg, stats, t0)
        return
    L = _kvlen(cache)
    lat = R["latent_cols"].to(bb.device)
    s = int(cfg["sink"])
    if int(lat.max()) >= L or s >= L or bool((lat == s).any()):
        stats["skip"] = f"latent {lat.min().item()}..{lat.max().item()} / sink {s} vs cache {L}"
        yield stats
        _write(cfg, stats, t0)
        return
    stats.update({"latent_cols": [int(lat.min()), int(lat.max())], "sink": s, "ground": R.get("ground_skip") or ("yes" if "ground" in R else "no")})
    vstar = R.get("vstar")
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
        return
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, repeat_kv
    n_layers = len(bb.lm.layers)
    acc = {"a_s": 0.0, "C_s": 0.0, "o": 0.0, "lat_nat": 0.0, "lat_new": 0.0, "lat_would": 0.0, "msg_nat": 0.0, "msg_new": 0.0, "msg_would": 0.0,
           "mR": 0.0, "gR": 0.0, "mt": 0.0, "cos_mg": 0.0, "n": 0,
           "ms": 0.0, "rR": 0.0, "rperp": 0.0, "cos_rot": 0.0, "delta_rel": 0.0, "dv_rel": 0.0}
    applied = redistributes()
    sp = readout()
    a_s_layer = [0.0] * n_layers
    n_layer = [0] * n_layers
    fal_layer = [None] * n_layers      # cos(dV_bar, mu_vis - mu_text) at the first hooked row
    falv_layer = [None] * n_layers     # cos(dV_bar, mu_vis)
    rot_layer = [0.0] * n_layers       # cos(m_s, m~_s) accumulated per layer
    ms_layer = [0.0] * n_layers        # |m_s| per head, accumulated per layer
    vis_d = vis_cols.to(bb.device) if fu else None
    fast = fu and os.environ.get("VLMAS_RNLCR_FAST", "").strip() == "1"
    if fast:
        stats["fast"] = True
    idx_sl = torch.cat([torch.tensor([s], device=bb.device, dtype=lat.dtype), lat]) if fu else None
    fast2 = fu and os.environ.get("VLMAS_RNLCR_FAST", "").strip() == "2"
    stats_level = os.environ.get("VLMAS_RNLCR_STATS", "lite").strip() or "lite"
    impl_name = getattr(bb.lm.config, "_attn_implementation", None) or "eager"
    if fast2 and impl_name == "eager":
        fast2, fast = False, True
        stats["fast"] = True
        stats["fast2_fallback"] = "eager"
    if fast2:
        stats["fast"] = 2

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
                rho = _relay4_gate_rho(rho, bb, li)
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

    def make_post(li):
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
            cos, sin = pe
            h_sel = hidden.index_select(1, sel)
            q = module.q_norm(module.q_proj(h_sel).view(1, Ts, -1, Hd)).transpose(1, 2)
            q, _ = apply_rotary_pos_emb(q, torch.zeros_like(q), cos.index_select(1, sel), sin.index_select(1, sel))
            with torch.no_grad():
                K = repeat_kv(keys, G).float()
                Vv = repeat_kv(values, G).float()
                scores = torch.matmul(q.float(), K.transpose(-2, -1)) * module.scaling        # [1,H,Ts,Lc]
                if T > 1:
                    past = Lc - T
                    col = torch.arange(Lc, device=keys.device)[None, None, None, :]
                    row = (past + sel)[None, None, :, None]
                    scores = scores.masked_fill(col > row, float("-inf"))
                alpha = torch.softmax(scores, dim=-1)
                new, a_s = redistribute(alpha, lat, s, scores)
                flat = lambda x: x.transpose(1, 2).reshape(1, Ts, -1)
                wd = module.o_proj.weight.dtype
                if fu:
                    Vs = repeat_kv(vstar[li].to(values.device), G).float()                   # [1,H,m,D] grounded values
                    V_lat = Vv[..., lat, :]
                    _rho = _relay4_gate_rho(torch.softmax(scores[..., lat], dim=-1), bb, li)
                    dvbar = torch.matmul(_rho, Vs - V_lat)                                    # [1,H,Ts,D]
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
                elif sp:
                    ctx_nat = torch.matmul(alpha, Vv)                                        # [1,H,Ts,D]
                    Vs = repeat_kv(vstar[li].to(values.device), G).float()                   # [1,H,m,D]
                    m_R, g_R, m_t = sp_readout(alpha[..., lat], Vv[..., lat, :], Vs, norm_preserve=(md == "spglr"))
                    ctx_new = ctx_nat - m_R + m_t
                    acc["mR"] += float(m_R.norm(dim=-1).mean()); acc["gR"] += float(g_R.norm(dim=-1).mean())
                    acc["mt"] += float(m_t.norm(dim=-1).mean())
                    acc["cos_mg"] += float(torch.nn.functional.cosine_similarity(m_R, g_R, dim=-1).mean())
                else:
                    ctx_new = torch.matmul(new, Vv)                                          # [1,H,Ts,D]
                if ctx_new is not None:
                    o_new = module.o_proj(flat(ctx_new).to(wd)).float()
                o_nat = attn_out.index_select(1, sel).float()
                lat_ctx_nat = torch.matmul(alpha[..., lat], Vv[..., lat, :])
                lat_ctx_new = lat_ctx_nat if fu else (m_t if sp else torch.matmul(new[..., lat], Vv[..., lat, :]))
                sink_ctx = alpha[..., s].unsqueeze(-1) * Vv[..., s, :].unsqueeze(2)          # [1,H,Ts,D]
                msg_nat = module.o_proj(flat(lat_ctx_nat).to(wd)).float().norm(dim=-1).mean()
                msg_new = module.o_proj(flat(lat_ctx_new).to(wd)).float().norm(dim=-1).mean()
                C_s = module.o_proj(flat(sink_ctx).to(wd)).float().norm(dim=-1).mean()
                acc["a_s"] += float(a_s.mean()); acc["C_s"] += float(C_s); acc["o"] += float(o_nat.norm(dim=-1).mean())
                lat_nat_v, lat_would_v = float(alpha[..., lat].sum(-1).mean()), float(new[..., lat].sum(-1).mean())
                acc["lat_nat"] += lat_nat_v; acc["lat_would"] += lat_would_v; acc["lat_new"] += (lat_would_v if applied else lat_nat_v)
                acc["msg_nat"] += float(msg_nat); acc["msg_would"] += float(msg_new); acc["msg_new"] += (float(msg_new) if (applied or sp) else float(msg_nat))
                acc["n"] += 1
                a_s_layer[li] += float(a_s.mean()); n_layer[li] += 1
            stats["calls"] += 1
            stats["rows"] += Ts
            if not (applied or sp or fu):
                return output
            out = attn_out.clone()
            out[:, sel, :] = o_new.to(attn_out.dtype)
            return (out, *output[1:]) if isinstance(output, tuple) else out
        return post

    pre2 = {}

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
    try:
        yield stats
    finally:
        for h in handles:
            h.remove()
        if restore is not None:
            restore()
        n = max(acc["n"], 1)
        stats.update({k: _g(v / n) for k, v in acc.items() if k != "n"})
        stats["a_s_by_layer"] = [_g(a / max(c, 1)) for a, c in zip(a_s_layer, n_layer)]
        if fu:
            if fast or fast2:
                fal_layer[:] = [None if v is None else _g(v) for v in fal_layer]
                falv_layer[:] = [None if v is None else _g(v) for v in falv_layer]
            stats["fal_by_layer"] = fal_layer
            stats["falv_by_layer"] = falv_layer
            stats["cos_rot_by_layer"] = [_g(a / max(c, 1)) for a, c in zip(rot_layer, n_layer)]
            stats["ms_by_layer"] = [_g(a / max(c, 1)) for a, c in zip(ms_layer, n_layer)]
            vals = [v for v in fal_layer if v is not None]
            stats["fal_mean"] = _g(sum(vals) / len(vals)) if vals else None
            valsv = [v for v in falv_layer if v is not None]
            stats["falv_mean"] = _g(sum(valsv) / len(valsv)) if valsv else None
        stats["rows"] = stats["rows"] // max(n_layers, 1)
        if "ground" in R:
            stats["ground_stats"] = {k: R["ground"][k] for k in ("loss", "cosP", "cosN", "dz_rel", "kv_rel") if k in R["ground"]}
        _write(cfg, stats, t0)


# --------------------------------------------------------------------------- RELR


def _layer_window(spec: str, n: int) -> list[int]:
    if spec in ("", "all"):
        return list(range(n))
    a, b = (int(x) for x in spec.split("-")) if "-" in spec else (int(spec), int(spec))
    return list(range(max(0, min(a, b)), min(n - 1, max(a, b)) + 1))


def _answerer_prompt_ids(bb, system_prompt, user_prompt, json_prefix) -> torch.Tensor:
    from vision_text_mas.latent_terminal import _assistant_prompt
    prompt = _assistant_prompt(bb.processor, system_prompt=system_prompt or "", user_prompt=user_prompt or "", json_prefix=json_prefix)
    return bb.processor.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(bb.device)


def relr_teacher(bb, cache, position_cursor: int, ids: torch.Tensor, lat: torch.Tensor, vis: torch.Tensor, layers) -> dict:
    """Native prefill of the Answerer prompt on `cache` (cropped back afterwards); boundary-row statistics per layer."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, repeat_kv
    L = _kvlen(cache)
    out = {li: None for li in layers}
    lat_d, vis_d = lat.to(bb.device), vis.to(bb.device)

    def make_post(li):
        def post(module, args, kwargs, output):
            hidden, pe, c = _call_parts(args, kwargs)
            if hidden is None or pe is None or c is None:
                return output
            T = int(hidden.shape[1])
            sel = torch.tensor([T - 1], device=hidden.device)
            Hd, G = module.head_dim, module.num_key_value_groups
            cos, sin = pe
            q = module.q_norm(module.q_proj(hidden.index_select(1, sel)).view(1, 1, -1, Hd)).transpose(1, 2)
            q, _ = apply_rotary_pos_emb(q, torch.zeros_like(q), cos.index_select(1, sel), sin.index_select(1, sel))
            keys, values = c.layers[li].keys, c.layers[li].values
            with torch.no_grad():
                K = repeat_kv(keys, G).float()
                Vv = repeat_kv(values, G).float()
                sc = (torch.matmul(q.float(), K.transpose(-2, -1)) * module.scaling)[0, :, 0, :]      # [H, Lc]
                al = torch.softmax(sc, dim=-1)
                Lc = int(sc.shape[-1])
                other = torch.ones(Lc, dtype=torch.bool, device=sc.device)
                other[lat_d] = False
                lse_other = torch.logsumexp(sc[:, other], dim=-1)                                    # [H]
                wd = module.o_proj.weight.dtype
                proj = lambda ctx: module.o_proj(ctx.reshape(1, 1, -1).to(wd)).float().reshape(-1)  # [H,D] -> [d]
                m_V = proj(torch.matmul(al[:, vis_d].unsqueeze(1), Vv[0, :, vis_d, :]).squeeze(1))
                m_R = proj(torch.matmul(al[:, lat_d].unsqueeze(1), Vv[0, :, lat_d, :]).squeeze(1))
                o = proj(torch.matmul(al.unsqueeze(1), Vv[0]).squeeze(1))
                out[li] = {"q": q[0, :, 0, :].float().clone(), "lse_other": lse_other.clone(), "m_V": m_V.clone(),
                           "m_R": m_R.clone(), "o_norm": float(o.norm()), "lat_mass": float(al[:, lat_d].sum(-1).mean()),
                           "vis_mass": float(al[:, vis_d].sum(-1).mean()), "scaling": float(module.scaling), "G": int(G)}
            return output
        return post

    handles = [bb.lm.layers[li].self_attn.register_forward_hook(make_post(li), with_kwargs=True) for li in layers]
    try:
        with torch.no_grad():
            bb.lm(inputs_embeds=bb.lm.embed_tokens(ids).unsqueeze(0), position_ids=bb._text_positions(int(ids.numel()), start=int(position_cursor)),
                  past_key_values=cache, use_cache=True)
    finally:
        for h in handles:
            h.remove()
        cache.crop(L)
    return out


def _prefix_cache(cache, L0: int):
    from transformers.cache_utils import DynamicCache
    tmp = DynamicCache()
    for li, layer in enumerate(cache.layers):
        tmp.update(layer.keys[..., :L0, :], layer.values[..., :L0, :], li)
    return tmp


def relr_message(bb, out_cache, L0: int, m: int, teacher: dict, layers) -> tuple[torch.Tensor, torch.Tensor]:
    """m_R^l(Z') for every teacher layer from the replayed latent K'/V' (differentiable) -> (loss_msg, per-layer cos)."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import repeat_kv
    total = 0.0
    coss = []
    for li in layers:
        t = teacher[li]
        Kp = repeat_kv(out_cache.layers[li].keys[..., L0:L0 + m, :], t["G"])[0].float()      # [H, m, D]
        Vp = repeat_kv(out_cache.layers[li].values[..., L0:L0 + m, :], t["G"])[0].float()
        s_lat = torch.matmul(Kp, t["q"].unsqueeze(-1)).squeeze(-1) * t["scaling"]              # [H, m]
        lse = torch.logaddexp(t["lse_other"], torch.logsumexp(s_lat, dim=-1))                  # [H]
        al = torch.exp(s_lat - lse.unsqueeze(-1))                                               # [H, m]
        ctx = torch.matmul(al.unsqueeze(1), Vp).squeeze(1)                                      # [H, D]
        mod = bb.lm.layers[li].self_attn
        m_R = mod.o_proj(ctx.reshape(1, 1, -1).to(mod.o_proj.weight.dtype)).float().reshape(-1)
        total = total + (m_R - t["m_V"]).pow(2).sum() / (t["m_V"].pow(2).sum() + 1e-8)
        coss.append(torch.nn.functional.cosine_similarity(m_R, t["m_V"], dim=0))
    return total, torch.stack(coss)


@torch.no_grad()
def kv_affinity(cache, lat: torch.Tensor, vis: torch.Tensor, lat_kv=None, exclude_first: bool = True) -> dict:
    """Where do the latent KV columns sit: closer to the visual KV or to the text KV?  (per layer x kv-head, averaged)

    cosV_vis / cosV_text : cosine of each latent VALUE to the visual / text value centroid (mean over layers, heads, latents)
    cosK_vis / cosK_text : same for KEYS (post-RoPE, so keys carry position; values do not)
    nnV_vis_frac / nnK_vis_frac : fraction of (layer, head, latent) whose nearest cached column (cosine) is a visual column
    lat_kv: optional list of (K [1,Hk,m,D], V [1,Hk,m,D]) per layer to evaluate instead of the cache's latent columns.
    """
    dev = cache.layers[0].keys.device
    L = int(cache.layers[0].keys.shape[-2])
    lat_d, vis_d = lat.to(dev), vis.to(dev)
    text = torch.ones(L, dtype=torch.bool, device=dev)
    text[lat_d] = False; text[vis_d] = False
    if exclude_first:
        text[0] = False
    text_d = text.nonzero(as_tuple=True)[0]
    acc = {"cosV_vis": 0.0, "cosV_text": 0.0, "cosK_vis": 0.0, "cosK_text": 0.0, "nnV_vis": 0.0, "nnK_vis": 0.0, "n": 0}
    for li, layer in enumerate(cache.layers):
        K, V = layer.keys[0].float(), layer.values[0].float()                  # [Hk, L, D]
        if lat_kv is not None:
            Kl, Vl = lat_kv[li][0][0].float(), lat_kv[li][1][0].float()
        else:
            Kl, Vl = K[:, lat_d, :], V[:, lat_d, :]                            # [Hk, m, D]
        for X, Xl, tagc, tagn in ((V, Vl, ("cosV_vis", "cosV_text"), "nnV_vis"), (K, Kl, ("cosK_vis", "cosK_text"), "nnK_vis")):
            cv = torch.nn.functional.normalize(X[:, vis_d, :].mean(1), dim=-1)         # [Hk, D] visual centroid
            ct = torch.nn.functional.normalize(X[:, text_d, :].mean(1), dim=-1)
            xl = torch.nn.functional.normalize(Xl, dim=-1)                              # [Hk, m, D]
            acc[tagc[0]] += float((xl * cv.unsqueeze(1)).sum(-1).mean())
            acc[tagc[1]] += float((xl * ct.unsqueeze(1)).sum(-1).mean())
            Xn = torch.nn.functional.normalize(X, dim=-1)                               # [Hk, L, D]
            sims = torch.matmul(xl, Xn.transpose(-1, -2))                               # [Hk, m, L]
            sims[:, :, lat_d] = -2.0
            if exclude_first:
                sims[:, :, 0] = -2.0
            nn = sims.argmax(-1)                                                         # [Hk, m]
            isvis = torch.zeros(L, dtype=torch.bool, device=dev); isvis[vis_d] = True
            acc[tagn] += float(isvis[nn].float().mean())
        acc["n"] += 1
    n = max(acc["n"], 1)
    return {"cosV_vis": _g(acc["cosV_vis"] / n), "cosV_text": _g(acc["cosV_text"] / n), "cosK_vis": _g(acc["cosK_vis"] / n),
            "cosK_text": _g(acc["cosK_text"] / n), "nnV_vis_frac": _g(acc["nnV_vis"] / n), "nnK_vis_frac": _g(acc["nnK_vis"] / n)}


def relr_run(bb, cache, position_cursor: int, R: dict, cfg: dict, stats: dict, *, system_prompt=None, user_prompt=None,
             json_prefix=None, prompt_ids=None) -> None:
    lat = R["latent_cols"]
    m = int(R["m"])
    L0 = int(lat.min())
    L = _kvlen(cache)
    c0 = int(R.get("cursor0", -1))
    vis = R.get("vis_cols")
    traj = R.get("traj")
    if vis is None or int(vis.numel()) == 0 or traj is None or c0 < 0 or int(lat.max()) + 1 > L:
        raise RuntimeError("missing visual columns / native trajectory / cursor")
    n_close = L - (L0 + m)
    tok = bb.processor.tokenizer
    close_ids = None
    for txt in ("<|im_end|>\n", "</think>\n<|im_end|>\n"):
        ids = tok.encode(txt, add_special_tokens=False)
        if len(ids) == n_close:
            close_ids = ids
            break
    if n_close and close_ids is None:
        raise RuntimeError(f"cannot identify {n_close} turn-close tokens")
    layers = _layer_window(cfg["relr_layers"], len(bb.lm.layers))
    ids = prompt_ids if prompt_ids is not None else _answerer_prompt_ids(bb, system_prompt, user_prompt, json_prefix)
    teacher = relr_teacher(bb, cache, position_cursor, ids, lat, vis, layers)
    cos_nat = float(torch.stack([torch.nn.functional.cosine_similarity(teacher[li]["m_R"], teacher[li]["m_V"], dim=0) for li in layers]).mean())
    ratio_nat = float(torch.stack([teacher[li]["m_R"].norm() / (teacher[li]["m_V"].norm() + 1e-8) for li in layers]).mean())
    dev = bb.device
    zdtype = traj[0].dtype
    Z0 = torch.stack([z.to(dev) for z in traj]).float()
    positions = bb._text_positions(m, start=c0)
    rms = float(Z0.pow(2).mean().sqrt())
    losses, losses_msg = [], []
    with torch.enable_grad():
        Z = Z0.clone().requires_grad_(True)
        opt = torch.optim.Adam([Z], lr=cfg["relr_lr"] * rms)
        for _ in range(int(cfg["relr_steps"])):
            pre = _prefix_cache(cache, L0)
            o = bb.lm(inputs_embeds=Z.to(zdtype).unsqueeze(0), position_ids=positions, past_key_values=pre, use_cache=True)
            loss_msg, _ = relr_message(bb, o.past_key_values, L0, m, teacher, layers)
            loss = loss_msg + cfg["relr_lambda"] * (Z - Z0).pow(2).sum() / (Z0.pow(2).sum() + 1e-8)
            g, = torch.autograd.grad(loss, Z)
            Z.grad = g
            opt.step()
            opt.zero_grad()
            losses.append(float(loss)); losses_msg.append(float(loss_msg))
            del o, pre, g, loss
        with torch.no_grad():
            pre = _prefix_cache(cache, L0)
            o = bb.lm(inputs_embeds=Z.to(zdtype).unsqueeze(0), position_ids=positions, past_key_values=pre, use_cache=True)
            loss_msg, _ = relr_message(bb, o.past_key_values, L0, m, teacher, layers)
            losses.append(float(loss_msg + cfg["relr_lambda"] * (Z - Z0).pow(2).sum() / (Z0.pow(2).sum() + 1e-8))); losses_msg.append(float(loss_msg))
            del o, pre
    Zs = Z.detach()
    # commit: crop, replay Z*, re-append the turn-close tokens
    with torch.no_grad():
        nat = [(l.keys[..., L0:L0 + m, :].clone(), l.values[..., L0:L0 + m, :].clone()) for l in cache.layers]
        cache.crop(L0)
        bb.lm(inputs_embeds=Zs.to(zdtype).unsqueeze(0), position_ids=positions, past_key_values=cache, use_cache=True)
        if n_close:
            ce = bb.lm.embed_tokens(torch.tensor(close_ids, device=dev)).unsqueeze(0)
            bb.lm(inputs_embeds=ce, position_ids=bb._text_positions(n_close, start=c0 + m), past_key_values=cache, use_cache=True)
        if _kvlen(cache) != L:
            raise RuntimeError(f"commit left the cache at {_kvlen(cache)} != {L}")
        knum = kden = vnum = vden = 0.0
        for (k0, v0), l in zip(nat, cache.layers):
            k1, v1 = l.keys[..., L0:L0 + m, :], l.values[..., L0:L0 + m, :]
            knum += float((k1.float() - k0.float()).pow(2).sum()); kden += float(k0.float().pow(2).sum())
            vnum += float((v1.float() - v0.float()).pow(2).sum()); vden += float(v0.float().pow(2).sum())
        post = relr_teacher(bb, cache, position_cursor, ids, lat, vis, layers)
        aff_nat = kv_affinity(cache, lat, vis, lat_kv=nat)
        aff_new = kv_affinity(cache, lat, vis)
    stats["affinity"] = {"native": aff_nat, "new": aff_new}
    stats["text_mass"] = _g(sum(1.0 - teacher[li]["lat_mass"] - teacher[li]["vis_mass"] for li in layers) / len(layers))
    cos_new = float(torch.stack([torch.nn.functional.cosine_similarity(post[li]["m_R"], teacher[li]["m_V"], dim=0) for li in layers]).mean())
    ratio_new = float(torch.stack([post[li]["m_R"].norm() / (teacher[li]["m_V"].norm() + 1e-8) for li in layers]).mean())
    stats.update({
        "applied": True, "n_close": n_close, "steps": int(cfg["relr_steps"]), "layers": [layers[0], layers[-1]],
        "loss": [_g(v) for v in losses], "loss_msg": [_g(v) for v in losses_msg],
        "cos_nat": _g(cos_nat), "cos_new": _g(cos_new), "ratio_nat": _g(ratio_nat), "ratio_new": _g(ratio_new),
        "o_nat": _g(sum(teacher[li]["o_norm"] for li in layers) / len(layers)), "o_new": _g(sum(post[li]["o_norm"] for li in layers) / len(layers)),
        "lat_nat": _g(sum(teacher[li]["lat_mass"] for li in layers) / len(layers)), "lat_new": _g(sum(post[li]["lat_mass"] for li in layers) / len(layers)),
        "vis_mass": _g(sum(teacher[li]["vis_mass"] for li in layers) / len(layers)),
        "dz_rel": _g((Zs - Z0).norm() / Z0.norm().clamp_min(1e-12)),
        "kv_rel": {"k": _g(math.sqrt(knum / max(kden, 1e-30))), "v": _g(math.sqrt(vnum / max(vden, 1e-30)))},
        "cos_by_layer_new": [_g(torch.nn.functional.cosine_similarity(post[li]["m_R"], teacher[li]["m_V"], dim=0)) for li in layers],
    })


def summary(stats: dict) -> str:
    if "skip" in stats:
        return f"mode={stats['mode']} SKIP ({stats['skip']})"
    ab = stats.get("a_s_by_layer", [])
    if stats["mode"] == "relr":
        return (f"mode=relr applied={stats.get('applied')} latent={stats.get('latent_cols')} n_close={stats.get('n_close')} "
                f"loss {stats['loss'][0]:.4g}->{stats['loss'][-1]:.4g} (msg {stats['loss_msg'][0]:.4g}->{stats['loss_msg'][-1]:.4g}) "
                f"cos(mR,mV) {stats['cos_nat']:.3f}->{stats['cos_new']:.3f} |mR|/|mV| {stats['ratio_nat']:.3g}->{stats['ratio_new']:.3g} "
                f"|o| {stats['o_nat']:.3g}->{stats['o_new']:.3g} lat_mass {stats['lat_nat']:.2e}->{stats['lat_new']:.2e} dz {stats['dz_rel']:.3g} "
                f"kv_rel k {stats['kv_rel']['k']:.3g} v {stats['kv_rel']['v']:.3g} steps={stats['steps']} sec={stats['sec']}"
                + (f" | affinity V(vis/text) nat {stats['affinity']['native']['cosV_vis']:.3f}/{stats['affinity']['native']['cosV_text']:.3f}"
                   f" -> new {stats['affinity']['new']['cosV_vis']:.3f}/{stats['affinity']['new']['cosV_text']:.3f}"
                   f" K nat {stats['affinity']['native']['cosK_vis']:.3f}/{stats['affinity']['native']['cosK_text']:.3f}"
                   f" -> new {stats['affinity']['new']['cosK_vis']:.3f}/{stats['affinity']['new']['cosK_text']:.3f}"
                   f" nnV_vis {stats['affinity']['native']['nnV_vis_frac']:.2f}->{stats['affinity']['new']['nnV_vis_frac']:.2f}"
                   f" mass lat/vis/text {stats['lat_nat']:.1e}/{stats['vis_mass']:.3f}/{stats['text_mass']:.3f}" if "affinity" in stats else ""))
    if stats["mode"] == "fulr":
        fl = stats.get("fal_by_layer") or []
        rl = stats.get("cos_rot_by_layer") or []
        pick = lambda xs, i: (xs[i] if len(xs) > i and xs[i] is not None else float("nan"))
        return (f"mode=fulr applied={stats['applied']} ground={stats.get('ground')} latent={stats.get('latent_cols')} sink={stats.get('sink')} "
                f"a_s={stats['a_s']:.4f} C_s/|o|={stats['C_s']:.3g}/{stats['o']:.3g} |m_s|h={stats['ms']:.3g} |r_R|={stats['rR']:.3g} |r_perp|={stats['rperp']:.3g} "
                f"cos(m_s,m~)={stats['cos_rot']:.4f} (L27 {pick(rl, 27):.4f}, L35 {pick(rl, 35):.4f}) |o'-o|/|o|={stats['delta_rel']:.3g} dV_rel={stats['dv_rel']:.3g} "
                f"latent mass {stats['lat_nat']:.2e} | falsifier cos(dV,mu_v-mu_t) mean {stats.get('fal_mean')} (L27 {pick(fl, 27):.3f}, L35 {pick(fl, 35):.3f}) "
                f"cos(dV,mu_v) {stats.get('falv_mean')} rows={stats['rows']} sec={stats['sec']}")
    if stats["mode"] in SP_MODES:
        return (f"mode={stats['mode']} applied={stats['applied']} ground={stats.get('ground')} latent={stats.get('latent_cols')} "
                f"a_s={stats['a_s']:.4f} latent mass {stats['lat_nat']:.2e} |m_R|={stats['mR']:.4g} |g_R|={stats['gR']:.4g} |m~|={stats['mt']:.4g} "
                f"cos(m_R,g_R)={stats['cos_mg']:.3f} latent msg(W_O) {stats['msg_nat']:.3g}->{stats['msg_new']:.3g} rows={stats['rows']} sec={stats['sec']}")
    return (f"mode={stats['mode']} applied={stats['applied']} ground={stats.get('ground')} latent={stats.get('latent_cols')} sink={stats.get('sink')} "
            f"a_s={stats['a_s']:.4f} (L27 {ab[27] if len(ab) > 27 else float('nan'):.3f}, L35 {ab[35] if len(ab) > 35 else float('nan'):.3f}) "
            f"C_s/|o|={stats['C_s']:.3g}/{stats['o']:.3g} latent mass {stats['lat_nat']:.2e}->{stats['lat_new']:.4f} (would {stats['lat_would']:.4f}) "
            f"latent msg {stats['msg_nat']:.3g}->{stats['msg_new']:.3g} (would {stats['msg_would']:.3g}) rows={stats['rows']} sec={stats['sec']}")
