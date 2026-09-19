"""State-Backed WSI Support Handoff (SBSH) — state-forming support influence at the Reasoner boundary.

Notion C1/C2 Method Evolution 11.9 "State-Backed WSI Support Handoff" (C2 candidate).

Support gate. Every physical support G_g (one retained crop) gets one exposure coordinate u_g (native
u = 0). While the Reasoner latent steps are replayed from the cache state they started from, the
attention logit of every Reasoner latent query to every visual column j in G_g is shifted by u_g,
in EVERY decoder layer and EVERY latent step (one shared coordinate, no layer choice):
    logit_{t,j}^{(l)} <- logit_{t,j}^{(l)} + u_g
With z_R the final Reasoner latent state (last-layer hidden of the last latent step) and
zbar = z_R / ||z_R||,
    S_g = || d zbar / d u_g ||^2  at u = 0
Estimators
    hutchinson  (page)  S_g ~ 1/K sum_k (d (r_k . zbar) / d u_g)^2, r_k Rademacher — K backward passes
                        through ONE graph-retaining replay, all supports at once.
    fd          (check) central difference per support, (zbar(+eps e_g) - zbar(-eps e_g)) / (2 eps),
                        2G no-grad replays. Used to validate the autograd estimate on the real model.
The live cache is never touched (the replay runs on a deep copy cropped to the Reasoner start).

Handoff = memory/rsvmh.py with VLMAS_KV_RESTAGE_ORDER=state_backed (ascending S -> largest S nearest
the Answerer) or state_backed_rev (direction control), on the terminal restage path.

Env
    VLMAS_SBSH=1                 compute S at the end of the Reasoner latent block (also implied by
                                 VLMAS_KV_RESTAGE_ORDER=state_backed|state_backed_rev)
    VLMAS_SBSH_EST               hutchinson (default) | fd | both
    VLMAS_SBSH_K                 Rademacher probes (default 16)
    VLMAS_SBSH_EPS               fd step in logit units (default 0.1)
    VLMAS_SBSH_SEED              probe seed (default 0)
The Reasoner probe that carries the support bookkeeping is created by the backbone when VLMAS_RSS is
set (use VLMAS_RSS=<jsonl> VLMAS_RSS_NO_LOO=1; the rss record then also stores the SBSH result).
Off = no-op.
"""
from __future__ import annotations

import contextlib
import math
import os
import time
from copy import deepcopy

import torch

STATE_ORDERS = ("state_backed", "state_backed_rev")


def enabled() -> bool:
    v = os.environ.get("VLMAS_SBSH", "").strip()
    return (v not in ("", "0")) or os.environ.get("VLMAS_KV_RESTAGE_ORDER", "").strip() in STATE_ORDERS


# --------------------------------------------------------------------------- pure


def support_bias(u: torch.Tensor, groups, length: int) -> torch.Tensor:
    """[length] additive logit bias: u[g] on the columns of group g, 0 elsewhere (differentiable in u)."""
    bias = torch.zeros(int(length), dtype=u.dtype, device=u.device)
    for g, cols in enumerate(groups):
        cols = cols.to(u.device).long()
        bias = bias.index_add(0, cols, u[g].expand(int(cols.numel())))
    return bias


def combine_mask(mask, bias: torch.Tensor, q_len: int, dtype: torch.dtype) -> torch.Tensor:
    """Existing attention mask (None / bool keep-mask / additive float) + per-key bias -> additive 4-D mask."""
    L = int(bias.numel())
    row = bias.to(dtype).view(1, 1, 1, L)
    if mask is None:
        return row.expand(1, 1, int(q_len), L)
    m = mask[..., :L]
    if m.dtype == torch.bool:
        m = torch.zeros(m.shape, dtype=dtype, device=m.device).masked_fill(~m, torch.finfo(dtype).min)
    else:
        m = m.to(dtype)
    return m + row


def rademacher(k: int, d: int, seed: int, device) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    return (torch.randint(0, 2, (int(k), int(d)), generator=g, dtype=torch.int64) * 2 - 1).to(device=device, dtype=torch.float32)


def hutchinson_energy(zbar: torch.Tensor, u: torch.Tensor, probes: torch.Tensor) -> torch.Tensor:
    """1/K sum_k (d (r_k . zbar)/du)^2 per coordinate of u. `zbar` must be a graph output of `u`."""
    acc = torch.zeros(u.shape, dtype=torch.float64, device=u.device)
    K = int(probes.shape[0])
    for k in range(K):
        (g,) = torch.autograd.grad((probes[k].to(zbar.dtype) * zbar).sum(), u, retain_graph=k < K - 1)
        acc += g.detach().double() ** 2
    return acc / K


def rank_corr(a, b) -> float | None:
    a = torch.as_tensor(a, dtype=torch.float64)
    b = torch.as_tensor(b, dtype=torch.float64)
    if a.numel() < 3:
        return None
    ra = a.argsort().argsort().double()
    rb = b.argsort().argsort().double()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = float(ra.norm() * rb.norm())
    return None if den == 0 else float((ra * rb).sum()) / den


# --------------------------------------------------------------------------- gate hooks


@contextlib.contextmanager
def support_gate(lm, groups, u: torch.Tensor):
    """Pre-hooks on every decoder self-attention: add support_bias(u) to the attention mask."""
    handles = []

    def hook(module, args, kwargs):
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        kv = kwargs.get("past_key_values")
        if hidden is None or kv is None:
            raise RuntimeError("SBSH gate: hidden_states / past_key_values not passed as kwargs")
        layer = kv.layers[module.layer_idx]
        past = int(layer.keys.shape[-2]) if getattr(layer, "keys", None) is not None and layer.keys.dim() == 4 else 0
        q_len = int(hidden.shape[1])
        bias = support_bias(u, groups, past + q_len)
        kwargs["attention_mask"] = combine_mask(kwargs.get("attention_mask"), bias, q_len, hidden.dtype)
        return args, kwargs

    for layer in lm.layers:
        handles.append(layer.self_attn.register_forward_pre_hook(hook, with_kwargs=True))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def replay_latents(bb, base_cache, h0, cursor: int, m: int, groups, u: torch.Tensor):
    """Re-run the m Reasoner latent steps on a copy of `base_cache` with the support gate u. Returns z_R [D]."""
    c = deepcopy(base_cache)
    h = h0
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        kernel = sdpa_kernel(SDPBackend.MATH)
    except Exception:  # pragma: no cover - older torch
        kernel = contextlib.nullcontext()
    with kernel, support_gate(bb.lm, groups, u):
        for step in range(int(m)):
            le = bb._apply_realign(h)
            o = bb.lm(inputs_embeds=le, position_ids=bb._text_positions(1, int(cursor) + step),
                      past_key_values=c, use_cache=True, output_hidden_states=True)
            c = o.past_key_values
            h = o.hidden_states[-1][:, -1:, :]
    z = h[0, 0]
    if z.dtype in (torch.float16, torch.bfloat16):
        z = z.float()
    del c
    return z


# --------------------------------------------------------------------------- runtime


def compute(probe, output) -> dict:
    """Called by rss_diag.ReasonerProbe.after_step at the last latent step. Never raises."""
    from memory.rss_diag import support_columns

    t0 = time.time()
    rec: dict = {}
    try:
        bb = probe.bb
        if probe.image_ids is None:
            return {"skip": probe.note or "no support ids"}
        if probe.note and "eviction" in probe.note:
            return {"skip": probe.note}
        dev = probe.h0.device
        pairs = support_columns(probe.vis.to(dev), probe.image_ids)
        supports = [int(i) for i, _ in pairs]
        groups = [cols for _, cols in pairs]
        G = len(groups)
        est = os.environ.get("VLMAS_SBSH_EST", "hutchinson").strip() or "hutchinson"
        K = int(os.environ.get("VLMAS_SBSH_K", "16"))
        eps = float(os.environ.get("VLMAS_SBSH_EPS", "0.1"))
        seed = int(os.environ.get("VLMAS_SBSH_SEED", "0"))
        rec.update({"supports": supports, "est": est, "K": K, "eps": eps, "m": probe.m, "pre_len": probe.pre_len})

        base = deepcopy(output.past_key_values)
        base.crop(int(probe.pre_len))
        z_native = output.hidden_states[-1][0, -1].detach()
        z_native = z_native.float() if z_native.dtype in (torch.float16, torch.bfloat16) else z_native

        if est in ("hutchinson", "both"):
            t1 = time.time()
            with torch.enable_grad():
                u = torch.zeros(G, dtype=z_native.dtype, device=dev, requires_grad=True)
                z = replay_latents(bb, base, probe.h0, probe.cursor, probe.m, groups, u)
                zbar = z / z.norm()
                rec["floor_cos"] = float(1.0 - torch.nn.functional.cosine_similarity(z.detach(), z_native, dim=0))
                probes = rademacher(K, int(zbar.numel()), seed, dev)
                S = hutchinson_energy(zbar, u, probes)
            rec["S"] = [float(x) for x in S]
            rec["hutchinson_sec"] = round(time.time() - t1, 2)
            del z, zbar, u

        if est in ("fd", "both"):
            t2 = time.time()
            S_fd = []
            with torch.no_grad():
                for g in range(G):
                    e = torch.zeros(G, dtype=z_native.dtype, device=dev)
                    e[g] = eps
                    zp = replay_latents(bb, base, probe.h0, probe.cursor, probe.m, groups, e)
                    zm = replay_latents(bb, base, probe.h0, probe.cursor, probe.m, groups, -e)
                    J = (zp / zp.norm() - zm / zm.norm()) / (2.0 * eps)
                    S_fd.append(float(J.double().pow(2).sum()))
                if "floor_cos" not in rec:
                    z0 = replay_latents(bb, base, probe.h0, probe.cursor, probe.m, groups,
                                        torch.zeros(G, dtype=z_native.dtype, device=dev))
                    rec["floor_cos"] = float(1.0 - torch.nn.functional.cosine_similarity(z0, z_native, dim=0))
            rec["S_fd"] = S_fd
            rec["fd_sec"] = round(time.time() - t2, 2)
            if "S" not in rec:
                rec["S"] = S_fd
        if "S" in rec and "S_fd" in rec and est == "both":
            rec["rho_hutch_fd"] = rank_corr(rec["S"], rec["S_fd"])
        px = getattr(probe, "proxy", None) or {}
        if px.get("supports") == supports:
            rec["rho_S_mass"] = rank_corr(rec["S"], px.get("mass_sum_heads"))
            rec["rho_S_shat"] = rank_corr(rec["S"], px.get("s_hat"))
        elif px:
            rec["note"] = "proxy supports differ from SBSH supports"
        del base
    except Exception as exc:  # measurement must never break the pipeline
        rec["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        rec["sec"] = round(time.time() - t0, 2)
        S = rec.get("S")
        print(f"[SBSH] supports={len(rec.get('supports', []))} est={rec.get('est')} K={rec.get('K')} "
              f"floor_cos={rec.get('floor_cos')} S_max/med="
              f"{(max(S) / max(sorted(S)[len(S) // 2], 1e-30)) if S else None} "
              f"rho_mass={rec.get('rho_S_mass')} rho_shat={rec.get('rho_S_shat')} rho_fd={rec.get('rho_hutch_fd')} "
              f"skip={rec.get('skip')} error={rec.get('error')} sec={rec['sec']}", flush=True)
    return rec


__all__ = ["STATE_ORDERS", "enabled", "support_bias", "combine_mask", "rademacher", "hutchinson_energy",
           "rank_corr", "support_gate", "replay_latents", "compute"]
