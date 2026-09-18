"""C1 #21 NOVA — Null-Aware Visual Volume Allocation. Notion 3ddd771b-c9e2-81e1.

Opt-in: VLMAS_C1_QASC=1 VLMAS_C1_SELECT=nova, dispatched from memory/qasc_select.py at the Reasoner
prefill site (backbone/qwen3vl.py, the QASC / NTRS / SSDA site). Off = no call, byte-identical.

Page §1  crops -> FULL vision encoder -> NOVA selection -> LLM visual prefill / persistent visual KV.
         z_i = h_i^(L_V) / ||h_i^(L_V)||, h = merger output (pooler_output), the embedding the decoder
         consumes. No early-vision FLOP saving is claimed; what shrinks is the prefill length.
Page §2-§5 are the SEC-LD (C1 #19, memory/secld_select.py) objective verbatim and are REUSED, not
re-implemented:
         r_i, m_i (gray > tau_p, cell >= tau_c)      b_i = G_sigma * Pad_rep(m)      e_i = 1 - b_i
         q_i = sqrt(e_i) z_i        F(S) = sum_g log det(I + sum_{j in S cap V_g} q_j q_j^T),  |S| <= B
         exact greedy  Delta(j|S) = log(1 + e_j z_j^T M_{g(j)}(S)^{-1} z_j),  no per-crop quota,
         no support normalisation.
The only differences from SEC-LD are the selection point (full vision output instead of the Pruning-B
4-block boundary) and hence z.

Env
  VLMAS_NOVA_KEEP=0.25    budget B = round(keep * N)          (page §0 audit budget)
  VLMAS_NOVA_MODE=nova    nova | logdet (e_i == 1, §8 falsifier 4) | etopk (e_i Top-k, §8 falsifier 3)
  VLMAS_NOVA_TAU_P=215    VLMAS_NOVA_TAU_C=0.9    VLMAS_NOVA_SIGMA=2.0    (§2 estimator; sweep targets)
  VLMAS_NOVA_MIN=0        per-support minimum. Page: 0 (0-token crops allowed); >0 = runtime fallback.
Pixels: the un-cast fp32 pixel_values the backbone keeps as `_secld_pixel_values_fp32` (stored when
VLMAS_C1_SELECT=nova as well as for SEC-LD) so gray is bit-identical to the 50-case audit; falls back to
the bf16 model input (gray error <= 0.25 of a level) and reports which one in the log.
Log: `[NOVA] kept B/N per_support=[...] sd= empty_crops= m_all= m_sel= e_all= e_all_lt01= e_sel= e_sel_min=
b_sel= gain_first= gain_last= nov_first= nov_last= nov_min= erank_overlap= sec= mode= keep_ratio= mags=
pixels= total_sec=` (secld_select.summary + NOVA fields; erank_overlap = |S cap e-Top-B| / B).
"""
from __future__ import annotations

import math
import os
import time

import torch
import torch.nn.functional as F

from memory import secld_select as _sec

MODES = ("nova", "logdet", "etopk")


def enabled() -> bool:
    return os.environ.get("VLMAS_C1_SELECT", "").strip() == "nova"


def config() -> dict:
    def f(name, default):
        v = os.environ.get(name, "").strip()
        return float(v) if v else default
    mode = os.environ.get("VLMAS_NOVA_MODE", "nova").strip() or "nova"
    if mode not in MODES:
        raise ValueError(f"VLMAS_NOVA_MODE={mode!r} not in {MODES}")
    return {
        "tau_p": f("VLMAS_NOVA_TAU_P", 215.0),
        "tau_c": f("VLMAS_NOVA_TAU_C", 0.9),
        "sigma": f("VLMAS_NOVA_SIGMA", 2.0),
        "keep": f("VLMAS_NOVA_KEEP", 0.25),
        "min_per_support": int(os.environ.get("VLMAS_NOVA_MIN", "0") or 0),
        "mode": mode,
    }


def budget_for(n: int, keep: float) -> int:
    """Page §4 fixed budget B = round(keep * N) (SEC-LD's rule), at least 1 when N > 0."""
    return max(1, int(round(float(keep) * int(n)))) if int(n) > 0 else 0


# --------------------------------------------------------------------------- pure


def _etopk_order(e: torch.Tensor, support: torch.Tensor, budget: int, first_per_support: int) -> list:
    """e ranking, ties -> lowest index; optional forced first picks per support (same rule as the greedy)."""
    N = int(e.numel())
    e_cpu, s_cpu = e.detach().double().cpu(), support.detach().cpu()
    chosen = torch.zeros(N, dtype=torch.bool)
    order = []
    n_sup = int(s_cpu.max()) + 1 if N else 0
    for _ in range(max(0, int(first_per_support))):
        for g in range(n_sup):
            if len(order) >= budget:
                break
            score = e_cpu.masked_fill(chosen | (s_cpu != g), float("-inf"))
            if not torch.isfinite(score.max()):
                continue
            j = int(torch.argmax(score))                      # argmax returns the first maximal index
            chosen[j] = True
            order.append(j)
    while len(order) < budget:
        score = e_cpu.masked_fill(chosen, float("-inf"))
        j = int(torch.argmax(score))
        chosen[j] = True
        order.append(j)
    return order


@torch.no_grad()
def select_tokens(pixel_values: torch.Tensor, grid_thw, feats: torch.Tensor, counts, budget: int,
                  cfg: dict | None = None, *, merge: int = 2, patch: int = 16, temporal: int = 2) -> dict:
    """One Reasoner prefill: feats [N, d] = full-vision merger output (one row per merged token, same
    order as the pixel merge groups), counts per crop. Returns the secld_select.select info dict
    (+ "mode"). mode nova = secld_select.select verbatim; logdet = e_i == 1 through the same greedy;
    etopk = plain e ranking under the same budget / minimum."""
    cfg = config() if cfg is None else cfg
    mode = cfg["mode"]
    if mode == "nova":
        info = _sec.select(pixel_values, grid_thw, feats, counts, int(budget), cfg,
                           merge=merge, patch=patch, temporal=temporal)
        info["mode"] = mode
        return info
    t0 = time.time()
    counts = [int(c) for c in counts]
    N = int(feats.shape[0])
    if sum(counts) != N:
        raise ValueError(f"token_counts {sum(counts)} != feats {N}")
    ev = _sec.evidence(pixel_values, grid_thw, cfg, merge=merge, patch=patch, temporal=temporal)
    e = ev["e"].to(feats.device)
    if int(e.numel()) != N:
        raise ValueError(f"evidence cells {int(e.numel())} != feats {N}")
    z = F.normalize(feats.to(torch.float32), dim=-1)
    support = torch.repeat_interleave(torch.arange(len(counts), device=feats.device),
                                      torch.tensor(counts, dtype=torch.long, device=feats.device))
    B = max(0, min(int(budget), N))
    mn = int(cfg.get("min_per_support", 0))
    if mode == "logdet":
        order, gains, d2s = _sec.support_logdet_greedy(z, support, B, first_per_support=mn)
        nov = [d - 1.0 for d in d2s]                          # e == 1 -> novelty = z^T M^-1 z
    elif mode == "etopk":
        order = torch.tensor(_etopk_order(e, support, B, mn), dtype=torch.long)
        gains, nov = [], []
    else:
        raise ValueError(mode)
    keep = torch.zeros(N, dtype=torch.bool, device=feats.device)
    keep[order.to(feats.device)] = True
    per = [int(keep[support == g].sum()) for g in range(len(counts))]
    e_cpu, b_cpu, m_cpu = e.detach().cpu(), ev["b"].detach().cpu(), ev["m"].detach().cpu()
    sel = order.cpu()
    e_sel = e_cpu[sel]
    top_e = torch.topk(e_cpu, B).indices if B > 0 else torch.empty(0, dtype=torch.long)
    erank_overlap = float(keep.cpu()[top_e].float().mean()) if B > 0 else float("nan")
    nov_t = torch.tensor(nov, dtype=torch.float64)
    k10 = max(1, B // 10)
    nan = float("nan")
    return {
        "keep": keep, "order": order, "gains": gains, "per_support": per, "budget": B, "n": N,
        "e": e_cpu, "b": b_cpu, "m": m_cpu, "mode": mode,
        "empty_crops": sum(1 for v in per if v == 0),
        "e_all_mean": float(e_cpu.mean()), "e_all_lt01": float((e_cpu < 0.1).float().mean()),
        "m_all_frac": float(m_cpu.mean()), "m_sel_frac": float(m_cpu[sel].mean()) if B else nan,
        "e_sel_mean": float(e_sel.mean()) if B else nan,
        "e_sel_min": float(e_sel.min()) if B else nan,
        "b_sel_mean": float(b_cpu[sel].mean()) if B else nan,
        "gain_first": float(sum(gains[:k10]) / k10) if gains else nan,
        "gain_last": float(sum(gains[-k10:]) / k10) if gains else nan,
        "nov_first": float(nov_t[:k10].mean()) if nov_t.numel() else nan,
        "nov_last": float(nov_t[-k10:].mean()) if nov_t.numel() else nan,
        "nov_min": float(nov_t.min()) if nov_t.numel() else nan,
        "erank_overlap": erank_overlap,
        "sec": time.time() - t0,
    }


# --------------------------------------------------------------------------- runtime


@torch.no_grad()
def select_vision_features(bb, pixel_values, grid_thw):
    """Full vision tower + NOVA. Returns (retained merged features [B, d], keep mask [N]); same
    contract and survivor bookkeeping as qasc_select / ssda_select.select_vision_features."""
    t0 = time.time()
    cfg = config()
    feats = bb.vlm.get_image_features(pixel_values, grid_thw, return_dict=True).pooler_output
    feats = torch.cat(list(feats), dim=0) if isinstance(feats, (list, tuple)) else feats
    unit = int(bb.visual.spatial_merge_unit)
    merge = int(round(math.sqrt(unit)))
    thw = [tuple(int(v) for v in row) for row in torch.as_tensor(grid_thw).tolist()]
    counts = [t * h * w // unit for t, h, w in thw]
    N = int(feats.shape[0])
    if sum(counts) != N:
        raise RuntimeError(f"NOVA: grid tokens {sum(counts)} != features {N}")
    pv = getattr(bb, "_secld_pixel_values_fp32", None)
    src = "fp32"
    if pv is None or tuple(pv.shape) != tuple(pixel_values.shape):
        pv, src = pixel_values, str(pixel_values.dtype).replace("torch.", "")
    budget = budget_for(N, cfg["keep"])
    info = select_tokens(pv.to(feats.device), grid_thw, feats, counts, budget, cfg, merge=merge)
    keep = info["keep"].to(feats.device)
    kept_cpu = keep.detach().cpu()
    image_of = torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(counts, dtype=torch.long))
    bb._prefill_prune_survivor_scores = info["e"].to(torch.float32)[kept_cpu]
    bb._prefill_prune_survivor_group_ids = torch.nonzero(kept_cpu, as_tuple=True)[0]
    bb._prefill_prune_survivor_image_ids = image_of[kept_cpu]
    mags = list(getattr(bb, "_prune_image_magnifications", ()) or ())
    print(f"[NOVA] {_sec.summary(info)} mode={cfg['mode']} keep_ratio={cfg['keep']:.3f} mags={mags} "
          f"pixels={src} total_sec={time.time() - t0:.2f}", flush=True)
    return feats[keep], keep


__all__ = ["MODES", "enabled", "config", "budget_for", "select_tokens", "select_vision_features"]
