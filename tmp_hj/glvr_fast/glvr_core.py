"""GLVR canonical-A core math (model-free).

All functions take plain tensors so they can be unit tested without the pipeline.
Shapes follow the Answerer terminal row at one layer:
  logits : [H, N]   attention logits of the terminal query over the cache (pre-softmax, scaled)
  values : [H, N, D] value vectors already repeat_kv'd to query heads
Groups are index tensors into the cache axis.

Definitions (Notion GLVR Canonical A, 2026-09-18):
  rho_G   = sum_{c in G} alpha_c                      native source mass of group G
  m_G     = sum_{c in G} alpha_c V_c                  native source message of group G
  g_k     = alpha_{R_k} / rho_R                       conditional step distribution over the m latent slots
  u_V,k   = sum_{j in V} P^{R->V}_{k,j} V_j           conditional visual read of grounded latent k (from replay)
  u_relay = g^T U_V
  o'      = o_nat + lambda * (rho_B * u_relay - m_B)  B = donor domain (canonical: H_C)
"""
from __future__ import annotations

import torch


def attn_weights(logits: torch.Tensor) -> torch.Tensor:
    """[H, N] logits -> [H, N] softmax weights in fp32."""
    return torch.softmax(logits.float(), dim=-1)


def group_mass_message(alpha: torch.Tensor, values: torch.Tensor,
                       idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Native mass and message of one provenance group.

    alpha [H, N], values [H, N, D], idx [k] -> (rho [H], m [H, D]).
    """
    if idx.numel() == 0:
        h = alpha.shape[0]
        return (alpha.new_zeros(h), values.new_zeros(h, values.shape[-1]).float())
    a = alpha[:, idx]                                   # [H, k]
    v = values[:, idx, :].float()                       # [H, k, D]
    return a.sum(-1), torch.einsum("hk,hkd->hd", a, v)


def step_distribution(alpha: torch.Tensor, latent_idx: torch.Tensor,
                      eps: float = 1e-12) -> tuple[torch.Tensor, torch.Tensor]:
    """Answerer -> grounded latent conditional weights.

    Returns (rho_R [H], g [H, m]).  g sums to one per head; if the latent mass is
    numerically zero the weights fall back to uniform so the relay stays defined.
    """
    a = alpha[:, latent_idx]                            # [H, m]
    rho = a.sum(-1)                                     # [H]
    g = a / rho.clamp_min(eps).unsqueeze(-1)
    flat = rho <= eps
    if bool(flat.any()):
        g = torch.where(flat.unsqueeze(-1), torch.full_like(g, 1.0 / a.shape[-1]), g)
    return rho, g


def conditional_visual_read(logits: torch.Tensor, values: torch.Tensor,
                            vis_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One replay row -> (rho_V [H], u_V [H, D]).

    rho_V is the row's visual mass over the *full* causal prefix; u_V is the
    within-visual conditional read, i.e. m_V / rho_V.
    """
    alpha = attn_weights(logits)
    rho, m = group_mass_message(alpha, values, vis_idx)
    u = m / rho.clamp_min(1e-12).unsqueeze(-1)
    return rho, u


def relay_message(g: torch.Tensor, U_V: torch.Tensor) -> torch.Tensor:
    """g [H, m], U_V [H, m, D] -> u_relay [H, D]."""
    return torch.einsum("hm,hmd->hd", g.float(), U_V.float())


def apply_relay(o_nat: torch.Tensor, rho_B: torch.Tensor, m_B: torch.Tensor,
                u_relay: torch.Tensor, lam: float) -> torch.Tensor:
    """Source-local interpolation of the donor branch only.

    o_nat [H, D] pre-o_proj head outputs of the terminal row, rho_B [H], m_B [H, D].
    lam = 0 returns o_nat unchanged; lam = 1 replaces the donor content entirely.
    """
    delta = rho_B.unsqueeze(-1) * u_relay - m_B
    return o_nat.float() + float(lam) * delta


def native_head_output(alpha: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Reference implementation of the native attention output [H, D]."""
    return torch.einsum("hn,hnd->hd", alpha, values.float())


def relay_weight_view(alpha: torch.Tensor, donor_idx: torch.Tensor,
                      vis_idx: torch.Tensor, gamma: torch.Tensor,
                      lam: float) -> torch.Tensor:
    """Token-level view of the same operator, used only to verify apply_relay.

    gamma [H, m, |vis|] holds the replay conditional distributions P^{R->V} of the
    m grounded latents, already mixed by g outside this helper (so gamma here is
    [H, |vis|]).  The donor group keeps its native mass but loses lam of its
    content, which is handed to the visual columns.
    """
    w = alpha.clone().float()
    rho_B = w[:, donor_idx].sum(-1)                     # [H]
    w[:, donor_idx] = w[:, donor_idx] * (1.0 - lam)
    w[:, vis_idx] = w[:, vis_idx] + lam * rho_B.unsqueeze(-1) * gamma.float()
    return w
