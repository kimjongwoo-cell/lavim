"""Sender-priority relay destination weights (minimal experiment module).

Builds the Answerer-handoff destination distribution for the reallocation
variants compared in the sender-semantic-priority experiment:

  uniform                  w_r = 1 / N_visual
  sender_priority          w_r ∝ clamp(sender_priority_r, 0)
  shuffled_sender_priority w_r ∝ permute(clamp(sender_priority, 0))_r

The canonical "key_norm" source never reaches this module: the engine passes
no override and the reallocation hook keeps its existing behaviour, so the
canonical path stays byte-identical.
"""

from __future__ import annotations

import zlib

import torch

STATIC_RELAY_SOURCES = (
    "uniform",
    "sender_priority",
    "shuffled_sender_priority",
)

_EPS = 1e-12


def deterministic_shuffle_seed(base_seed: int, columns: torch.Tensor) -> int:
    """Case-deterministic seed: global seed mixed with the visual column set."""
    payload = ",".join(str(int(c)) for c in columns.tolist()).encode("ascii")
    return (int(base_seed) * 1_000_003 + zlib.crc32(payload)) % (2**31)


def build_destination_weights(
    *,
    relay_source: str,
    columns: torch.Tensor,
    sender_priority: torch.Tensor,
    base_seed: int,
) -> tuple[torch.Tensor | None, dict]:
    """Return (normalized weights aligned to `columns`, audit dict).

    Weights are None when the request cannot be honored (missing/degenerate
    priority for the priority-based sources); the audit dict then records the
    explicit fallback so the caller can revert to the canonical destination.
    """
    count = int(columns.numel())
    audit: dict = {
        "relay_source": relay_source,
        "n_visual": count,
        "fallback_used": False,
        "fallback_reason": None,
        "shuffle_seed": None,
    }
    if count == 0:
        audit["fallback_used"] = True
        audit["fallback_reason"] = "no_visual_columns"
        return None, audit
    if relay_source == "uniform":
        weights = torch.full((count,), 1.0 / count, dtype=torch.float32)
        audit["weights_norm"] = weights.tolist()
        return weights, audit
    if relay_source not in ("sender_priority", "shuffled_sender_priority"):
        audit["fallback_used"] = True
        audit["fallback_reason"] = f"unknown_relay_source:{relay_source}"
        return None, audit
    if sender_priority.numel() != count:
        audit["fallback_used"] = True
        audit["fallback_reason"] = (
            f"priority_length_mismatch:{int(sender_priority.numel())}!={count}"
        )
        return None, audit
    weights = sender_priority.detach().float().cpu().clamp_min(0.0)
    total = float(weights.sum())
    if not torch.isfinite(weights).all() or total <= _EPS:
        audit["fallback_used"] = True
        audit["fallback_reason"] = "degenerate_priority"
        return None, audit
    if relay_source == "shuffled_sender_priority":
        seed = deterministic_shuffle_seed(base_seed, columns)
        generator = torch.Generator().manual_seed(seed)
        weights = weights[torch.randperm(count, generator=generator)]
        audit["shuffle_seed"] = seed
    weights = weights / (weights.sum() + _EPS)
    audit["weights_norm"] = weights.tolist()
    return weights, audit


def rank_correlation(a: torch.Tensor, b: torch.Tensor) -> float | None:
    """Spearman rank correlation without scipy (average-free simple ranks)."""
    if a.numel() != b.numel() or a.numel() < 2:
        return None
    ar = torch.argsort(torch.argsort(a)).float()
    br = torch.argsort(torch.argsort(b)).float()
    ac = ar - ar.mean()
    bc = br - br.mean()
    denom = float(ac.norm() * bc.norm())
    if denom <= _EPS:
        return None
    return float((ac @ bc) / denom)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float | None:
    if a.numel() != b.numel() or a.numel() == 0:
        return None
    denom = float(a.float().norm() * b.float().norm())
    if denom <= _EPS:
        return None
    return float(a.float() @ b.float() / denom)
