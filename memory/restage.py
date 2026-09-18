"""KV re-staging helpers: move cached visual K/V columns to recency by
re-rotating their RoPE phase (position test of the refeed effect — same
content, same computation, new position; no re-encoding, no text).

The delta rotation is computed with the angle-subtraction identity on the
model's OWN cos/sin tables, elementwise per rotary dim:
    cos(b-a) = cos(b)cos(a) + sin(b)sin(a)
    sin(b-a) = sin(b)cos(a) - cos(b)sin(a)
so the MRoPE section interleave never has to be reconstructed here — whatever
layout the rotary embedding emits, the delta stays in the same layout.
Rotations about the same axis compose additively, hence
    R(b) k_raw = R(b-a) (R(a) k_raw) = R(b-a) k_cached.
"""
from __future__ import annotations

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def delta_cos_sin(
    cos_old: torch.Tensor,
    sin_old: torch.Tensor,
    cos_new: torch.Tensor,
    sin_new: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Elementwise cos/sin of (new - old) from the two rotation tables."""
    cos_delta = cos_new * cos_old + sin_new * sin_old
    sin_delta = sin_new * cos_old - cos_new * sin_old
    return cos_delta, sin_delta


def rerotate_keys(
    keys: torch.Tensor,
    cos_delta: torch.Tensor,
    sin_delta: torch.Tensor,
) -> torch.Tensor:
    """Apply the delta rotation to cached keys.

    keys [B, H, n, D]; cos/sin [B, n, D] (or [n, D]) — broadcast over heads.
    """
    if cos_delta.dim() == 2:
        cos_delta = cos_delta.unsqueeze(0)
        sin_delta = sin_delta.unsqueeze(0)
    cos_delta = cos_delta.unsqueeze(1).to(keys.dtype)
    sin_delta = sin_delta.unsqueeze(1).to(keys.dtype)
    return keys * cos_delta + rotate_half(keys) * sin_delta
