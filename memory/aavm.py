"""AAVM — Acquisition-Addressed Visual Memory (method draft 0908).

Each retained crop p is re-addressed with the acquisition event g(p) that
retrieved it: the evidence-query token span T_g whose key centroid becomes the
crop's retrieval address. Values are never touched.

    a_g   = mean over T_g of pre-RoPE keys          (acquisition address, v1)
            -- v2 (VLMAS_AAVM=2) uses the pre-RoPE QUERY representation
               u_g = mean over T_g of q_pre instead: the actual retrieval
               direction at acquisition time, GQA-reduced to the KV heads
               (reduce_query_heads). Everything below is shared by v1/v2.
    k_p   = mean over crop p of pre-RoPE keys       (current visual centroid)
    c_p   = ||k_p|| * a_g / ||a_g||                 (direction swapped, norm kept)
    dk_p  = c_p - k_p                               (minimum-perturbation shift)

`dk_p` is the exact solution of  min ||K~ - K||_F  s.t. mean(K~) = c_p, so the
within-crop key geometry is preserved exactly:  k~_i - k~_j = k_i - k_j.

Position handling (the part the method text leaves implicit): the address is
computed in the PRE-RoPE (position-free) space so the direction carries
semantics rather than the phase of wherever the query tokens happened to sit.
The shift is then rotated ONCE by the crop's representative position and added
uniformly to the cached (post-RoPE) keys, which keeps the crop-uniform
addressing factor exp[q^T dk_p] exact for every token of the crop — adding a
pre-RoPE shift before re-rotation would instead make it vary token by token.
"""
from __future__ import annotations

import torch

from memory.restage import rotate_half


def crop_spans(kept_counts: list[int]) -> list[tuple[int, int]]:
    """Half-open (start, end) of each crop's block on the kept token axis."""
    spans: list[tuple[int, int]] = []
    offset = 0
    for count in kept_counts:
        spans.append((offset, offset + count))
        offset += count
    return spans


def address_shift(
    crop_keys_raw: torch.Tensor,
    address_raw: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """dk = c - k_bar for one crop, per head.

    crop_keys_raw [H, n, D] pre-RoPE keys of the crop; address_raw [H, D] the
    acquisition-event key centroid. Returns [H, D].
    """
    if crop_keys_raw.dim() != 3 or address_raw.dim() != 2:
        raise ValueError("crop keys must be [H, n, D] and address [H, D]")
    if crop_keys_raw.shape[0] != address_raw.shape[0]:
        raise ValueError("head counts must match")
    if crop_keys_raw.shape[-1] != address_raw.shape[-1]:
        raise ValueError("head dims must match")
    centroid = crop_keys_raw.mean(dim=1)                      # [H, D]
    centroid_norm = centroid.norm(dim=-1, keepdim=True)       # [H, 1]
    address_norm = address_raw.norm(dim=-1, keepdim=True)     # [H, 1]
    target = centroid_norm * address_raw / address_norm.clamp_min(eps)
    return target - centroid


def reduce_query_heads(
    query_heads: torch.Tensor,
    num_kv_heads: int,
) -> torch.Tensor:
    """GQA reduction [H_q, D] -> [H_kv, D]: mean over each KV head's q-group.

    Under grouped-query attention every KV head serves H_q/H_kv query heads;
    the event address that KV head is asked to serve is the mean of the
    retrieval directions of exactly those query heads.
    """
    if query_heads.dim() != 2:
        raise ValueError("query heads must be [H_q, D]")
    h_q, dim = query_heads.shape
    if num_kv_heads <= 0 or h_q % num_kv_heads != 0:
        raise ValueError(f"H_q={h_q} not divisible by H_kv={num_kv_heads}")
    return query_heads.view(num_kv_heads, h_q // num_kv_heads, dim).mean(dim=1)


def rotate_vector(
    vector: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Rotate a per-head vector [H, D] by one position's cos/sin [D]."""
    cos = cos.to(vector.dtype).unsqueeze(0)
    sin = sin.to(vector.dtype).unsqueeze(0)
    return vector * cos + rotate_half(vector) * sin


def apply_crop_shift(
    keys: torch.Tensor,
    span: tuple[int, int],
    shift_rotated: torch.Tensor,
) -> torch.Tensor:
    """Add one rotated shift [H, D] to every key of a crop block, in place.

    keys [B, H, N, D] cached keys; span indexes the crop on the N axis.
    """
    start, end = span
    keys[:, :, start:end, :] = (
        keys[:, :, start:end, :] + shift_rotated.unsqueeze(0).unsqueeze(2)
    ).to(keys.dtype)
    return keys


def shuffled_addresses(
    addresses: list[torch.Tensor],
    seed: int = 42,
) -> list[torch.Tensor]:
    """Falsification control: give each crop another crop's address.

    A derangement when possible, so no crop keeps its own acquisition address.
    """
    import random

    count = len(addresses)
    if count < 2:
        return list(addresses)
    order = list(range(count))
    rng = random.Random(seed)
    for _ in range(64):
        rng.shuffle(order)
        if all(order[i] != i for i in range(count)):
            break
    else:
        order = order[1:] + order[:1]
    return [addresses[order[i]] for i in range(count)]
