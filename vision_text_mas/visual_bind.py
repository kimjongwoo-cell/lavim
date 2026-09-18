"""Visual Binding Step helpers (env-gated experiment; off = no-op).

The last Reasoner latent step z_m can be turned into a Visual Binding Step:
at selected decoder layers its attention source is RESTRICTED to the WSI
visual KV columns (plus the token itself), so the latent state appended to
the cache — the message the Answerer consumes through the existing handoff —
is forced to carry visual evidence. No extra tokens, no refeed, no projector:
only the generation of the final existing latent step changes.

Pure helpers live here so they can be unit-tested without a model. The
context manager that installs the per-layer attention-mask hooks lives in
backbone/qwen3vl.py (_visual_bind_step).

Env contract:
  VLMAS_VISUAL_BIND=1                 enable (anything else = off)
  VLMAS_VISUAL_BIND_LAYERS=18        comma list of layer indices; 'fX' entries
                                      are fractions of depth (f0.5 -> mid);
                                      empty/invalid -> [num_layers // 2]
  VLMAS_VISUAL_BIND_MODE=matched     matched | wrong_slide | random_v
"""
from __future__ import annotations

import torch

BIND_MODES = ("matched", "wrong_slide", "shuffled_v", "random_v")


def shuffle_permutation(n: int, seed: int) -> torch.Tensor:
    """Deterministic permutation of n visual columns (shuffled-V control)."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randperm(n, generator=generator)


def parse_bind_layers(spec: str, num_layers: int) -> list[int]:
    """Parse the layer list; fractions allowed; 'all' = every decoder layer
    (the paper-method Sec. 3.4 setting — no layer-selection hyperparameter);
    fallback = middle layer."""
    if (spec or "").strip().lower() == "all":
        return list(range(num_layers))
    out: set[int] = set()
    for token in (spec or "").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            if token.startswith("f"):
                fraction = float(token[1:])
                index = round(fraction * (num_layers - 1))
            else:
                index = int(token)
        except ValueError:
            continue
        if 0 <= index < num_layers:
            out.add(index)
    if not out:
        out = {num_layers // 2}
    return sorted(out)


def build_bind_mask(
    kv_len: int,
    vis_cols: torch.Tensor,
    self_col: int | None,
    dtype: torch.dtype,
    device,
) -> torch.Tensor:
    """[1,1,1,kv_len] additive mask: 0 at visual cols (+ self when self_col is
    not None), -inf elsewhere. self_col=None is the spec-faithful Sec. 3.4.2
    read (softmax normalized over R* only)."""
    mask = torch.full(
        (1, 1, 1, kv_len), torch.finfo(dtype).min, dtype=dtype, device=device
    )
    mask[..., vis_cols] = 0
    if self_col is not None:
        mask[..., self_col] = 0
    return mask


def matched_noise_like(values: torch.Tensor, seed: int) -> torch.Tensor:
    """Deterministic Gaussian with the tensor's scalar mean/std (shape control)."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    noise = torch.randn(values.shape, generator=generator, dtype=torch.float32)
    stats = values.float()
    noise = noise * stats.std().item() + stats.mean().item()
    return noise.to(device=values.device, dtype=values.dtype)
