"""Level 3/4 probe: does the Reasoner state depend on the slide, and does the
Answerer use it?

Level 1 (attention mass) and Level 2 (read contribution) are measured inside
memory/access_diag.py. Those say how much visual gets INTO the Reasoner. They
cannot say whether what got in is slide-specific, nor whether the Answerer
consumes it. This module supplies the two remaining levels:

    Level 3  state dependence   1 - cos(R_true[t], R_wrong[t]) per latent step
    Level 4  causal answer use  answer flip / accuracy delta between the arms

Both come from running the SAME cases twice and comparing offline:

    arm A  normal                      -> z_1..z_m, answer
    arm B  VLMAS_RPATH=wrong_visual    -> z_1..z_m, answer

Arm B freezes the first case's visual K/V as a donor and substitutes it at the
Reasoner boundary, so every later case reasons over another slide's evidence
while its question, prompt and cache geometry stay identical. If Level 3 is
~0 the Reasoner never encoded slide identity; if Level 3 is large but Level 4
is ~0 we are in the third cell of the diagnostic table -- Reasoner visually
dependent, Answerer visually insensitive.

Env:
    VLMAS_RPATH=wrong_visual   swap visual K/V for a frozen donor at the
                               Reasoner boundary (donor case itself unswapped)
    VLMAS_DUMP_ZALL=<dir>      dump all m latent states per case as .pt
"""
from __future__ import annotations

import os
from pathlib import Path

import torch


def wrong_visual_enabled() -> bool:
    return os.environ.get("VLMAS_RPATH", "").strip() == "wrong_visual"


def dump_dir() -> str:
    return os.environ.get("VLMAS_DUMP_ZALL", "").strip()


@torch.no_grad()
def swap_visual_for_donor(backbone, cache, visual_columns) -> str:
    """Replace the visual K/V columns with a frozen donor slide's.

    The first case that reaches this captures its own visual K/V as the donor
    and is left untouched (exclude it from analysis, exactly as the restage
    wrong-slide control does). Later cases get the donor written in.
    """
    if not wrong_visual_enabled() or visual_columns is None:
        return ""
    cols = visual_columns.to(device=backbone.device, dtype=torch.long)
    n = int(cols.numel())
    if n == 0:
        return ""
    donor = getattr(backbone, "_rpath_donor", None)
    if donor is None:
        backbone._rpath_donor = {
            index: (
                layer.keys[:, :, cols, :].detach().clone(),
                layer.values[:, :, cols, :].detach().clone(),
            )
            for index, layer in enumerate(cache.layers)
        }
        backbone._rpath_is_donor_case = True
        return " (donor case: own visual, NOT swapped)"
    backbone._rpath_is_donor_case = False
    for index, layer in enumerate(cache.layers):
        if index not in donor:
            continue
        donor_k, donor_v = donor[index]
        take = min(n, int(donor_k.shape[2]))
        target = cols[:take]
        layer.keys.index_copy_(2, target, donor_k[:, :, :take, :].to(layer.keys.dtype))
        layer.values.index_copy_(
            2, target, donor_v[:, :, :take, :].to(layer.values.dtype))
    return f" (WRONG-VISUAL donor, n={n})"


def dump_latent_states(backbone, trajectory: list) -> None:
    """Persist z_1..z_m so Level 3 can compare arms case by case."""
    target = dump_dir()
    if not target or not trajectory:
        return
    root = Path(target)
    root.mkdir(parents=True, exist_ok=True)
    index = getattr(backbone, "_zall_index", 0)
    backbone._zall_index = index + 1
    stacked = torch.stack([t.detach().float().cpu().reshape(-1) for t in trajectory])
    torch.save(stacked, root / f"z_{index:04d}.pt")


def state_dependence(true_path: str, wrong_path: str) -> dict:
    """Offline Level 3: per-step 1 - cos between the two arms' latent states."""
    a = torch.load(true_path, map_location="cpu")
    b = torch.load(wrong_path, map_location="cpu")
    steps = min(int(a.shape[0]), int(b.shape[0]))
    values = []
    for t in range(steps):
        x, y = a[t].float(), b[t].float()
        denom = (x.norm() * y.norm()).clamp_min(1e-9)
        values.append(float(1.0 - (x @ y) / denom))
    return {"steps": steps, "per_step": values,
            "mean": sum(values) / len(values) if values else 0.0}


__all__ = [
    "dump_latent_states", "dump_dir", "state_dependence",
    "swap_visual_for_donor", "wrong_visual_enabled",
]
