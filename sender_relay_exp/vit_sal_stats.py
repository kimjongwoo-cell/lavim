"""Position-bias audit statistics for the vision-encoder salience a_i (numpy only).

a_i (Notion "Question-Agnostic Salience-Coverage Log-Det Selection", C1 #14) is the
attention a merged visual token receives from the other tokens of its own crop inside the
frozen vision tower, averaged over heads and queries. The Validation paragraph asks, before
a_i may enter F(S)=log det(I + sum a_i z_i z_i^T), for the mean 5x / 20x heatmaps and the
correlation of a_i with token index and with distance to the crop centre.

Everything here is per crop: the vision tower's attention is block-diagonal over crops
(cu_seqlens), so a_i sums to 1 inside every crop and says nothing about allocation across
crops. Maps are rescaled by the crop's token count so that uniform attention reads 1.0.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def average_ranks(v) -> np.ndarray:
    """0-based ranks; tied values share their average rank."""
    v = np.asarray(v, dtype=float).ravel()
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(v.size, dtype=float)
    ranks[order] = np.arange(v.size, dtype=float)
    _, inverse = np.unique(v, return_inverse=True)
    sums = np.bincount(inverse, weights=ranks)
    counts = np.bincount(inverse)
    return (sums / counts)[inverse]


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return None if den == 0.0 else float(a @ b) / den


def spearman(x, y) -> float | None:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError(f"length mismatch {x.size} vs {y.size}")
    if x.size < 3:
        return None
    return _corr(average_ranks(x), average_ranks(y))


def partial_spearman(x, y, z) -> float | None:
    """Spearman(x, y) after regressing the rank of z out of both ranks.

    Used to ask whether a centre effect survives once tissue content is held fixed
    (tissue tends to sit in the middle of a tissue-selected crop).
    """
    rx, ry, rz = (average_ranks(np.asarray(a, dtype=float)) for a in (x, y, z))
    rz = rz - rz.mean()
    zz = float(rz @ rz)
    if zz == 0.0:
        return spearman(x, y)

    def resid(r):
        r = r - r.mean()
        return r - (float(r @ rz) / zz) * rz

    return _corr(resid(rx), resid(ry))


def grid_coords(h: int, w: int) -> dict[str, np.ndarray]:
    """Raster token index, row, col and Euclidean distance to the grid centre."""
    idx = np.arange(h * w)
    row, col = np.divmod(idx, w)
    dist = np.hypot(row - (h - 1) / 2.0, col - (w - 1) / 2.0)
    return {"idx": idx.astype(float), "row": row.astype(float),
            "col": col.astype(float), "dist": dist}


def crop_spans(thw, merge: int = 2) -> list[tuple[int, int, int, int]]:
    """image_grid_thw -> [(start, end, H, W)] over the merged-token sequence."""
    spans, start = [], 0
    for t, h, w in np.asarray(thw, dtype=int).tolist():
        hh, ww = h // merge, w // merge
        n = t * hh * ww
        spans.append((start, start + n, hh, ww))
        start += n
    return spans


def crop_map(values, span) -> np.ndarray:
    """Slice one crop and rescale so uniform attention = 1."""
    a, b, _, _ = span
    seg = np.asarray(values, dtype=float)[a:b]
    total = seg.sum()
    return seg * (seg.size / total) if total > 0 else seg


def centre_border_ratio(m2d) -> float:
    """Mean of the central half-square over mean of the outermost ring."""
    m = np.asarray(m2d, dtype=float)
    h, w = m.shape
    inner = m[h // 4: h - h // 4, w // 4: w - w // 4].mean()
    ring = np.ones_like(m, dtype=bool)
    ring[1:-1, 1:-1] = False
    return float(inner / m[ring].mean())


def loo_template_r(maps, groups) -> list[float | None]:
    """Spearman of each map with the mean of all maps from OTHER groups (cases).

    High values mean a fixed spatial template, the same whatever the tissue.
    Leaving out the whole case avoids crediting two crops of one slide.
    """
    maps = np.asarray(maps, dtype=float).reshape(len(groups), -1)
    groups = np.asarray(groups)
    out: list[float | None] = []
    for i in range(len(groups)):
        other = groups != groups[i]
        if not other.any():
            out.append(None)
            continue
        out.append(spearman(maps[i], maps[other].mean(axis=0)))
    return out


def tissue_blocks(image: Image.Image, h: int, w: int, patch: int = 32) -> np.ndarray:
    """Per merged token: fraction of pixels with HSV saturation >= 26 (geometry.tissue_fraction rule)."""
    img = image.convert("RGB").resize((w * patch, h * patch), Image.Resampling.BICUBIC)
    sat = np.asarray(img.convert("HSV"))[..., 1] >= 26
    return sat.reshape(h, patch, w, patch).mean(axis=(1, 3)).ravel()


def summarize(values) -> dict:
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if v.size == 0:
        return {"n": 0}
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return {"n": int(v.size), "median": float(med), "q1": float(q1), "q3": float(q3),
            "mean": float(v.mean()), "frac_neg": float((v < 0).mean())}
