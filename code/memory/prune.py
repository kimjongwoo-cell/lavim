"""Safe visual-KV pruning — drop only confidently-useless vision tokens.

RVIS (arXiv:2604.12358) shows top-k saliency pruning fails on reasoning because
the relevant visual region SHIFTS during decoding. But that shift happens among
TISSUE regions, never to glass/background. So dropping only (background AND
low-saliency) + (attention-sink/register) tokens compresses the visual KV
without the RVIS failure mode — every tissue/diagnostic token is kept.

This is the *safe* alternative to method.md's intra-pass top-k drop (which is
kept only as an ablation baseline to reproduce the RVIS degradation).
"""

from __future__ import annotations

import copy
import math
from typing import assert_never

import numpy as np
import torch

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


def safe_prune_mask(
    tissue_frac: np.ndarray,        # [nV] per-token tissue fraction (0..1)
    saliency: np.ndarray,           # [nV] per-token saliency (higher = important)
    sink_map: np.ndarray | None,    # [nV] universal-attention (higher = register/sink)
    *,
    tissue_th: float = 0.1,         # < this => background candidate
    sal_lo_pct: float = 50.0,       # background dropped only if saliency <= this pctile
    sink_frac: float = 0.05,        # top fraction of universal-attn treated as sink
    keep_salient_pct: float = 90.0, # never drop tokens above this saliency pctile
) -> np.ndarray:
    """Return boolean KEEP mask over the nV vision tokens.

    drop = (background AND low-saliency) OR (sink/register), minus a top-saliency
    rescue so a salient-but-background token is never dropped."""
    nV = len(tissue_frac)
    sal = np.asarray(saliency, float)

    is_bg = np.asarray(tissue_frac, float) < tissue_th
    is_low_sal = sal <= np.percentile(sal, sal_lo_pct)

    if sink_map is not None and len(sink_map) == nV:
        k = max(1, int(round(sink_frac * nV)))
        sink_thresh = np.sort(np.asarray(sink_map, float))[-k]
        is_sink = np.asarray(sink_map, float) >= sink_thresh
    else:
        is_sink = np.zeros(nV, dtype=bool)

    drop = (is_bg & is_low_sal) | is_sink
    # rescue: never drop a clearly-salient token (even if it reads as background)
    rescue = sal >= np.percentile(sal, keep_salient_pct)
    drop = drop & ~rescue
    return ~drop   # keep mask


# ── inputs for the keep mask ────────────────────────────────────────────────────
def universal_attention(q2v: dict, layers: list[int]) -> np.ndarray | None:
    """Mean q2v over heads & query rows over layers -> [nV]; high = register/sink."""
    acc = []
    for li in layers:
        if li not in q2v:
            continue
        a = np.asarray(q2v[li].float())                       # [H, nq, nV]
        a = a / np.clip(a.sum(-1, keepdims=True), 1e-9, None)
        acc.append(a.mean(axis=(0, 1)))
    return np.mean(acc, 0) if acc else None


def token_tissue_fraction(images: list, roi_spans: list) -> np.ndarray:
    """Per-token tissue fraction [nV] from an HSV-saturation tissue mask per crop."""
    fracs = []
    for r in roi_spans:
        img = images[r.id] if r.id < len(images) else images[0]
        arr = np.array(img.convert("RGB"))
        if _HAS_CV2:
            sat = cv2.medianBlur(cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)[:, :, 1], 7)
            mask = sat > 8
        else:
            a = arr.astype(float); mx = a.max(2); mn = a.min(2)
            mask = (np.where(mx > 0, (mx - mn) / np.clip(mx, 1, None), 0) * 255) > 8
        H, W = mask.shape
        for i in range(r.h):
            for j in range(r.w):
                y0, y1 = int(i * H / r.h), int((i + 1) * H / r.h)
                x0, x1 = int(j * W / r.w), int((j + 1) * W / r.w)
                fracs.append(mask[y0:y1, x0:x1].mean())
    return np.array(fracs)


def token_morphology_strata(images: list, roi_spans: list) -> np.ndarray:
    """Assign H&E-informed morphology proxy strata to visual tokens.

    The categories are deliberately image-derived rather than a diagnostic label:
    1=nuclear-rich, 2=eosin/stroma-rich, 3=low-cellularity/necrotic-like.  They
    provide a deterministic diversity constraint for Pruning v2 and never claim a
    tumour/stroma/necrosis diagnosis.  Background tiles are assigned stratum 0.
    """
    strata: list[np.ndarray] = []
    for span in roi_spans:
        image = images[span.id] if span.id < len(images) else images[0]
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        brightness = rgb.mean(axis=2)
        chroma = rgb.max(axis=2) - rgb.min(axis=2)
        nuclear = np.clip(blue - red + (1.0 - brightness) * 0.35, 0.0, None)
        eosin = np.clip(red - blue + chroma * 0.25, 0.0, None)
        low_cellularity = np.clip(1.0 - chroma - nuclear * 0.35 - eosin * 0.20, 0.0, None)
        score = np.stack((nuclear, eosin, low_cellularity), axis=-1)
        token_strata = np.zeros((span.h, span.w), dtype=np.int8)
        height, width = score.shape[:2]
        for row in range(span.h):
            for column in range(span.w):
                y0, y1 = int(row * height / span.h), int((row + 1) * height / span.h)
                x0, x1 = int(column * width / span.w), int((column + 1) * width / span.w)
                tile = score[y0:y1, x0:x1]
                if tile.size != 0:
                    token_strata[row, column] = int(tile.mean(axis=(0, 1)).argmax()) + 1
        strata.append(token_strata.reshape(-1))
    return np.concatenate(strata) if strata else np.zeros(0, dtype=np.int8)


def flatten_saliency(saliency: dict, roi_spans: list) -> np.ndarray:
    """{roi_id: [h,w]} -> [nV] in ROI/crop order."""
    out = [np.asarray(saliency[r.id]).reshape(-1) for r in roi_spans if r.id in saliency]
    return np.concatenate(out) if out else np.zeros(0)


def roi_keep_mask(grid: np.ndarray, b_span: int = 32, radius: int = 1) -> np.ndarray:
    """method.md Π_i for ONE ROI on its [h,w] saliency grid:
    `J_i = TopK({ŝ_j}, B_span)`, then `Π_i = ∪_{j∈J_i}([j-r,j+r] ∩ S_i)` as a
    ±r 2D neighbor expansion. Returns bool [h,w] (True = retained)."""
    g = np.asarray(grid)
    h, w = g.shape
    flat = g.reshape(-1)
    k = min(b_span, flat.size)
    top = np.argsort(flat)[-k:]
    keep = np.zeros((h, w), dtype=bool)
    for idx in top:
        i, j = idx // w, idx % w
        keep[max(0, i - radius): i + radius + 1,
             max(0, j - radius): j + radius + 1] = True
    return keep


def method_prune_mask(saliency: dict, roi_spans: list, b_span: int = 32,
                      radius: int = 1) -> np.ndarray:
    """Concatenated keep mask over all ROIs (intra-pass pruning use). Per-ROI
    rule = `roi_keep_mask`. RVIS-risky (drops non-top tissue) — kept as the
    method-literal ablation; memory.unit.build_pi reuses the same per-ROI rule."""
    masks = [roi_keep_mask(saliency[r.id], b_span, radius).reshape(-1)
             for r in roi_spans if r.id in saliency]
    return np.concatenate(masks) if masks else np.zeros(0, dtype=bool)


def flat_topk_keep(sal: np.ndarray, roi_of_token: np.ndarray,
                   b_span: int = 32, radius: int = 1,
                   keep_ratio: float | None = None) -> np.ndarray:
    """Per-ROI top-B_span keep over a FLAT saliency vector (grid-agnostic).

    Unlike `method_prune_mask` (which needs rectangular [h,w] grids), this works on
    any current token set via a `roi_of_token` map, so it survives repeated pruning
    where the grid is no longer rectangular. Neighbor expansion is the 1-D local
    pool of method.md's Pool_κ (κ=2·radius+1) within each ROI's token order."""
    if keep_ratio is not None and not 0.0 < keep_ratio <= 1.0:
        raise ValueError("keep_ratio must be in (0, 1]")
    keep = np.zeros(sal.shape[0], dtype=bool)
    for r in np.unique(roi_of_token):
        idx = np.where(roi_of_token == r)[0]
        sr = sal[idx]
        k = min(
            sr.size,
            max(1, math.ceil(sr.size * keep_ratio))
            if keep_ratio is not None else b_span,
        )
        for j in np.argsort(sr)[-k:]:
            lo = max(0, j - radius)
            hi = min(sr.size, j + radius + 1)
            keep[idx[lo:hi]] = True
    return keep


def hierarchy_prefill_keep_mask(
    scores: torch.Tensor,
    *,
    token_counts: tuple[int, ...],
    parent_indices: tuple[int, ...],
    keep_ratio: float | None,
    min_tokens_per_image: int,
    significance_sigma: float | None = None,
) -> torch.Tensor:
    """Select morphology evidence while preserving every coarse/fine branch.

    Every observed image retains a minimum evidence quota. Fine-patch importance
    is propagated to its x5 parent branch before the remaining global budget is
    assigned, so high-magnification evidence cannot become context-free.
    """
    if scores.ndim != 1:
        raise ValueError("hierarchy prefill scores must be one-dimensional")
    if sum(token_counts) != scores.numel():
        raise ValueError("per-image token counts must cover every visual score")
    if len(parent_indices) != len(token_counts):
        raise ValueError("every image must provide one parent index")
    if keep_ratio is not None and not 0.0 < keep_ratio <= 1.0:
        raise ValueError("hierarchy prefill keep ratio must be in (0, 1]")
    if significance_sigma is not None and significance_sigma < 0.0:
        raise ValueError("hierarchy prefill significance sigma must be non-negative")
    if keep_ratio is None and significance_sigma is None:
        raise ValueError("hierarchy prefill requires a budget or significance threshold")
    if min_tokens_per_image < 1:
        raise ValueError("per-image evidence quota must be positive")

    image_ranges: list[torch.Tensor] = []
    image_scores: list[torch.Tensor] = []
    offset = 0
    for count in token_counts:
        indices = torch.arange(offset, offset + count, device=scores.device)
        image_ranges.append(indices)
        values = scores[indices]
        top_count = max(1, math.ceil(count * 0.10))
        image_scores.append(values.topk(top_count).values.mean())
        offset += count

    roots = tuple(
        parent if 0 <= parent < len(token_counts) else index
        for index, parent in enumerate(parent_indices)
    )
    branch_scores: dict[int, torch.Tensor] = {}
    for index, root in enumerate(roots):
        current = branch_scores.get(root)
        branch_scores[root] = (
            image_scores[index]
            if current is None
            else torch.maximum(current, image_scores[index])
        )

    adjusted = scores.float().clone()
    keep = torch.zeros(scores.numel(), dtype=torch.bool, device=scores.device)
    for index, indices in enumerate(image_ranges):
        adjusted[indices] += branch_scores[roots[index]]
        quota = min(indices.numel(), min_tokens_per_image)
        order = torch.argsort(adjusted[indices], descending=True, stable=True)
        keep[indices[order[:quota]]] = True

    if significance_sigma is not None:
        for indices in image_ranges:
            values = adjusted[indices]
            threshold = values.mean() + significance_sigma * values.std(unbiased=False)
            keep[indices] |= values >= threshold
        return keep

    assert keep_ratio is not None
    budget = min(scores.numel(), max(int(keep.sum().item()), math.ceil(scores.numel() * keep_ratio)))
    remaining = budget - int(keep.sum().item())
    if remaining > 0:
        candidates = torch.nonzero(~keep, as_tuple=False).flatten()
        order = torch.argsort(adjusted[candidates], descending=True, stable=True)
        keep[candidates[order[:remaining]]] = True
    return keep


def dynamic_ratio_topk_keep(
    sal: np.ndarray,
    roi_of_token: np.ndarray,
    *,
    b_span: int = 32,
    radius: int = 1,
    min_ratio: float = 0.05,
    max_ratio: float = 0.20,
) -> np.ndarray:
    """Keep an entropy-adaptive fraction independently inside each ROI.

    Diffuse saliency keeps more tokens; concentrated saliency keeps fewer. The
    ratio is recomputed from the current controller score at every prune event.
    ``b_span`` remains a small-token fallback for legacy callers.
    """
    if not 0.0 < min_ratio <= max_ratio <= 1.0:
        raise ValueError("dynamic ratio bounds must satisfy 0 < min <= max <= 1")
    keep = np.zeros(sal.shape[0], dtype=bool)
    for roi in np.unique(roi_of_token):
        idx = np.where(roi_of_token == roi)[0]
        values = np.asarray(sal[idx], dtype=np.float64)
        shifted = values - values.min()
        weights = shifted + np.finfo(np.float64).eps
        probabilities = weights / weights.sum()
        entropy = -float(np.sum(probabilities * np.log(probabilities)))
        max_entropy = math.log(max(2, idx.size))
        concentration = min(1.0, max(0.0, entropy / max_entropy))
        ratio = min_ratio + (max_ratio - min_ratio) * concentration
        k = min(idx.size, max(1, math.ceil(idx.size * ratio)))
        for local in np.argsort(values)[-k:]:
            lower = max(0, local - radius)
            upper = min(idx.size, local + radius + 1)
            keep[idx[lower:upper]] = True
    return keep


def keep_mask_for_block(mode: str, sal: np.ndarray, roi_of_token: np.ndarray, *,
                        tissue: np.ndarray | None = None,
                        sink: np.ndarray | None = None,
                        b_span: int = 32, radius: int = 1,
                        tissue_th: float = 0.1, sink_frac: float = 0.05,
                        random_seed: int = 0) -> np.ndarray:
    """Grid-agnostic keep-mask for ONE freshly-added vision block (any vision
    entry). The shared decision seam used by both the streaming controller and the
    one-shot per-entry prune; a cross-iteration retention strategy can later wrap
    this. none→keep all; safe→drop bg+sink (RVIS-safe); topk→per-ROI flat top-k;
    random→per-ROI RANDOM members at topk's EQUAL budget (chance baseline)."""
    if mode == "safe":
        if tissue is None:
            return np.ones(sal.shape[0], dtype=bool)   # no tissue signal → keep all
        return safe_prune_mask(tissue, sal, sink, tissue_th=tissue_th,
                               sink_frac=sink_frac)
    if mode == "topk":
        return flat_topk_keep(sal, roi_of_token, b_span=b_span, radius=radius)
    if mode == "random":
        return random_floor(sal, roi_of_token, b_span=b_span, radius=radius,
                            rng=np.random.default_rng(random_seed))
    return np.ones(sal.shape[0], dtype=bool)


def random_floor(sal, roi_of_token, b_span: int = 32, radius: int = 1, rng=None):
    """Keep the SAME per-ROI count as `flat_topk_keep` but with RANDOM membership.

    Baseline for --carry_prune random: isolates whether saliency-guided eviction
    beats chance at an EQUAL KV budget (identical compression, random victims). Uses
    flat_topk_keep only to read each ROI's kept COUNT, then samples that many cols
    uniformly per ROI."""
    if rng is None:
        rng = np.random.default_rng(0)
    ref = np.asarray(flat_topk_keep(sal, roi_of_token, b_span=b_span, radius=radius))
    roi_of_token = np.asarray(roi_of_token)
    keep = np.zeros(len(sal), dtype=bool)
    for r in np.unique(roi_of_token):
        idx = np.where(roi_of_token == r)[0]
        k = int(ref[idx].sum())                        # match topk's kept count for this ROI
        if k >= idx.size:
            keep[idx] = True
        elif k > 0:
            keep[rng.choice(idx, size=k, replace=False)] = True
    return keep


def mature_prune_mask(sal, regidx, age, cur_iter: int, delay: int,
                      pruned_iters: set, mode: str, *,
                      b_span: int = 32, radius: int = 1,
                      tissue=None, sink=None,
                      tissue_th: float = 0.1, sink_frac: float = 0.05,
                      stream_mode: str = "snap", drop_per_event: int = 64,
                      random_seed: int = 0):
    """Carry-level DELAYED vision-KV prune keep-mask over the CURRENT vision cols.

    Each vision col carries `regidx` (owning registry ROI) and `age` (the iteration
    it entered). A block = all cols sharing an entry iteration `e`; it becomes
    eligible once `cur_iter - e >= delay` (recency grace). Cols in blocks still
    within grace (or already in `pruned_iters`) are kept verbatim. Unlike the
    intra-pass streaming controller this runs AFTER the Reasoner, so it never nulls
    the Reasoner's saliency/registry → composes with the ground_cos gate + joint
    rescore (which the intra-pass path starves).

      - stream_mode='snap'    : prune the block to its top-b_span+radius floor in ONE
                                call, then mark it done (`matured`).
      - stream_mode='gradual' : each call drop the lowest `drop_per_event` cols
                                OUTSIDE that fixed per-ROI floor; mark done only when
                                the block reaches the floor (descends over iterations).

    Returns (keep_mask[bool over all cols], newly_done:set[int]). Score is used
    as-is (frozen entry saliency); a grace-accumulate variant folds l2v upstream.
    """
    sal = np.asarray(sal, dtype=float)
    regidx = np.asarray(regidx)
    age = np.asarray(age)
    keep = np.ones(sal.shape[0], dtype=bool)
    matured: set = set()
    if sal.shape[0] == 0:
        return keep, matured
    for e in sorted({int(a) for a in age.tolist()}):
        if e in pruned_iters or (cur_iter - e) < delay:
            continue                                   # already done, or still in grace
        block = np.where(age == e)[0]
        if block.size == 0:
            matured.add(e)                             # nothing left → done
            continue
        if mode == "random":                           # baseline: equal budget, random victims
            rng = np.random.default_rng((int(random_seed) + 1) * 1_000_003
                                        + cur_iter * 10_007 + e)
            floor = random_floor(sal[block], regidx[block], b_span, radius, rng)
            droppable = block[~np.asarray(floor, dtype=bool)]
            if droppable.size == 0:
                matured.add(e)
                continue
            if stream_mode == "gradual":
                order = rng.permutation(droppable)     # random victims (not lowest-saliency)
                keep[order[:drop_per_event]] = False
                if droppable.size <= drop_per_event:
                    matured.add(e)
            else:                                      # snap: drop all outside the random floor
                keep[droppable] = False
                matured.add(e)
        elif stream_mode == "gradual" and mode == "topk":
            # descend toward the fixed per-ROI floor by drop_per_event, per call
            floor = np.asarray(flat_topk_keep(sal[block], regidx[block],
                                              b_span, radius), dtype=bool)
            droppable = block[~floor]                  # cols outside the floor (global idx)
            if droppable.size == 0:
                matured.add(e)                         # already at floor → done
                continue
            order = droppable[np.argsort(sal[droppable])]   # lowest frozen score first
            keep[order[:drop_per_event]] = False
            if droppable.size <= drop_per_event:
                matured.add(e)                         # this call finishes the descent
        else:                                          # snap: to floor in one shot
            sub = keep_mask_for_block(
                mode, sal[block], regidx[block],
                tissue=(tissue[block] if tissue is not None else None),
                sink=(sink[block] if sink is not None else None),
                b_span=b_span, radius=radius, tissue_th=tissue_th, sink_frac=sink_frac)
            keep[block[~np.asarray(sub, dtype=bool)]] = False
            matured.add(e)
    return keep, matured


def compute_keep_mask(mode: str, sal: np.ndarray, tissue: np.ndarray,
                      sink: np.ndarray | None, *, saliency: dict | None = None,
                      roi_spans: list | None = None, b_span: int = 32, radius: int = 1,
                      tissue_th: float = 0.1, sink_frac: float = 0.05) -> np.ndarray:
    """Dispatch keep mask. none=keep all; safe=drop bg+sink (RVIS-safe);
    topk=method.md per-ROI Π_i (RVIS-risky baseline)."""
    if mode == "safe":
        return safe_prune_mask(tissue, sal, sink, tissue_th=tissue_th, sink_frac=sink_frac)
    if mode == "topk":
        if saliency is None or roi_spans is None:
            raise ValueError("topk mode needs saliency grids + roi_spans")
        return method_prune_mask(saliency, roi_spans, b_span=b_span, radius=radius)
    return np.ones(len(sal), dtype=bool)


# ── M5 budget-transfer: selection → keep_mask over carried vision columns ────────
def keep_mask_for_selection(carried_vis_cols, kept_abs_cols) -> np.ndarray:
    """M5 budget-transfer contract: keep_mask[i] = (carried_vis_cols[i] is kept).

    `carried_vis_cols` are the absolute KV columns of ALL vision tokens threaded
    through the carry loop (in append order); `kept_abs_cols` is the union of the
    bank-selected units' retained-token columns (`unit.meta['vis_cols']`). The mask
    is aligned to vis_cols ORDER, exactly how `apply_kv_prune` consumes it (it
    indexes `vis_cols[i]` with `keep_mask[i]`) — see memory/test_prune_carry.py.
    """
    kept = {int(c) for c in (kept_abs_cols or [])}
    return np.array([int(c) in kept for c in (carried_vis_cols or [])], dtype=bool)


# ── per-handoff span prune: keep {V,L} (drop prompt) at every carry step ─────────
def prune_handoff(past_kv, prev_len: int, delta_spans, keep_types,
                  carried_cols, expected_len: int, device):
    """Drop non-kept span columns WITHIN a freshly-appended delta block.

    Carry semantics: an agent appended `delta_spans` (ordered (type,n) tuples,
    e.g. [("P",P),("V",n),("P",Q),("L",m)]) on top of a `prev_len`-column prefix
    that is already clean (only kept types survive there). We keep the whole
    prefix verbatim (tagged "_") and, inside the delta, keep only span types in
    `keep_types` — so drop_prompt=`{"V","L"}` drops the two P blocks, latent_only=
    `{"L"}` also drops V. The prefix's internal structure is irrelevant (monolithic
    keep), so NO global span_map bookkeeping is needed.

    `carried_cols` are absolute vision-KV columns threaded for S8 reallocation;
    they are remapped through the survivor map (dropped ones removed). Returns
    (pruned_kv, new_len, new_carried_cols, survivor_mask) where survivor_mask is a
    bool array over the INPUT carried_cols (so the caller can filter its parallel
    saliency list). No-op (identity) when nothing is dropped.

    `expected_len` guards alignment: if prev_len+sum(delta) != actual cache length
    the delta does not describe the appended block and we refuse to prune.
    """
    import numpy as np
    keep = {"_"} | {str(t) for t in keep_types}
    span_map = [("_", int(prev_len))] + [(str(t), int(n)) for t, n in delta_spans]
    total = sum(n for _, n in span_map)
    if total != int(expected_len):
        raise ValueError(
            f"prune_handoff span/length mismatch: prev_len={prev_len} + "
            f"delta={sum(n for _, n in delta_spans)} = {total} != cache "
            f"len {expected_len}; refusing to gather (would corrupt KV).")

    keep_cols, col = [], 0
    for t, n in span_map:
        if t in keep:
            keep_cols.extend(range(col, col + n))
        col += n

    carried = [int(c) for c in (carried_cols or [])]
    if len(keep_cols) == total:                 # nothing dropped → identity
        return past_kv, total, carried, np.ones(len(carried), dtype=bool)

    import torch
    idx = torch.tensor(keep_cols, device=device, dtype=torch.long)
    pk = copy.deepcopy(past_kv)
    layers = pk.layers if hasattr(pk, "layers") else None
    if layers is not None:
        for layer in layers:
            # index_select already returns a contiguous allocation. Calling
            # contiguous() again can retain the old KV and create a second full
            # copy during streaming eviction, which is enough to exceed 48 GiB.
            old_keys, old_values = layer.keys, layer.values
            layer.keys = old_keys.index_select(2, idx)
            layer.values = old_values.index_select(2, idx)
            del old_keys, old_values
    pos = {c: j for j, c in enumerate(keep_cols)}
    mask = np.array([c in pos for c in carried], dtype=bool)
    new_cols = [pos[c] for c in carried if c in pos]
    return pk, len(keep_cols), new_cols, mask


# ── apply: drop unkept vision columns from the KV cache ──────────────────────────
def apply_kv_prune(past_kv, past_len: int, vis_cols, keep_mask: np.ndarray,
                   spans: list, device, *, in_place: bool = False):
    """Drop unkept vision KV columns. Kept columns keep their baked-in rotary phase
    (post-rotary cache), so pos_cursor is unchanged and holes in position are fine.
    Returns (pruned_kv, new_len, new_spans, new_vis_cols).

    ``in_place`` avoids duplicating the entire multi-layer cache during an
    online eviction event.  Use it only when the caller immediately replaces
    its cache reference (the latent streaming path); the default retains the
    copy-on-write behavior used by offline callers and tests.
    """
    vis_cols = [int(c) for c in (vis_cols.tolist() if hasattr(vis_cols, "tolist") else vis_cols)]
    drop = {vis_cols[i] for i in range(len(vis_cols)) if not keep_mask[i]}
    if not drop:
        return past_kv, past_len, spans, vis_cols

    keep_cols = [c for c in range(past_len) if c not in drop]
    # rebuild span_map by RLE over surviving column types
    types = []
    for t, n in spans:
        types += [t] * n
    new_types = [types[c] for c in keep_cols] if len(types) >= past_len else None
    new_spans = []
    if new_types is not None:
        rt, rn = None, 0
        for t in new_types:
            if t == rt:
                rn += 1
            else:
                if rt is not None:
                    new_spans.append((rt, rn))
                rt, rn = t, 1
        if rt is not None:
            new_spans.append((rt, rn))
    else:
        new_spans = spans

    idx = torch.tensor(keep_cols, device=device, dtype=torch.long)
    pk = past_kv if in_place else copy.deepcopy(past_kv)
    layers = pk.layers if hasattr(pk, "layers") else None
    if layers is not None:
        for layer in layers:
            # index_select already materializes a contiguous tensor. Repeating
            # contiguous() makes a second full KV copy at every prune event.
            layer.keys = layer.keys.index_select(2, idx)
            layer.values = layer.values.index_select(2, idx)
    pos = {c: j for j, c in enumerate(keep_cols)}
    new_vis = [pos[c] for c in vis_cols if c in pos]
    return pk, len(keep_cols), new_spans, new_vis


# ── intra-pass streaming prune controller (model-agnostic) ───────────────────────
class PruneController:
    """Model-agnostic intra-pass visual-KV prune policy + per-step saliency dump.

    Owns everything that is NOT model-specific: the EMA heavy-hitter score
    (method.md s_j^(τ) = H2O accumulated attention), the warmup/every-k schedule
    (H2O streaming eviction), and the keep-mask rule (safe / topk via the existing
    `compute_keep_mask` primitives). A backbone's latent loop only feeds it the
    per-step latent→vision attention and applies the returned mask with
    `apply_kv_prune` — so a new model reuses this policy verbatim.

    Two independent modes (set via args):
      - `--prune_dump`         : record per-step EMA saliency (read-only, NO eviction).
                                 Powers the saliency-stability probe (step-k vs step-m).
      - `--prune_every k > 0`  : actually evict every k steps after `--prune_warmup`
                                 (the H2O streaming path). Needs `--kv_prune safe|topk`.

    With both off, `build_prune_controller` returns None → zero overhead, the
    backbone path is byte-identical to before (one-shot prune in the Reasoner still
    applies as today).
    """

    def __init__(self, args) -> None:
        self.every     = int(getattr(args, "prune_every", 0) or 0)
        self.warmup    = int(getattr(args, "prune_warmup", 0) or 0)
        self.lam       = float(getattr(args, "prune_ema_lambda", 0.5))
        self.mode      = getattr(args, "kv_prune", "none")
        self.b_span    = int(getattr(args, "kv_prune_b_span", 32))
        self.keep_ratio = getattr(args, "prune_keep_ratio", None)
        self.dynamic_keep_ratio = bool(getattr(args, "prune_dynamic_ratio", False))
        self.event_steps = getattr(args, "prune_event_steps", None)
        self.step_adaptive = bool(getattr(args, "prune_step_adaptive", False))
        self.first_event_keep_ratio = getattr(args, "prune_first_event_keep_ratio", None)
        self.stable_streaming = bool(getattr(args, "prune_stable_streaming", False))
        self.stable_low_steps = int(getattr(args, "prune_stable_low_steps", 3) or 3)
        self.stable_observe_every = int(
            getattr(args, "prune_stable_observe_every", 1) or 1
        )
        self.stable_single_event = bool(
            getattr(args, "prune_stable_single_event", False)
        )
        self.stable_dynamic_survivors = bool(
            getattr(args, "prune_stable_dynamic_survivors", False)
        )
        self.stable_dynamic_min_keep = float(
            getattr(args, "prune_stable_dynamic_min_keep", 0.10) or 0.10
        )
        self.stable_dynamic_max_keep = float(
            getattr(args, "prune_stable_dynamic_max_keep", 0.45) or 0.45
        )
        self.stable_event_at_end = bool(
            getattr(args, "prune_stable_event_at_end", False)
        )
        self.hierarchy_aware = bool(getattr(args, "prune_hierarchy_aware", False))
        self.step_keep_ratios = getattr(args, "prune_step_keep_ratios", None)
        self.relative_saliency_floor = getattr(
            args, "prune_relative_saliency_floor", None
        )
        self._active_event_step: int | None = None
        self._initial_token_count = 0
        self._gpu_ema: torch.Tensor | None = None
        self._gpu_max: torch.Tensor | None = None
        self._gpu_low_streak: torch.Tensor | None = None
        self._gpu_roi: torch.Tensor | None = None
        self.radius    = int(getattr(args, "kv_prune_radius", 1))
        self.tissue_th = float(getattr(args, "kv_prune_tissue_th", 0.1))
        self.sink_frac = float(getattr(args, "kv_prune_sink_frac", 0.05))
        self.dump      = bool(getattr(args, "prune_dump", False))
        # streaming eviction is only on with a real rule AND a positive interval
        self.evict     = self.every > 0 and self.mode in ("safe", "topk")
        self.history: list[dict] = []
        self.evicted = False
        # optional Rel(q2v) in the eviction score (gated; off = l2v-EMA only, unchanged)
        self.use_rel = bool(getattr(args, "prune_stream_rel", False))
        self.lam_r   = float(getattr(args, "saliency_lam_r", 1.0))
        self.lam_s   = float(getattr(args, "saliency_lam_s", 0.2))
        self.rel = None
        # gradual floor-descent (opt-in): each event drop the lowest `drop_per_event`
        # tokens OUTSIDE the fixed top-b_span+radius floor, so streaming reaches the
        # SAME floor smoothly. snap (default) = one-shot to floor (unchanged).
        self.stream_mode    = getattr(args, "prune_stream_mode", "snap")
        self.drop_per_event = int(getattr(args, "prune_drop_per_event", 64) or 64)
        # head-axis reduce of the per-step latent->vision map before the EMA.
        # mean (default) = head-mean, byte-identical to the old .mean(0) path;
        # max = head-max (matches roi_saliency head_mode=max); contrast = head-max
        # - head-mean (>=0), the ②contrast surrogate. Feeds self.s → affects both the
        # eviction score (_score) and the final_saliency rebuild.
        self.reduce = getattr(args, "stream_saliency_reduce", "mean")
        # --stream_reduce_head_first: when set, the backbone reduces heads PER LAYER
        # (via _reduce_heads) then averages over layers → mean_l(reduce_h), matching
        # roi_saliency's Eq.(reason-attn) order. Off (default) → backbone layer-means
        # first and observe() head-reduces the [H,nV] map (reduce_h(mean_l)) → the
        # current byte-identical path. Read by the backbone hook; observe() itself is
        # order-agnostic (passes a 1-D pre-reduced vector straight through).
        self.reduce_head_first = bool(getattr(args, "stream_reduce_head_first", False))
        # --stream_rebuild_ec: final_saliency (the --stream_recompute_saliency rebuild)
        # uses the eviction score e_c = lam_r*Rel* + lam_f*a_c instead of the focus EMA
        # a_c alone, so registry/G_i/z_i keep the dual-branch (relevance+focus) structure
        # of the one-shot s_c. Off (default) → a_c only (self.s), byte-identical. Only
        # differs when prune_stream_rel is on (else _score() returns self.s anyway).
        self.rebuild_ec = bool(getattr(args, "stream_rebuild_ec", False))
        # --stream_rel_precise: compute the static Rel* as the TRUE relevance branch
        # (head-max over all collected full-attn layers, renorm+contrast over all rows,
        # averaged over qq rows only) instead of the cheap head-mean/mid-band/all-rows
        # version. Off (default) → cheap Rel*, byte-identical. Needs role_spans in start().
        self.rel_precise = bool(getattr(args, "stream_rel_precise", False))
        self.head_mode   = getattr(args, "saliency_head_mode", "max")
        # --saliency_bg_correct: replace the static Rel*'s query-row contrast with the
        # template-bias correction Rel* = [mu(qq) - rho(B)]_+ = Eq.(reason-branch), so
        # streaming eviction reuses the SAME corrected relevance as the dual-branch
        # importance. Needs bg_role_spans in start(). Off → legacy contrast (identical).
        self.bg_correct  = bool(getattr(args, "saliency_bg_correct", False))
        self.morphology_aware = bool(getattr(args, "prune_morphology_aware", False))
        self.morphology_quota = int(getattr(args, "prune_morphology_quota", 1) or 1)

    def configure_for_latent_steps(self, latent_steps: int) -> None:
        """Place dynamic two-stage pruning relative to the rollout horizon."""
        if not self.step_adaptive:
            return
        if self.stable_streaming:
            if self.stable_event_at_end:
                self.event_steps = (latent_steps,)
                return
            second = max(1, math.ceil(0.7 * latent_steps))
            if self.stable_single_event:
                first_stable = 1 + (
                    (self.stable_low_steps - 1) * self.stable_observe_every
                )
                self.event_steps = (min(latent_steps, first_stable),)
                return
            first = max(1, math.ceil(0.4 * latent_steps))
            self.event_steps = (first, min(latent_steps, max(first + 1, second)))
            return
        first = max(1, latent_steps // 5)
        second = max(first + 1, (3 * latent_steps) // 5)
        self.event_steps = (first, min(latent_steps, second))

    # state for one pass; built from model-supplied primitives (still agnostic)
    def start(self, *, grids, vis_idx, sms: int, images, q2v, l_mid_layers,
              role_spans=None, bg_role_spans=None, image_magnifications=(),
              image_parent_indices=(),
              morphology_enabled: bool = True) -> None:
        n = int(vis_idx.numel()) if hasattr(vis_idx, "numel") else len(vis_idx)
        self._initial_token_count = n
        self.history = []
        self.evicted = False
        # per-token → ROI map from the patch grid (tokens = t*h*w / sms^2 per crop)
        roi: list[int] = []
        spatial_quadrants: list[int] = []
        gh_gw: list[tuple[int, int]] = []
        if grids is not None:
            for i, (t, h, w) in enumerate(grids.tolist()):
                grid_h = int(h // sms)
                grid_w = int(w // sms)
                c = int(t * grid_h * grid_w)
                roi += [i] * c
                gh_gw.append((grid_h, grid_w))
                for token in range(c):
                    within_plane = token % (grid_h * grid_w)
                    row, col = divmod(within_plane, grid_w)
                    spatial_quadrants.append(
                        2 * min(1, 2 * row // max(1, grid_h))
                        + min(1, 2 * col // max(1, grid_w))
                    )
        if len(roi) != n:                       # fallback: treat as one ROI
            roi = [0] * n
            gh_gw = []
            spatial_quadrants = [0] * n
        self.roi_of_token = np.asarray(roi, dtype=np.int64)
        parents = tuple(int(value) for value in image_parent_indices)
        self.parent_roi = np.asarray(
            [parents[index] if index < len(parents) else -1 for index in range(len(gh_gw))],
            dtype=np.int64,
        )
        self.spatial_quadrant = np.asarray(spatial_quadrants, dtype=np.int8)
        self.s = np.zeros(n, dtype=np.float64)  # EMA saliency over current tokens
        self._gpu_ema = None
        self._gpu_max = None
        self._gpu_low_streak = None
        self._gpu_roi = None
        self._gpu_spatial_quadrant = None
        self._gpu_morphology_strata = None
        self.morphology_strata = None
        self.scale_of_token = None
        if self.morphology_aware and morphology_enabled and gh_gw and images:
            from memory.saliency import RoiSpan
            spans, start = [], 0
            for index, (gh, gw) in enumerate(gh_gw):
                spans.append(RoiSpan(id=index, start=start, length=gh * gw, h=gh, w=gw))
                start += gh * gw
            try:
                strata = token_morphology_strata(images, spans)
                scales = tuple(int(value) for value in image_magnifications)
                scale_parts = [
                    np.full(span.length, scales[index] if index < len(scales) else 0, dtype=np.int16)
                    for index, span in enumerate(spans)
                ]
                scale_of_token = np.concatenate(scale_parts) if scale_parts else np.zeros(0, dtype=np.int16)
                if strata.shape[0] == n and scale_of_token.shape[0] == n:
                    self.morphology_strata = strata
                    self.scale_of_token = scale_of_token
                    active_scales = sorted({int(value) for value in scale_of_token if value > 0})
                    print(
                        "[PruneCtrl/v2] morphology strata enabled "
                        f"(scales={active_scales or ['unlabeled']}, quota={self.morphology_quota}, "
                        "budget=fixed)"
                    )
            except (IndexError, TypeError, ValueError):
                self.morphology_strata = None
                self.scale_of_token = None
        self.tissue = None
        self.sink = None
        if self.mode == "safe" and gh_gw:       # safe rule needs tissue + sink
            from memory.saliency import RoiSpan
            spans, start = [], 0
            for i, (gh, gw) in enumerate(gh_gw):
                spans.append(RoiSpan(id=i, start=start, length=gh * gw, h=gh, w=gw))
                start += gh * gw
            try:
                self.tissue = token_tissue_fraction(images, spans)
            except Exception as e:              # never let tissue masking break a run
                print(f"[PruneCtrl] tissue fraction failed ({e}); safe→keep all")
            mid = [L for L in q2v if 14 <= L <= 27] or list(q2v)
            self.sink = universal_attention(q2v, mid)
        # optional static Rel(q2v): question->vision, renorm + contrast (sink-suppressed).
        # Gated by --prune_stream_rel; None → score stays l2v-EMA only (unchanged).
        self.rel = None
        if self.use_rel and q2v:
            try:
                # template-bias correction Rel* = [mu(qq) - rho(B)]_+ (Eq. reason-branch):
                # head-max per layer -> layer-mean -> renorm -> mu-rho, mirroring
                # memory.saliency.roi_saliency (max-max). Gated by --saliency_bg_correct
                # + bg_role_spans; else falls through to the legacy contrast path below.
                _bg = None
                if self.bg_correct and bg_role_spans is not None and role_spans is not None:
                    from memory.saliency import build_qq_mask
                    n_q = int(next(iter(q2v.values())).shape[1])
                    _qqm = build_qq_mask(n_q, role_spans)
                    _bgm = build_qq_mask(n_q, bg_role_spans)
                    _qq = _qqm.cpu().numpy().astype(bool) if _qqm is not None else None
                    _bg = _bgm.cpu().numpy().astype(bool) if _bgm is not None else None
                if _bg is not None and _bg.any():
                    acc = [(np.asarray(q2v[li].float()).max(axis=0) if self.head_mode == "max"
                            else np.asarray(q2v[li].float()).mean(axis=0)) for li in q2v]
                    abar = np.mean(acc, 0)                                       # [nq, nV]
                    abar = abar / np.clip(abar.sum(-1, keepdims=True), 1e-9, None)  # renorm
                    mu = abar[_qq].mean(0) if (_qq is not None and _qq.any()) else abar.mean(0)
                    self.rel = np.clip(mu - abar[_bg].mean(0), 0.0, None)        # [nV]
                    print(f"[PruneCtrl] Rel*: template-bias correction "
                          f"(qq={int(_qq.sum()) if _qq is not None else 0}, bg={int(_bg.sum())} rows)")
                elif self.rel_precise and role_spans is not None:
                    # TRUE relevance branch (Eq. reason-branch): head-max over ALL
                    # collected full-attn layers, renorm + contrast over all rows,
                    # then average over qq (question/answer) rows only.
                    from memory.saliency import build_qq_mask
                    n_q = int(next(iter(q2v.values())).shape[1])
                    qq = build_qq_mask(n_q, role_spans)
                    qq = qq.cpu().numpy().astype(bool) if qq is not None else None
                    acc = []
                    for li in q2v:                                # all collected layers
                        a = np.asarray(q2v[li].float())           # [H, nq, nV]
                        a = (a.max(axis=0) if self.head_mode == "max"
                             else a.mean(axis=0))                 # head reduce → [nq, nV]
                        a = a / np.clip(a.sum(-1, keepdims=True), 1e-9, None)   # renorm
                        if a.shape[0] >= 2:                       # contrast over all rows
                            a = np.clip(a - a.mean(axis=0, keepdims=True), 0.0, None)
                        rows = a[qq] if (qq is not None and qq.any()) else a
                        acc.append(rows.mean(axis=0))             # mean over qq rows
                    self.rel = np.mean(acc, 0) if acc else None
                else:
                    mid = [L for L in q2v if 14 <= L <= 27] or list(q2v)
                    acc = []
                    for li in mid:
                        a = np.asarray(q2v[li].float())               # [H, nq, nV]
                        a = a.mean(axis=0)                             # [nq, nV]
                        a = a / np.clip(a.sum(-1, keepdims=True), 1e-9, None)
                        if a.shape[0] >= 2:
                            a = np.clip(a - a.mean(axis=0, keepdims=True), 0.0, None)
                        acc.append(a.mean(axis=0))                     # [nV]
                    self.rel = np.mean(acc, 0) if acc else None
                if self.rel is not None and self.rel.shape[0] != n:
                    self.rel = None                               # size guard
            except Exception as e:
                print(f"[PruneCtrl] rel(q2v) failed ({e}); streaming uses l2v-EMA only")
                self.rel = None

    @staticmethod
    def _minmax(x: np.ndarray) -> np.ndarray:
        lo, hi = float(np.min(x)), float(np.max(x))
        return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)

    def _score(self) -> np.ndarray:
        """Eviction score: l2v-EMA alone, or lam_r*Rel + lam_s*EMA when --prune_stream_rel."""
        if self.rel is None:
            return self.s
        return self.lam_r * self._minmax(self.rel) + self.lam_s * self._minmax(self.s)

    def _reduce_heads(self, mat: np.ndarray) -> np.ndarray:
        """Reduce a [H, nV] per-head latent→vision map to [nV] per --stream_saliency_
        reduce. mean (default) is byte-identical to the old head-mean; max/contrast
        recover roi_saliency's head-max / ②contrast. 1-D input (already reduced,
        e.g. tests / older callers) passes through unchanged."""
        if mat.ndim == 1:
            return mat
        if self.reduce == "max":
            return mat.max(axis=0)
        if self.reduce == "contrast":
            return np.clip(mat.max(axis=0) - mat.mean(axis=0), 0.0, None)
        return mat.mean(axis=0)                 # mean (default)

    def observe(self, step: int, vec: np.ndarray) -> None:
        """Accumulate this step's per-token latent→vision saliency (EMA). `vec` is
        [H, nV] (backbone passes head-resolved → reduced here per --stream_saliency_
        reduce) or [nV] (already reduced)."""
        vec = self._reduce_heads(np.asarray(vec, dtype=np.float64))
        if vec.shape[0] != self.s.shape[0]:
            return                              # size drift guard (shouldn't happen)
        self.s = self.lam * self.s + (1.0 - self.lam) * vec
        if self.dump:
            self.history.append({"step": int(step),
                                 "saliency": self.s.copy().tolist()})

    def observe_gpu(self, step: int, vec: torch.Tensor) -> None:
        """Accumulate V3 evidence stability without a per-step host transfer."""
        del step
        score = vec.float()
        if score.ndim == 2:
            match self.reduce:
                case "max":
                    score = score.max(dim=0).values
                case "contrast":
                    score = (score.max(dim=0).values - score.mean(dim=0)).clamp_min(0.0)
                case "mean":
                    score = score.mean(dim=0)
                case unreachable:
                    assert_never(unreachable)
        if self._gpu_ema is None:
            self._gpu_ema = torch.zeros_like(score)
            self._gpu_max = torch.zeros_like(score)
            self._gpu_low_streak = torch.zeros_like(score, dtype=torch.int16)
            self._gpu_roi = torch.as_tensor(
                self.roi_of_token,
                dtype=torch.long,
                device=score.device,
            )
            self._gpu_spatial_quadrant = torch.as_tensor(
                self.spatial_quadrant,
                dtype=torch.long,
                device=score.device,
            )
            if self.morphology_strata is not None:
                self._gpu_morphology_strata = torch.as_tensor(
                    self.morphology_strata,
                    dtype=torch.long,
                    device=score.device,
                )
        if score.numel() != self._gpu_ema.numel():
            return
        self._gpu_ema.mul_(self.lam).add_(score, alpha=1.0 - self.lam)
        self._gpu_max = torch.maximum(self._gpu_max, score)
        low = torch.zeros_like(score, dtype=torch.bool)
        for roi_id in torch.unique(self._gpu_roi):
            roi_mask = self._gpu_roi == roi_id
            threshold = score[roi_mask].median()
            low[roi_mask] = score[roi_mask] <= threshold
        self._gpu_low_streak = torch.where(
            low,
            self._gpu_low_streak + 1,
            torch.zeros_like(self._gpu_low_streak),
        )

    def should_observe(self, step: int) -> bool:
        """Return whether this latent step contributes to stable V3 evidence."""
        return step % self.stable_observe_every == 0

    def maybe_keep_mask_gpu(self, step: int) -> torch.Tensor | None:
        """Select stable V3 survivors on GPU at the two physical prune events."""
        if not self.is_eviction_step(step) or self._gpu_ema is None:
            return None
        if self._gpu_max is None or self._gpu_low_streak is None or self._gpu_roi is None:
            return None
        score = 0.7 * self._gpu_ema + 0.3 * self._gpu_max
        if self.hierarchy_aware and self.parent_roi.size:
            score = score.clone()
            for child_roi, parent_roi in enumerate(self.parent_roi.tolist()):
                if parent_roi < 0:
                    continue
                child = torch.nonzero(self._gpu_roi == child_roi, as_tuple=False).flatten()
                parent = torch.nonzero(self._gpu_roi == parent_roi, as_tuple=False).flatten()
                if child.numel() == 0 or parent.numel() == 0:
                    continue
                anchor = parent[score[parent].argmax()]
                inherited = score[child].max()
                score[anchor] = torch.maximum(
                    score[anchor],
                    torch.nextafter(inherited, torch.full_like(inherited, float("inf"))),
                )
        keep = torch.zeros_like(score, dtype=torch.bool)
        if self.stable_dynamic_survivors:
            if self.hierarchy_aware:
                provisional = self._gpu_low_streak < self.stable_low_steps
                for roi_id in torch.unique(self._gpu_roi):
                    roi_indices = torch.nonzero(self._gpu_roi == roi_id, as_tuple=False).flatten()
                    count = int(roi_indices.numel())
                    roi_score = score[roi_indices]
                    spread = (roi_score - roi_score.min()).clamp_min(0.0)
                    probabilities = spread / spread.sum().clamp_min(torch.finfo(spread.dtype).tiny)
                    entropy = -(probabilities * probabilities.clamp_min(torch.finfo(spread.dtype).tiny).log()).sum() / math.log(max(2, count))
                    ratio = self.stable_dynamic_min_keep + (self.stable_dynamic_max_keep - self.stable_dynamic_min_keep) * entropy.clamp(0.0, 1.0)
                    budget = max(min(count, self.b_span), int(torch.ceil(ratio * count).item()))
                    order = torch.argsort(score[roi_indices], descending=True, stable=True)
                    provisional[roi_indices[order[:budget]]] = True
                total_budget = int(provisional.sum().item())
                order = torch.argsort(score, descending=True, stable=True)
                keep[order[:total_budget]] = True
                print(f"[PruneCtrl/v4] hierarchy-aware single Top-K: {total_budget}/{score.numel()}")
                return keep
            keep |= self._gpu_low_streak < self.stable_low_steps
            for roi_id in torch.unique(self._gpu_roi):
                roi_indices = torch.nonzero(self._gpu_roi == roi_id, as_tuple=False).flatten()
                count = int(roi_indices.numel())
                floor = min(count, self.b_span)
                order = torch.argsort(score[roi_indices], descending=True, stable=True)
                keep[roi_indices[order[:floor]]] = True
                roi_score = score[roi_indices]
                spread = (roi_score - roi_score.min()).clamp_min(0.0)
                spread_sum = spread.sum()
                probabilities = spread / spread_sum.clamp_min(torch.finfo(spread.dtype).tiny)
                entropy = -(
                    probabilities * probabilities.clamp_min(torch.finfo(spread.dtype).tiny).log()
                ).sum() / math.log(max(2, count))
                entropy = torch.where(
                    spread_sum > 0,
                    entropy,
                    torch.ones_like(entropy),
                )
                dynamic_ratio = self.stable_dynamic_min_keep + (
                    self.stable_dynamic_max_keep - self.stable_dynamic_min_keep
                ) * entropy.clamp(0.0, 1.0)
                dynamic_budget = torch.ceil(dynamic_ratio * count).to(torch.long)
                rank = torch.empty(count, dtype=torch.long, device=score.device)
                rank[order] = torch.arange(count, dtype=torch.long, device=score.device)
                keep[roi_indices[rank < dynamic_budget]] = True
                if self.morphology_aware and self._gpu_spatial_quadrant is not None:
                    roi_quadrants = self._gpu_spatial_quadrant[roi_indices]
                    for quadrant in torch.unique(roi_quadrants):
                        candidates = roi_indices[roi_quadrants == quadrant]
                        best = candidates[score[candidates].argmax()]
                        keep[best] = True
                if self._gpu_morphology_strata is not None:
                    roi_strata = self._gpu_morphology_strata[roi_indices]
                    for stratum in torch.unique(roi_strata):
                        candidates = roi_indices[roi_strata == stratum]
                        order = torch.argsort(score[candidates], descending=True, stable=True)
                        keep[candidates[order[:self.morphology_quota]]] = True
            return keep
        first_event = self.event_steps is not None and step + 1 == self.event_steps[0]
        first_event_ratio = self.first_event_keep_ratio or 0.5
        for roi_id in torch.unique(self._gpu_roi):
            roi_indices = torch.nonzero(self._gpu_roi == roi_id, as_tuple=False).flatten()
            count = int(roi_indices.numel())
            budget = (
                max(1, math.ceil(first_event_ratio * count))
                if first_event
                else min(count, max(self.b_span, math.ceil(0.15 * count)))
            )
            order = torch.argsort(score[roi_indices], descending=True, stable=True)
            keep[roi_indices[order[:budget]]] = True
        keep |= self._gpu_low_streak < self.stable_low_steps
        return keep

    def commit_gpu(self, keep: torch.Tensor) -> None:
        """Compact V3 GPU state after the backbone compacts the physical cache."""
        if self._gpu_ema is None or self._gpu_max is None:
            return
        if self._gpu_low_streak is None or self._gpu_roi is None:
            return
        self._gpu_ema = self._gpu_ema[keep]
        self._gpu_max = self._gpu_max[keep]
        self._gpu_low_streak = self._gpu_low_streak[keep]
        self._gpu_roi = self._gpu_roi[keep]
        if self._gpu_spatial_quadrant is not None:
            self._gpu_spatial_quadrant = self._gpu_spatial_quadrant[keep]
        if self._gpu_morphology_strata is not None:
            self._gpu_morphology_strata = self._gpu_morphology_strata[keep]
        keep_cpu = keep.detach().cpu().numpy()
        self.s = self._gpu_ema.detach().cpu().numpy().astype(np.float64, copy=False)
        self.roi_of_token = self.roi_of_token[keep_cpu]
        self.spatial_quadrant = self.spatial_quadrant[keep_cpu]
        if self.morphology_strata is not None:
            self.morphology_strata = self.morphology_strata[keep_cpu]
        if self.scale_of_token is not None:
            self.scale_of_token = self.scale_of_token[keep_cpu]
        self.evicted = True

    def maybe_keep_mask(self, step: int) -> np.ndarray | None:
        """Return a boolean keep-mask over current vision tokens, or None to skip."""
        if not self.is_eviction_step(step):
            return None
        self._active_event_step = step + 1
        if self.mode == "safe":
            if self.tissue is None:
                return None
            keep = safe_prune_mask(
                self.tissue,
                self._score(),
                self.sink,
                tissue_th=self.tissue_th,
                sink_frac=self.sink_frac,
            )
            return self._bounded_keep(keep)
        if self.stream_mode == "gradual":       # topk: descend to the floor smoothly
            return self._gradual_keep()
        return self._topk_keep()                # topk snap: per-ROI flat top-k + 1D radius

    def is_eviction_step(self, step: int) -> bool:
        """Return whether this zero-based latent step is scheduled to prune KV."""
        if not self.evict:
            return False
        one_based_step = step + 1
        if self.event_steps is not None:
            return one_based_step in self.event_steps
        return one_based_step > self.warmup and one_based_step % self.every == 0

    def allow_reallocation(self, step: int) -> bool:
        """Keep pruning-event saliency natural rather than reallocation-biased."""
        return not self.is_eviction_step(step)

    def _topk_keep(self) -> np.ndarray:
        if self.hierarchy_aware and self.relative_saliency_floor is not None:
            return self._hierarchy_significance_keep()
        if self.hierarchy_aware and self.keep_ratio is not None:
            return self._hierarchy_fixed_ratio_keep()
        if self.dynamic_keep_ratio:
            if (
                self.first_event_keep_ratio is not None
                and self.event_steps is not None
                and self._active_event_step == self.event_steps[0]
            ):
                keep = flat_topk_keep(
                    self._score(), self.roi_of_token, self.b_span, self.radius,
                    self.first_event_keep_ratio,
                )
            else:
                keep = dynamic_ratio_topk_keep(
                    self._score(), self.roi_of_token, b_span=self.b_span,
                    radius=self.radius,
                )
        else:
            keep = flat_topk_keep(
                self._score(), self.roi_of_token, self.b_span, self.radius,
                self.keep_ratio,
            )
        return self._with_morphology_quota(keep)

    def _hierarchy_significance_keep(self) -> np.ndarray:
        """Drop only evidence that is weak relative to current visual evidence.

        This is deliberately not a Top-K budget. Uniformly useful evidence keeps
        every token; only tokens below the configured fraction of the strongest
        current visual evidence mean are removed. The mean is the uniform-
        attention baseline, so this measures evidence weakness rather than a
        requested survivor count.
        Surviving fine evidence also protects the strongest coarse parent anchor.
        """
        score = np.asarray(self._score(), dtype=np.float64)
        keep = np.ones(score.size, dtype=bool)
        floor = float(self.relative_saliency_floor)
        tiny = np.finfo(np.float64).tiny
        values = np.maximum(score, 0.0)
        mean = float(values.mean()) if values.size else 0.0
        if mean > tiny:
            keep = values >= floor * mean

        for child_roi, parent_roi in enumerate(self.parent_roi.tolist()):
            if parent_roi < 0:
                continue
            child = np.flatnonzero(self.roi_of_token == child_roi)
            parent = np.flatnonzero(self.roi_of_token == parent_roi)
            if child.size == 0 or parent.size == 0 or not np.any(keep[child]):
                continue
            keep[int(parent[np.argmax(score[parent])])] = True

        print(
            f"[PruneCtrl/v4] reasoning step {self._active_event_step}: "
            f"hierarchy-aware significance {int(keep.sum())}/{score.size} "
            f"(uniform-mean floor={floor:.0%}; no fixed budget)"
        )
        return keep

    def _hierarchy_fixed_ratio_keep(self) -> np.ndarray:
        """Keep one global budget while closing selected fine branches to parents.

        The Navigator already provides exact child-to-parent ROI indices, so no
        learned graph/scorer is needed. A fine ROI can enter the global Top-K only
        after its parent's strongest coarse token is ranked just above that fine
        branch. This retains the same total token budget and performs one gather.
        """
        score = np.asarray(self._score(), dtype=np.float64)
        count = int(score.size)
        if count == 0:
            return np.zeros(0, dtype=bool)
        target_ratio = float(self.keep_ratio)
        if self.step_keep_ratios is not None and self._active_event_step is not None:
            event_index = (
                self.event_steps.index(self._active_event_step)
                if self.event_steps is not None
                and self._active_event_step in self.event_steps
                else self._active_event_step - 1
            )
            if event_index < len(self.step_keep_ratios):
                target_ratio = float(self.step_keep_ratios[event_index])
        # Decimal ratios such as 0.55 can land infinitesimally above an integer
        # in binary floating point (55.00000000000001), which would retain one
        # extra token.  Keep the configured absolute stage budget exact.
        absolute_budget = math.ceil(
            target_ratio * self._initial_token_count - 1e-12
        )
        budget = max(1, min(count, absolute_budget))
        adjusted = score.copy()
        parent_anchors: dict[int, int] = {}
        for child_roi, parent_roi in enumerate(self.parent_roi.tolist()):
            if parent_roi < 0:
                continue
            child = np.flatnonzero(self.roi_of_token == child_roi)
            parent = np.flatnonzero(self.roi_of_token == parent_roi)
            if child.size == 0 or parent.size == 0:
                continue
            anchor = int(parent[np.argmax(score[parent])])
            inherited = float(np.max(score[child]))
            adjusted[anchor] = np.nextafter(
                max(float(adjusted[anchor]), inherited),
                np.inf,
            )
            parent_anchors[child_roi] = anchor

        order = np.argsort(-adjusted, kind="stable")
        keep = np.zeros(count, dtype=bool)
        keep[order[:budget]] = True

        # The propagated score normally makes closure automatic. Enforce it
        # explicitly for ties, while swapping inside the same fixed budget.
        required = {
            anchor
            for child_roi, anchor in parent_anchors.items()
            if np.any(keep[self.roi_of_token == child_roi])
        }
        for anchor in sorted(required):
            if keep[anchor]:
                continue
            victims = np.flatnonzero(keep)
            victims = np.asarray(
                [index for index in victims if int(index) not in required],
                dtype=np.int64,
            )
            if victims.size == 0:
                break
            victim = int(victims[np.argmin(adjusted[victims])])
            keep[victim] = False
            keep[anchor] = True
        print(
            f"[PruneCtrl/v4] reasoning step {self._active_event_step}: "
            f"hierarchy-aware Top-K {int(keep.sum())}/{count} "
            f"(target={target_ratio:.0%} of initial {self._initial_token_count})"
        )
        return keep

    def _with_morphology_quota(self, keep: np.ndarray) -> np.ndarray:
        """Swap morphology representatives into the fixed attention survivor budget."""
        if (
            not self.morphology_aware
            or self.morphology_strata is None
            or self.scale_of_token is None
        ):
            return keep
        score = self._score()
        representative = np.zeros(keep.shape[0], dtype=bool)
        for scale in np.unique(self.scale_of_token):
            for stratum in np.unique(self.morphology_strata):
                if stratum == 0:
                    continue
                candidates = np.where(
                    (self.scale_of_token == scale) & (self.morphology_strata == stratum)
                )[0]
                if candidates.size:
                    top = candidates[np.argsort(score[candidates])[-self.morphology_quota:]]
                    representative[top] = True
        missing = np.where(representative & ~keep)[0]
        for candidate in missing[np.argsort(score[missing])[::-1]]:
            victims = np.where(keep & ~representative)[0]
            if victims.size == 0:
                break
            victim = victims[np.argmin(score[victims])]
            keep[victim] = False
            keep[candidate] = True
        return keep

    def _bounded_keep(self, requested_keep: np.ndarray) -> np.ndarray:
        """Limit one safe-prune event to the lowest-score requested victims."""
        requested_drop = np.where(~requested_keep)[0]
        if requested_drop.size <= self.drop_per_event:
            return requested_keep
        score = self._score()
        victims = requested_drop[np.argsort(score[requested_drop])[:self.drop_per_event]]
        bounded = np.ones(requested_keep.shape[0], dtype=bool)
        bounded[victims] = False
        return bounded

    def _gradual_keep(self) -> np.ndarray:
        """Drop the globally-lowest `drop_per_event` tokens that lie OUTSIDE the fixed
        per-ROI top-b_span+radius floor, so repeated events descend to that SAME floor
        smoothly instead of snapping in one shot. Every ROI keeps ≥ its floor (the
        floor set, incl. radius neighbors, is never evictable); already at floor →
        keep-all no-op. Score = EMA accumulated (adaptive victim across events)."""
        score = self._score()
        floor = (
            dynamic_ratio_topk_keep(
                score, self.roi_of_token, b_span=self.b_span, radius=self.radius,
            )
            if self.dynamic_keep_ratio else flat_topk_keep(
                score, self.roi_of_token, self.b_span, self.radius, self.keep_ratio,
            )
        )
        floor = self._with_morphology_quota(floor)
        droppable = np.where(~floor)[0]         # surplus outside the fixed floor
        keep = np.ones(score.shape[0], dtype=bool)
        if droppable.size == 0:
            return keep                         # already at floor → no-op
        order = droppable[np.argsort(score[droppable])]   # lowest score first
        keep[order[:self.drop_per_event]] = False
        return keep

    def commit(self, keep: np.ndarray) -> None:
        """Compact internal per-token state after the backbone applied `keep`."""
        self.s = self.s[keep]
        self.roi_of_token = self.roi_of_token[keep]
        if self.tissue is not None:
            self.tissue = self.tissue[keep]
        if self.sink is not None:
            self.sink = self.sink[keep]
        if self.rel is not None:
            self.rel = self.rel[keep]
        if self.morphology_strata is not None:
            self.morphology_strata = self.morphology_strata[keep]
        if self.scale_of_token is not None:
            self.scale_of_token = self.scale_of_token[keep]
        self.evicted = True

    def final_saliency(self):
        """Post-streaming saliency over the SURVIVING tokens, rebuilt from the EMA
        accumulated score (`self.s`) + ROI map (`self.roi_of_token`) — both kept
        correctly compacted through every eviction. Lets the Reasoner's registry /
        ground_cos survive intra-pass streaming (the raw q2v/l2v are unusable
        post-evict). Returns (roi_spans, {roi_id: [1,n] grid}); ([], {}) if empty.
        Survivors stay ROI-contiguous (evict preserves order) → spans by run-scan."""
        from memory.saliency import RoiSpan
        if self.s is None or len(self.s) == 0 or self.roi_of_token is None:
            return [], {}
        roi = np.asarray(self.roi_of_token)
        # --stream_rebuild_ec: rebuild from e_c (relevance+focus) vs a_c alone (default).
        s = np.asarray(self._score() if self.rebuild_ec else self.s, dtype=np.float32)
        grids, spans, off, j, n = {}, [], 0, 0, len(roi)
        while j < n:
            r = int(roi[j]); k = 0
            while j < n and int(roi[j]) == r:
                k += 1; j += 1
            grids[r] = s[off:off + k].reshape(1, k)
            spans.append(RoiSpan(id=r, start=off, length=k, h=1, w=k))
            off += k
        return spans, grids


def build_prune_controller(args) -> "PruneController | None":
    """Factory: None unless streaming eviction OR the saliency dump is requested.
    None ⇒ the backbone runs its original (one-shot) path with zero overhead."""
    if args is None:
        return None
    every = int(getattr(args, "prune_every", 0) or 0)
    mode = getattr(args, "kv_prune", "none")
    dump = bool(getattr(args, "prune_dump", False))
    if not ((every > 0 and mode in ("safe", "topk")) or dump):
        return None
    return PruneController(args)
