"""S2 — text-query-driven visual-token saliency (training-free).

Turns the Reasoner's already-extracted attention maps into a per-ROI saliency
grid that downstream stages (S3 within-ROI span retention, S4 latent summary,
S5 bank scoring) consume.

Design (see method discussion):

  ① layer/head aggregate, in fp32                  -> [Q, n_vis]
  ② sink removal
       (a) cross-slice renorm   : restore a distribution over vision tokens
                                   (the backbone slice dropped BOS/system mass)
       (b) contrast             : ReLU(a_qj - mean_q a_qj), kills register/sink
                                   patches every query attends to regardless
  ③ reduce queries -> Rel (question/choice/diag, PRIMARY)
                      Sal (latent steps, SECONDARY; latent collapses, weak)
       s_j = lam_r * minmax(Rel_j) + lam_s * minmax(Sal_j)
  ④ per-ROI 2D local pool (SNR + morphology)       -> {roi_id: [h, w]}

Every knob lives in `SaliencyConfig` so ablations are parameters, not code
forks.  The only ablations that need code *outside* this file are:
  - score="qk_proxy"  -> backbone must expose query vectors (hook); see
                         `qk_proxy_fn` arg, left as an interface stub.
  - L_att (scoring layers) -> reuse the existing `--l_mid_layers` flag; the
                         dicts passed in already contain exactly those layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Callable, Optional

import torch

_EPS = 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# config
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SaliencyConfig:
    """All S2 knobs. Add a field here + an argparse flag to expose a new ablation."""
    source:   str   = "rel"     # "rel" | "sal" | "both"
    score:    str   = "probs"   # "probs" | "qk_proxy"   (proxy needs a backbone hook)
    renorm:   bool  = True      # ②(a) cross-slice renorm
    contrast: bool  = True      # ②(b) register/sink suppression (query-row β_c; legacy)
    # ②(b') template-bias correction: subtract a query-independent background prior
    # rho_c = Phi_c(B) (renorm mean over fixed template rows B) from each branch,
    # S = [mu - rho]_+, instead of the query-row contrast. Needs bg_mask at call.
    # OFF (default) → the legacy `contrast` path runs, byte-identical.
    bg_correct: bool = False
    pool:     str   = "2d"      # "none" | "1d" | "2d"
    kappa:    int   = 3         # pool window (odd)
    lam_r:    float = 1.0       # Rel weight
    lam_s:    float = 0.2       # Sal weight (low: latent collapses)
    qq_only:  bool  = True      # use only Q_q rows (needs qq_mask) vs all text rows
    norm:     str   = "minmax"  # per-ROI output norm: "minmax" | "rank" | "none"
    # ── head aggregation (①). Uniform mean over heads was measured to KILL the
    # signal (IMPLEMENTATION_PLAN §1: tissue_lift<0, entropy≈1.0) because a few
    # specialized heads carry it and averaging drowns them in the rest. LACO uses
    # max-over-heads for the same reason.
    head_mode: str = "mean"     # "mean" | "max" | "topk" | "fixed"
    head_topk: int = 4          # topk: # of most-peaked heads kept per layer
    head_fixed: str = ""        # fixed: "layer:head,layer:head" e.g. "15:30,13:12"
    # layer aggregation (①, over the collected layers). Default mean = unchanged.
    # max = LACO CHSA (2605.22504) style; with head_mode=max this is max_{l,h}
    # (rare-but-critical cue preserved instead of diluted by averaging).
    layer_mode: str = "mean"    # "mean" | "max"

    @classmethod
    def from_args(cls, args: Any) -> "SaliencyConfig":
        """Build from an argparse namespace, reading `--saliency_<field>` flags
        when present and falling back to defaults otherwise."""
        kw = {}
        for f in fields(cls):
            v = getattr(args, f"saliency_{f.name}", None)
            if v is not None:
                kw[f.name] = v
        return cls(**kw)


# ─────────────────────────────────────────────────────────────────────────────
# S1 — per-ROI span bookkeeping
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RoiSpan:
    """One ROI's contiguous block inside the visual-token sequence.

    Vision tokens are laid out ROI-by-ROI at the front of the sequence, so ROI
    `id` owns columns [start, start+length) and reshapes to a (h, w) patch grid.
    """
    id: int
    start: int            # first visual-token column (within the V block)
    length: int           # number of visual tokens (= h * w when t == 1)
    h: int                # merged-grid height
    w: int                # merged-grid width
    bbox: Any = None      # level-0 BoundingBoxAction (provenance), optional
    meta: dict = field(default_factory=dict)


def spans_from_regidx(regidx, roi_hw: dict) -> list["RoiSpan"]:
    """Survivor-aware RoiSpans from a per-vision-col registry-index array.

    Under KV pruning the initial `h*w` no longer equals a ROI's surviving token
    count, so reconstructing spans from `(h, w)` overshoots and the cross-pass
    rescore guard (`n_vis == Σ length`) trips. Instead we scan CONTIGUOUS runs of
    `regidx` (vision tokens stay ROI-contiguous + prune preserves order), so each
    span's `length` = that ROI's surviving count and `Σ length == len(regidx)`.

    `roi_hw` = {registry_idx: (h, w)}; kept on the span so `roi_saliency` can 2D-
    reshape when a ROI happens to be un-pruned, and fall back to 1×N otherwise.
    Returns spans in column order (start = running offset).
    """
    spans: list[RoiSpan] = []
    start, j, n = 0, 0, len(regidx)
    while j < n:
        i = int(regidx[j])
        length = 0
        while j < n and int(regidx[j]) == i:
            length += 1
            j += 1
        h, w = roi_hw.get(i, (1, length))
        spans.append(RoiSpan(id=i, start=start, length=length, h=int(h), w=int(w)))
        start += length
    return spans


def build_roi_spans(
    grids: torch.Tensor,           # [n_patches, 3] raw (t, h, w) per image
    spatial_merge_size: int,
    bboxes: Optional[list] = None,
) -> list[RoiSpan]:
    """Derive per-ROI spans from the encoder's image_grid_thw.

    Merged token count per image = t * (h // sms) * (w // sms).
    """
    spans: list[RoiSpan] = []
    start = 0
    sms = int(spatial_merge_size)
    for i, g in enumerate(grids):
        t, h, w = int(g[0]), int(g[1]), int(g[2])
        gh, gw = h // sms, w // sms
        n = t * gh * gw
        bbox = bboxes[i] if (bboxes is not None and i < len(bboxes)) else None
        spans.append(RoiSpan(id=i, start=start, length=n, h=gh, w=gw, bbox=bbox))
        start += n
    return spans


# ─────────────────────────────────────────────────────────────────────────────
# ① aggregate layers + heads  → [Q, n_vis] in fp32
# ─────────────────────────────────────────────────────────────────────────────
def parse_head_fixed(spec: str) -> dict[int, list[int]]:
    """'15:30,13:12' → {15:[30], 13:[12]}; repeats on a layer merge ('15:30;15:8')."""
    out: dict[int, list[int]] = {}
    if not spec:
        return out
    for part in str(spec).replace(";", ",").split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        li, hi = part.split(":", 1)
        out.setdefault(int(li), []).append(int(hi))
    return out


def _head_scores(a: torch.Tensor, head_axis: int) -> torch.Tensor:
    """Per-head peakiness = mean over query rows of the vision-normalized row max.
    A specialized head concentrates on few patches (high max); a diffuse/background
    head spreads out (low max). Returns [H]."""
    x = a.float().movedim(head_axis, 0)                    # [H, ..., n_vis]
    flat = x.reshape(x.shape[0], -1, x.shape[-1])          # [H, Q, n_vis]
    p = flat / flat.sum(-1, keepdim=True).clamp_min(_EPS)  # normalize over vision
    return p.amax(-1).mean(-1)                             # [H]


def _reduce_heads(a: torch.Tensor, head_axis: int, cfg: SaliencyConfig,
                  layer: int, fixed_map: dict) -> Optional[torch.Tensor]:
    """Collapse the head axis per cfg.head_mode. Returns the head-reduced tensor
    (head axis removed), or None when 'fixed' has no head for this layer."""
    a = a.float()
    mode = cfg.head_mode
    if mode == "mean":
        return a.mean(dim=head_axis)
    if mode == "max":                                      # LACO-style; peaky heads survive
        return a.amax(dim=head_axis)
    if mode == "topk":
        s = _head_scores(a, head_axis)                     # [H]
        k = max(1, min(int(cfg.head_topk), int(s.numel())))
        idx = s.topk(k).indices
        return a.index_select(head_axis, idx).mean(dim=head_axis)
    if mode == "fixed":
        heads = fixed_map.get(int(layer)) if fixed_map else None
        if not heads:
            return None
        idx = torch.tensor([h for h in heads if 0 <= h < a.shape[head_axis]],
                           dtype=torch.long)
        if idx.numel() == 0:
            return None
        return a.index_select(head_axis, idx).mean(dim=head_axis)
    raise ValueError(f"unknown head_mode={mode!r}")


def _agg_layers(mats: list[torch.Tensor], mode: str) -> torch.Tensor:
    """Collapse the layer axis of stacked per-layer maps. mode 'mean' (default,
    unchanged) averages; 'max' takes the per-element max over layers (LACO CHSA).
    With head_mode='max' upstream, max here yields max_{l,h} (= joint max, since
    max over layers of max over heads)."""
    s = torch.stack(mats, dim=0)
    return s.amax(dim=0) if mode == "max" else s.mean(dim=0)


def _agg_q2v(q2v: dict[int, torch.Tensor], cfg: SaliencyConfig,
             fixed_map: dict) -> Optional[torch.Tensor]:
    """{layer: [H, n_q, n_vis]} → head-reduce (cfg.head_mode) then mean|max over
    layers (cfg.layer_mode) → [n_q, n_vis] fp32."""
    if not q2v:
        return None
    mats = [r for li, a in q2v.items()
            if (r := _reduce_heads(a, 0, cfg, li, fixed_map)) is not None]
    return _agg_layers(mats, cfg.layer_mode) if mats else None


def _agg_l2v(l2v: dict[int, torch.Tensor], cfg: SaliencyConfig,
             fixed_map: dict) -> Optional[torch.Tensor]:
    """{layer: [m, H, n_vis]} → head-reduce (cfg.head_mode) then mean|max over
    layers (cfg.layer_mode) → [m, n_vis] fp32. The m latent steps stay as the
    query axis so ②(b) contrast has a baseline; reduction over m happens in ③."""
    if not l2v:
        return None
    mats = [r for li, a in l2v.items()
            if (r := _reduce_heads(a, 1, cfg, li, fixed_map)) is not None]
    return _agg_layers(mats, cfg.layer_mode) if mats else None


# ─────────────────────────────────────────────────────────────────────────────
# ② sink removal
# ─────────────────────────────────────────────────────────────────────────────
def _renorm(x: torch.Tensor) -> torch.Tensor:
    """②(a) per-query renorm over vision tokens → rows sum to 1."""
    return x / x.sum(dim=-1, keepdim=True).clamp_min(_EPS)


def _contrast(x: torch.Tensor) -> torch.Tensor:
    """②(b) subtract per-token baseline over the query axis, clamp at 0.

    A register/sink patch is attended by *every* query, so its column mean is
    high and it cancels; a content patch attended only by relevant queries
    survives. Needs >1 query row to be meaningful.
    """
    if x.shape[0] < 2:
        return x
    return (x - x.mean(dim=0, keepdim=True)).clamp_min(0.0)


def _reduce(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """③ average the selected query rows → [n_vis]."""
    if mask is not None and mask.any():
        x = x[mask]
    return x.mean(dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# normalization + pooling
# ─────────────────────────────────────────────────────────────────────────────
def _minmax(x: torch.Tensor) -> torch.Tensor:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo).clamp_min(_EPS)


def _minmax_ref(x: torch.Tensor, ref: Optional[tuple] = None) -> tuple:
    """minmax with an OPTIONAL external (lo, hi) reference.

    Phase-1 (raw cross-ROI score): the ③ per-source minmax that makes Rel/Sal
    comparable is GLOBAL over the token set it sees. For the initial multi-ROI
    pass that global set spans every ROI, so the combined `s` is cross-ROI
    comparable. An APPENDED crop is scored alone, so its own (lo, hi) would be
    within-crop → not comparable to the initial ROIs. Passing the initial pass's
    (lo, hi) here puts the append on the SAME ruler. Returns (normed, (lo, hi))
    so the initial pass can hand its ref forward."""
    if ref is not None:
        lo, hi = float(ref[0]), float(ref[1])
    else:
        lo, hi = float(x.min()), float(x.max())
    span = (hi - lo)
    span = span if span > _EPS else _EPS
    return (x - lo) / span, (lo, hi)


def _normalize(x: torch.Tensor, how: str) -> torch.Tensor:
    if how == "minmax":
        return _minmax(x)
    if how == "rank":
        order = x.argsort().argsort().float()
        return order / max(len(order) - 1, 1)
    return x


def _pool_grid(g: torch.Tensor, cfg: SaliencyConfig) -> torch.Tensor:
    """Local pool a [h, w] saliency grid (avg). 1D operates along width only."""
    if cfg.pool == "none" or cfg.kappa <= 1:
        return g
    k = cfg.kappa | 1                     # force odd
    pad = k // 2
    x = g[None, None]                     # [1,1,h,w]
    if cfg.pool == "2d":
        x = torch.nn.functional.avg_pool2d(x, kernel_size=k, stride=1, padding=pad)
    else:  # "1d": pool along width, raster rows (note: wraps at row ends — 2d preferred)
        x = torch.nn.functional.avg_pool2d(x, kernel_size=(1, k), stride=1, padding=(0, pad))
    return x[0, 0]


# ─────────────────────────────────────────────────────────────────────────────
# main entry
# ─────────────────────────────────────────────────────────────────────────────
def roi_saliency(
    q2v: dict[int, torch.Tensor],            # {layer: [H, n_q, n_vis]}
    l2v: dict[int, torch.Tensor],            # {layer: [m, H, n_vis]}
    roi_spans: list[RoiSpan],
    cfg: SaliencyConfig,
    qq_mask: Optional[torch.Tensor] = None,  # bool [n_q]; None → all text rows
    bg_mask: Optional[torch.Tensor] = None,  # bool [n_q]; template rows B for --saliency_bg_correct
    qk_proxy_fn: Optional[Callable[[], tuple[torch.Tensor, torch.Tensor]]] = None,
    norm_ref: Optional[dict] = None,         # {"rel":(lo,hi), "sal":(lo,hi)} external ③ ref
    return_raw: bool = False,                # also return pre-④ `s`-grids + the ③ ref used
):
    """Return {roi_id: saliency grid [h, w]} (fp32, normalized per cfg.norm).

    `qk_proxy_fn` is the S2 ablation hook: a callable returning (Rel, Sal) raw
    [..., n_vis] scores from pre-softmax q·k. Not wired yet (needs the backbone
    to expose query vectors); cfg.score="qk_proxy" without it raises.

    Phase-1 cross-ROI raw score: pass `return_raw=True` to also get
    {roi_id: raw_grid} (the pooled ③-combined `s` BEFORE the per-ROI ④ minmax —
    cross-ROI comparable because ③'s minmax spans all ROIs of THIS call) and the
    ③ reference {"rel":(lo,hi),"sal":(lo,hi)}. Feed that ref back as `norm_ref`
    when scoring a later APPENDED crop alone so it lands on the SAME ruler
    (returns (out, raw, ref) instead of just out).
    """
    # token-level Rel / Sal over the full vision sequence -------------------------
    if cfg.score == "qk_proxy":
        if qk_proxy_fn is None:
            raise NotImplementedError(
                "score='qk_proxy' needs query vectors from the backbone "
                "(hook not yet wired); use score='probs' or pass qk_proxy_fn."
            )
        rel_tok, sal_tok = qk_proxy_fn()                     # [n_q,n_vis], [m,n_vis]
    else:
        fixed_map = parse_head_fixed(cfg.head_fixed)
        rel_tok = _agg_q2v(q2v, cfg, fixed_map)             # [n_q, n_vis] or None
        sal_tok = _agg_l2v(l2v, cfg, fixed_map)             # [m, n_vis] or None

    n_vis = roi_spans[-1].start + roi_spans[-1].length if roi_spans else 0

    def _branch(x: Optional[torch.Tensor], mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x is None:
            return None
        if cfg.renorm:
            x = _renorm(x)
        if cfg.contrast:
            x = _contrast(x)
        return _reduce(x, mask)                              # [n_vis]

    use_rel = cfg.source in ("rel", "both")
    use_sal = cfg.source in ("sal", "both")

    # ②(b') template-bias correction: S^b = [ mu^b - rho ]_+ with a SHARED background
    # prior rho = Phi(B) (renorm mean over template rows B of the SAME q2v aggregation,
    # so head_mode carries — max_h → max-max). Replaces the query-row contrast. rho is
    # estimated once from the text rows and subtracted from BOTH branches. Falls back
    # to the legacy _branch path when off / no bg_mask / no rel_tok (byte-identical).
    _bg = (getattr(cfg, "bg_correct", False) and bg_mask is not None
           and rel_tok is not None and bool(bg_mask.any()))
    if _bg:
        rel_r = _renorm(rel_tok) if cfg.renorm else rel_tok
        rho = _reduce(rel_r, bg_mask)                        # [n_vis] rho_c = Phi_c(B)

        def _branch_bg(x: Optional[torch.Tensor], mask: Optional[torch.Tensor],
                       renormed: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
            if x is None:
                return None
            xr = renormed if renormed is not None else (_renorm(x) if cfg.renorm else x)
            return (_reduce(xr, mask) - rho).clamp_min(0.0)   # [mu - rho]_+

        rel = _branch_bg(rel_tok, qq_mask if cfg.qq_only else None, rel_r) if use_rel else None
        sal = _branch_bg(sal_tok, None) if use_sal else None
    else:
        rel = _branch(rel_tok, qq_mask if cfg.qq_only else None) if use_rel else None
        sal = _branch(sal_tok, None) if use_sal else None

    # combine (minmax each first so Rel/Sal scales are comparable) ----------------
    # ③ minmax uses an external ref when given (norm_ref) so an appended crop is
    # normalized on the initial pass's GLOBAL scale, not its own within-crop range.
    _ref = norm_ref or {}
    s = torch.zeros(n_vis, dtype=torch.float32)
    rel_ref = sal_ref = None
    if rel is not None:
        rel_n, rel_ref = _minmax_ref(rel, _ref.get("rel"))
        s = s + cfg.lam_r * rel_n
    if sal is not None:
        sal_n, sal_ref = _minmax_ref(sal, _ref.get("sal"))
        s = s + cfg.lam_s * sal_n

    # slice per ROI → reshape → pool → (raw, ④-normalize) ------------------------
    out: dict[int, torch.Tensor] = {}
    raw: dict[int, torch.Tensor] = {}
    for r in roi_spans:
        flat = s[r.start:r.start + r.length]
        if r.h * r.w != r.length:                            # safety: fall back to 1×N
            grid = flat.view(1, -1)
        else:
            grid = flat.view(r.h, r.w)
        grid = _pool_grid(grid, cfg)
        if return_raw:
            raw[r.id] = grid.clone()                         # pre-④ (cross-ROI comparable)
        out[r.id] = _normalize(grid.reshape(-1), cfg.norm).view_as(grid)
    if return_raw:
        return out, raw, {"rel": rel_ref, "sal": sal_ref}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Q_q mask — role tagging of text query rows  (the one real plumbing dependency)
# ─────────────────────────────────────────────────────────────────────────────
def build_qq_mask(n_q: int, role_spans: Optional[list[tuple[int, int]]] = None
                  ) -> Optional[torch.Tensor]:
    """Boolean mask over the n_q text query rows selecting question/choice/
    diagnosis tokens.

    role_spans: list of (start, end) token index ranges to KEEP. When None
    (MVP), returns None → caller averages over *all* text rows. Precise tagging
    requires the embed step to emit these ranges (see reasoner wiring TODO).
    """
    if role_spans is None:
        return None
    mask = torch.zeros(n_q, dtype=torch.bool)
    for a, b in role_spans:
        mask[max(0, a):min(n_q, b)] = True
    return mask if mask.any() else None


def build_bg_mask(tail_ids, tokenizer, targets, n_q: int) -> Optional[torch.Tensor]:
    """Boolean mask over the n_q post-image text rows marking the template rows B
    for --saliency_bg_correct. `targets` = fixed, task-invariant template strings
    (e.g. REASONER_TAIL); each is subsequence-matched in `tail_ids` (tail-relative
    == the n_q axis). Rows contain no question-/slide-/region-specific info.
    Returns None (→ correction inert) when nothing matches / inputs missing."""
    if tail_ids is None or tokenizer is None or not targets:
        return None
    idl = tail_ids.tolist() if hasattr(tail_ids, "tolist") else list(tail_ids)
    nt = len(idl)
    mask = torch.zeros(n_q, dtype=torch.bool)
    for tgt in targets:
        tgt = (str(tgt) if tgt is not None else "").strip()
        if not tgt:
            continue
        for variant in (" " + tgt, "\n" + tgt, tgt):
            try:
                seq = tokenizer.encode(variant, add_special_tokens=False)
            except Exception:
                seq = []
            L = len(seq)
            if 0 < L <= nt:
                hit = next((i for i in range(0, nt - L + 1) if idl[i:i + L] == seq), None)
                if hit is not None:
                    mask[max(0, hit):min(n_q, hit + L)] = True
                    break
    return mask if mask.any() else None
