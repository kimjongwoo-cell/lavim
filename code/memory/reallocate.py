"""S8 — training-free attention reallocation (ViF-style), production hook.

Ported from stage1_saliency_probe/diag_reallocation.py (isolation-validated:
decode visual share 3.5%→16–25% with alpha 0.3–0.5, no garbage). Moves a
fraction `alpha` of each query's attention mass FROM non-vision keys (text/latent)
TO vision keys, in mid+deep layers. No key is removed (question preserved) — only
re-weighted, so it is safe on structured decode.

Usage (Diagnosis decode):
    from memory.reallocate import reallocating
    with reallocating(vis_cols, alpha=0.15,
                      layers=parse_layers("14-30", num_layers(backbone))):
        ...manual eager decode loop through eager attention...

The context manager installs a monkeypatch on Qwen3-VL's eager attention and
restores it on exit. The hook only fires while CTX["on"] and the current layer is
in CTX["layers"] and CTX["vis"] is set — so it is inert unless explicitly entered.
IMPORTANT: the decode must run through EAGER attention for the patch to fire.
The model is configured with the eager implementation at load time; callers do
not need to materialize and return full attention tensors merely to use this
hook or its lightweight pre-intervention mass probe.
"""
from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import TypedDict

import torch
import torch.nn.functional as F
import transformers.models.qwen3_vl.modeling_qwen3_vl as Qmod

_ORIG = Qmod.eager_attention_forward
_INSTALLED = False

# Per-step α SHAPE for the Verifier refinement schedule (phase0k, n=20, m=5):
# deficit-driven α [0.275,0.260,0.234,0.215,0.217] normalized to mean 1.0, so the
# schedule α_base*SHAPE has the SAME mean as the constant α_base → schedule-vs-constant
# is a matched-mean A/B (pure shape effect). Higher early (r_v lowest at step 0),
# eased later. m!=5 → np.interp in the caller.
VERIFIER_RAMP_SHAPE = [1.145, 1.082, 0.974, 0.895, 0.903]


@dataclass(slots=True)
class ReallocationProbe:
    """Low-memory mean of natural vision attention seen by active hook calls."""

    total_mass: float = 0.0
    observations: int = 0
    total_after: float = 0.0
    observations_after: int = 0

    def observe(self, mass: float) -> None:
        self.total_mass += float(mass)
        self.observations += 1

    def observe_after(self, mass: float) -> None:
        """Record post-intervention vision mass for mechanism reporting."""
        self.total_after += float(mass)
        self.observations_after += 1

    @property
    def mean_mass(self) -> float | None:
        if self.observations == 0:
            return None
        return self.total_mass / self.observations

    @property
    def mean_after(self) -> float | None:
        if self.observations_after == 0:
            return None
        return self.total_after / self.observations_after


class _ReallocationState(TypedDict):
    on: bool
    layers: set[int]
    alpha: float
    vis: torch.Tensor | None
    vis_w: torch.Tensor | None
    visual_patch_ids: torch.Tensor | None
    parent_patch_indices: tuple[int, ...]
    parent_anchor_strength: float
    relay_ratio: float
    relay_source: str
    relay_mode: str
    mode: str
    decode_only: bool
    prefill_only: bool
    probe: ReallocationProbe | None


CTX: _ReallocationState = {
    "on": False,
    "layers": set(),
    "alpha": 0.0,
    "vis": None,
    "vis_w": None,
    "visual_patch_ids": None,
    "parent_patch_indices": (),
    "parent_anchor_strength": 0.0,
    "relay_ratio": 1.0,
    "relay_source": "attention",
    "relay_mode": "topk",
    "mode": "realloc",
    "decode_only": False,
    "prefill_only": False,
    "probe": None,
}


def _vision_attention_mass(aw: torch.Tensor, vis_idx: torch.Tensor) -> float:
    """Measure pre-intervention attention mass assigned to visual keys."""
    key_count = aw.shape[-1]
    if vis_idx.dtype == torch.bool and vis_idx.ndim == 2:
        recipients = torch.zeros(
            (aw.shape[0], key_count),
            dtype=torch.bool,
            device=aw.device,
        )
        width = min(key_count, int(vis_idx.shape[1]))
        recipients[:, :width] = vis_idx[:, :width].to(device=aw.device)
        recipient_mask = recipients[:, None, None, :]
    else:
        columns = vis_idx[vis_idx < key_count]
        recipient_mask = torch.zeros(key_count, dtype=torch.bool, device=aw.device)
        recipient_mask[columns] = True
    return float((aw.float() * recipient_mask).sum(dim=-1).mean().item())


def set_active_alpha(a: float) -> None:
    """Mutate the live reallocation strength mid-context (per-step schedule).
    No-op unless a reallocating() block is active, so callers outside a context
    (or with α_base=0 → patch not installed) are unaffected."""
    if CTX["on"]:
        CTX["alpha"] = float(a)


def get_active_alpha() -> float:
    """Current live α (for save/restore around a per-step schedule)."""
    return float(CTX["alpha"])


def mask_vision(aw, vis_idx):
    """aw: [b,h,q,k] post-softmax. ZERO the attention on vision columns and
    renormalize each query row over the surviving (non-vision) keys, so the query
    attends AS IF vision were absent — a true vision-OFF counterfactual (vs
    reallocate()'s re-weighting). Rows whose surviving mass is ~0 are left
    unchanged (degenerate guard)."""
    k = aw.shape[-1]
    v = vis_idx[vis_idx < k]
    keep = torch.ones(k, dtype=torch.bool, device=aw.device)
    keep[v] = False
    af = aw.float() * keep                       # zero vision cols
    denom = af.sum(-1, keepdim=True)
    af = torch.where(denom > 1e-9, af / denom.clamp_min(1e-9), aw.float())
    return af.to(aw.dtype)


def _visual_relay_mask(
    attention: torch.Tensor,
    visual_mask: torch.Tensor,
    relay_ratio: float,
    relay_scores: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select query-conditioned visual relays independently per batch/head/query."""
    expanded_visual = visual_mask.expand_as(attention)
    if relay_ratio >= 1.0:
        return expanded_visual
    relay = torch.zeros_like(expanded_visual)
    scores = (
        attention
        if relay_scores is None
        else relay_scores[:, None, None, :].expand_as(attention)
    ).masked_fill(~expanded_visual, float("-inf"))
    for batch_index in range(attention.shape[0]):
        visual_count = int(visual_mask[batch_index].sum().item())
        relay_count = max(1, math.ceil(visual_count * relay_ratio))
        indices = scores[batch_index].topk(relay_count, dim=-1).indices
        relay[batch_index].scatter_(-1, indices, True)
    return relay & expanded_visual


def reallocate(
    aw,
    vis_idx,
    alpha,
    vis_w=None,
    relay_ratio: float = 1.0,
    relay_scores: torch.Tensor | None = None,
    relay_mode: str = "topk",
    visual_patch_ids: torch.Tensor | None = None,
    parent_patch_indices: tuple[int, ...] = (),
    parent_anchor_strength: float = 0.0,
):
    """aw: [b,h,q,k] post-softmax. Move `alpha` of non-vision mass to vision cols.

    relay_mode="topk" concentrates the moved mass on the relay_ratio fraction of
    visual columns ranked by relay_scores (ViF-style selection).
    relay_mode="proportional" spreads it across ALL visual columns proportionally
    to their per-column saliency scores (AttnReal-style soft redistribution), so
    global tissue context is preserved while stable regions still gain mass.

    Distribution among vision cols (P1-3, `--attn_realloc_weight`):
      • vis_w is None  → proportional to each col's EXISTING attention weight
        (amplifies whatever the decode already looks at — incl. sink/background).
      • vis_w given    → proportional to per-col SALIENCY (aligned to vis_idx), so
        the moved mass lands on diagnostically salient tokens regardless of the
        current (mis-grounded) distribution. Falls back to existing-weight when the
        saliency vector is degenerate (all ~0), to avoid leaking mass."""
    k = aw.shape[-1]
    if isinstance(vis_idx, torch.Tensor) and vis_idx.dtype == torch.bool and vis_idx.ndim == 2:
        rec = torch.zeros((aw.shape[0], k), dtype=torch.bool, device=aw.device)
        width = min(k, int(vis_idx.shape[1]))
        rec[:, :width] = vis_idx[:, :width].to(device=aw.device)
        rec = rec[:, None, None, :]
        donor = ~rec
        af = aw.float()
        has_recipients = rec.any(dim=-1, keepdim=True)
        moved = float(alpha) * has_recipients * (af * donor).sum(-1, keepdim=True)
        af = af * torch.where(donor, 1.0 - float(alpha) * has_recipients, 1.0)
        if relay_mode == "proportional" and relay_scores is not None:
            weights = relay_scores[:, None, None, :].expand_as(af) * rec
            w_sum = weights.sum(-1, keepdim=True).clamp_min(1e-9)
            return (af + moved * (weights / w_sum)).to(aw.dtype)
        relay = _visual_relay_mask(af, rec, relay_ratio, relay_scores)
        rec_w = af * relay
        rec_sum = rec_w.sum(-1, keepdim=True).clamp_min(1e-9)
        return (af + moved * (rec_w / rec_sum)).to(aw.dtype)
    mask = vis_idx < k
    v = vis_idx[mask]
    rec = torch.zeros(k, dtype=torch.bool, device=aw.device)
    rec[v] = True
    donor = ~rec
    af = aw.float()
    donor_mass = (af * donor).sum(-1, keepdim=True)
    moved = alpha * donor_mass
    af = af * torch.where(donor, 1.0 - alpha, 1.0)

    if vis_w is not None:
        w_full = torch.zeros(k, dtype=torch.float32, device=aw.device)
        w_full[v] = vis_w[mask].float().clamp_min(0.0)
        w_sum = w_full.sum()
        if w_sum > 1e-9:
            # Preserve the original query-conditioned relevance and use the
            # pathology hierarchy only as a multiplicative structural prior.
            # Static weights alone erase which morphology the current query was
            # already reading and caused unrelated coarse/fine evidence to gain
            # equal influence.
            weighted = af * w_full
            weighted_sum = weighted.sum(-1, keepdim=True)
            static_fallback = w_full / w_sum
            recipients = torch.where(
                weighted_sum > 1e-9,
                weighted / weighted_sum.clamp_min(1e-9),
                static_fallback,
            )
            af = af + moved * recipients
            return af.to(aw.dtype)
    if (
        visual_patch_ids is not None
        and visual_patch_ids.numel() == v.numel()
        and parent_anchor_strength > 0.0
    ):
        patch_full = torch.full((k,), -1, dtype=torch.long, device=aw.device)
        patch_full[v] = visual_patch_ids[mask].to(device=aw.device, dtype=torch.long)
        natural_visual = af * rec
        context = torch.zeros_like(natural_visual)
        for child, parent in enumerate(parent_patch_indices):
            if parent < 0:
                continue
            child_mask = patch_full == child
            parent_mask = patch_full == parent
            if not bool(child_mask.any()) or not bool(parent_mask.any()):
                continue
            child_mass = (natural_visual * child_mask).sum(-1, keepdim=True)
            parent_attention = natural_visual * parent_mask
            parent_total = parent_attention.sum(-1, keepdim=True)
            parent_uniform = parent_mask.to(dtype=af.dtype) / parent_mask.sum().clamp_min(1)
            parent_distribution = torch.where(
                parent_total > 1e-9,
                parent_attention / parent_total.clamp_min(1e-9),
                parent_uniform,
            )
            context = context + float(parent_anchor_strength) * child_mass * parent_distribution
        recipient_scores = natural_visual + context
        recipient_total = recipient_scores.sum(-1, keepdim=True)
        if bool((recipient_total > 1e-9).any()):
            fallback = natural_visual / natural_visual.sum(-1, keepdim=True).clamp_min(1e-9)
            recipients = torch.where(
                recipient_total > 1e-9,
                recipient_scores / recipient_total.clamp_min(1e-9),
                fallback,
            )
            return (af + moved * recipients).to(aw.dtype)
    # existing-weight redistribution (default / degenerate-saliency fallback)
    visual_mask = rec[None, None, None, :].expand(aw.shape[0], 1, 1, k)
    relay = _visual_relay_mask(af, visual_mask, relay_ratio, relay_scores)
    rec_w = af * relay
    rec_sum = rec_w.sum(-1, keepdim=True).clamp_min(1e-9)
    af = af + moved * (rec_w / rec_sum)
    return af.to(aw.dtype)


def _patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    ks = Qmod.repeat_kv(key, module.num_key_value_groups)
    vs = Qmod.repeat_kv(value, module.num_key_value_groups)
    aw = torch.matmul(query, ks.transpose(2, 3)) * scaling
    if attention_mask is not None:
        aw = aw + attention_mask[:, :, :, : ks.shape[-2]]
    aw = F.softmax(aw, dim=-1, dtype=torch.float32).to(query.dtype)
    li = getattr(module, "layer_idx", -1)
    q_len = query.shape[-2]
    scope_ok = (
        (not CTX.get("decode_only", False) or q_len == 1)
        and (not CTX.get("prefill_only", False) or q_len > 1)
    )
    if CTX["on"] and li in CTX["layers"] and CTX["vis"] is not None and scope_ok:
        probe = CTX.get("probe")
        if probe is not None:
            probe.observe(_vision_attention_mass(aw, CTX["vis"]))
        if CTX.get("mode") == "mask":
            aw = mask_vision(aw, CTX["vis"])
        else:
            relay_scores = (
                ks.float().norm(dim=-1).mean(dim=1)
                if CTX["relay_source"] == "key_norm"
                else None
            )
            aw = reallocate(
                aw,
                CTX["vis"],
                CTX["alpha"],
                CTX["vis_w"],
                CTX["relay_ratio"],
                relay_scores,
                CTX.get("relay_mode", "topk"),
                CTX.get("visual_patch_ids"),
                CTX.get("parent_patch_indices", ()),
                CTX.get("parent_anchor_strength", 0.0),
            )
        if probe is not None:
            probe.observe_after(_vision_attention_mass(aw, CTX["vis"]))
    aw = F.dropout(aw, p=dropout, training=module.training)
    out = torch.matmul(aw, vs).transpose(1, 2).contiguous()
    return out, aw


def install():
    global _INSTALLED
    if not _INSTALLED:
        Qmod.eager_attention_forward = _patched
        _INSTALLED = True


def uninstall():
    global _INSTALLED
    if _INSTALLED:
        Qmod.eager_attention_forward = _ORIG
        _INSTALLED = False


@contextlib.contextmanager
def reallocating(vis_cols, alpha: float, layers, device=None, vis_weight=None,
                 decode_only: bool = False, relay_ratio: float = 1.0,
                 relay_source: str = "attention", relay_mode: str = "topk",
                 prefill_only: bool = False, visual_patch_ids=None,
                 parent_patch_indices: tuple[int, ...] = (),
                 parent_anchor_strength: float = 0.0):
    """Activate reallocation for the duration of the block.

    vis_cols:   1-D LongTensor/list of absolute vision-key columns, or a [B,K] bool
                mask when direct-HF micro-batching packs slides with different KV maps.
    alpha:      fraction of non-vision mass moved to vision (0 = no-op).
    layers:     iterable of layer indices to reallocate (e.g. range(14, 31)).
    vis_weight: optional per-col saliency aligned to vis_cols (P1-3). None → moved
                mass distributed by existing attention weight; given → by saliency.
    """
    if vis_cols is None or len(vis_cols) == 0:
        yield ReallocationProbe()  # inert: no visual keys to observe or reallocate
        return
    dev = device if device is not None else "cpu"
    if isinstance(vis_cols, torch.Tensor) and vis_cols.dtype == torch.bool and vis_cols.ndim == 2:
        vt = vis_cols.to(device=dev, dtype=torch.bool)
    else:
        vt = torch.as_tensor(vis_cols, dtype=torch.long, device=dev)
    vw = None
    if vis_weight is not None and len(vis_weight) == len(vis_cols):
        vw = torch.as_tensor(vis_weight, dtype=torch.float32, device=dev)
    patch_ids = None
    if visual_patch_ids is not None and len(visual_patch_ids) == len(vis_cols):
        patch_ids = torch.as_tensor(visual_patch_ids, dtype=torch.long, device=dev)
    prev = CTX.copy()
    probe = ReallocationProbe()
    CTX["on"] = True
    CTX["layers"] = set(int(x) for x in layers)
    CTX["alpha"] = float(alpha)
    CTX["vis"] = vt
    CTX["vis_w"] = vw
    CTX["visual_patch_ids"] = patch_ids
    CTX["parent_patch_indices"] = tuple(int(index) for index in parent_patch_indices)
    CTX["parent_anchor_strength"] = float(parent_anchor_strength)
    CTX["relay_ratio"] = float(relay_ratio)
    CTX["relay_source"] = relay_source
    CTX["relay_mode"] = relay_mode
    CTX["mode"] = "realloc"
    CTX["decode_only"] = bool(decode_only)
    CTX["prefill_only"] = bool(prefill_only)
    CTX["probe"] = probe
    install()
    try:
        yield probe
    finally:
        CTX["on"] = prev["on"]
        CTX["layers"] = prev["layers"]
        CTX["alpha"] = prev["alpha"]
        CTX["vis"] = prev["vis"]
        CTX["vis_w"] = prev.get("vis_w")
        CTX["visual_patch_ids"] = prev.get("visual_patch_ids")
        CTX["parent_patch_indices"] = prev.get("parent_patch_indices", ())
        CTX["parent_anchor_strength"] = prev.get("parent_anchor_strength", 0.0)
        CTX["relay_ratio"] = prev.get("relay_ratio", 1.0)
        CTX["relay_source"] = prev.get("relay_source", "attention")
        CTX["relay_mode"] = prev.get("relay_mode", "topk")
        CTX["mode"] = prev.get("mode", "realloc")
        CTX["decode_only"] = prev.get("decode_only", False)
        CTX["prefill_only"] = prev.get("prefill_only", False)
        CTX["probe"] = prev.get("probe")
        if not CTX["on"]:
            uninstall()


@contextlib.contextmanager
def masking(vis_cols, layers, device=None):
    """Zero vision attention for the duration of the block (true vision-OFF
    counterfactual). Mirror of reallocating() but installs the mask branch; `alpha`
    is unused. Inert when vis_cols is empty."""
    if vis_cols is None or len(vis_cols) == 0:
        yield  # inert: no vision to mask
        return
    dev = device if device is not None else "cpu"
    vt = torch.as_tensor(vis_cols, dtype=torch.long, device=dev)
    prev = CTX.copy()
    CTX["on"] = True
    CTX["layers"] = set(int(x) for x in layers)
    CTX["vis"] = vt
    CTX["mode"] = "mask"
    install()
    try:
        yield
    finally:
        CTX["on"] = prev["on"]
        CTX["layers"] = prev["layers"]
        CTX["alpha"] = prev["alpha"]
        CTX["vis"] = prev["vis"]
        CTX["vis_w"] = prev.get("vis_w")
        CTX["mode"] = prev.get("mode", "realloc")
        CTX["decode_only"] = prev.get("decode_only", False)
        if not CTX["on"]:
            uninstall()


def num_layers(backbone) -> int:
    """LM depth of the LIVE backbone — the basis every relative layer spec
    ('all', 'early'/'mid'/'late') resolves against. Every backbone exposes
    `self.lm` (the decoder) with `.layers`; `full_attn_indices` is the fallback
    (its max+1 is the depth even when the band itself is filtered, as on qwen35).
    Raises rather than guessing: a wrong depth silently shifts the thirds."""
    lm = getattr(backbone, "lm", None)
    layers = getattr(lm, "layers", None)
    if layers is not None:
        return len(layers)
    fai = getattr(backbone, "full_attn_indices", None)
    if fai:
        return int(max(fai)) + 1
    raise ValueError(
        f"cannot determine LM depth of {type(backbone).__name__} (no lm.layers / "
        f"full_attn_indices) — pass n_layers explicitly to parse_layers()")


def _third(n_layers: int, k: int) -> range:
    """k-th third (0=early, 1=mid, 2=late) of an n_layers-deep stack. Boundaries
    are floor(k*n/3), so the thirds partition range(n_layers) exactly for any n
    (n=36 → 0-11 / 12-23 / 24-35; n=32 → 0-9 / 10-20 / 21-31)."""
    return range((k * n_layers) // 3, ((k + 1) * n_layers) // 3)


_THIRDS = {"early": 0, "mid": 1, "late": 2}


def parse_layers(spec: str, n_layers: int) -> set:
    """Parse a layer spec against the live model depth `n_layers` (see num_layers).

    Parts are comma-separable and may be mixed ('mid,late', '0-3,late'):
      'all' / '*'                → every layer
      'early' / 'mid' / 'late'   → thirds of the stack, derived from n_layers
      '14-30'                    → inclusive range
      '18'                       → single layer
    Indices outside [0, n_layers) are dropped with a warning — a band written for
    a deeper model would otherwise silently apply to fewer layers than intended.

    NOTE 'mid' here is the middle THIRD (n=36 → 12-23), which is NOT the default
    S8 band '14-30' (mid+deep). They are different bands by design.
    """
    spec = str(spec).strip().lower()
    out: set = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if part in ("all", "*"):
            out.update(range(n_layers))
        elif part in _THIRDS:
            out.update(_third(n_layers, _THIRDS[part]))
        elif "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    keep = {li for li in out if 0 <= li < n_layers}
    if keep != out:
        print(f"[reallocate] parse_layers({spec!r}): dropped out-of-range layers "
              f"{sorted(out - keep)} (model depth {n_layers})")
    return keep
