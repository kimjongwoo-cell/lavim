"""AAVM runtime: capture acquisition addresses and re-address visual keys.

Kept out of ``backbone/qwen3vl.py`` on purpose — the backbone only opens a
capture context around its Reasoner prefill and makes one call at the
consolidation boundary; everything else lives here.

Pipeline position (method draft 0908):

    Visual bank -> C1 (Observation-Stratified compression)
                -> AAVM (this module)
                -> Reasoner x m latent steps
                -> Answerer

so the re-addressed evidence is what the Reasoner integrates, rather than
something the terminal Answerer is asked to read out of mid-cache.

Space handling. Keys are cached AFTER RoPE, but an address direction taken
from rotated vectors carries the phase of wherever the query tokens sat. So
both the crop centroid and the acquisition address are formed in the PRE-RoPE
(position-free) key space -- ``k_norm(k_proj(h))``, exactly what the model
rotates -- and only the resulting shift is rotated, once, by the crop's
representative position before being added uniformly to the cached keys. That
keeps the crop-uniform addressing factor exp[q^T dk_p] exact for every token
of the crop, which a pre-RoPE shift added before re-rotation would not.

Env:
    VLMAS_AAVM=1              enable v1: address = pre-RoPE KEY centroid of the
                              evidence-query span (off/unset = byte-identical)
    VLMAS_AAVM=2              enable v2: address = pre-RoPE QUERY representation
                              u_g = mean over T_g of q_pre (the retrieval
                              direction at acquisition time), GQA-reduced to
                              the KV heads. Shift math identical to v1.
    VLMAS_AAVM_CONTROL=shuffled_address   falsification arm (derangement)
    VLMAS_AAVM_LAYERS=all|<i,j,k>         layers to re-address (default all)
"""
from __future__ import annotations

import contextlib
import os

import torch

from memory.aavm import (
    address_shift,
    apply_crop_shift,
    crop_spans,
    reduce_query_heads,
    rotate_vector,
    shuffled_addresses,
)


def mode() -> str:
    return os.environ.get("VLMAS_AAVM", "").strip()


def enabled() -> bool:
    return mode() in ("1", "2")


def _layer_indices(backbone) -> list[int]:
    spec = os.environ.get("VLMAS_AAVM_LAYERS", "all").strip()
    total = len(backbone.lm.layers)
    if spec in ("", "all"):
        return list(range(total))
    picked = [int(part) for part in spec.replace(" ", "").split(",") if part]
    return [index for index in picked if 0 <= index < total]


def _pre_rope_keys(module, hidden: torch.Tensor) -> torch.Tensor:
    """k_norm(k_proj(h)) -> [B, H_kv, L, D]; the model's own pre-RoPE keys."""
    shape = (*hidden.shape[:-1], -1, module.head_dim)
    return module.k_norm(module.k_proj(hidden).view(shape)).transpose(1, 2)


def _pre_rope_queries(module, hidden: torch.Tensor) -> torch.Tensor:
    """q_norm(q_proj(h)) -> [B, H_q, L, D]; the model's own pre-RoPE queries."""
    shape = (*hidden.shape[:-1], -1, module.head_dim)
    return module.q_norm(module.q_proj(hidden).view(shape)).transpose(1, 2)


def _num_kv_heads(module) -> int:
    return int(module.k_proj.out_features) // int(module.head_dim)


@contextlib.contextmanager
def capture_visual_raw_keys(backbone, vision_columns_abs: torch.Tensor):
    """Collect pre-RoPE keys and rotation phases at the visual columns.

    Yields {"keys": {layer: [H, nV, D]}, "cos": [nV, D], "sin": [nV, D]} filled
    during the wrapped prefill. Chunked prefills accumulate in column order.
    """
    store: dict[str, object] = {}
    if not enabled() or vision_columns_abs.numel() == 0:
        yield store
        return
    layers = _layer_indices(backbone)
    columns = vision_columns_abs.detach().to(
        device=backbone.device, dtype=torch.long)
    parts: dict[int, list[torch.Tensor]] = {index: [] for index in layers}
    phase: list[tuple[torch.Tensor, torch.Tensor]] = []
    handles: list[object] = []

    def make_hook(layer_index: int):
        def hook(module, args, kwargs, _output):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            position_embeddings = kwargs.get(
                "position_embeddings", args[1] if len(args) > 1 else None)
            cache = kwargs.get(
                "past_key_values", args[3] if len(args) > 3 else None)
            if hidden is None or cache is None or position_embeddings is None:
                return
            chunk_len = int(hidden.shape[1])
            start = int(cache.layers[layer_index].keys.shape[2]) - chunk_len
            local = columns - start
            inside = local[(local >= 0) & (local < chunk_len)]
            if inside.numel() == 0:
                return
            raw = _pre_rope_keys(module, hidden)          # [1, H, L, D]
            parts[layer_index].append(
                raw[0].index_select(1, inside).float().cpu())
            if layer_index == layers[0]:
                cos, sin = position_embeddings
                phase.append((
                    cos[0].index_select(0, inside).float().cpu(),
                    sin[0].index_select(0, inside).float().cpu(),
                ))

        return hook

    for index in layers:
        handles.append(
            backbone.lm.layers[index].self_attn.register_forward_hook(
                make_hook(index), with_kwargs=True))
    try:
        yield store
    finally:
        for handle in handles:
            handle.remove()
        if phase and all(parts[index] for index in layers):
            store["keys"] = {
                index: torch.cat(parts[index], dim=1) for index in layers
            }
            store["cos"] = torch.cat([c for c, _ in phase], dim=0)
            store["sin"] = torch.cat([s for _, s in phase], dim=0)
        else:
            print(f"[AAVM] capture EMPTY: nV={int(columns.numel())} "
                  f"phase_chunks={len(phase)} layers_hit="
                  f"{sum(1 for i in layers if parts[i])}/{len(layers)}",
                  flush=True)


@torch.no_grad()
def query_address(backbone, text: str) -> dict[int, torch.Tensor]:
    """Pre-RoPE key centroid of one evidence-query span, per layer.

    The span is the evidence_query text alone -- no chat template, no JSON
    syntax, no coordinates -- which is what the method defines as T_g.
    """
    layers = _layer_indices(backbone)
    tokens = backbone.processor.tokenizer(
        text, return_tensors="pt", add_special_tokens=False)
    ids = tokens["input_ids"].to(backbone.device)
    if int(ids.shape[1]) == 0:
        return {}
    captured: dict[int, torch.Tensor] = {}
    handles: list[object] = []

    def make_hook(layer_index: int):
        def hook(module, args, kwargs, _output):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            if hidden is None:
                return
            raw = _pre_rope_keys(module, hidden)          # [1, H, L, D]
            captured[layer_index] = raw[0].mean(dim=1).float().cpu()  # [H, D]

        return hook

    for index in layers:
        handles.append(
            backbone.lm.layers[index].self_attn.register_forward_hook(
                make_hook(index), with_kwargs=True))
    try:
        _ = backbone.lm(
            inputs_embeds=backbone.lm.embed_tokens(ids),
            position_ids=backbone._make_position_ids(int(ids.shape[1])),
            use_cache=False,
        )
    finally:
        for handle in handles:
            handle.remove()
    return captured


@torch.no_grad()
def query_address_v2(backbone, text: str) -> dict[int, torch.Tensor]:
    """v2 address: pre-RoPE QUERY representation of the evidence-query span.

    u_g^{l,h} = mean over r in T_g of q_{r,pre}^{l,h} — the retrieval direction
    the model actually formed for this evidence requirement — GQA-reduced from
    the H_q query heads to the H_kv key heads each address must serve. Same
    span contract as v1 (evidence_query text only: no chat template, no JSON
    syntax, no coordinates).
    """
    layers = _layer_indices(backbone)
    tokens = backbone.processor.tokenizer(
        text, return_tensors="pt", add_special_tokens=False)
    ids = tokens["input_ids"].to(backbone.device)
    if int(ids.shape[1]) == 0:
        return {}
    captured: dict[int, torch.Tensor] = {}
    handles: list[object] = []

    def make_hook(layer_index: int):
        def hook(module, args, kwargs, _output):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            if hidden is None:
                return
            raw = _pre_rope_queries(module, hidden)       # [1, H_q, L, D]
            u_q = raw[0].mean(dim=1).float().cpu()        # [H_q, D]
            captured[layer_index] = reduce_query_heads(
                u_q, _num_kv_heads(module))               # [H_kv, D]

        return hook

    for index in layers:
        handles.append(
            backbone.lm.layers[index].self_attn.register_forward_hook(
                make_hook(index), with_kwargs=True))
    try:
        _ = backbone.lm(
            inputs_embeds=backbone.lm.embed_tokens(ids),
            position_ids=backbone._make_position_ids(int(ids.shape[1])),
            use_cache=False,
        )
    finally:
        for handle in handles:
            handle.remove()
    return captured


def _deterministic_direction(
    like: torch.Tensor,
    crop_index: int,
    layer_index: int,
) -> torch.Tensor:
    """A reproducible random unit direction shaped like an address [H, D]."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1_000_003 * (crop_index + 1) + layer_index)
    noise = torch.randn(like.shape, generator=generator)
    return (noise / noise.norm(dim=-1, keepdim=True).clamp_min(1e-6)).to(like.device)


@torch.no_grad()
def apply_aavm(
    backbone,
    cache,
    *,
    capture: dict,
    keep_mask: torch.Tensor | None,
    kept_counts: list[int],
    crop_queries: list[str],
    cache_columns: torch.Tensor,
) -> int:
    """Re-address each crop's cached visual keys. Returns crops re-addressed.

    ``capture`` is what capture_visual_raw_keys collected over ALL visual
    columns of the prefill; ``keep_mask`` (C1's decision, or None when C1 is
    off) selects the retained subset those cached columns correspond to.
    """
    if not enabled():
        return 0
    if not capture or not kept_counts:
        print(f"[AAVM] SKIP boundary: capture={'yes' if capture else 'EMPTY'} "
              f"crops={len(kept_counts)} events="
              f"{sum(1 for q in crop_queries if q)}", flush=True)
        return 0
    raw_all = capture.get("keys")
    cos_all, sin_all = capture.get("cos"), capture.get("sin")
    if raw_all is None or cos_all is None or sin_all is None:
        print("[AAVM] SKIP: no prefill capture", flush=True)
        return 0
    layers = sorted(raw_all)
    n_captured = int(next(iter(raw_all.values())).shape[1])
    if keep_mask is not None:
        keep_cpu = keep_mask.detach().cpu().bool()
        if int(keep_cpu.numel()) != n_captured:
            print(f"[AAVM] SKIP: keep {int(keep_cpu.numel())} != captured "
                  f"{n_captured}", flush=True)
            return 0
        index = keep_cpu.nonzero(as_tuple=True)[0]
        raw_all = {li: k.index_select(1, index) for li, k in raw_all.items()}
        cos_all = cos_all.index_select(0, index)
        sin_all = sin_all.index_select(0, index)
    total_kept = sum(kept_counts)
    if int(next(iter(raw_all.values())).shape[1]) != total_kept:
        print(f"[AAVM] SKIP: kept {total_kept} != selected "
              f"{int(next(iter(raw_all.values())).shape[1])}", flush=True)
        return 0
    if int(cache_columns.numel()) != total_kept:
        print(f"[AAVM] SKIP: cache cols {int(cache_columns.numel())} != "
              f"{total_kept}", flush=True)
        return 0

    spans = crop_spans(kept_counts)
    extract = query_address_v2 if mode() == "2" else query_address
    addresses_by_text: dict[str, dict[int, torch.Tensor]] = {}
    for text in dict.fromkeys(q for q in crop_queries if q):
        addresses_by_text[text] = extract(backbone, text)
    per_crop = [addresses_by_text.get(q or "", {}) for q in crop_queries]
    control = os.environ.get("VLMAS_AAVM_CONTROL", "")
    if control == "shuffled_address":
        per_crop = shuffled_addresses(per_crop)
    elif control == "random_norm":
        # Diagnostic 2: same displacement magnitude, NO acquisition semantics.
        # If this loses as much as the real address, the damage comes from
        # moving the crop centroid at all, not from where it is moved to.
        per_crop = [
            {
                li: _deterministic_direction(vec, crop_index, li)
                for li, vec in address.items()
            }
            for crop_index, address in enumerate(per_crop)
        ]

    columns = cache_columns.detach().to(device=backbone.device, dtype=torch.long)
    moved = 0
    for crop_index, (span, address) in enumerate(zip(spans, per_crop)):
        if not address or span[1] <= span[0]:
            continue
        middle = (span[0] + span[1]) // 2
        cos_rep = cos_all[middle].to(backbone.device)
        sin_rep = sin_all[middle].to(backbone.device)
        crop_cols = columns[span[0]:span[1]]
        for layer_index in layers:
            if layer_index not in address:
                continue
            raw_crop = raw_all[layer_index][:, span[0]:span[1], :].to(
                backbone.device)
            shift = address_shift(raw_crop, address[layer_index].to(backbone.device))
            if control == "identity":
                # Diagnostic 1: run every step (hooks, address forward, rotate,
                # index_copy) but apply a ZERO displacement. Must reproduce the
                # un-addressed run exactly; anything else is a plumbing defect.
                shift = torch.zeros_like(shift)
            shift_rotated = rotate_vector(shift, cos_rep, sin_rep)
            keys = cache.layers[layer_index].keys
            block = keys.index_select(2, crop_cols)
            block = (block + shift_rotated.unsqueeze(0).unsqueeze(2)).to(keys.dtype)
            keys.index_copy_(2, crop_cols, block)
        moved += 1
    print(f"[AAVM] v{mode()} re-addressed {moved}/{len(kept_counts)} crops "
          f"over {len(layers)} layers (control={control or 'none'}, "
          f"events={len(addresses_by_text)})", flush=True)
    return moved


__all__ = [
    "apply_aavm",
    "capture_visual_raw_keys",
    "enabled",
    "mode",
    "query_address",
    "query_address_v2",
    "crop_spans",
    "apply_crop_shift",
]
