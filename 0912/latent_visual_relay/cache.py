"""Insert selected original latent K/V after the original visual block."""

from dataclasses import dataclass

import torch
from pvcr.errors import LayoutError
from pvcr.layout import Neighborhoods
from transformers.cache_utils import Cache, DynamicLayer


@dataclass(frozen=True, slots=True)
class SourceBlock:
    latent_columns: torch.Tensor
    visual_columns: torch.Tensor
    layout: Neighborhoods | None = None


@dataclass(frozen=True, slots=True)
class InsertionAudit:
    before: int
    after: int
    insertion_index: int
    source_columns: list[int]
    selected_head_entries: int
    source_equal: bool
    original_cache_equal: bool


@dataclass(frozen=True, slots=True)
class InsertedBlock:
    columns: torch.Tensor
    allowed: dict[int, torch.Tensor]
    audit: InsertionAudit
    sender: str = ""
    family_ids: torch.Tensor | None = None
    layout: Neighborhoods | None = None


@dataclass(frozen=True, slots=True)
class LayerView:
    owner: DynamicLayer
    keys: torch.Tensor
    values: torch.Tensor


def layer_view(cache: Cache, index: int) -> LayerView:
    """Narrow the external cache API to the initialized Qwen cache contract."""
    layer = cache.layers[index]
    if (
        not isinstance(layer, DynamicLayer)
        or layer.keys is None
        or layer.values is None
    ):
        raise LayoutError("Latent visual relay requires initialized DynamicLayer K/V")
    return LayerView(layer, layer.keys, layer.values)


def insert_selected(
    cache: Cache, source: SourceBlock, selected: dict[int, torch.Tensor]
) -> InsertedBlock:
    """Copy selected latent heads into new cache rows; retain all original rows."""
    length = int(cache.get_seq_length())
    insertion = int(source.visual_columns[-1]) + 1
    if not (0 < insertion <= length) or source.latent_columns.numel() == 0:
        raise LayoutError("Missing source visual/latent block")
    masks = torch.stack(list(selected.values()))
    if masks.ndim in (4, 5):
        addresses = masks.any(dim=(0, 1, 2)).nonzero()
    else:
        raise LayoutError(
            "Relay selections must be [layer,batch,head,step] or family-aware"
        )
    steps = addresses[:, -1]
    if length + addresses.shape[0] > 12288:
        raise LayoutError("Latent relay insertion would exceed fixed context12288")
    positions = source.latent_columns.to(steps.device).index_select(0, steps)
    if positions.numel() and (positions.min() < insertion or positions.max() >= length):
        raise LayoutError("Relay source is not in the sender latent block")
    rows = int(addresses.shape[0])
    family_ids = (
        addresses[:, 0].detach().cpu()
        if masks.ndim == 5
        else torch.full((rows,), -1, dtype=torch.long)
    )
    allowed: dict[int, torch.Tensor] = {}
    pending: list[LayerView] = []
    source_equal = True
    original_equal = True
    selected_count = 0
    for index in range(len(cache.layers)):
        original = layer_view(cache, index)
        keys, values = original.keys, original.values
        if keys.shape[-2] != length or values.shape[-2] != length:
            raise LayoutError("Native cache layer lengths disagree")
        native_k = keys.index_select(-2, positions.to(keys.device))
        native_v = values.index_select(-2, positions.to(values.device))
        selection = selected.get(index)
        mask = (
            torch.zeros(*keys.shape[:2], rows, dtype=torch.bool, device=keys.device)
            if selection is None
            else selection.to(keys.device).index_select(-1, steps.to(keys.device))
        )
        if selection is not None and selection.ndim == 4:
            families = addresses[:, 0].to(keys.device)
            mask = selection.to(keys.device)[..., families, steps.to(keys.device)]
        allowed[index] = mask
        selected_count += int(mask.sum())
        copied_k = native_k.masked_fill(~mask.unsqueeze(-1), 0)
        copied_v = native_v.masked_fill(~mask.unsqueeze(-1), 0)
        out_k = torch.cat(
            (keys[:, :, :insertion], copied_k, keys[:, :, insertion:]), dim=-2
        )
        out_v = torch.cat(
            (values[:, :, :insertion], copied_v, values[:, :, insertion:]), dim=-2
        )
        source_equal &= bool(
            torch.equal(out_k[:, :, insertion : insertion + rows][mask], native_k[mask])
            and torch.equal(
                out_v[:, :, insertion : insertion + rows][mask], native_v[mask]
            )
        )
        original_equal &= bool(
            torch.equal(
                torch.cat(
                    (out_k[:, :, :insertion], out_k[:, :, insertion + rows :]),
                    dim=-2,
                ),
                keys,
            )
            and torch.equal(
                torch.cat(
                    (out_v[:, :, :insertion], out_v[:, :, insertion + rows :]),
                    dim=-2,
                ),
                values,
            )
        )
        pending.append(LayerView(original.owner, out_k, out_v))
    if not source_equal or not original_equal:
        raise LayoutError("Latent source/original cache preservation check failed")
    for updated in pending:
        updated.owner.keys = updated.keys
        updated.owner.values = updated.values
    audit = InsertionAudit(
        length,
        int(cache.get_seq_length()),
        insertion,
        positions.cpu().tolist(),
        selected_count,
        source_equal,
        original_equal,
    )
    return InsertedBlock(
        torch.arange(insertion, insertion + rows),
        allowed,
        audit,
        family_ids=family_ids,
        layout=source.layout,
    )
