"""Measure grounding of native latent rows without synthesizing their values."""

from dataclasses import dataclass, field
from typing import Literal

import torch
from PIL import Image
from pvcr.errors import LayoutError
from pvcr.layout import ImageMetadata, Neighborhoods, build_neighborhoods
from transformers.cache_utils import Cache

from latent_visual_relay.cache import SourceBlock, layer_view
from latent_visual_relay.selection import (
    contrastive_context_margin,
    contrastive_query_margin,
    grounding_scores,
    relational_grounding_scores,
    select_heads,
    select_attention_context_pairs,
    select_contextual_pairs,
    select_relational_fraction,
    select_visual_closer_pairs,
    select_visual_fraction,
    select_relational_heads,
)

type Mode = Literal[
    "relational",
    "spatial",
    "shuffled",
    "visual",
    "contrastive",
    "contrastive_context",
    "attention_context",
    "attention_context_read",
    "off",
]


def select_prompt_query_rows(
    prompt_queries: torch.Tensor, columns: torch.Tensor, start: int
) -> torch.Tensor:
    """Select absolute query addresses using indices colocated with cached Q."""
    indices = columns.to(device=prompt_queries.device) - start
    return prompt_queries.index_select(-2, indices)


@dataclass(frozen=True, slots=True)
class Bank:
    source: SourceBlock
    selected: dict[int, torch.Tensor]


@dataclass(slots=True)
class Capture:
    """Mutable per-sender accumulator; observation never alters native outputs."""

    stage: str
    images: tuple[Image.Image, ...]
    mode: Mode
    grids: torch.Tensor | None = None
    visual_columns: torch.Tensor | None = None
    layout: Neighborhoods | None = None
    observing: bool = False
    step: int = 0
    query_heads: int = 0
    scores: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    positions: dict[int, list[int]] = field(default_factory=dict)
    selection_scores: dict[int, torch.Tensor] = field(default_factory=dict)
    prompt_queries: dict[int, tuple[int, torch.Tensor]] = field(default_factory=dict)

    def grid_hook(
        self,
        module: torch.nn.Module,
        args: tuple[torch.Tensor, ...],
        kwargs: dict[str, torch.Tensor],
    ) -> None:
        """Observe the installed vision module's native hook ABI."""
        del module
        grid = kwargs.get("grid_thw")
        if grid is None and len(args) > 1:
            grid = args[1]
        if grid is None:
            raise LayoutError("Vision forward did not expose grid_thw")
        self.grids = grid.detach().cpu().clone()

    def prepare(self, columns: torch.Tensor) -> None:
        if self.layout is not None:
            return
        if self.grids is None:
            raise LayoutError("Missing sender image grids")
        metadata = tuple(
            ImageMetadata.model_validate(
                {
                    "patch_id": image.info.get("latent_patch_id", ""),
                    "parent_id": image.info.get("latent_parent_patch_id", ""),
                    "magnification": image.info.get("latent_magnification", 0),
                    "box": image.info.get("latent_box"),
                }
            )
            for image in self.images
        )
        self.visual_columns = columns.detach().cpu().clone()
        variant: Literal[
            "spatial", "shuffled", "visual", "paired_context", "off"
        ] = (
            "spatial"
            if self.mode == "relational"
            else "paired_context"
            if self.mode in ("attention_context", "attention_context_read")
            else "visual"
            if self.mode in ("contrastive", "contrastive_context")
            else self.mode
        )
        self.layout = build_neighborhoods(
            columns, self.grids, metadata, variant=variant
        )

    def record(self, layer: int, probabilities: torch.Tensor, position: int) -> None:
        if self.layout is None:
            raise LayoutError("Missing sender spatial layout")
        positions = self.positions.setdefault(layer, [])
        if len(positions) != self.step:
            raise LayoutError("More than one attention observation per latent step")
        positions.append(position)
        score = (
            relational_grounding_scores(probabilities, self.layout)
            if self.mode
            in ("relational", "attention_context", "attention_context_read")
            else grounding_scores(
                probabilities, self.layout, spatial=self.mode != "visual"
            )
        )
        self.scores.setdefault(layer, []).append(score.detach())

    def record_prompt_queries(
        self, layer: int, queries: torch.Tensor, key_length: int
    ) -> None:
        """Save actual prefill Q vectors with their absolute cache address span."""
        if queries.shape[-2] < 2 or layer in self.prompt_queries:
            return
        start = key_length - queries.shape[-2]
        self.prompt_queries[layer] = (
            start,
            queries.detach().to(device="cpu", dtype=torch.float32),
        )

    def finish(self, cache: Cache) -> Bank:
        if (
            self.step != 10
            or sorted(self.positions) != list(range(8, 21))
            or self.visual_columns is None
        ):
            raise LayoutError("Expected10 steps at all13 mid layers")
        columns = self.positions[8]
        if len(columns) != 10 or any(
            positions != columns for positions in self.positions.values()
        ):
            raise LayoutError("Latent cache addresses disagree across layers")
        if columns != list(range(columns[0], columns[0] + 10)):
            raise LayoutError("Observed sender latent block is not contiguous")
        source = SourceBlock(torch.tensor(columns), self.visual_columns, self.layout)
        if self.mode in ("contrastive", "contrastive_context"):
            selected = self._select_visual_closer_kv(
                cache, source, contextual=self.mode == "contrastive_context"
            )
            return Bank(source, selected)
        scores = {
            layer: torch.stack(steps, dim=-1) for layer, steps in self.scores.items()
        }
        if self.mode in ("attention_context", "attention_context_read"):
            for layer, score in scores.items():
                self.selection_scores[layer] = score.amax(dim=-2).detach().cpu()
            return Bank(
                source,
                {
                    layer: select_attention_context_pairs(score)
                    for layer, score in scores.items()
                },
            )
        selected = {
            layer: (
                select_relational_heads(score)
                if self.mode == "relational"
                else select_visual_fraction(score)
            )
            for layer, score in scores.items()
        }
        if self.mode == "relational":
            selected = select_relational_fraction(scores, selected)
        return Bank(source, selected)

    def _select_visual_closer_kv(
        self, cache: Cache, source: SourceBlock, *, contextual: bool = False
    ) -> dict[int, torch.Tensor]:
        """Keep latent K/V pairs closer to visual than non-visual cache context."""
        length = int(cache.get_seq_length())
        excluded = torch.zeros(length, dtype=torch.bool)
        excluded[source.visual_columns] = True
        excluded[source.latent_columns] = True
        text_columns = (~excluded).nonzero().flatten()
        if text_columns.numel() == 0:
            raise LayoutError("Contrastive relay requires non-visual text KV")
        selected: dict[int, torch.Tensor] = {}
        for layer in range(8, 21):
            keys = layer_view(cache, layer).keys
            if layer not in self.prompt_queries:
                raise LayoutError(f"Missing actual prefill queries for layer {layer}")
            start, prompt_query = self.prompt_queries[layer]
            end = start + prompt_query.shape[-2]
            visual_columns = source.visual_columns.cpu()
            visual_in_prefill = visual_columns[
                (visual_columns >= start) & (visual_columns < end)
            ]
            text_start = max(start, int(visual_columns.max()) + 1)
            text_in_prefill = torch.arange(text_start, end)
            if visual_in_prefill.numel() == 0 or text_in_prefill.numel() == 0:
                raise LayoutError("Contrastive relay requires visual and text prefill Q")
            query_columns = (
                self.layout.columns
                if contextual and self.layout is not None
                else visual_in_prefill
            )
            visual_queries = select_prompt_query_rows(
                prompt_query, query_columns, start
            )
            text_queries = select_prompt_query_rows(
                prompt_query, text_in_prefill, start
            )
            visual_queries = visual_queries.to(keys.device)
            text_queries = text_queries.to(keys.device)
            latent_keys = keys.index_select(
                -2, source.latent_columns.to(keys.device)
            )
            if contextual:
                if self.layout is None:
                    raise LayoutError("Missing visual-context family layout")
                core = self.layout.core.to(device=keys.device, dtype=visual_queries.dtype)
                context = self.layout.context.to(
                    device=keys.device, dtype=visual_queries.dtype
                )
                local_queries = torch.einsum(
                    "bhqd,fq->bhfd", visual_queries, core
                ) / core.sum(-1).clamp_min(1).view(1, 1, -1, 1)
                context_queries = torch.einsum(
                    "bhqd,fq->bhfd", visual_queries, context
                ) / context.sum(-1).clamp_min(1).view(1, 1, -1, 1)
                family_margins = contrastive_context_margin(
                    local_queries, context_queries, text_queries, latent_keys
                )
                margin = family_margins.amax(dim=-2)
            else:
                margin = contrastive_query_margin(
                    visual_queries, text_queries, latent_keys
                )
            self.selection_scores[layer] = margin.detach().cpu()
            selected[layer] = (
                select_contextual_pairs(margin)
                if contextual
                else select_visual_closer_pairs(margin)
            )
        return selected
