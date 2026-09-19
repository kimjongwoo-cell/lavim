"""Collect each sender latent query without intervening in its computation."""

from dataclasses import dataclass, field

import torch
from pydantic import BaseModel, ConfigDict

from pvcr.core import Contribution
from pvcr.errors import LayoutError
from pvcr.layout import ImageMetadata, Neighborhoods, Variant, build_neighborhoods
from pvcr.receiver import RelayBank


class SenderAudit(BaseModel):
    model_config = ConfigDict(frozen=True)
    stage: str
    kind: str
    columns: list[int]
    layers: list[int]
    values_shape: list[int]
    query_heads: int
    kv_heads: int
    queries_per_kv_head: int
    visual_mass_by_step: list[float]
    value_norm_ratio_by_step: list[float]
    support_by_step: list[float]
    source_image_count: int
    source_visual_tokens: int


@dataclass
class Capture:
    """Mutable accumulator scoped to one role append, reset for every case."""

    stage: str
    metadata: tuple[ImageMetadata, ...]
    variant: Variant
    grids: torch.Tensor | None = None
    layout: Neighborhoods | None = None
    step: int = 0
    observing: bool = False
    query_heads: int = 0
    positions: dict[int, list[int]] = field(default_factory=dict)
    values: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    masses: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    ratios: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    supports: dict[int, list[torch.Tensor]] = field(default_factory=dict)

    def prepare(self, columns: torch.Tensor) -> None:
        if self.layout is None:
            if self.grids is None:
                raise LayoutError("Visual forward did not expose grid_thw")
            self.layout = build_neighborhoods(
                columns, self.grids, self.metadata, variant=self.variant
            )

    def record(
        self,
        layer: int,
        result: Contribution,
        original: torch.Tensor,
        position: int,
        *,
        query_heads: int,
    ) -> None:
        """Store one synthetic row and compact diagnostics per layer/step."""
        entries = self.positions.setdefault(layer, [])
        if self.query_heads not in (0, query_heads):
            raise LayoutError("PVCR query head count changed within sender")
        self.query_heads = query_heads
        if len(entries) != self.step:
            raise LayoutError("PVCR latent layer called more than once per step")
        entries.append(position)
        row = original if self.variant == "original" else result.value
        self.values.setdefault(layer, []).append(row.detach().clone())
        self.masses.setdefault(layer, []).append(result.visual_mass.detach().mean())
        ratio = result.value.float().norm(dim=-1) / original.float().norm(
            dim=-1
        ).clamp_min(1e-12)
        self.ratios.setdefault(layer, []).append(ratio.detach().mean())
        self.supports.setdefault(layer, []).append(result.support.detach().mean())

    def finish(self) -> tuple[RelayBank, SenderAudit]:
        """Reject incomplete captures and produce a bank aligned to cache positions."""
        if (
            self.step != 10
            or self.layout is None
            or sorted(self.values) != list(range(8, 21))
        ):
            raise LayoutError("PVCR requires 10 observed steps at all 13 mid layers")
        positions = self.positions[8]
        if any(rows != positions for rows in self.positions.values()):
            raise LayoutError("PVCR layer cache addresses disagree")
        values = {
            layer: torch.stack(rows, dim=2) for layer, rows in self.values.items()
        }
        if any(not torch.isfinite(value).all() for value in values.values()):
            raise LayoutError("PVCR produced nonfinite visual contributions")
        audit = SenderAudit(
            stage=self.stage,
            kind=self.layout.kind,
            columns=positions,
            layers=sorted(values),
            values_shape=list(values[8].shape),
            query_heads=self.query_heads,
            kv_heads=values[8].shape[1],
            queries_per_kv_head=self.query_heads // values[8].shape[1],
            visual_mass_by_step=torch.stack(
                [torch.stack(x) for x in self.masses.values()]
            )
            .mean(0)
            .cpu()
            .tolist(),
            value_norm_ratio_by_step=torch.stack(
                [torch.stack(x) for x in self.ratios.values()]
            )
            .mean(0)
            .cpu()
            .tolist(),
            support_by_step=torch.stack(
                [torch.stack(x) for x in self.supports.values()]
            )
            .mean(0)
            .cpu()
            .tolist(),
            source_image_count=len(self.metadata),
            source_visual_tokens=int(self.layout.columns.numel()),
        )
        return RelayBank(torch.tensor(positions, dtype=torch.long), values), audit
