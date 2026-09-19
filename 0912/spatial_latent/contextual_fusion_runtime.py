"""Ten-step Reasoner schedule for pathology-style cross-scale fusion."""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import torch

from spatial_latent.contextual_fusion_attention import (
    CrossScaleFusionStep,
    cross_scale_fusion_sdpa,
)
from spatial_latent.grouping import GroupMode, LayoutError
from spatial_latent.relation_attention import RelationStep, relation_step_sdpa
from spatial_latent.runtime import SpatialExperiment, StepTrace
from vision_text_mas import latent_hybrid_variants as hybrid


class ContextualFusionExperiment(SpatialExperiment):
    """Encode four parent-child neighborhoods before global latent synthesis."""

    def __init__(
        self,
        mode: GroupMode,
        output_root: Path,
        gain: float = 4.0,
        fusion_strength: float = 0.4,
    ) -> None:
        super().__init__(mode, gain)
        self.output_root = output_root
        self.fusion_strength = fusion_strength
        self.fusion_scores: list[float] = []

    @contextmanager
    def latent_step(
        self,
        backbone: hybrid._VariantBackbone,
        vision_columns: torch.Tensor,
        visual_patch_ids: torch.Tensor | None = None,
        parent_patch_indices: tuple[int, ...] = (),
    ) -> Iterator[None]:
        """Fuse four local neighborhoods, consolidate them, then synthesize globally."""
        del visual_patch_ids, parent_patch_indices
        if not getattr(backbone, "_prune_morphology_enabled", False):
            yield None
            return
        step_index = len(self.steps)
        if step_index >= 10:
            raise LayoutError("Contextual fusion requires exactly 10 Reasoner steps")
        if step_index == 0:
            self._prepare(backbone, vision_columns)
            self.fusion_scores.clear()
        if step_index < 4:
            group = self.groups[step_index]
            parent = self.columns[group[0]]
            children = torch.cat([self.columns[index] for index in group[1:]])
            definition = CrossScaleFusionStep(parent, children, self.fusion_strength)
            with cross_scale_fusion_sdpa(definition) as probe:
                yield None
            self.fusion_scores.append(probe.mean_agreement)
            patch_indices = group
            visual_tokens = int(parent.numel() + children.numel())
        else:
            if step_index < 8:
                definition = RelationStep(
                    vision_columns[:0], True, math.log(self.gain)
                )
            elif step_index == 8:
                definition = RelationStep(
                    vision_columns[:0],
                    False,
                    math.log(self.gain),
                    tail_offsets=(2, 4, 6, 8),
                )
            else:
                definition = RelationStep(vision_columns[:0], True, math.log(self.gain))
            with relation_step_sdpa(definition) as probe:
                yield None
            patch_indices = ()
            visual_tokens = 0
        if probe.applied_calls != 13:
            raise LayoutError(
                f"Step {step_index}: fusion SDPA fired {probe.applied_calls}, "
                "expected 13"
            )
        self.steps.append(
            StepTrace(
                step=step_index + 1,
                patch_indices=patch_indices,
                visual_tokens=visual_tokens,
                biased_layer_calls=probe.applied_calls,
                cache_tokens=probe.key_count,
            )
        )
