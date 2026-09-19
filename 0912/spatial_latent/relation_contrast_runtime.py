"""Runtime for explicit parent-child contrast latent reasoning."""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import torch

from spatial_latent.grouping import GroupMode, LayoutError
from spatial_latent.relation_attention import RelationStep, relation_step_sdpa
from spatial_latent.relation_contrast_attention import (
    ContrastStep,
    cross_scale_contrast_sdpa,
)
from spatial_latent.relation_runtime import RelationExperiment
from spatial_latent.runtime import StepTrace
from vision_text_mas import latent_hybrid_variants as hybrid


class ContrastRelationExperiment(RelationExperiment):
    """Form relation rows with normalized fine-minus-context attention outputs."""

    def __init__(
        self,
        mode: GroupMode,
        output_root: Path,
        gain: float = 4.0,
        answerer_alpha: float = 0.1,
        contrast_strength: float = 0.1,
    ) -> None:
        super().__init__(mode, output_root, gain, answerer_alpha)
        self.contrast_strength = contrast_strength

    @contextmanager
    def latent_step(
        self,
        backbone: hybrid._VariantBackbone,
        vision_columns: torch.Tensor,
        visual_patch_ids: torch.Tensor | None = None,
        parent_patch_indices: tuple[int, ...] = (),
    ) -> Iterator[None]:
        """Alternate context rows with explicit cross-scale contrast rows."""
        del visual_patch_ids, parent_patch_indices
        if not getattr(backbone, "_prune_morphology_enabled", False):
            yield None
            return
        step_index = len(self.steps)
        if step_index >= 10:
            raise LayoutError("Contrast experiment requires exactly 10 Reasoner steps")
        if step_index == 0:
            self._prepare(backbone, vision_columns)
        if step_index < 8 and step_index % 2 == 1:
            group = self.groups[step_index // 2]
            parent = self.columns[group[0]]
            children = torch.cat([self.columns[index] for index in group[1:]])
            definition = ContrastStep(
                parent_columns=parent,
                child_columns=children,
                strength=self.contrast_strength,
            )
            with cross_scale_contrast_sdpa(definition) as probe:
                yield None
            patch_indices = group
            visual_tokens = int(parent.numel() + children.numel())
        else:
            if step_index < 8:
                group = self.groups[step_index // 2]
                patch_indices = group[:1]
                columns = self.columns[group[0]]
                definition = RelationStep(
                    columns=columns,
                    include_previous=False,
                    log_gain=math.log(self.gain),
                )
            elif step_index == 8:
                patch_indices = ()
                columns = vision_columns[:0]
                definition = RelationStep(
                    columns=columns,
                    include_previous=False,
                    log_gain=math.log(self.gain),
                    tail_offsets=(2, 4, 6, 8),
                )
            else:
                patch_indices = ()
                columns = vision_columns[:0]
                definition = RelationStep(
                    columns=columns,
                    include_previous=True,
                    log_gain=math.log(self.gain),
                )
            with relation_step_sdpa(definition) as probe:
                yield None
            visual_tokens = int(columns.numel())
        if probe.applied_calls != 13:
            raise LayoutError(
                f"Step {step_index}: contrast SDPA fired {probe.applied_calls}, "
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
