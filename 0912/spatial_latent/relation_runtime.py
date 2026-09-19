"""Isolated runtime for forming and reading cross-scale relation latent rows."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path

import torch

from spatial_latent.attention import BiasProbe
from spatial_latent.grouping import GroupMode, LayoutError
from spatial_latent.relation_attention import (
    RelationStep,
    relation_step_sdpa,
    reserved_relation_sdpa,
)
from spatial_latent.runtime import SpatialExperiment, StepTrace
from vision_text_mas import latent_hybrid_variants as hybrid
from vision_text_mas.latent_qwen_engine import LatentQwenEngine


class RelationExperiment(SpatialExperiment):
    """Alternate context/relation steps and reserve their Answerer contribution."""

    def __init__(
        self,
        mode: GroupMode,
        output_root: Path,
        gain: float = 4.0,
        answerer_alpha: float = 0.1,
    ) -> None:
        super().__init__(mode, gain)
        self.output_root = output_root
        self.answerer_alpha = answerer_alpha

    def answerer_readout(
        self, latent_columns: torch.Tensor
    ) -> AbstractContextManager[BiasProbe]:
        """Return the terminal latent-KV readout for this experiment."""
        return reserved_relation_sdpa(
            latent_columns[[1, 3, 5, 7]], alpha=self.answerer_alpha
        )

    @contextmanager
    def latent_step(
        self,
        backbone: hybrid._VariantBackbone,
        vision_columns: torch.Tensor,
        visual_patch_ids: torch.Tensor | None = None,
        parent_patch_indices: tuple[int, ...] = (),
    ) -> Iterator[None]:
        """Create context, relation and synthesis rows in the ten-step Base budget."""
        del visual_patch_ids, parent_patch_indices
        if not getattr(backbone, "_prune_morphology_enabled", False):
            yield None
            return
        step_index = len(self.steps)
        if step_index >= 10:
            raise LayoutError("Relation experiment requires exactly 10 Reasoner steps")
        if step_index == 0:
            self._prepare(backbone, vision_columns)
        if step_index < 8:
            group = self.groups[step_index // 2]
            relation_step = step_index % 2 == 1
            patch_indices = group[1:] if relation_step else group[:1]
            columns = torch.cat([self.columns[index] for index in patch_indices])
            definition = RelationStep(
                columns=columns,
                include_previous=relation_step,
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
        if probe.applied_calls != 13:
            raise LayoutError(
                f"Step {step_index}: relation SDPA fired {probe.applied_calls}, "
                "expected 13"
            )
        self.steps.append(
            StepTrace(
                step=step_index + 1,
                patch_indices=patch_indices,
                visual_tokens=int(columns.numel()),
                biased_layer_calls=probe.applied_calls,
                cache_tokens=probe.key_count,
            )
        )

    @contextmanager
    def installed(self) -> Iterator[None]:
        """Install relation formation plus terminal readout around Base execution."""
        experiment = self
        original_decode = LatentQwenEngine.decode

        def decode(
            self: LatentQwenEngine,
            *,
            cache,
            position_cursor: int,
            system_prompt: str,
            user_prompt: str,
            max_new_tokens: int,
            json_prefix: str | None,
            json_schema: str | None = None,
        ) -> str:
            terminal = json_prefix is not None and json_prefix.strip().startswith(
                '{"answer"'
            )
            if not terminal:
                return original_decode(
                    self,
                    cache=cache,
                    position_cursor=position_cursor,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_new_tokens=max_new_tokens,
                    json_prefix=json_prefix,
                    json_schema=json_schema,
                )
            latent_columns = self._rpath_latent_cols
            relation_columns = latent_columns[[1, 3, 5, 7]]
            if experiment.answerer_alpha == 0.0:
                output = original_decode(
                    self,
                    cache=cache,
                    position_cursor=position_cursor,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_new_tokens=max_new_tokens,
                    json_prefix=json_prefix,
                    json_schema=json_schema,
                )
                applied_calls = 0
                maximum_cache_tokens = 0
            else:
                with experiment.answerer_readout(latent_columns) as probe:
                    output = original_decode(
                        self,
                        cache=cache,
                        position_cursor=position_cursor,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        max_new_tokens=max_new_tokens,
                        json_prefix=json_prefix,
                        json_schema=json_schema,
                    )
                if probe.applied_calls < 1:
                    raise LayoutError(
                        "Answerer did not read the reserved relation pathway"
                    )
                applied_calls = probe.applied_calls
                maximum_cache_tokens = probe.key_count
            case_index = int(getattr(self, "_rpath_case_index", -1))
            path = (
                experiment.output_root / "relation_readout" / f"{case_index:03d}.json"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "dataset_index": case_index,
                        "relation_columns": relation_columns.tolist(),
                        "alpha": experiment.answerer_alpha,
                        "applied_layer_calls": applied_calls,
                        "maximum_cache_tokens": maximum_cache_tokens,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            return output

        LatentQwenEngine.decode = decode
        try:
            with super().installed():
                yield None
        finally:
            LatentQwenEngine.decode = original_decode
