"""Runtime for agreement-gated cross-scale latent evidence."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import torch

from spatial_latent.grouping import GroupMode, LayoutError
from spatial_latent.relation_attention import RelationStep, relation_step_sdpa
from spatial_latent.relation_consistency_attention import (
    ConsistencyStep,
    consistency_gate_sdpa,
)
from spatial_latent.relation_runtime import RelationExperiment
from spatial_latent.runtime import StepTrace
from vision_text_mas import latent_hf_ablation_cli as cli
from vision_text_mas import latent_hybrid_variants as hybrid
from vision_text_mas.contracts import CaseInput, RunResult


class ConsistencyRelationExperiment(RelationExperiment):
    """Preserve low-power context and add high-power evidence only on agreement."""

    def __init__(
        self,
        mode: GroupMode,
        output_root: Path,
        gain: float = 4.0,
        answerer_alpha: float = 0.1,
        consistency_strength: float = 0.1,
    ) -> None:
        super().__init__(mode, output_root, gain, answerer_alpha)
        self.consistency_strength = consistency_strength
        self.consistency_scores: list[float] = []

    @contextmanager
    def latent_step(
        self,
        backbone: hybrid._VariantBackbone,
        vision_columns: torch.Tensor,
        visual_patch_ids: torch.Tensor | None = None,
        parent_patch_indices: tuple[int, ...] = (),
    ) -> Iterator[None]:
        """Alternate parent context rows with agreement-gated child evidence rows."""
        del visual_patch_ids, parent_patch_indices
        if not getattr(backbone, "_prune_morphology_enabled", False):
            yield None
            return
        step_index = len(self.steps)
        if step_index >= 10:
            raise LayoutError(
                "Consistency experiment requires exactly 10 Reasoner steps"
            )
        if step_index == 0:
            self._prepare(backbone, vision_columns)
            self.consistency_scores.clear()
        if step_index < 8 and step_index % 2 == 1:
            group = self.groups[step_index // 2]
            parent = self.columns[group[0]]
            children = torch.cat([self.columns[index] for index in group[1:]])
            definition = ConsistencyStep(parent, children, self.consistency_strength)
            with consistency_gate_sdpa(definition) as probe:
                yield None
            self.consistency_scores.append(probe.mean_agreement)
            patch_indices = group
            visual_tokens = int(parent.numel() + children.numel())
        else:
            if step_index < 8:
                group = self.groups[step_index // 2]
                patch_indices = group[:1]
                columns = self.columns[group[0]]
                definition = RelationStep(columns, False, math.log(self.gain))
            elif step_index == 8:
                patch_indices = ()
                columns = vision_columns[:0]
                definition = RelationStep(
                    columns, False, math.log(self.gain), tail_offsets=(2, 4, 6, 8)
                )
            else:
                patch_indices = ()
                columns = vision_columns[:0]
                definition = RelationStep(columns, True, math.log(self.gain))
            with relation_step_sdpa(definition) as probe:
                yield None
            visual_tokens = int(columns.numel())
        if probe.applied_calls != 13:
            raise LayoutError(
                f"Step {step_index}: consistency SDPA fired {probe.applied_calls}, "
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

    @contextmanager
    def installed(self) -> Iterator[None]:
        """Persist per-case agreement before the inherited Answerer readout exits."""
        with super().installed():
            original_case = cli.run_case_with_retries

            def run_case(
                *,
                case: CaseInput,
                max_attempts: int,
                operation: Callable[[], RunResult],
                metrics_path: Path,
            ) -> RunResult:
                result = original_case(
                    case=case,
                    max_attempts=max_attempts,
                    operation=operation,
                    metrics_path=metrics_path,
                )
                path = (
                    self.output_root
                    / "consistency_traces"
                    / f"{case.dataset_index:03d}.json"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(
                        {
                            "dataset_index": case.dataset_index,
                            "group_agreement": self.consistency_scores,
                            "mean_agreement": sum(self.consistency_scores)
                            / len(self.consistency_scores),
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return result

            cli.run_case_with_retries = run_case
            try:
                yield None
            finally:
                cli.run_case_with_retries = original_case
