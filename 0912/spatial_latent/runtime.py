"""Process-local adapters for the unmodified Base pipeline."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict

from spatial_latent.attention import BiasSettings, guided_sdpa
from spatial_latent.grouping import (
    GroupMode,
    LayoutError,
    PatchGroups,
    make_groups,
    patch_columns,
)
from vision_text_mas import latent_hf_ablation_cli as cli
from vision_text_mas import latent_hybrid_variants as hybrid
from vision_text_mas.contracts import CaseInput, RunResult

type Box = tuple[int, int, int, int]


class PatchLayout(BaseModel):
    """Parse patch metadata at the legacy runtime adapter boundary."""

    model_config = ConfigDict(frozen=True)
    parents: tuple[int, ...]
    magnifications: tuple[int, ...]
    boxes: tuple[Box | None, ...]


@dataclass(frozen=True, slots=True)
class StepTrace:
    step: int
    patch_indices: tuple[int, ...]
    visual_tokens: int
    biased_layer_calls: int
    cache_tokens: int


@dataclass(frozen=True, slots=True)
class CaseTrace:
    dataset_index: int
    mode: GroupMode
    gain: float
    groups: PatchGroups
    original_parents: tuple[int, ...]
    patch_token_counts: tuple[int, ...]
    steps: tuple[StepTrace, ...]


class SpatialExperiment:
    """Own per-case mutable step counters and traces, never model weights or KV."""

    def __init__(self, mode: GroupMode, gain: float = 2.0) -> None:
        self.mode = mode
        self.gain = gain
        self.settings = BiasSettings(log_gain=math.log(gain))
        self.steps: list[StepTrace] = []
        self.groups: PatchGroups = ()
        self.parents: tuple[int, ...] = ()
        self.columns: tuple[torch.Tensor, ...] = ()

    def _prepare(
        self, backbone: hybrid._VariantBackbone, vision_columns: torch.Tensor
    ) -> None:
        """Check true geometric containment before changing any group identity."""
        layout = PatchLayout.model_validate(
            {
                "parents": getattr(backbone, "_prune_image_parent_indices", ()),
                "magnifications": getattr(backbone, "_prune_image_magnifications", ()),
                "boxes": getattr(backbone, "_prune_image_boxes", ()),
            }
        )
        self.parents = layout.parents
        real_groups = make_groups(self.parents, GroupMode.SPATIAL)
        boxes = layout.boxes
        magnifications = layout.magnifications
        if len(boxes) != 12 or len(magnifications) != 12:
            raise LayoutError("Missing the fixed protocol's patch metadata")
        for root, *children in real_groups:
            parent_box = boxes[root]
            if parent_box is None or magnifications[root] != 5:
                raise LayoutError(f"Invalid 5x parent metadata at patch {root}")
            x, y, width, height = parent_box
            for child in children:
                child_box = boxes[child]
                if child_box is None or magnifications[child] != 20:
                    raise LayoutError(f"Invalid 20x child metadata at patch {child}")
                cx, cy, cw, ch = child_box
                if not (
                    x <= cx
                    and y <= cy
                    and cx + cw <= x + width
                    and cy + ch <= y + height
                ):
                    raise LayoutError(
                        f"Patch {child} lies outside its declared parent {root}"
                    )
        self.groups = make_groups(self.parents, self.mode)
        self.columns = patch_columns(vision_columns, expected_patches=12)

    @contextmanager
    def latent_step(
        self,
        backbone: hybrid._VariantBackbone,
        vision_columns: torch.Tensor,
        visual_patch_ids: torch.Tensor | None = None,
        parent_patch_indices: tuple[int, ...] = (),
    ) -> Iterator[None]:
        """Implement the existing latent_reallocation callback without shared edits."""
        del visual_patch_ids, parent_patch_indices
        if not getattr(backbone, "_prune_morphology_enabled", False):
            yield None
            return
        step = len(self.steps)
        if step >= 10:
            raise LayoutError("Spatial experiment requires exactly 10 Reasoner steps")
        if step == 0:
            self._prepare(backbone, vision_columns)
        group = self.groups[step // 2] if step < 8 else ()
        selected = (
            torch.cat([self.columns[i] for i in group]) if group else vision_columns[:0]
        )
        with guided_sdpa(selected, self.settings) as probe:
            yield None
        expected = 13 if group and self.gain != 1.0 else 0
        if probe.applied_calls != expected:
            raise LayoutError(
                f"Step {step}: SDPA bias fired {probe.applied_calls}, "
                f"expected {expected}"
            )
        self.steps.append(
            StepTrace(
                step=step + 1,
                patch_indices=group,
                visual_tokens=int(selected.numel()),
                biased_layer_calls=probe.applied_calls,
                cache_tokens=probe.key_count,
            )
        )

    @contextmanager
    def installed(self) -> Iterator[None]:
        """Install adapters only while this isolated command is running."""
        original_step = hybrid.latent_reallocation
        original_case = cli.run_case_with_retries

        def run_case(
            *,
            case: CaseInput,
            max_attempts: int,
            operation: Callable[[], RunResult],
            metrics_path: Path,
        ) -> RunResult:
            """Match the upstream case boundary and attach an auditable case ID."""

            def observed_operation() -> RunResult:
                self.steps.clear()
                self.groups = ()
                self.columns = ()
                result = operation()
                if len(self.steps) != 10:
                    raise LayoutError(
                        f"Case {case.dataset_index}: observed {len(self.steps)} "
                        "Reasoner steps"
                    )
                trace = CaseTrace(
                    dataset_index=case.dataset_index,
                    mode=self.mode,
                    gain=self.gain,
                    groups=self.groups,
                    original_parents=self.parents,
                    patch_token_counts=tuple(
                        int(part.numel()) for part in self.columns
                    ),
                    steps=tuple(self.steps),
                )
                path = (
                    metrics_path.parent
                    / "spatial_traces"
                    / f"{case.dataset_index:03d}.json"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(asdict(trace), indent=2) + "\n", encoding="utf-8"
                )
                return result

            return original_case(
                case=case,
                max_attempts=max_attempts,
                operation=observed_operation,
                metrics_path=metrics_path,
            )

        hybrid.latent_reallocation = self.latent_step
        cli.run_case_with_retries = run_case
        try:
            yield None
        finally:
            hybrid.latent_reallocation = original_step
            cli.run_case_with_retries = original_case
