"""Single-call Navigator role for the OnePass eight-patch pipeline."""

from __future__ import annotations

from functools import partial
import os
from typing import Annotated, final

from PIL import Image
from pydantic import BeforeValidator, TypeAdapter

from vision_text_mas.navigation.navigation_contracts import EvidencePlan
from vision_text_mas.navigation.navigation_prompts import (
    ONE_SHOT_NAVIGATOR_SYSTEM,
    one_shot_navigator_prompt,
)
from vision_text_mas.navigation.onepass_navigation_contracts import (
    OneShotNavigation,
    recover_one_shot_navigation_slots,
    repair_one_shot_navigation_output,
)
from vision_text_mas.navigation.onepass_navigator_schema import one_shot_navigation_json_schema
from vision_text_mas.qwen_client import ParsedRoleCall, QwenJsonClient

NAVIGATOR_CONTROL_TOKEN_BUDGET = int(
    os.getenv("VLMAS_NAVIGATOR_CONTROL_TOKENS", "512")
)


@final
class OneShotNavigatorAgent:
    """Choose all planned x5 regions and x20 targets in one response."""

    def __init__(
        self,
        client: QwenJsonClient,
        *,
        control_tokens: int | None = None,
    ) -> None:
        self._client = client
        self._control_tokens = (
            NAVIGATOR_CONTROL_TOKEN_BUDGET
            if control_tokens is None
            else control_tokens
        )

    def select(
        self,
        *,
        thumbnail: Image.Image,
        root_grid: Image.Image,
        evidence_plan: EvidencePlan,
        root_candidate_ids: frozenset[int],
    ) -> ParsedRoleCall[OneShotNavigation]:
        """Return the complete multi-scale navigation allocation once."""
        scale = evidence_plan.scale_plan

        def fallback(
            outputs: tuple[str, ...],
            _last_error: str,
        ) -> OneShotNavigation:
            """Keep the fixed crop contract executable after bounded JSON repair."""
            recovered_slots = (
                recover_one_shot_navigation_slots(outputs[-1]) if outputs else {}
            )
            recovered = repair_one_shot_navigation_output(
                recovered_slots,
                root_candidate_ids=root_candidate_ids,
                overview_count=scale.overview_patch_count,
                detail_count=scale.detail_patch_count,
            )
            if not isinstance(recovered, dict):
                raise RuntimeError("one-shot Navigator fallback must be a JSON object")
            if not recovered_slots:
                recovered["selection_reason"] = "deterministic eligible-tissue fallback"
            return OneShotNavigation.model_validate(recovered)

        def validate(output: OneShotNavigation) -> str | None:
            if len(output.root_ids) != scale.overview_patch_count:
                return "root IDs must match planned x5 overview count"
            if len(output.detail_cell_ids) != scale.detail_patch_count:
                return "x20 IDs must match planned x20 detail count"
            if set(output.root_ids).difference(root_candidate_ids):
                return "root IDs must be visible root-grid candidates"
            return None

        return self._client.generate_json(
            role="navigator",
            images=(thumbnail, root_grid),
            image_labels=("whole-slide-thumbnail", "root-grid"),
            system_prompt=ONE_SHOT_NAVIGATOR_SYSTEM,
            user_prompt=one_shot_navigator_prompt(
                evidence_plan=evidence_plan,
                root_candidate_ids=root_candidate_ids,
            ),
            output_adapter=TypeAdapter(
                Annotated[
                    OneShotNavigation,
                    BeforeValidator(
                        partial(
                            repair_one_shot_navigation_output,
                            root_candidate_ids=root_candidate_ids,
                            overview_count=scale.overview_patch_count,
                            detail_count=scale.detail_patch_count,
                        )
                    ),
                ]
            ),
            final_tokens=self._control_tokens,
            json_schema=one_shot_navigation_json_schema(
                root_candidate_ids=root_candidate_ids,
                overview_count=scale.overview_patch_count,
                detail_count=scale.detail_patch_count,
            ),
            semantic_validator=validate,
            fallback_factory=fallback,
            fallback_on_parse_failure=True,
        )
