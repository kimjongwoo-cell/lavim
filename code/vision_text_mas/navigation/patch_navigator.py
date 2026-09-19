"""Plan-conditioned Patch Navigator role."""

from __future__ import annotations

from functools import partial
from typing import Annotated

from PIL import Image
from pydantic import BeforeValidator, TypeAdapter

from vision_text_mas.errors import FailureCode
from vision_text_mas.navigation.navigation_contracts import EvidencePlan, RankedSelection
from vision_text_mas.navigation.navigation_prompts import (
    PATCH_NAVIGATOR_SYSTEM,
    patch_navigator_prompt,
)
from vision_text_mas.navigation.patch_navigator_schema import patch_navigator_json_schema
from vision_text_mas.qwen_client import ParsedRoleCall, QwenJsonClient
from vision_text_mas.output_repair import repair_ranked_selection_output


class PatchNavigatorAgent:
    """Rank candidate IDs against a validated evidence plan."""

    def __init__(self, client: QwenJsonClient) -> None:
        self._client = client

    def select(
        self,
        *,
        image: Image.Image,
        image_label: str,
        evidence_plan: EvidencePlan,
        candidate_ids: frozenset[int],
        selection_count: int,
    ) -> ParsedRoleCall[RankedSelection]:
        """Return exactly one available, unique ranked reserve."""

        def validate(output: RankedSelection) -> str | None:
            if len(output.ranked) != selection_count:
                return f"exactly {selection_count} ranked candidates are required"
            invalid = set(output.ids).difference(candidate_ids)
            if invalid:
                return (
                    f"unavailable IDs {sorted(invalid)}; available IDs are "
                    f"{sorted(candidate_ids)}"
                )
            return None

        return self._client.generate_json(
            role="patch_navigator",
            images=(image,),
            image_labels=(image_label,),
            system_prompt=PATCH_NAVIGATOR_SYSTEM,
            user_prompt=patch_navigator_prompt(
                evidence_plan=evidence_plan,
                candidate_ids=candidate_ids,
                selection_count=selection_count,
            ),
            output_adapter=TypeAdapter(
                Annotated[
                    RankedSelection,
                    BeforeValidator(
                        partial(
                            repair_ranked_selection_output,
                            candidate_ids=candidate_ids,
                            selection_count=selection_count,
                        )
                    ),
                ]
            ),
            final_tokens=512,
            json_schema=patch_navigator_json_schema(
                candidate_ids=candidate_ids,
                selection_count=selection_count,
            ),
            semantic_validator=validate,
            semantic_failure_code=FailureCode.SELECTION_CONTRACT,
        )
