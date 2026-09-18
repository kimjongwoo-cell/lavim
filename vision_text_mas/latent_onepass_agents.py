"""Planner and Reasoner adapters that retain all intermediate content in KV."""

from __future__ import annotations

from typing import Literal, Protocol

from PIL import Image

from vision_text_mas.contracts import (
    EvidenceRound,
    PatchMetadata,
    PatchObservation,
    ReasonerReport,
    RoleCall,
)
from vision_text_mas.latent_agent_prompts import (
    LATENT_REASONER_SYSTEM,
    latent_reasoner_prompt,
    prompt2_enabled,
)
from vision_text_mas.latent_onepass import announce_latent_upstream, fixed_patch_plan
from vision_text_mas.navigation_contracts import EvidencePlan
from vision_text_mas.navigation_prompts import (
    EVIDENCE_PLANNER_SYSTEM,
    evidence_planner_prompt,
)
from vision_text_mas.prompts import REASONER_SYSTEM, reasoner_prompt
from vision_text_mas.qwen_client import ParsedRoleCall


def _acquisition_event_queries(patches) -> dict[str, str]:
    """Read round_N_acquisition_events.json written beside the crop images."""
    if not patches:
        return {}
    import json as _json

    first = patches[0]
    target = (
        first.image_path.parent.parent
        / f"round_{first.round_number}_acquisition_events.json"
    )
    if not target.is_file():
        return {}
    try:
        payload = _json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    mapping = payload.get("patch_event_query")
    return mapping if isinstance(mapping, dict) else {}


class AppendOnlyClient(Protocol):
    """Capability used by intermediate latent roles with no decoded prose."""

    def append_only(
        self,
        *,
        role: str,
        images: tuple[Image.Image, ...],
        image_labels: tuple[str, ...],
        system_prompt: str,
        user_prompt: str,
    ) -> RoleCall: ...


class LatentEvidencePlannerAgent:
    """Write question and thumbnail analysis to KV; expose tool-only scale metadata."""

    def __init__(self, client: AppendOnlyClient) -> None:
        self._client = client

    def plan(
        self,
        *,
        question: str,
        thumbnail: Image.Image,
        missing_evidence: str | None,
        patch_budget: Literal[8, 12, 25] = 8,
    ) -> ParsedRoleCall[EvidencePlan]:
        """Retain Planner content in latent state and return fixed crop allocation."""
        prompt = evidence_planner_prompt(
            question=question,
            missing_evidence=missing_evidence,
            patch_budget=patch_budget,
        )
        system_prompt = EVIDENCE_PLANNER_SYSTEM
        record = self._client.append_only(
            role="evidence_planner",
            images=(thumbnail,),
            image_labels=("whole-slide-thumbnail",),
            system_prompt=system_prompt,
            user_prompt=prompt,
        )
        plan = fixed_patch_plan(question, patch_budget=patch_budget)
        # VLMAS_NAV_QUERY_FROM_PLAN: the Planner really emits the x5/x20
        # descriptors (clone decode of its own latent state) instead of the
        # placeholder strings, so the search Navigator can use them as queries.
        import os as _os
        if (_os.environ.get("VLMAS_NAV_QUERY_FROM_PLAN", "0").strip() in ("1", "2") \
                or _os.environ.get("VLMAS_NAV2", "0") == "1") \
                and hasattr(self._client, "decode_plan_targets"):
            targets = self._client.decode_plan_targets(question)
            if targets is not None:
                data = plan.model_dump()
                data["scale_plan"]["overview_target"] = targets[0][:240]
                data["scale_plan"]["detail_target"] = targets[1][:240]
                plan = EvidencePlan.model_validate(data)
        return ParsedRoleCall(value=plan, record=record)


class LatentReasonerAgent:
    """Place all materialized patches in KV without producing a prose report."""

    def __init__(self, client: AppendOnlyClient) -> None:
        self._client = client

    def observe(
        self,
        *,
        question: str,
        evidence_round: EvidenceRound,
        evidence_plan: EvidencePlan,
    ) -> ParsedRoleCall[ReasonerReport]:
        """Write the exact planned visual bundle into the shared latent state."""
        admitted_patches = evidence_round.patches
        # AAVM: the acquisition-event query that retrieved each crop, written by
        # the search Navigator next to the round's patches. Absent for the grid
        # Navigator (no per-crop evidence query exists there).
        event_queries = _acquisition_event_queries(admitted_patches)
        images: list[Image.Image] = []
        for patch in admitted_patches:
            with Image.open(patch.image_path) as source:
                image = source.convert("RGB")
                image.info["latent_event_query"] = event_queries.get(
                    str(patch.patch_id), "")
                image.info["latent_patch_id"] = str(patch.patch_id)
                image.info["latent_parent_patch_id"] = str(patch.source_x5_anchor)
                image.info["latent_magnification"] = int(patch.magnification.value)
                image.info["latent_box"] = (
                    int(patch.box.x), int(patch.box.y),
                    int(patch.box.width), int(patch.box.height),
                )
                images.append(image)
        if prompt2_enabled():
            prompt = latent_reasoner_prompt(
                question=question,
                patches=admitted_patches,
            )
            system_prompt = LATENT_REASONER_SYSTEM
        else:
            standard_prompt = reasoner_prompt(
                question=question,
                patches=admitted_patches,
                evidence_plan=evidence_plan,
            )
            prompt = announce_latent_upstream(
                standard_prompt,
                prefix="Evidence plan: ",
                note="The evidence plan is provided in latent KV representation format; it "
                "may contain irrelevant detail, so use only what helps and do not restate it.",
                stage="reasoner",
            )
            system_prompt = REASONER_SYSTEM
        record = self._client.append_only(
            role="reasoner",
            images=tuple(images),
            image_labels=tuple(str(patch.patch_id) for patch in admitted_patches),
            system_prompt=system_prompt,
            user_prompt=prompt,
        )
        carried = "carried in latent state"
        report = ReasonerReport(
            patches=tuple(
                PatchObservation(
                    patch_id=patch.patch_id,
                    quality=carried,
                    architecture=carried,
                    cellularity=carried,
                    cytology=carried,
                    stroma=carried,
                    necrosis=carried,
                    boundary=carried,
                    artifact=carried,
                    question_relevance=carried,
                    confidence=0.0,
                )
                for patch in admitted_patches
            ),
            consistent_findings=(),
            contradictions=(),
            unresolved_evidence=(),
            diagnostically_usable=tuple(
                patch.patch_id for patch in admitted_patches
            ),
        )
        return ParsedRoleCall(value=report, record=record)
