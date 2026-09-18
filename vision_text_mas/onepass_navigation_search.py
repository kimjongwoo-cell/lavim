"""Opt-in search-driven Navigator: query -> frozen tool -> observe -> query.

Enabled with VLMAS_NAV_SEARCH=1 (off = byte-identical default pipeline). The
Navigator stops picking numbered grid cells and instead acts as a controller
over a frozen WsiSearchTool:

    Planner latent KV
      -> Navigator decodes one x5 evidence query        (role "navigator")
      -> tool.search_coarse -> k5 x5 crops
      -> Navigator observes the x5 crops, decodes one
         x20 evidence query on an isolated branch       (role "navigator_detail")
      -> tool.search_fine inside each parent -> k20 x20 crops per parent
      -> final {x5 + x20} bank enters the shared cache once, downstream,
         exactly like any other evidence round (sequential acquisition,
         batched reasoning).

Invariants: Planner->Navigator communication stays latent KV; queries are
decoded only as tool arguments; the x5 observation precedes the x20 query;
hop images for the fine query never enter the shared cache (isolated branch);
tool scores are never forwarded to the Reasoner.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import Field, TypeAdapter

from vision_text_mas.contracts import (
    CandidateId,
    FrozenModel,
    NavigationAction,
    PatchId,
    PatchMetadata,
    RoleCall,
    RootCellId,
)
from vision_text_mas.errors import FailureCode, PipelineFailure
from vision_text_mas.geometry import Magnification
from vision_text_mas.navigation_contracts import EvidencePlan, SelectionTrace
from vision_text_mas.navigation_render import SlideLike, save_image
from vision_text_mas.onepass_navigation_roots import RootGrid
from vision_text_mas.qwen_client import QwenJsonClient
from vision_text_mas.wsi_search_tool import (
    NavigationResult,
    WSIObservation,
    WsiSearchTool,
    build_search_tool,
)

SEARCH_NAVIGATOR_SYSTEM = (
    "You are Navigator. Emit one retrieval query describing the visual "
    "evidence a frozen search tool should find; never answer or diagnose."
)

MINIMUM_BANK_SIZE = 5  # EvidenceRound lower bound; backfilled with extra x5.


class EvidenceQuery(FrozenModel):
    """The single structured tool argument decoded per acquisition phase."""

    evidence_query: str = Field(min_length=1)


class FineQueryEvent(FrozenModel):
    """One per-parent fine acquisition event: address query + its parent."""

    parent_id: str = Field(min_length=1)
    evidence_query: str = Field(min_length=1)


class FineQueryArray(FrozenModel):
    """VLMAS_NAV_EVENTS: one second-hop call emitting a query per x5 parent.

    Each element is an acquisition event g = (T_g, P_g): its evidence_query is
    the token span T_g whose key centroid AAVM reuses as the retrieval address,
    and the x20 crops the parent-restricted search returns are P_g. Keeping
    them anchor-specific (not scale-shared) is what makes the address
    observation-specific rather than a mere x5/x20 scale tag.
    """

    fine_queries: tuple[FineQueryEvent, ...] = Field(min_length=1, max_length=8)


def event_fine_prompt(
    evidence_plan: EvidencePlan, *, parent_ids: tuple[str, ...]
) -> str:
    scale = evidence_plan.scale_plan
    listing = ", ".join(parent_ids)
    template = json.dumps(
        {"fine_queries": [{"parent_id": parent_ids[0], "evidence_query": "..."}]}
    )
    return (
        f"Detail target at x20: {scale.detail_target}\n"
        f"Question focus: {evidence_plan.question_focus}\n"
        f"The {len(parent_ids)} attached images are the selected x5 regions, "
        f"labelled by parent_id in order: {listing}.\n"
        "For EACH parent region, considering what is visible in it and what "
        "diagnostic ambiguity remains there, describe in one sentence the "
        "visual evidence to retrieve at x20 inside that specific region. Name "
        "tissue appearance, not locations or coordinates. Give one entry per "
        "parent_id, in the same order. Do not answer the question. Output "
        f"ONLY JSON with this exact shape: {template}"
    )


def _decode_fine_events(
    client: QwenJsonClient,
    *,
    images: tuple,
    image_labels: tuple[str, ...],
    parent_ids: tuple[str, ...],
    evidence_plan: EvidencePlan,
) -> tuple[dict[str, str], RoleCall]:
    """One second-hop call: a fine evidence query per x5 parent (event)."""
    fallback_query = evidence_plan.scale_plan.detail_target

    def fallback(_outputs: tuple[str, ...], _last_error: str) -> FineQueryArray:
        return FineQueryArray(
            fine_queries=tuple(
                FineQueryEvent(parent_id=pid, evidence_query=fallback_query)
                for pid in parent_ids
            )
        )

    def validate(output: FineQueryArray) -> str | None:
        got = {event.parent_id for event in output.fine_queries}
        if got != set(parent_ids):
            return "fine_queries must cover exactly the listed parent_ids once"
        return None

    call = client.generate_json(
        role="navigator_detail",
        images=images,
        image_labels=image_labels,
        system_prompt=SEARCH_NAVIGATOR_SYSTEM,
        user_prompt=event_fine_prompt(evidence_plan, parent_ids=parent_ids),
        output_adapter=TypeAdapter(FineQueryArray),
        final_tokens=384,
        semantic_validator=validate,
        fallback_factory=fallback,
        fallback_on_parse_failure=True,
    )
    by_parent = {
        event.parent_id: event.evidence_query.strip()
        for event in call.value.fine_queries
    }
    # Any parent missing after a lenient parse falls back to the plan target.
    for pid in parent_ids:
        by_parent.setdefault(pid, fallback_query)
    return by_parent, call.record


def coarse_query_prompt(evidence_plan: EvidencePlan) -> str:
    scale = evidence_plan.scale_plan
    template = json.dumps({"evidence_query": "..."})
    # The "Validated evidence plan:" line is the role-"navigator" transport
    # convention: announce_latent_upstream swaps exactly this line for the
    # latent-handoff note, so its payload text is never sent.
    return (
        "Validated evidence plan: (carried in latent state)\n"
        f"Overview target at x5: {scale.overview_target}\n"
        f"Question focus: {evidence_plan.question_focus}\n"
        "Describe in one sentence the visual evidence a region-retrieval tool "
        "should find at x5 magnification to satisfy the overview target. "
        "Name tissue appearance, not locations or coordinates. Do not answer "
        f"the question. Output ONLY JSON with this exact key: {template}"
    )


def fine_query_prompt(evidence_plan: EvidencePlan, *, crop_count: int) -> str:
    scale = evidence_plan.scale_plan
    template = json.dumps({"evidence_query": "..."})
    return (
        f"Detail target at x20: {scale.detail_target}\n"
        f"Question focus: {evidence_plan.question_focus}\n"
        f"The {crop_count} attached images are the x5 regions already "
        "retrieved. Considering what is visible in them and what diagnostic "
        "ambiguity remains, describe in one sentence the visual evidence to "
        "retrieve at x20 magnification. Name tissue appearance, not locations "
        "or coordinates. Do not answer the question. Output ONLY JSON with "
        f"this exact key: {template}"
    )


def _decode_query(
    client: QwenJsonClient,
    *,
    role: str,
    images: tuple,
    image_labels: tuple[str, ...],
    user_prompt: str,
    fallback_query: str,
) -> tuple[str, RoleCall]:
    def fallback(_outputs: tuple[str, ...], _last_error: str) -> EvidenceQuery:
        """Keep the acquisition executable with the plan's own target text."""
        return EvidenceQuery(evidence_query=fallback_query)

    call = client.generate_json(
        role=role,
        images=images,
        image_labels=image_labels,
        system_prompt=SEARCH_NAVIGATOR_SYSTEM,
        user_prompt=user_prompt,
        output_adapter=TypeAdapter(EvidenceQuery),
        final_tokens=192,
        fallback_factory=fallback,
        fallback_on_parse_failure=True,
    )
    return call.value.evidence_query.strip(), call.record


def _to_patch(
    observation: WSIObservation,
    *,
    round_number: int,
    rank: int,
    anchor_ids: dict[str, PatchId],
    artifact_root: Path,
) -> PatchMetadata:
    patch_id = PatchId(f"R{round_number}-P{rank}")
    if observation.parent_crop_id is None:
        anchor_ids[observation.crop_id] = patch_id
        anchor = patch_id
    else:
        anchor = anchor_ids[observation.parent_crop_id]
    path = save_image(
        observation.image,
        artifact_root / f"round_{round_number}" / f"{patch_id}.png",
    )
    return PatchMetadata(
        patch_id=patch_id,
        round_number=round_number,
        rank=rank,
        action=NavigationAction.INITIAL,
        magnification=Magnification(observation.scale),
        box=observation.box,
        root_id=RootCellId(observation.root_id),
        source_x5_anchor=anchor,
        candidate_id=CandidateId(observation.candidate_id),
        tissue_fraction=observation.tissue_fraction,
        image_path=path,
    )


def _dump_provenance(
    result: NavigationResult,
    *,
    artifact_root: Path,
    round_number: int,
) -> None:
    payload = {
        "coarse_query": result.coarse_query,
        "fine_query": result.fine_query,
        "crops": [
            {
                "crop_id": crop.crop_id,
                "scale": crop.scale,
                "box": [crop.box.x, crop.box.y, crop.box.width, crop.box.height],
                "parent_crop_id": crop.parent_crop_id,
                "tissue_fraction": round(crop.tissue_fraction, 4),
            }
            for crop in result.all_crops
        ],
    }
    target = artifact_root / f"round_{round_number}_search_provenance.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    _ = target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_search_navigation(
    slide: SlideLike,
    *,
    client: QwenJsonClient,
    root_grid: RootGrid,
    evidence_plan: EvidencePlan,
    artifact_root: Path,
    round_number: int,
    tool: WsiSearchTool | None = None,
) -> tuple[tuple[PatchMetadata, ...], tuple[RoleCall, ...], SelectionTrace]:
    """Run the two-phase search acquisition and return one batched bank."""
    scale = evidence_plan.scale_plan
    k5 = int(os.environ.get("VLMAS_NAV_SEARCH_K5", "0")) or scale.overview_patch_count
    k20 = int(os.environ.get("VLMAS_NAV_SEARCH_K20", "2"))
    search_tool = tool if tool is not None else build_search_tool(root_grid=root_grid)

    # VLMAS_NAV_QUERY_FROM_PLAN: 1 = the x5 query IS the Planner's
    # overview_target (no Navigator decode; the Navigator still sees the x5
    # crops and writes the x20 query); 2 = both queries from the plan (pure
    # Planner-driven acquisition, diagnostic). Default 0 = decode both.
    plan_mode = os.environ.get("VLMAS_NAV_QUERY_FROM_PLAN", "0").strip()
    if plan_mode in ("1", "2"):
        coarse_query = scale.overview_target.strip()
        coarse_record = RoleCall(
            role="navigator", prompt="<query from plan: overview_target>",
            image_labels=(), reasoning="", final_outputs=(coarse_query,),
            format_repairs=0, physical_calls=1, elapsed_seconds=0.0)
        print(f"[NavSearch] x5 query from plan: {coarse_query[:100]}", flush=True)
    else:
        coarse_query, coarse_record = _decode_query(
            client,
            role="navigator",
            images=(),
            image_labels=(),
            user_prompt=coarse_query_prompt(evidence_plan),
            fallback_query=scale.overview_target,
        )
    coarse_crops = search_tool.search_coarse(slide, query=coarse_query, top_k=k5)
    if not coarse_crops:
        raise PipelineFailure(
            code=FailureCode.INSUFFICIENT_CANDIDATES,
            stage="navigator",
            detail="search tool returned no tissue-safe x5 crop",
        )

    events_mode = os.environ.get("VLMAS_NAV_EVENTS", "0") == "1"
    # crop_id -> the acquisition-event query (T_g) that retrieved this crop.
    crop_event_query: dict[str, str] = {crop.crop_id: coarse_query for crop in coarse_crops}
    if events_mode:
        parent_ids = tuple(crop.crop_id for crop in coarse_crops)
        by_parent, fine_record = _decode_fine_events(
            client,
            images=tuple(crop.image for crop in coarse_crops),
            image_labels=tuple(f"x5-{crop.crop_id}" for crop in coarse_crops),
            parent_ids=parent_ids,
            evidence_plan=evidence_plan,
        )
        fine_query = " || ".join(f"{pid}: {q}" for pid, q in by_parent.items())
        fine_crops = []
        for parent in coarse_crops:
            parent_query = by_parent[parent.crop_id]
            children = search_tool.search_fine(
                slide, parent=parent, query=parent_query, top_k=k20
            )
            for child in children:
                crop_event_query[child.crop_id] = parent_query
            fine_crops.extend(children)
    elif plan_mode == "2":
        fine_query = scale.detail_target.strip()
        fine_record = RoleCall(
            role="navigator_detail", prompt="<query from plan: detail_target>",
            image_labels=(), reasoning="", final_outputs=(fine_query,),
            format_repairs=0, physical_calls=1, elapsed_seconds=0.0)
        print(f"[NavSearch] x20 query from plan: {fine_query[:100]}", flush=True)
        fine_crops = []
        for parent in coarse_crops:
            children = search_tool.search_fine(
                slide, parent=parent, query=fine_query, top_k=k20
            )
            for child in children:
                crop_event_query[child.crop_id] = fine_query
            fine_crops.extend(children)
    else:
        fine_query, fine_record = _decode_query(
            client,
            role="navigator_detail",
            images=tuple(crop.image for crop in coarse_crops),
            image_labels=tuple(f"x5-{crop.crop_id}" for crop in coarse_crops),
            user_prompt=fine_query_prompt(evidence_plan, crop_count=len(coarse_crops)),
            fallback_query=scale.detail_target,
        )
        fine_crops = []
        for parent in coarse_crops:
            children = search_tool.search_fine(
                slide, parent=parent, query=fine_query, top_k=k20
            )
            for child in children:
                crop_event_query[child.crop_id] = fine_query
            fine_crops.extend(children)

    # EvidenceRound requires at least MINIMUM_BANK_SIZE patches; a sparse slide
    # can under-fill the fine phase, so extend the deterministic coarse ranking.
    coarse_list = list(coarse_crops)
    while len(coarse_list) + len(fine_crops) < MINIMUM_BANK_SIZE:
        extended = search_tool.search_coarse(
            slide,
            query=coarse_query,
            top_k=len(coarse_list) + 1,
        )
        if len(extended) <= len(coarse_list):
            raise PipelineFailure(
                code=FailureCode.INSUFFICIENT_CANDIDATES,
                stage="navigator",
                detail="search bank cannot reach the minimum evidence size",
            )
        coarse_list = list(extended)

    result = NavigationResult(
        coarse_query=coarse_query,
        fine_query=fine_query,
        coarse_crops=tuple(coarse_list),
        fine_crops=tuple(fine_crops),
    )
    _dump_provenance(result, artifact_root=artifact_root, round_number=round_number)
    # VLMAS_NAV_DETAIL_APPEND=1: the x5 observations were already appended to
    # the shared cache by the navigator_detail hop, so the Reasoner bank
    # carries only the x20 crops (same total visual columns, no duplicates).
    # Fall back to the full bank if the fine phase alone is under the minimum.
    import os as _os
    bank_crops = result.all_crops
    if _os.environ.get("VLMAS_NAV_DETAIL_APPEND", "") == "1" and len(fine_crops) >= MINIMUM_BANK_SIZE:
        bank_crops = tuple(fine_crops)
        print(f"[NavSearch] detail-append: bank = {len(bank_crops)} x20 crops "
              f"(x5 {len(coarse_list)} already in shared KV)", flush=True)

    anchor_ids: dict[str, PatchId] = {}
    patch_event_query: dict[str, str] = {}
    patches_list: list[PatchMetadata] = []
    bank_ids = {crop.crop_id for crop in bank_crops}
    # EvidenceRound requires bank IDs R{n}-P1..Pk contiguous, so bank crops
    # are numbered first; x5 crops left out of the bank (detail-append) are
    # still materialised afterwards with trailing ranks -- their PatchId is
    # the x20 anchor and the validator does not require anchors in the bank.
    ordered = list(bank_crops) + [c for c in result.all_crops if c.crop_id not in bank_ids]
    if any(c.parent_crop_id is not None and c.parent_crop_id not in bank_ids for c in bank_crops):
        # register the excluded x5 anchors before their children are built
        for rank, observation in enumerate(
                [c for c in result.all_crops if c.crop_id not in bank_ids], start=len(bank_crops) + 1):
            _to_patch(observation, round_number=round_number, rank=rank,
                      anchor_ids=anchor_ids, artifact_root=artifact_root)
        ordered = list(bank_crops)
    for rank, observation in enumerate(ordered, start=1):
        patch = _to_patch(
            observation,
            round_number=round_number,
            rank=rank,
            anchor_ids=anchor_ids,
            artifact_root=artifact_root,
        )
        if observation.crop_id not in bank_ids:
            continue
        patches_list.append(patch)
        patch_event_query[str(patch.patch_id)] = crop_event_query.get(
            observation.crop_id, coarse_query
        )
    patches = tuple(patches_list)
    # AAVM consumes this: each retained crop's acquisition-event query text,
    # whose per-layer/head key centroid becomes the crop's retrieval address.
    events_target = (
        artifact_root / f"round_{round_number}_acquisition_events.json"
    )
    events_target.parent.mkdir(parents=True, exist_ok=True)
    _ = events_target.write_text(
        json.dumps(
            {
                "events_mode": events_mode,
                "coarse_query": coarse_query,
                "patch_event_query": patch_event_query,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    selected_roots = tuple(dict.fromkeys(crop.root_id for crop in result.coarse_crops))
    trace = SelectionTrace(
        label=f"round-{round_number}-search-coarse",
        ranked_ids=selected_roots[:10],
        skipped_low_tissue_ids=(),
        selected_ids=selected_roots[:5],
    )
    return patches, (coarse_record, fine_record), trace
