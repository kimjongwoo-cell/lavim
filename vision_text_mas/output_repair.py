"""Conservative repairs at the untrusted structured-output boundary."""

from __future__ import annotations

from difflib import SequenceMatcher
import re
from typing import Final

from vision_text_mas.contracts import MAX_FINAL_EVIDENCE_PATCHES, PatchId
from vision_text_mas.evidence import EvidenceMemory
from vision_text_mas.errors import PipelineFailure


type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
SPLIT_CHOICE_LABEL: Final = re.compile(r"^[a-d]\s*[:.)-]\s*", re.IGNORECASE)
ALL_ABOVE_VARIANT: Final = re.compile(r"^all(?: of)?(?: the)? above$")
CHOICE_LIST_SEPARATOR: Final = re.compile(r"\s*(?:,|;|/|\band\b)\s*")
TYPO_CHOICE_MIN_LENGTH: Final = 8
TYPO_CHOICE_MIN_RATIO: Final = 0.94


def _trim_patch_ids(value: JsonValue) -> JsonValue:
    if not isinstance(value, list):
        return value
    unique: list[JsonValue] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            return value
        if item not in seen:
            unique.append(item)
            seen.add(item)
    return unique[:MAX_FINAL_EVIDENCE_PATCHES]


def _repair_findings(value: JsonValue) -> JsonValue:
    if not isinstance(value, list):
        return value
    repaired: list[JsonValue] = []
    for finding in value:
        if not isinstance(finding, dict):
            repaired.append(finding)
            continue
        normalized = dict(finding)
        if "patch_ids" in normalized:
            normalized["patch_ids"] = _trim_patch_ids(normalized["patch_ids"])
        repaired.append(normalized)
    return repaired


def repair_verifier_output(
    value: JsonValue,
    *,
    memory: EvidenceMemory,
) -> JsonValue:
    """Repair only citation cardinality and known patch lineage."""
    if not isinstance(value, dict):
        return value
    repaired = dict(value)
    action = repaired.get("action")
    match action:  # noqa: MATCH_OK - raw JSON tag remains open until Pydantic parsing.
        case "PASS":
            if "supporting_patches" in repaired:
                repaired["supporting_patches"] = _trim_patch_ids(
                    repaired["supporting_patches"]
                )
            if "verified_findings" in repaired:
                repaired["verified_findings"] = _repair_findings(
                    repaired["verified_findings"]
                )
        case "INSUFFICIENT":
            if "best_available_patches" in repaired:
                repaired["best_available_patches"] = _trim_patch_ids(
                    repaired["best_available_patches"]
                )
            if "verified_findings" in repaired:
                repaired["verified_findings"] = _repair_findings(
                    repaired["verified_findings"]
                )
        case "ZOOM_DETAIL":
            anchor = repaired.get("source_x5_anchor")
            if not isinstance(anchor, str):
                return repaired
            try:
                patch = memory.find_patch(PatchId(anchor))
            except PipelineFailure:
                return repaired
            repaired["source_x5_anchor"] = str(patch.source_x5_anchor)
        case "NEW_REGION" | _:
            pass
    return repaired


def _choice_key(value: str) -> str:
    normalized = " ".join(value.casefold().split())
    normalized = SPLIT_CHOICE_LABEL.sub("", normalized)
    return normalized.rstrip(" .,:;!?")


def _all_above_key(value: str) -> str:
    """Normalize only the fixed all-of-the-above cosmetic variant."""
    key = _choice_key(value)
    return "all above" if ALL_ABOVE_VARIANT.fullmatch(key) else key


def _all_above_from_listed_choices(
    answer: str,
    choices: tuple[str, ...],
) -> str | None:
    """Recover an all-of-the-above choice from two or more exact listed options."""
    all_above = tuple(
        choice for choice in choices if ALL_ABOVE_VARIANT.fullmatch(_choice_key(choice))
    )
    listed = tuple(
        part for part in CHOICE_LIST_SEPARATOR.split(_choice_key(answer)) if part
    )
    individual = frozenset(
        _choice_key(choice)
        for choice in choices
        if not ALL_ABOVE_VARIANT.fullmatch(_choice_key(choice))
    )
    if len(all_above) == 1 and len(listed) >= 2 and all(part in individual for part in listed):
        return all_above[0]
    return None


def _unique_typo_choice(answer: str, choices: tuple[str, ...]) -> str | None:
    """Restore one long answer when it differs from exactly one choice by a typo."""
    answer_key = _choice_key(answer)
    if len(answer_key) < TYPO_CHOICE_MIN_LENGTH:
        return None
    matching = tuple(
        choice
        for choice in choices
        if len(_choice_key(choice)) >= TYPO_CHOICE_MIN_LENGTH
        and SequenceMatcher(None, answer_key, _choice_key(choice)).ratio()
        >= TYPO_CHOICE_MIN_RATIO
    )
    return matching[0] if len(matching) == 1 else None


def _repair_answer_citations(
    value: JsonValue,
    *,
    cited_ids: tuple[PatchId, ...],
) -> JsonValue:
    """Keep answer citations within the verifier-selected evidence set."""
    trimmed = _trim_patch_ids(value)
    if not isinstance(trimmed, list) or not all(
        isinstance(item, str) for item in trimmed
    ):
        return trimmed
    allowed = frozenset(str(patch_id) for patch_id in cited_ids)
    supported: list[JsonValue] = [item for item in trimmed if item in allowed]
    if supported or not cited_ids:
        return supported
    fallback: list[JsonValue] = [str(cited_ids[0])]
    return fallback


def repair_ranked_selection_output(
    value: JsonValue,
    *,
    candidate_ids: frozenset[int],
    selection_count: int,
) -> JsonValue:
    """Retain valid Navigator ranking order and fill unavailable IDs locally."""
    if not isinstance(value, dict):
        return value
    raw_ranked = value.get("ranked")
    if not isinstance(raw_ranked, list):
        return value
    ranked: list[JsonValue] = []
    selected_ids: set[int] = set()
    for raw_candidate in raw_ranked:
        if not isinstance(raw_candidate, dict):
            continue
        candidate_id = raw_candidate.get("id")
        if (
            not isinstance(candidate_id, int)
            or isinstance(candidate_id, bool)
            or candidate_id not in candidate_ids
            or candidate_id in selected_ids
        ):
            continue
        reason = raw_candidate.get("match_reason")
        ranked.append(
            {
                "id": candidate_id,
                "match_reason": (
                    reason if isinstance(reason, str) and reason else "model-ranked"
                ),
            }
        )
        selected_ids.add(candidate_id)
    for candidate_id in sorted(candidate_ids):
        if len(ranked) == selection_count:
            break
        if candidate_id in selected_ids:
            continue
        ranked.append({"id": candidate_id, "match_reason": "eligible fallback"})
        selected_ids.add(candidate_id)
    repaired = dict(value)
    repaired["ranked"] = ranked
    return repaired


def repair_answer_output(
    value: JsonValue,
    *,
    choices: tuple[str, ...],
    cited_ids: tuple[PatchId, ...],
) -> JsonValue:
    """Repair answer citations and one uniquely matching cosmetic choice drift."""
    if not isinstance(value, dict):
        return value
    repaired = dict(value)
    if "decision" in repaired:
        # Qwen occasionally abbreviates the schema field. Normalize this only at
        # the AnswerOutput boundary, where ``decision`` is otherwise forbidden.
        if "decision_basis" not in repaired:
            repaired["decision_basis"] = repaired["decision"]
        del repaired["decision"]
    if "evidence_patches" in repaired:
        repaired["evidence_patches"] = _repair_answer_citations(
            repaired["evidence_patches"],
            cited_ids=cited_ids,
        )
    if not choices:
        return repaired
    answer = value.get("answer")
    if not isinstance(answer, str) or answer in choices:
        return repaired
    matching = tuple(
        choice for choice in choices if _choice_key(choice) == _choice_key(answer)
    )
    if not matching:
        matching = tuple(
            choice
            for choice in choices
            if _all_above_key(choice) == _all_above_key(answer)
        )
    if not matching:
        all_above = _all_above_from_listed_choices(answer, choices)
        matching = (all_above,) if all_above is not None else ()
    if not matching:
        typo_choice = _unique_typo_choice(answer, choices)
        matching = (typo_choice,) if typo_choice is not None else ()
    if len(matching) != 1:
        return repaired
    repaired["answer"] = matching[0]
    return repaired
