"""Runtime JSON Schema for semantically legal Verifier decisions."""

from __future__ import annotations

import json

from vision_text_mas.contracts import MAX_FINAL_EVIDENCE_PATCHES, PatchId


type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]


def _object_schema(properties: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _nonempty_string() -> dict[str, JsonValue]:
    return {"type": "string", "minLength": 1}


def _known_patch_id(known_ids: tuple[PatchId, ...]) -> dict[str, JsonValue]:
    return {"type": "string", "enum": [str(patch_id) for patch_id in known_ids]}


def _citation_list(known_ids: tuple[PatchId, ...]) -> dict[str, JsonValue]:
    return {
        "type": "array",
        "items": _known_patch_id(known_ids),
        "minItems": 1,
        "maxItems": MAX_FINAL_EVIDENCE_PATCHES,
    }


def _verified_findings(known_ids: tuple[PatchId, ...]) -> dict[str, JsonValue]:
    finding = _object_schema(
        {
            "finding": _nonempty_string(),
            "patch_ids": _citation_list(known_ids),
        }
    )
    return {"type": "array", "items": finding}


def _pass_schema(known_ids: tuple[PatchId, ...]) -> dict[str, JsonValue]:
    return _object_schema(
        {
            "action": {"type": "string", "const": "PASS"},
            "verified_findings": _verified_findings(known_ids),
            "supporting_patches": _citation_list(known_ids),
            "reason": _nonempty_string(),
        }
    )


def _new_region_schema() -> dict[str, JsonValue]:
    return _object_schema(
        {
            "action": {"type": "string", "const": "NEW_REGION"},
            "missing_evidence": _nonempty_string(),
            "retrieval_query": _nonempty_string(),
            "reason": _nonempty_string(),
        }
    )


def _zoom_schema(original_x5_ids: tuple[PatchId, ...]) -> dict[str, JsonValue]:
    return _object_schema(
        {
            "action": {"type": "string", "const": "ZOOM_DETAIL"},
            "source_x5_anchor": _known_patch_id(original_x5_ids),
            "missing_evidence": _nonempty_string(),
            "target_magnification": {"type": "integer", "enum": [20, 40]},
            "reason": _nonempty_string(),
        }
    )


def _insufficient_schema(known_ids: tuple[PatchId, ...]) -> dict[str, JsonValue]:
    return _object_schema(
        {
            "action": {"type": "string", "const": "INSUFFICIENT"},
            "best_available_patches": _citation_list(known_ids),
            "verified_findings": _verified_findings(known_ids),
            "uncertainty": _nonempty_string(),
        }
    )


def verifier_json_schema(
    *,
    known_ids: tuple[PatchId, ...],
    original_x5_ids: tuple[PatchId, ...],
    allow_pass: bool,
    final_round: bool,
) -> str:
    """Constrain generation to decisions that pass deterministic validation."""
    variants: list[JsonValue] = []
    if allow_pass:
        variants.append(_pass_schema(known_ids))
    if final_round:
        variants.append(_insufficient_schema(known_ids))
    else:
        variants.append(_new_region_schema())
        if original_x5_ids:
            variants.append(_zoom_schema(original_x5_ids))
    schema: dict[str, JsonValue] = {"oneOf": variants}
    return json.dumps(schema, separators=(",", ":"))
