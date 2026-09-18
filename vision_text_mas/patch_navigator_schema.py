"""Runtime JSON Schema for legal Patch Navigator rankings."""

from __future__ import annotations

import json


type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]


def patch_navigator_json_schema(
    *,
    candidate_ids: frozenset[int],
    selection_count: int,
) -> str:
    """Constrain every ranked ID to the visible grid and exact reserve size."""
    enum_values: list[JsonValue] = [value for value in sorted(candidate_ids)]
    id_schema: dict[str, JsonValue] = {
        "type": "integer",
        "enum": enum_values,
    }
    reason_schema: dict[str, JsonValue] = {"type": "string", "minLength": 1}
    candidate_properties: dict[str, JsonValue] = {
        "id": id_schema,
        "match_reason": reason_schema,
    }
    candidate: dict[str, JsonValue] = {
        "type": "object",
        "properties": candidate_properties,
        "required": ["id", "match_reason"],
        "additionalProperties": False,
    }
    schema: dict[str, JsonValue] = {
        "type": "object",
        "properties": {
            "ranked": {
                "type": "array",
                "items": candidate,
                "minItems": selection_count,
                "maxItems": selection_count,
            }
        },
        "required": ["ranked"],
        "additionalProperties": False,
    }
    return json.dumps(schema, separators=(",", ":"))
