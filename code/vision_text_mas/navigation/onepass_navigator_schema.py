"""Runtime JSON schema for one-call multiscale navigation."""

from __future__ import annotations

import json


def one_shot_navigation_json_schema(
    *,
    root_candidate_ids: frozenset[int],
    overview_count: int,
    detail_count: int,
) -> str:
    """Constrain the one Navigator response to its executable allocation."""
    root_ids = sorted(root_candidate_ids)
    schema = {
        "type": "object",
        "properties": {
            "root_ids": {
                "type": "array",
                "items": {"type": "integer", "enum": root_ids},
                "minItems": overview_count,
                "maxItems": overview_count,
            },
            "detail_cell_ids": {
                "type": "array",
                "items": {"type": "integer", "minimum": 1, "maximum": 16},
                "minItems": detail_count,
                "maxItems": detail_count,
            },
            "selection_reason": {"type": "string", "minLength": 1},
        },
        "required": ["root_ids", "detail_cell_ids", "selection_reason"],
        "additionalProperties": False,
    }
    return json.dumps(schema, separators=(",", ":"))
