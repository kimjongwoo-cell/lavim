"""Runtime JSON Schema for a semantically legal Answerer output."""

from __future__ import annotations

import json

from vision_text_mas.contracts import DecisionBasis, PatchId


type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]


def latent_answerer_json_schema(*, answer_options: tuple[str, ...]) -> str:
    """0902 grammar_json: flat {answer, rationale, confidence} object that mirrors the
    STRUCTURED_JSON prompt schema exactly (same keys, same order). With CHOICE
    options the answer is an enum of the exact choice texts; OPEN answers are any
    non-empty string. Used only by AnswererProtocol.GRAMMAR_JSON."""
    answer: dict[str, JsonValue] = {"type": "string", "minLength": 1, "maxLength": 200}
    if answer_options:
        answer = {"type": "string", "enum": list(answer_options)}
    properties: dict[str, JsonValue] = {
        "answer": answer,
        "rationale": {"type": "string", "minLength": 1, "maxLength": 600},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    }
    schema: dict[str, JsonValue] = {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
    return json.dumps(schema, separators=(",", ":"))


def answerer_json_schema(
    *,
    decision_basis: DecisionBasis,
    cited_ids: tuple[PatchId, ...],
    answer_options: tuple[str, ...],
) -> str:
    """Constrain answers and citations to values legal for this exact case."""
    answer: dict[str, JsonValue] = {"type": "string", "minLength": 1}
    if answer_options:
        answer = {"type": "string", "enum": list(answer_options)}
    properties: dict[str, JsonValue] = {
        "decision_basis": {"type": "string", "const": decision_basis.value},
        "answer": answer,
        "evidence_patches": {
            "type": "array",
            "items": {"type": "string", "enum": [str(item) for item in cited_ids]},
            "minItems": 1,
            "maxItems": len(cited_ids),
        },
        "rationale": {"type": "string", "minLength": 1},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    }
    schema: dict[str, JsonValue] = {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
    return json.dumps(schema, separators=(",", ":"))
