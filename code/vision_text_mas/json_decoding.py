"""Small JSON decoding primitives shared by generation clients."""

from __future__ import annotations

import json


def extract_json_object(raw: str) -> str:
    """Return the first complete JSON object, restoring a missing opener."""
    candidate = raw.strip()
    if not candidate.startswith("{"):
        candidate = "{" + candidate
    decoder = json.JSONDecoder()
    for start, character in enumerate(candidate):
        if character != "{":
            continue
        _, end = decoder.raw_decode(candidate[start:])
        return candidate[start : start + end]
    message = "no JSON object found"
    raise json.JSONDecodeError(message, candidate, 0)


def is_terminal_json_truncation(error: json.JSONDecodeError) -> bool:
    """Return whether JSON ended where a closing token was expected."""
    document = error.doc.rstrip()
    return bool(document) and (
        error.pos >= len(document) - 1
        or error.msg.startswith("Unterminated string")
    )
