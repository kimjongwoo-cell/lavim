"""Typed control decoding from a non-mutating latent-state clone."""

from __future__ import annotations

import json
import os
import time
from typing import Callable, Generic, Protocol, TypeVar

from pydantic import TypeAdapter, ValidationError

from vision_text_mas.contracts import RoleCall
from vision_text_mas.errors import FailureCode, PipelineFailure
from vision_text_mas.latent_state import LatentCaseState
from vision_text_mas.qwen_client import ParsedRoleCall


CacheT = TypeVar("CacheT")
OutputT = TypeVar("OutputT")
FallbackFactory = Callable[[tuple[str, ...], str], OutputT]
MAX_FORMAT_REPAIRS = int(os.getenv("VLMAS_MAX_FORMAT_REPAIRS", "2"))


class CloneDecoder(Protocol[CacheT]):
    """Capability required to read a latent state without advancing it."""

    def decode_on_clone(
        self,
        state: LatentCaseState[CacheT],
        *,
        system: str,
        prompt: str,
        max_new_tokens: int,
        json_prefix: str | None,
        json_schema: str | None,
    ) -> str: ...

    def decode_in_place(
        self,
        state: LatentCaseState[CacheT],
        *,
        system: str,
        prompt: str,
        max_new_tokens: int,
        json_prefix: str | None,
        json_schema: str | None,
    ) -> str: ...


class LatentJsonClient(Generic[CacheT]):
    """Validate bounded structured decodes while preserving the main KV state."""

    def __init__(self, backend: CloneDecoder[CacheT]) -> None:
        self._backend = backend

    @staticmethod
    def _extract_json_object(raw: str, json_prefix: str = "{") -> str:
        candidate = raw.strip()
        if not candidate.startswith("{"):
            # The decode prompt already carries json_prefix, so the raw text
            # resumes mid-object; restore the exact prefix before parsing.
            candidate = json_prefix + candidate
        decoder = json.JSONDecoder()
        last_error: json.JSONDecodeError | None = None
        for start, character in enumerate(candidate):
            if character != "{":
                continue
            try:
                _, end = decoder.raw_decode(candidate[start:])
            except json.JSONDecodeError as error:
                last_error = error
                continue
            return candidate[start : start + end]
        if last_error is not None:
            raise last_error
        raise json.JSONDecodeError("no JSON object found", candidate, 0)

    def decode_json(
        self,
        *,
        state: LatentCaseState[CacheT],
        role: str,
        system: str,
        prompt: str,
        repair_prompt: str | None = None,
        audit_prompt: str | None = None,
        json_prefix: str | None = "{",
        json_schema: str | None = None,
        consume_state: bool = False,
        output_adapter: TypeAdapter[OutputT],
        final_tokens: int,
        raw_semantic_validator: Callable[[str], str | None] | None = None,
        semantic_validator: Callable[[OutputT], str | None] | None = None,
        semantic_failure_code: FailureCode = FailureCode.MODEL_OUTPUT,
        fallback_factory: FallbackFactory[OutputT] | None = None,
        fallback_on_parse_failure: bool = False,
    ) -> ParsedRoleCall[OutputT]:
        """Decode once and parse one typed control without model retries."""
        started = time.perf_counter()
        outputs: list[str] = []
        active_prompt = prompt
        repair_base = prompt if repair_prompt is None else repair_prompt
        last_error = "invalid JSON"
        last_failure_code = FailureCode.MODEL_OUTPUT
        for repair_count in range(MAX_FORMAT_REPAIRS + 1):
            decode = (
                self._backend.decode_in_place
                if consume_state
                else self._backend.decode_on_clone
            )
            if json_schema is None:
                raw = decode(
                    state,
                    system=system,
                    prompt=active_prompt,
                    max_new_tokens=final_tokens,
                    json_prefix=json_prefix,
                )
            else:
                raw = decode(
                    state,
                    system=system,
                    prompt=active_prompt,
                    max_new_tokens=final_tokens,
                    json_prefix=json_prefix,
                    json_schema=json_schema,
                )
            outputs.append(raw)
            try:
                json_text = self._extract_json_object(raw, json_prefix or "{")
                raw_semantic_error = (
                    raw_semantic_validator(json_text)
                    if raw_semantic_validator is not None
                    else None
                )
                if raw_semantic_error is not None:
                    last_error = raw_semantic_error
                    last_failure_code = semantic_failure_code
                    active_prompt = (
                        f"{repair_base}\n\nThe previous control JSON violated the "
                        f"contract: {last_error}\nPrevious output: {raw}\n"
                        "Return only one corrected JSON object."
                    )
                    continue
                value = output_adapter.validate_json(json_text)
            except (json.JSONDecodeError, ValidationError) as error:
                last_error = str(error)
                if (
                    fallback_on_parse_failure
                    and fallback_factory is not None
                    and isinstance(error, json.JSONDecodeError)
                ):
                    value = fallback_factory(tuple(outputs), last_error)
                    return ParsedRoleCall(
                        value=value,
                        record=RoleCall(
                            role=role,
                            prompt=audit_prompt or prompt,
                            image_labels=(),
                            reasoning="",
                            final_outputs=tuple(outputs),
                            format_repairs=repair_count,
                            physical_calls=len(outputs),
                            elapsed_seconds=time.perf_counter() - started,
                        ),
                    )
                active_prompt = (
                    f"{repair_base}\n\nThe previous control JSON was invalid: "
                    f"{last_error}\nPrevious output: {raw}\n"
                    "Return only one corrected JSON object."
                )
                continue
            semantic_error = (
                semantic_validator(value) if semantic_validator is not None else None
            )
            if semantic_error is not None:
                last_error = semantic_error
                last_failure_code = semantic_failure_code
                active_prompt = (
                    f"{repair_base}\n\nThe previous control JSON violated the contract: "
                    f"{last_error}\nPrevious output: {raw}\n"
                    "Return only one corrected JSON object."
                )
                continue
            return ParsedRoleCall(
                value=value,
                record=RoleCall(
                    role=role,
                    prompt=audit_prompt or prompt,
                    image_labels=(),
                    reasoning="",
                    final_outputs=tuple(outputs),
                    format_repairs=repair_count,
                    physical_calls=repair_count + 1,
                    elapsed_seconds=time.perf_counter() - started,
                ),
            )
        if fallback_factory is not None:
            value = fallback_factory(tuple(outputs), last_error)
            return ParsedRoleCall(
                value=value,
                record=RoleCall(
                    role=role,
                    prompt=audit_prompt or prompt,
                    image_labels=(),
                    reasoning="",
                    final_outputs=tuple(outputs[-3:]),
                    format_repairs=MAX_FORMAT_REPAIRS,
                    physical_calls=len(outputs),
                    elapsed_seconds=time.perf_counter() - started,
                ),
            )
        raise PipelineFailure(
            code=last_failure_code,
            stage=role,
            detail=f"{last_error}; last_output={outputs[-1][:500]!r}",
        )
