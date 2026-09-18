"""Multi-image Qwen native-thinking and typed JSON finalization."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

from PIL import Image
from pydantic import TypeAdapter, ValidationError

from vision_text_mas.contracts import RoleCall
from vision_text_mas.errors import FailureCode, PipelineFailure
from vision_text_mas.json_decoding import (
    extract_json_object,
    is_terminal_json_truncation,
)
from vision_text_mas.qwen_backend import GenerationBackend, TransformersQwenBackend

OutputT = TypeVar("OutputT")
FallbackFactory = Callable[[tuple[str, ...], str], OutputT]
MAX_FORMAT_REPAIRS = int(os.getenv("VLMAS_MAX_FORMAT_REPAIRS", "2"))
MAX_EXECUTION_RETRIES = int(os.getenv("VLMAS_MAX_EXECUTION_RETRIES", "2"))
THINKING_TOKENS = 1_024


@dataclass(frozen=True, slots=True)
class ParsedRoleCall(Generic[OutputT]):
    """Validated semantic value plus its auditable call record."""

    value: OutputT
    record: RoleCall


class QwenJsonClient:
    """Run one native reasoning pass followed by bounded JSON finalization."""

    def __init__(
        self,
        backend: GenerationBackend,
        *,
        direct_json_roles: frozenset[str] = frozenset(),
        request_max_tokens: int | None = None,
    ) -> None:
        self._backend = backend
        self._direct_json_roles = direct_json_roles
        self._request_max_tokens = request_max_tokens

    @classmethod
    def from_pretrained(
        cls,
        model_path: Path,
        device: str,
        *,
        direct_json_roles: frozenset[str] = frozenset(),
        temperature: float = 0.6,
        top_p: float = 0.95,
        seed: int = 42,
        max_model_len: int = 8_192,
        request_max_tokens: int | None = None,
        constrained_json: bool = False,
    ) -> QwenJsonClient:
        """Create a client owning one shared Transformers backend."""
        return cls(
            TransformersQwenBackend.from_pretrained(
                model_path,
                device,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                max_model_len=max_model_len,
                constrained_json=constrained_json,
            ),
            direct_json_roles=direct_json_roles,
            request_max_tokens=request_max_tokens,
        )

    def _generate_with_retries(
        self,
        *,
        images: tuple[Image.Image, ...],
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
        json_prefix: str | None,
        json_schema: str | None,
        stage: str,
    ) -> tuple[str, int]:
        last_error = "model execution failed"
        for attempt in range(1, MAX_EXECUTION_RETRIES + 2):
            try:
                output = self._backend.generate(
                    images=images,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_new_tokens=(
                    min(self._request_max_tokens, max_new_tokens)
                    if self._request_max_tokens is not None
                    else max_new_tokens
                ),
                    json_prefix=json_prefix,
                    json_schema=json_schema,
                )
            except RuntimeError as error:
                last_error = str(error)
                continue
            return output, attempt
        raise PipelineFailure(
            code=FailureCode.MODEL_EXECUTION,
            stage=stage,
            detail=last_error,
        )

    def generate_final_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        final_tokens: int,
        terminal_prefix: str | None,
    ) -> RoleCall:
        """Generate one auditable terminal text response without latent state."""
        started = time.perf_counter()
        raw, physical_calls = self._generate_with_retries(
            images=(),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=final_tokens,
            json_prefix=terminal_prefix,
            json_schema=None,
            stage="answerer",
        )
        return RoleCall(
            role="answerer",
            prompt=user_prompt,
            image_labels=(),
            reasoning="",
            final_outputs=(raw,),
            format_repairs=0,
            physical_calls=physical_calls,
            elapsed_seconds=time.perf_counter() - started,
        )

    def generate_json(
        self,
        *,
        role: str,
        images: tuple[Image.Image, ...],
        image_labels: tuple[str, ...],
        system_prompt: str,
        user_prompt: str,
        output_adapter: TypeAdapter[OutputT],
        final_tokens: int,
        thinking_tokens: int = THINKING_TOKENS,
        json_schema: str | None = None,
        raw_semantic_validator: Callable[[str], str | None] | None = None,
        semantic_validator: Callable[[OutputT], str | None] | None = None,
        semantic_failure_code: FailureCode = FailureCode.MODEL_OUTPUT,
        fallback_factory: FallbackFactory[OutputT] | None = None,
        fallback_on_parse_failure: bool = False,
    ) -> ParsedRoleCall[OutputT]:
        """Generate and parse one role result without rethinking on format repair."""
        if len(images) != len(image_labels):
            raise PipelineFailure(
                code=FailureCode.MODEL_OUTPUT,
                stage=role,
                detail="image labels must match the image count",
            )
        started = time.perf_counter()
        if role in self._direct_json_roles:
            reasoning = ""
            physical_calls = 0
            final_prompt = user_prompt
        else:
            reasoning, physical_calls = self._generate_with_retries(
                images=images,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_new_tokens=thinking_tokens,
                json_prefix=None,
                json_schema=None,
                stage=role,
            )
            final_prompt = (
                f"{user_prompt}\n\nPrior visual reasoning:\n{reasoning}"
                "\n\nReturn the requested JSON object now."
            )
        outputs: list[str] = []
        output_schema = json_schema or json.dumps(output_adapter.json_schema())
        last_error = "invalid JSON"
        last_failure_code = FailureCode.MODEL_OUTPUT
        finalization_tokens = final_tokens
        for repair_count in range(MAX_FORMAT_REPAIRS + 1):
            try:
                raw, calls = self._generate_with_retries(
                    images=images,
                    system_prompt=system_prompt,
                    user_prompt=final_prompt,
                    max_new_tokens=finalization_tokens,
                    json_prefix="{",
                    json_schema=output_schema,
                    stage=role,
                )
            except PipelineFailure as failure:
                if fallback_factory is None:
                    raise
                value = fallback_factory(tuple(outputs), failure.detail)
                return ParsedRoleCall(
                    value=value,
                    record=RoleCall(
                        role=role,
                        prompt=user_prompt,
                        image_labels=image_labels,
                        reasoning=reasoning,
                        final_outputs=tuple(outputs[-3:])
                        or (f"deterministic fallback after {failure}",),
                        format_repairs=repair_count,
                        physical_calls=max(physical_calls, 1),
                        elapsed_seconds=time.perf_counter() - started,
                    ),
                )
            physical_calls += calls
            outputs.append(raw)
            try:
                json_text = extract_json_object(raw)
                raw_semantic_error = (
                    raw_semantic_validator(json_text)
                    if raw_semantic_validator is not None
                    else None
                )
                if raw_semantic_error is not None:
                    last_error = raw_semantic_error
                    last_failure_code = semantic_failure_code
                    final_prompt = (
                        f"{user_prompt}\n\nPrior visual reasoning:\n{reasoning}"
                        f"\n\nThe previous JSON violated the contract: {last_error}"
                        f"\nPrevious final output: {raw}"
                        "\nReturn only one corrected JSON object."
                    )
                    continue
                value = output_adapter.validate_json(json_text)
            except (json.JSONDecodeError, ValidationError) as error:
                last_error = str(error)
                last_failure_code = FailureCode.MODEL_OUTPUT
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
                            prompt=user_prompt,
                            image_labels=image_labels,
                            reasoning=reasoning,
                            final_outputs=tuple(outputs),
                            format_repairs=repair_count,
                            physical_calls=physical_calls,
                            elapsed_seconds=time.perf_counter() - started,
                        ),
                    )
                if isinstance(error, json.JSONDecodeError) and is_terminal_json_truncation(error):
                    finalization_tokens *= 2
                final_prompt = (
                    f"{user_prompt}\n\nPrior visual reasoning:\n{reasoning}"
                    f"\n\nThe previous JSON was invalid: {last_error}"
                    f"\nPrevious final output: {raw}"
                    "\nReturn only one corrected JSON object."
                )
                continue
            semantic_error = (
                semantic_validator(value) if semantic_validator is not None else None
            )
            if semantic_error is not None:
                last_error = semantic_error
                last_failure_code = semantic_failure_code
                final_prompt = (
                    f"{user_prompt}\n\nPrior visual reasoning:\n{reasoning}"
                    f"\n\nThe previous JSON violated the contract: {last_error}"
                    f"\nPrevious final output: {raw}"
                    "\nReturn only one corrected JSON object."
                )
                continue
            record = RoleCall(
                role=role,
                prompt=user_prompt,
                image_labels=image_labels,
                reasoning=reasoning,
                final_outputs=tuple(outputs),
                format_repairs=repair_count,
                physical_calls=physical_calls,
                elapsed_seconds=time.perf_counter() - started,
            )
            return ParsedRoleCall(value=value, record=record)
        if fallback_factory is not None:
            value = fallback_factory(tuple(outputs), last_error)
            return ParsedRoleCall(
                value=value,
                record=RoleCall(
                    role=role,
                    prompt=user_prompt,
                    image_labels=image_labels,
                    reasoning=reasoning,
                    final_outputs=tuple(outputs[-3:]),
                    format_repairs=MAX_FORMAT_REPAIRS,
                    physical_calls=physical_calls,
                    elapsed_seconds=time.perf_counter() - started,
                ),
            )
        raise PipelineFailure(
            code=last_failure_code,
            stage=role,
            detail=last_error,
        )
