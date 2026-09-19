"""Latent-only transport primitives for the four-agent OnePass topology."""

from __future__ import annotations

from copy import deepcopy
import time
from typing import Callable, Generic, Literal, Protocol, TypeVar

from PIL import Image
from pydantic import TypeAdapter

from vision_text_mas.contracts import RoleCall
from vision_text_mas.errors import FailureCode, PipelineFailure
from vision_text_mas.latent_client import FallbackFactory, LatentJsonClient
from vision_text_mas.latent_state import LatentCaseState
from vision_text_mas.navigation.navigation_contracts import EvidencePlan
from vision_text_mas.qwen_client import ParsedRoleCall


CacheT = TypeVar("CacheT")
OutputT = TypeVar("OutputT")


def _locate_upstream_line(lines: list[str], *, prefix: str, stage: str) -> int:
    """Return the sole line index carrying the prior-agent text payload."""
    matched = [index for index, line in enumerate(lines) if line.startswith(prefix)]
    if len(matched) != 1:
        raise PipelineFailure(
            code=FailureCode.MODEL_OUTPUT,
            stage=stage,
            detail=(
                "expected exactly one serialized upstream prompt line with "
                f"prefix {prefix!r}; found {len(matched)}"
            ),
        )
    return matched[0]


def remove_serialized_upstream_line(
    prompt: str,
    *,
    prefix: str,
    stage: str,
) -> str:
    """Remove the one prior-agent text payload that is carried by KV instead."""
    lines = prompt.splitlines(keepends=True)
    dropped = _locate_upstream_line(lines, prefix=prefix, stage=stage)
    return "".join(line for index, line in enumerate(lines) if index != dropped)


def announce_latent_upstream(
    prompt: str,
    *,
    prefix: str,
    note: str,
    stage: str,
) -> str:
    """Swap the KV-carried payload line for a short latent-handoff note.

    Mirrors the original LatentMAS convention of telling each downstream agent
    that its upstream context arrives in latent state, without re-feeding the
    serialized text the KV already holds.
    """
    lines = prompt.splitlines(keepends=True)
    index = _locate_upstream_line(lines, prefix=prefix, stage=stage)
    newline = "\n" if lines[index].endswith("\n") else ""
    lines[index] = f"{note}{newline}"
    return "".join(lines)


class LatentTransportBackend(Protocol[CacheT]):
    """The stateful operations required by the OnePass role transport."""

    def append_multimodal_latent(
        self,
        state: LatentCaseState[CacheT],
        *,
        stage: str,
        images: tuple[Image.Image, ...],
        system: str,
        prompt: str,
        latent_steps: int,
    ) -> LatentCaseState[CacheT]: ...

    def branch_multimodal_latent(
        self,
        state: LatentCaseState[CacheT],
        *,
        stage: str,
        images: tuple[Image.Image, ...],
        system: str,
        prompt: str,
        latent_steps: int,
    ) -> LatentCaseState[CacheT]: ...

    def decode_on_clone(
        self,
        state: LatentCaseState[CacheT],
        *,
        system: str,
        prompt: str,
        max_new_tokens: int,
        json_prefix: str | None,
        json_schema: str | None = None,
    ) -> str: ...

    def decode_in_place(
        self,
        state: LatentCaseState[CacheT],
        *,
        system: str,
        prompt: str,
        max_new_tokens: int,
        json_prefix: str | None,
        json_schema: str | None = None,
    ) -> str: ...

    def release(self, state: LatentCaseState[CacheT]) -> None: ...

class LatentRoleClient(Generic[CacheT]):
    """Carry Planner, Navigator, and Reasoner state without decoded prose."""

    def __init__(
        self,
        *,
        backend: LatentTransportBackend[CacheT],
        initial_state: LatentCaseState[CacheT],
        latent_steps: int,
        max_model_len: int,
        retain_navigator_kv: bool = True,
    ) -> None:
        self._backend = backend
        self._decoder = LatentJsonClient(backend)
        self._state = initial_state
        self._latent_steps = latent_steps
        self._max_model_len = max_model_len
        self._retain_navigator_kv = retain_navigator_kv

    @property
    def state(self) -> LatentCaseState[CacheT]:
        """Expose only the immutable, tensor-owning state snapshot."""
        return self._state

    def append_only(
        self,
        *,
        role: str,
        images: tuple[Image.Image, ...],
        image_labels: tuple[str, ...],
        system_prompt: str,
        user_prompt: str,
    ) -> RoleCall:
        """Advance one intermediate role without exposing textual reasoning."""
        if len(images) != len(image_labels):
            raise PipelineFailure(
                code=FailureCode.MODEL_OUTPUT,
                stage=role,
                detail="image labels must match the image count",
            )
        started = time.perf_counter()
        state = self._backend.append_multimodal_latent(
            self._state,
            stage=role,
            images=images,
            system=system_prompt,
            prompt=user_prompt,
            latent_steps=self._latent_steps,
        )
        if state.cache_length > self._max_model_len:
            raise PipelineFailure(
                code=FailureCode.MODEL_EXECUTION,
                stage=role,
                detail=(
                    "latent state exceeds the fixed context limit: "
                    f"{state.cache_length} > {self._max_model_len}"
                ),
            )
        self._state = state
        return RoleCall(
            role=role,
            prompt=user_prompt,
            image_labels=image_labels,
            reasoning="",
            final_outputs=("<carried-in-latent-state>",),
            format_repairs=0,
            physical_calls=1,
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
        thinking_tokens: int = 1_024,
        json_schema: str | None = None,
        raw_semantic_validator: Callable[[str], str | None] | None = None,
        semantic_validator: Callable[[OutputT], str | None] | None = None,
        semantic_failure_code: FailureCode = FailureCode.MODEL_OUTPUT,
        fallback_factory: FallbackFactory[OutputT] | None = None,
        fallback_on_parse_failure: bool = False,
    ) -> ParsedRoleCall[OutputT]:
        """Decode only Navigator tool JSON or the terminal Answerer JSON."""
        _ = thinking_tokens
        append_elapsed = 0.0
        match role:
            case "navigator":
                # Mirror upstream LatentMAS framing (Critic/Refiner/Judger): tell the
                # latent-consuming Navigator that its plan arrives in latent KV format
                # and may hold irrelevant detail — instead of silently dropping the line.
                transport_prompt = announce_latent_upstream(
                    user_prompt,
                    prefix="Validated evidence plan: ",
                    note="The evidence plan is provided in latent KV representation "
                    "format; it may contain irrelevant detail, so use only what helps "
                    "and do not restate it.",
                    stage=role,
                )
                if self._retain_navigator_kv:
                    audit = self.append_only(
                        role=role,
                        images=images,
                        image_labels=image_labels,
                        system_prompt=system_prompt,
                        user_prompt=transport_prompt,
                    )
                    active_state = self._state
                else:
                    started = time.perf_counter()
                    active_state = self._backend.branch_multimodal_latent(
                        self._state,
                        stage=role,
                        images=images,
                        system=system_prompt,
                        prompt=transport_prompt,
                        latent_steps=self._latent_steps,
                    )
                    if active_state.cache_length > self._max_model_len:
                        raise PipelineFailure(
                            code=FailureCode.MODEL_EXECUTION,
                            stage=role,
                            detail=(
                                "latent state exceeds the fixed context limit: "
                                f"{active_state.cache_length} > {self._max_model_len}"
                            ),
                        )
                    audit = RoleCall(
                        role=role,
                        prompt=transport_prompt,
                        image_labels=image_labels,
                        reasoning="",
                        final_outputs=("<isolated-from-latent-state>",),
                        format_repairs=0,
                        physical_calls=1,
                        elapsed_seconds=time.perf_counter() - started,
                    )
                append_elapsed = audit.elapsed_seconds
                consume_state = False
                prefix_calls = audit.physical_calls
                record_labels = audit.image_labels
                audit_prompt = transport_prompt
                repair_prompt = transport_prompt
            case "answerer":
                # The final Answerer consumes the already accumulated cache. Its
                # prompt is prefilled as normal text by the terminal decode, rather
                # than being converted into another latent-agent block.
                active_state = self._state
                consume_state = True
                prefix_calls = 0
                record_labels: tuple[str, ...] = ()
                audit_prompt = user_prompt
                repair_prompt = user_prompt
            case unsupported:
                raise PipelineFailure(
                    code=FailureCode.MODEL_OUTPUT,
                    stage=unsupported,
                    detail="latent OnePass only permits navigator or answerer JSON",
                )
        decoded = self._decoder.decode_json(
            state=active_state,
            role=role,
            # Both supported roles append their full prompt immediately above;
            # decoding must continue from that assistant boundary, not replay it.
            system=system_prompt,
            prompt=repair_prompt,
            repair_prompt=repair_prompt,
            audit_prompt=audit_prompt,
            json_prefix="{",
            json_schema=json_schema,
            consume_state=consume_state,
            output_adapter=output_adapter,
            final_tokens=final_tokens,
            raw_semantic_validator=raw_semantic_validator,
            semantic_validator=semantic_validator,
            semantic_failure_code=semantic_failure_code,
            fallback_factory=fallback_factory,
            fallback_on_parse_failure=fallback_on_parse_failure,
        )
        return ParsedRoleCall(
            value=decoded.value,
            record=decoded.record.model_copy(
                update={
                    "image_labels": record_labels,
                    "physical_calls": prefix_calls + decoded.record.physical_calls,
                    "elapsed_seconds": append_elapsed + decoded.record.elapsed_seconds,
                }
            ),
        )

    def release(self) -> None:
        """Release the per-case KV once the terminal answer is persisted."""
        self._backend.release(self._state)

    def generate_final_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        final_tokens: int,
        terminal_prefix: str | None = "<answer>",
    ) -> RoleCall:
        """Generate the final answer directly from the accumulated latent cache."""
        started = time.perf_counter()
        # A tag protocol teacher-forces "<answer>". The boxed protocol keeps this
        # unset so the model can emit the 0731 native-hybrid free-text readout.
        # Keep the accumulated inter-agent state as an immutable branch point.
        # ``transformers.generate`` may extend a mutable DynamicCache in place;
        # decoding on a clone prevents a failed long Judger readout from poisoning
        # a format-repair attempt that must reconsider the same latent evidence.
        raw = self._backend.decode_in_place(
            deepcopy(self._state),
            system=system_prompt,
            prompt=user_prompt,
            max_new_tokens=final_tokens,
            json_prefix=terminal_prefix,
        )
        return RoleCall(
            role="answerer",
            prompt=user_prompt,
            image_labels=(),
            reasoning="",
            final_outputs=(raw,),
            format_repairs=0,
            physical_calls=1,
            elapsed_seconds=time.perf_counter() - started,
        )


def fixed_patch_plan(
    question: str, *, patch_budget: Literal[8, 25]
) -> EvidencePlan:
    """Supply deterministic multi-scale tool metadata, not planner prose."""
    if patch_budget == 8:
        overview_count, detail_count, anchor_count = 3, 5, 2
    else:
        overview_count, detail_count, anchor_count = 5, 20, 5
    return EvidencePlan.model_validate(
        {
            "question_focus": question,
            "needed_visual_information": ("carried in latent planner state",),
            "thumbnail_observations": (),
            "search_instruction": "use the preceding latent planner state",
            "success_criteria": (
                f"obtain {overview_count} x5 and {detail_count} x20 tissue patches",
            ),
            "detail_trigger": "fixed OnePass multi-scale allocation",
            "scale_plan": {
                "overview_target": "latent-planned x5 architecture",
                "overview_patch_count": overview_count,
                "detail_target": "latent-planned x20 cellular detail",
                "detail_patch_count": detail_count,
                "detail_anchor_count": anchor_count,
                "scale_success_criteria": (
                    f"all {patch_budget} patches materialized",
                ),
                "patch_budget": patch_budget,
            },
        }
    )


def fixed_eight_patch_plan(question: str) -> EvidencePlan:
    """Supply the original three-x5 plus five-x20 allocation."""
    return fixed_patch_plan(question, patch_budget=8)
