"""Pruning-free interface for advancing and reading latent Qwen state."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from PIL import Image

from vision_text_mas.latent_state import LatentAudit, LatentCaseState


CacheT = TypeVar("CacheT")


@dataclass(frozen=True, slots=True)
class LatentAppendResult(Generic[CacheT]):
    """Raw state returned by one model prompt plus latent continuation."""

    cache: CacheT
    cache_length: int
    position_cursor: int
    added_tokens: int


class LatentEngine(Protocol[CacheT]):
    """Physical model operations required by Latent VL-MAS."""

    def append(
        self,
        *,
        cache: CacheT | None,
        cache_length: int,
        position_cursor: int,
        images: tuple[Image.Image, ...],
        system_prompt: str,
        user_prompt: str,
        latent_steps: int,
        stage: str = "",
    ) -> LatentAppendResult[CacheT]: ...

    def decode(
        self,
        *,
        cache: CacheT | None,
        position_cursor: int,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
        json_prefix: str | None,
        json_schema: str | None = None,
    ) -> str: ...

    def release(self, cache: CacheT | None) -> None: ...

class LatentBackendAdapter(Generic[CacheT]):
    """Keep state bookkeeping separate from the physical Qwen implementation."""

    def __init__(self, engine: LatentEngine[CacheT]) -> None:
        self._engine = engine

    @staticmethod
    def snapshot(state: LatentCaseState[CacheT]) -> LatentCaseState[CacheT]:
        """Deep-copy mutable KV tensors for a stable branch point."""
        return LatentCaseState(
            slide_id=state.slide_id,
            dataset_index=state.dataset_index,
            kv=deepcopy(state.kv),
            cache_length=state.cache_length,
            position_cursor=state.position_cursor,
            audits=state.audits,
        )

    def append_role_latent(
        self,
        state: LatentCaseState[CacheT],
        *,
        stage: str,
        system: str,
        prompt: str,
        latent_steps: int,
    ) -> LatentCaseState[CacheT]:
        """Append a text role prompt and its latent continuation."""
        return self._append(
            state,
            stage=stage,
            images=(),
            system=system,
            prompt=prompt,
            latent_steps=latent_steps,
        )

    def append_patch_latent(
        self,
        state: LatentCaseState[CacheT],
        *,
        stage: str,
        image: Image.Image,
        label: str,
        latent_steps: int,
    ) -> LatentCaseState[CacheT]:
        """Append one accepted patch and its latent continuation."""
        return self._append(
            state,
            stage=stage,
            images=(image,),
            system="",
            prompt=label,
            latent_steps=latent_steps,
        )

    def append_multimodal_latent(
        self,
        state: LatentCaseState[CacheT],
        *,
        stage: str,
        images: tuple[Image.Image, ...],
        system: str,
        prompt: str,
        latent_steps: int,
    ) -> LatentCaseState[CacheT]:
        """Append one role prompt with all images supplied to that role."""
        return self._append(
            state,
            stage=stage,
            images=images,
            system=system,
            prompt=prompt,
            latent_steps=latent_steps,
        )

    def branch_multimodal_latent(
        self,
        state: LatentCaseState[CacheT],
        *,
        stage: str,
        images: tuple[Image.Image, ...],
        system: str,
        prompt: str,
        latent_steps: int,
    ) -> LatentCaseState[CacheT]:
        """Advance an isolated candidate-selection branch from the main state."""
        branch = self.snapshot(state)
        return self.append_multimodal_latent(
            branch,
            stage=stage,
            images=images,
            system=system,
            prompt=prompt,
            latent_steps=latent_steps,
        )

    def _append(
        self,
        state: LatentCaseState[CacheT],
        *,
        stage: str,
        images: tuple[Image.Image, ...],
        system: str,
        prompt: str,
        latent_steps: int,
    ) -> LatentCaseState[CacheT]:
        result = self._engine.append(
            cache=state.kv,
            cache_length=state.cache_length,
            position_cursor=state.position_cursor,
            images=images,
            system_prompt=system,
            user_prompt=prompt,
            latent_steps=latent_steps,
            stage=stage,
        )
        return state.advance(
            kv=result.cache,
            cache_length=result.cache_length,
            position_cursor=result.position_cursor,
            audit=LatentAudit(
                stage=stage,
                added_tokens=result.added_tokens,
                latent_steps=latent_steps,
            ),
        )

    def decode_on_clone(
        self,
        state: LatentCaseState[CacheT],
        *,
        system: str,
        prompt: str,
        max_new_tokens: int,
        json_prefix: str | None,
        json_schema: str | None = None,
    ) -> str:
        """Decode control text from a deep copy of the main KV state."""
        return self._engine.decode(
            cache=deepcopy(state.kv),
            position_cursor=state.position_cursor,
            system_prompt=system,
            user_prompt=prompt,
            max_new_tokens=max_new_tokens,
            json_prefix=json_prefix,
            json_schema=json_schema,
        )

    def decode_in_place(
        self,
        state: LatentCaseState[CacheT],
        *,
        system: str,
        prompt: str,
        max_new_tokens: int,
        json_prefix: str | None,
        json_schema: str | None = None,
    ) -> str:
        """Consume the case KV for the terminal Answerer readout."""
        return self._engine.decode(
            cache=state.kv,
            position_cursor=state.position_cursor,
            system_prompt=system,
            user_prompt=prompt,
            max_new_tokens=max_new_tokens,
            json_prefix=json_prefix,
            json_schema=json_schema,
        )


    def release(self, state: LatentCaseState[CacheT]) -> None:
        """Release model-owned tensors for one completed or failed case."""
        self._engine.release(state.kv)
