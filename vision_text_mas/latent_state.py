"""Immutable per-case state for latent agent communication."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Generic, TypeVar


CacheT = TypeVar("CacheT")


@dataclass(frozen=True, slots=True)
class LatentAudit:
    """One state-advancing latent or multimodal append."""

    stage: str
    added_tokens: int
    latent_steps: int


@dataclass(frozen=True, slots=True)
class LatentStateRegressionError(RuntimeError):
    """A KV update attempted to move its cache or position backwards."""

    field: str
    previous: int
    proposed: int

    def __str__(self) -> str:
        return (
            f"latent {self.field} cannot decrease: "
            f"{self.previous} -> {self.proposed}"
        )


@dataclass(frozen=True, slots=True)
class LatentCaseState(Generic[CacheT]):
    """Opaque KV tensors plus typed monotonic metadata for one WSI case."""

    slide_id: str
    dataset_index: int
    kv: CacheT | None = None
    cache_length: int = 0
    position_cursor: int = 0
    audits: tuple[LatentAudit, ...] = ()

    @classmethod
    def empty(cls, slide_id: str, dataset_index: int) -> LatentCaseState[CacheT]:
        """Create an isolated state with no model cache."""
        return cls(slide_id=slide_id, dataset_index=dataset_index)

    def advance(
        self,
        *,
        kv: CacheT,
        cache_length: int,
        position_cursor: int,
        audit: LatentAudit,
    ) -> LatentCaseState[CacheT]:
        """Return the next state after enforcing monotonic cache coordinates."""
        if cache_length < self.cache_length:
            raise LatentStateRegressionError(
                field="cache length",
                previous=self.cache_length,
                proposed=cache_length,
            )
        if position_cursor < self.position_cursor:
            raise LatentStateRegressionError(
                field="position cursor",
                previous=self.position_cursor,
                proposed=position_cursor,
            )
        return replace(
            self,
            kv=kv,
            cache_length=cache_length,
            position_cursor=position_cursor,
            audits=(*self.audits, audit),
        )
