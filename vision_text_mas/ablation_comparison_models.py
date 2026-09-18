"""Typed evidence and output models for the corrected WSI-MAS comparison."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import ClassVar, override

from pydantic import BaseModel, ConfigDict, field_validator


class MethodName(StrEnum):
    """The fixed four-way ablation methods."""

    BASE = "base"
    PRUNING = "pruning"
    REALLOCATION = "reallocation"
    BOTH = "both"


class EvaluationSummary(BaseModel):
    """Metrics emitted by the shared WSI-VQA evaluator."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    num_rows: int
    total_mcq: int
    total_open: int
    total_accuracy: float
    mcq_accuracy: float
    open_substring_accuracy: float
    open_token_f1: float
    open_BLEU_1: float
    open_BLEU_2: float
    open_BLEU_3: float
    open_BLEU_4: float
    open_METEOR: float
    open_ROUGE_L: float
    avg_inference_time_sec: float
    input_files: tuple[Path, ...]


class EvaluationContract(BaseModel):
    """Evaluator identity and completeness contract."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    contract_id: str
    evaluator: str
    metric_scope: str
    required_unique_results: int
    result_roots: tuple[Path, ...]

    def fixed_signature(self) -> tuple[str | int, ...]:
        """Return evaluator fields that must match while excluding method paths."""
        return (
            self.contract_id,
            self.evaluator,
            self.metric_scope,
            self.required_unique_results,
        )


class ProtocolManifest(BaseModel):
    """Normalized fields that must remain fixed across interventions."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    backend: str
    pipeline_mode: str
    agent_count: int
    max_rounds: int
    patch_budget: int
    navigator_tool_decodes: int
    terminal_answer_decodes: int
    terminal_answer_marker: bool
    intermediate_transport: str
    thinking_roles: tuple[str, ...]
    latent_steps: int
    transport_mode: str
    realign_method: str
    deterministic_cuda: bool
    answerer_do_sample: bool
    answerer_thinking: bool
    answerer_max_new_tokens: int
    terminal_no_repeat_ngram_size: int = 0
    terminal_repetition_penalty: float = 1.0
    terminal_antiloop_method: str = "none"
    answerer_rationale: bool = False
    answerer_protocol: str | None = None
    terminal_protocol: str | None = None
    canonical_open_options: bool
    save_navigation_pngs: bool = False
    max_model_len: int
    temperature: float
    top_p: float
    seed: int
    variant: str | None = None
    pruning: bool = False
    reallocation: bool = False

    @field_validator("answerer_rationale", "save_navigation_pngs", mode="before")
    @classmethod
    def _normalize_legacy_optional_flag(cls, value: bool | None) -> bool:
        """Treat historical null flags as the disabled state they represented."""
        return False if value is None else value

    def fixed_signature(
        self,
    ) -> tuple[str | int | float | bool | tuple[str, ...], ...]:
        """Return only protocol fields that interventions cannot change."""
        terminal = self.answerer_protocol or self.terminal_protocol
        if terminal is None:
            raise ComparisonError("missing answerer/terminal protocol")
        return (
            self.backend,
            self.pipeline_mode,
            self.agent_count,
            self.max_rounds,
            self.patch_budget,
            self.navigator_tool_decodes,
            self.terminal_answer_decodes,
            self.terminal_answer_marker,
            self.intermediate_transport,
            self.thinking_roles,
            self.latent_steps,
            self.transport_mode,
            self.realign_method,
            self.deterministic_cuda,
            self.answerer_do_sample,
            self.answerer_thinking,
            self.answerer_max_new_tokens,
            self.terminal_no_repeat_ngram_size,
            self.terminal_repetition_penalty,
            self.terminal_antiloop_method,
            self.answerer_rationale,
            terminal,
            self.canonical_open_options,
            self.save_navigation_pngs,
            self.max_model_len,
            self.temperature,
            self.top_p,
            self.seed,
        )


class EvaluationRow(BaseModel):
    """The evaluator row fields used to prove IDs and successful latency."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    question_id: str
    inference_time_sec: float


class MetricDistribution(BaseModel):
    """Success-only distribution emitted by the Direct-HF runner."""
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    mean: float
    median: float
    min: float
    max: float


class EfficiencySummary(BaseModel):
    """Per-method efficiency measurements with failed attempts separated."""
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    successful_cases: int
    failed_attempts_excluded: int
    case_wall_seconds: MetricDistribution
    peak_allocated_bytes: MetricDistribution
    peak_reserved_bytes: MetricDistribution
    role_call_seconds: MetricDistribution
    physical_calls: MetricDistribution
    format_repairs: MetricDistribution


class MethodMeasurement(BaseModel):
    """One method's final metrics and deltas from the base."""
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    method: MethodName
    num_rows: int
    total_accuracy: float
    delta_total_accuracy_pp: float
    mcq_accuracy: float
    open_substring_accuracy: float
    open_token_f1: float
    open_BLEU_1: float
    open_BLEU_2: float
    open_BLEU_3: float
    open_BLEU_4: float
    open_METEOR: float
    open_ROUGE_L: float
    mean_success_latency_sec: float
    median_success_latency_sec: float
    mean_role_call_latency_sec: float
    median_role_call_latency_sec: float
    mean_peak_allocated_gib: float
    mean_peak_reserved_gib: float
    mean_physical_calls: float
    mean_format_repairs: float
    failed_attempts_excluded: int
    speedup_vs_base: float
    accuracy_improved: bool
    latency_improved: bool


class ComparisonReport(BaseModel):
    """Auditable final comparison across the exact same evaluated IDs."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    contract_id: str
    expected_cases: int
    same_ids_verified: bool
    question_ids_sha256: str
    methods: tuple[MethodMeasurement, ...]


@dataclass(frozen=True, slots=True)
class MethodSource:
    """Filesystem sources for one method."""

    method: MethodName
    summary: Path
    evaluation_manifest: Path
    protocol: Path
    efficiency_summary: Path


@dataclass(frozen=True, slots=True)
class LoadedMethod:
    """Parsed and validated evidence for one method."""

    source: MethodSource
    summary: EvaluationSummary
    contract: EvaluationContract
    protocol: ProtocolManifest
    efficiency: EfficiencySummary
    question_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ComparisonError(Exception):
    """A comparison cannot be claimed from inconsistent evidence."""

    detail: str

    @override
    def __str__(self) -> str:
        return self.detail
