"""Atomic, append-preserving per-case artifact storage."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from vision_text_mas.contracts import (
    CaseInput,
    EvidenceRound,
    ReasonerReport,
    RoleCall,
    RunResult,
    VerifierDecision,
)
from vision_text_mas.evidence import EvidenceMemory, render_evidence_board
from vision_text_mas.errors import FailureCode, PipelineFailure

if TYPE_CHECKING:
    from vision_text_mas.io_pipeline import AsyncJsonWriter
from vision_text_mas.navigation.navigation_contracts import EvidencePlan, SelectionTrace


ArtifactT = TypeVar("ArtifactT")


class FailureArtifact(BaseModel):
    """Serializable technical failure that is never converted to an answer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: FailureCode
    stage: str = Field(min_length=1)
    detail: str = Field(min_length=1)
    dataset_index: int = Field(ge=0)
    slide_id: str = Field(min_length=1)


class ArtifactStore:
    """Write validated JSON and evidence boards without deleting partial work."""

    def __init__(self, root: Path, *, json_writer: AsyncJsonWriter | None = None) -> None:
        self._root = root
        self._json_writer = json_writer
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def _write_typed(
        self,
        relative_path: Path,
        value: ArtifactT,
        adapter: TypeAdapter[ArtifactT],
    ) -> Path:
        path = self._root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            payload = adapter.dump_json(value, indent=2)
            if self._json_writer is None:
                temporary.write_bytes(payload)
                temporary.replace(path)
            else:
                self._json_writer.submit(path, payload)
        except OSError as error:
            raise PipelineFailure(
                code=FailureCode.IMAGE_IO,
                stage="artifacts",
                detail=f"failed to write {path}: {error}",
            ) from error
        return path

    def write_navigator_calls(
        self,
        round_number: int,
        calls: tuple[RoleCall, ...],
    ) -> Path:
        return self._write_typed(
            Path(f"round_{round_number}/navigator_calls.json"),
            calls,
            TypeAdapter(tuple[RoleCall, ...]),
        )

    def write_plan(self, round_number: int, plan: EvidencePlan, call: RoleCall) -> None:
        self._write_typed(
            Path(f"round_{round_number}/evidence_plan.json"),
            plan,
            TypeAdapter(EvidencePlan),
        )
        self._write_typed(
            Path(f"round_{round_number}/evidence_planner_call.json"),
            call,
            TypeAdapter(RoleCall),
        )

    def write_selection_traces(
        self, round_number: int, traces: tuple[SelectionTrace, ...]
    ) -> Path:
        return self._write_typed(
            Path(f"round_{round_number}/selection_traces.json"),
            traces,
            TypeAdapter(tuple[SelectionTrace, ...]),
        )

    def write_round(self, evidence_round: EvidenceRound) -> Path:
        return self._write_typed(
            Path(f"round_{evidence_round.round_number}/evidence.json"),
            evidence_round,
            TypeAdapter(EvidenceRound),
        )

    def write_reasoner(
        self,
        round_number: int,
        report: ReasonerReport,
        call: RoleCall,
    ) -> tuple[Path, Path]:
        report_path = self._write_typed(
            Path(f"round_{round_number}/reasoner_report.json"),
            report,
            TypeAdapter(ReasonerReport),
        )
        call_path = self._write_typed(
            Path(f"round_{round_number}/reasoner_call.json"),
            call,
            TypeAdapter(RoleCall),
        )
        return report_path, call_path

    def write_verifier(
        self,
        round_number: int,
        decision: VerifierDecision,
        call: RoleCall,
    ) -> tuple[Path, Path]:
        decision_path = self._write_typed(
            Path(f"round_{round_number}/verifier_decision.json"),
            decision,
            TypeAdapter(VerifierDecision),
        )
        call_path = self._write_typed(
            Path(f"round_{round_number}/verifier_call.json"),
            call,
            TypeAdapter(RoleCall),
        )
        return decision_path, call_path

    def write_answerer(self, call: RoleCall) -> Path:
        return self._write_typed(Path("answerer_call.json"), call, TypeAdapter(RoleCall))

    def write_result(self, result: RunResult) -> Path:
        path = self._write_typed(Path("result.json"), result, TypeAdapter(RunResult))
        self._archive_recovered_failure()
        return path

    def _archive_recovered_failure(self) -> None:
        """Retain a retried failure without marking the completed case as failed."""
        active = self._root / "failure.json"
        if not active.exists():
            return
        recovered = self._root / "failure.recovered.json"
        index = 1
        while recovered.exists():
            recovered = self._root / f"failure.recovered.{index}.json"
            index += 1
        try:
            active.replace(recovered)
        except OSError as error:
            raise PipelineFailure(
                code=FailureCode.IMAGE_IO,
                stage="artifacts",
                detail=f"failed to archive recovered failure {active}: {error}",
            ) from error

    def write_failure(self, failure: PipelineFailure, case: CaseInput) -> Path:
        artifact = FailureArtifact(
            code=failure.code,
            stage=failure.stage,
            detail=failure.detail,
            dataset_index=case.dataset_index,
            slide_id=case.item.slide_id,
        )
        return self._write_typed(
            Path("failure.json"),
            artifact,
            TypeAdapter(FailureArtifact),
        )

    def render_board(self, memory: EvidenceMemory) -> Path:
        return render_evidence_board(
            memory,
            self._root / f"evidence_board_round_{len(memory.rounds)}.png",
        )
