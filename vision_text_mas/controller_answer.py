"""Final answer boundary for the deterministic controller."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from vision_text_mas.artifacts import ArtifactStore
from vision_text_mas.contracts import (
    CaseInput,
    DecisionBasis,
    ReasonerReport,
    RoleCall,
    RunResult,
    StateTransition,
)
from vision_text_mas.evidence import EvidenceMemory
from vision_text_mas.planned_evidence import AnswerEvidence
from vision_text_mas.role_protocols import AnswererRole


def finalize_answer(
    answerer: AnswererRole,
    artifacts: ArtifactStore,
    *,
    case: CaseInput,
    thumbnail: Image.Image,
    memory: EvidenceMemory,
    board: Path,
    reports: tuple[ReasonerReport, ...],
    decision: AnswerEvidence,
    basis: DecisionBasis,
    calls: tuple[RoleCall, ...],
    transitions: tuple[StateTransition, ...],
) -> RunResult:
    """Invoke Answerer once and persist the terminal typed result."""
    call = answerer.answer(
        question=case.item.question,
        choices=case.item.choices,
        thumbnail=thumbnail,
        memory=memory,
        evidence_board=board,
        reasoner_reports=reports,
        verifier_decision=decision,
        decision_basis=basis,
    )
    artifacts.write_answerer(call.record)
    result = RunResult(
        dataset_index=case.dataset_index,
        slide_id=case.item.slide_id,
        gold_answer=case.item.answer,
        answer=call.value,
        decision_basis=basis,
        round_count=len(memory.rounds),
        patches=memory.patches,
        transitions=transitions,
        role_calls=(*calls, call.record),
    )
    artifacts.write_result(result)
    # per-case efficiency sidecar (VLMAS_EFF_DUMP=1): accumulated FLOPs/TTFT
    # since the previous dump + peak GPU memory; counters reset so each case's
    # file is its own delta. Never blocks a case on instrumentation errors.
    import os as _os
    if _os.environ.get("VLMAS_EFF_DUMP", "") == "1":
        try:
            import torch as _torch

            from backbone import efficiency as _eff_mod
            backbone = getattr(_eff_mod, "ACTIVE", None)
            if backbone is not None:
                payload = dict(backbone.eff_summary())
                if _torch.cuda.is_available():
                    payload["peak_gpu_mem_bytes"] = int(
                        _torch.cuda.max_memory_allocated())
                    _torch.cuda.reset_peak_memory_stats()
                if payload:
                    artifacts.write_efficiency(payload)
                backbone.reset_eff()
        except Exception as error:  # noqa: BLE001
            print(f"[Efficiency] dump skipped: {error!r}", flush=True)
    return result
