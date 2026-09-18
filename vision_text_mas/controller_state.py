"""Pure legality helpers for controller state transitions."""

from __future__ import annotations

from typing_extensions import assert_never

from vision_text_mas.contracts import (
    InsufficientDecision,
    NewRegionDecision,
    PassDecision,
    StateTransition,
    VerifierDecision,
    ZoomDetailDecision,
)
from vision_text_mas.errors import FailureCode, PipelineFailure


def transition(round_number: int, decision: VerifierDecision) -> StateTransition:
    """Convert one validated Verifier decision into an audit transition."""
    match decision:
        case PassDecision() | NewRegionDecision() | ZoomDetailDecision():
            reason = decision.reason
        case InsufficientDecision():
            reason = decision.uncertainty
        case _ as unreachable:
            assert_never(unreachable)
    return StateTransition(round_number=round_number, action=decision.action, reason=reason)


def illegal(detail: str) -> None:
    """Raise the controller's typed transition failure."""
    raise PipelineFailure(
        code=FailureCode.SELECTION_CONTRACT,
        stage="controller",
        detail=detail,
    )


def require_navigation_round(round_number: int, max_rounds: int) -> None:
    """Reject navigation after the configured final evidence round."""
    if round_number >= max_rounds:
        illegal("navigation is forbidden after the final evidence round")
