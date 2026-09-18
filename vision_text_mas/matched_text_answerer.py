"""0808 Answerer contract with explicit text rather than latent transport."""

from __future__ import annotations

from pathlib import Path
from typing import Final, final

from PIL import Image
from pydantic import TypeAdapter

from vision_text_mas.contracts import (
    AnswerOutput,
    DecisionBasis,
    ReasonerReport,
    RoleCall,
)
from vision_text_mas.evidence import EvidenceMemory
from vision_text_mas.latent_agent_prompts import PROMPT2_HANDOFF
from vision_text_mas.latent_answerer import (
    AnswererProtocol,
    LatentAnswererAgent,
)
from vision_text_mas.planned_evidence import AnswerEvidence
from vision_text_mas.qwen_client import ParsedRoleCall, QwenJsonClient


_LATENT_HANDOFF: Final = (
    "Use only the pathology evidence available in this turn; it may contain "
    "irrelevant detail, so use only what helps answer. "
    "Do not request more evidence."
)
def text_transport_prompt(prompt: str, *, evidence_json: str) -> str:
    """Replace only the transport sentence while preserving the answer contract."""
    for handoff in (_LATENT_HANDOFF, PROMPT2_HANDOFF):
        if handoff in prompt:
            replacement = f"{handoff}\nTransferred evidence: {evidence_json}"
            return prompt.replace(handoff, replacement, 1)
    return prompt


@final
class MatchedTextAnswererAgent:
    """Delegate 0808 parsing while injecting the Reasoner's visible text report."""

    def __init__(
        self,
        client: QwenJsonClient,
        *,
        final_tokens: int,
        protocol: AnswererProtocol,
        canonical_open_options: bool,
    ) -> None:
        self._client = client
        self._final_tokens = final_tokens
        self._protocol = protocol
        self._canonical_open_options = canonical_open_options

    def answer(
        self,
        *,
        question: str,
        choices: tuple[str, ...],
        thumbnail: Image.Image,
        memory: EvidenceMemory,
        evidence_board: Path,
        reasoner_reports: tuple[ReasonerReport, ...],
        verifier_decision: AnswerEvidence,
        decision_basis: DecisionBasis,
    ) -> ParsedRoleCall[AnswerOutput]:
        """Answer using the same terminal contract with text-transferred evidence."""
        evidence_json = TypeAdapter(tuple[ReasonerReport, ...]).dump_json(
            reasoner_reports
        ).decode()
        transport = _TextTransportClient(self._client, evidence_json=evidence_json)
        delegate = LatentAnswererAgent(
            transport,
            final_tokens=self._final_tokens,
            protocol=self._protocol,
            canonical_open_options=self._canonical_open_options,
        )
        return delegate.answer(
            question=question,
            choices=choices,
            thumbnail=thumbnail,
            memory=memory,
            evidence_board=evidence_board,
            reasoner_reports=reasoner_reports,
            verifier_decision=verifier_decision,
            decision_basis=decision_basis,
        )


@final
class _TextTransportClient:
    """Translate only the latent handoff sentence before stateless HF decoding."""

    def __init__(self, client: QwenJsonClient, *, evidence_json: str) -> None:
        self._client = client
        self._evidence_json = evidence_json

    def generate_final_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        final_tokens: int,
        terminal_prefix: str | None,
    ) -> RoleCall:
        """Decode one final response with the visible Reasoner report inserted."""
        return self._client.generate_final_text(
            system_prompt=system_prompt,
            user_prompt=text_transport_prompt(
                user_prompt,
                evidence_json=self._evidence_json,
            ),
            final_tokens=final_tokens,
            # Empty prefix closes Qwen's default <think> block without forcing an
            # answer token. This matches the 0808 --no-answerer-thinking setting.
            terminal_prefix="" if terminal_prefix is None else terminal_prefix,
        )
