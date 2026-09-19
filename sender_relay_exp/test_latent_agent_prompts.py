"""Structural contracts for centralized latent-agent prompts."""

from vision_text_mas.latent_agent_prompts import (
    AnswererProtocol,
    latent_answerer_prompt,
    prompt2_enabled,
)
from vision_text_mas.matched_text_answerer import text_transport_prompt
from vision_text_mas.latent_qwen_engine import _reasoner_attention_targets


def test_structured_json_protocol_routes_to_json_envelope() -> None:
    # Given
    protocol = AnswererProtocol.STRUCTURED_JSON

    # When
    prompt = latent_answerer_prompt(
        question="Which tissue is shown?",
        choices=("Colon", "Liver"),
        include_rationale=True,
        protocol=protocol,
    )

    # Then
    assert '{"answer":' in prompt


def test_tag_protocol_routes_to_answer_tag() -> None:
    # Given
    protocol = AnswererProtocol.TAG

    # When
    prompt = latent_answerer_prompt(
        question="Which tissue is shown?",
        choices=("Colon", "Liver"),
        include_rationale=False,
        protocol=protocol,
    )

    # Then
    assert "<answer>" in prompt


def test_prompt2_text_transport_injects_transferred_evidence() -> None:
    # Given
    prompt = latent_answerer_prompt(
        question="Which tissue is shown?",
        choices=("Colon", "Liver"),
        protocol=AnswererProtocol.STRUCTURED_JSON,
    )

    # When
    transferred = text_transport_prompt(prompt, evidence_json='[{"patches":[]}]')

    # Then
    assert 'Transferred evidence: [{"patches":[]}]' in transferred


def test_prompt2_reasoner_exposes_pruning_attention_targets() -> None:
    # Given
    prompt = (
        "Question stem: Which tissue is shown?\n"
        "Assess every labeled patch for question-relevant visible morphology, "
        "including architecture, cellularity, cytology, stroma, necrosis, and "
        "tissue boundary."
    )

    # When
    questions, backgrounds = _reasoner_attention_targets(prompt)

    # Then
    assert questions == ["Question stem: Which tissue is shown?"]
    assert len(backgrounds) == 1


def test_legacy_prompt_set_remains_default(monkeypatch) -> None:
    # Given
    monkeypatch.delenv("VLMAS_PROMPT_SET", raising=False)

    # When / Then
    assert not prompt2_enabled()


def test_prompt2_activates_when_explicitly_selected(monkeypatch) -> None:
    # Given
    monkeypatch.setenv("VLMAS_PROMPT_SET", "prompt2")

    # When / Then
    assert prompt2_enabled()

