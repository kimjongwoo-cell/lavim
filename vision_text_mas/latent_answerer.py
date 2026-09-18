"""JSON-envelope Judger readout from the accumulated Latent VLMAS state."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Final, Protocol, final

from PIL import Image
from typing_extensions import assert_never

from vision_text_mas.contracts import (
    AnswerOutput,
    DecisionBasis,
    InsufficientDecision,
    NewRegionDecision,
    PassDecision,
    PatchId,
    ReasonerReport,
    RoleCall,
    ZoomDetailDecision,
)
from vision_text_mas.evidence import EvidenceMemory
from vision_text_mas.errors import FailureCode, PipelineFailure
from vision_text_mas.latent_agent_prompts import (
    LATENT_JUDGER_SYSTEM as PROMPT2_LATENT_JUDGER_SYSTEM,
    AnswererProtocol,
    latent_answerer_prompt as prompt2_latent_answerer_prompt,
    prompt2_enabled,
)
from vision_text_mas.planned_evidence import AnswerEvidence, PlannedEvidence
from vision_text_mas.prompts import canonical_open_answer_options
from vision_text_mas.answerer_schema import latent_answerer_json_schema
from vision_text_mas.qwen_client import ParsedRoleCall


LATENT_JUDGER_SYSTEM = (
    "You are Judger. Use the pathology evidence available in this turn and return one "
    "grounded final answer.\n"
    "Rules:\n"
    "1. For biomarker questions (HER2, PR, ER, Ki-67 etc.): output the observed "
    "status or score only (e.g. 'Positive', 'Negative', '2+', 'High'). Do not "
    "define the term.\n"
    "2. When the question asks about survival time or other quantitative results, "
    "output ONLY one numerical value in the answer field. Never output N/A, No, "
    "Not applicable, Unknown, or a refusal for a quantitative question.\n"
    "3. For vital status questions: output 'Alive' or 'Dead' only.\n"
    "4. For histological type questions: commit to a specific diagnosis.\n"
    "5. Keep an open-ended answer to one short phrase or one number.\n"
    "6. If choices are provided, your answer must be exactly one of the given "
    "options. Never generate an answer outside the provided choices."
)


def active_judger_system() -> str:
    """Select prompt2 only when explicitly requested by the environment."""
    return PROMPT2_LATENT_JUDGER_SYSTEM if prompt2_enabled() else LATENT_JUDGER_SYSTEM


# Room to reason step by step THEN emit the `{"answer": ...}` object. The old 32
# left no room to commit after the Thinking model's reasoning, so it never
# reached the JSON.
FINAL_ANSWER_TOKEN_BUDGET = 512
MAX_BOXED_ANSWER_CHARACTERS: Final = 256
MAX_BOXED_ANSWER_WORDS: Final = 32
MAX_TERMINAL_REPAIRS: Final = int(os.getenv("VLMAS_MAX_TERMINAL_REPAIRS", "3"))
# 0915: the 0806 "empty answer -> re-decode with a forced single digit" repair is
# REMOVED by default (it fabricated choice 1/2 and could never pick choice >= 10).
# An empty terminal answer is stored as this sentinel and scores as wrong.
# VLMAS_TERMINAL_DIGIT_REPAIR=1 restores the old behaviour for reproduction only.
NO_ANSWER_SENTINEL: Final = "<no-answer>"


def _digit_repair_enabled() -> bool:
    return os.environ.get("VLMAS_TERMINAL_DIGIT_REPAIR", "0").strip() == "1"


# generate_final_text teacher-forces this JSON opener for the flat JSON protocol so
# a greedy Thinking model commits to the `{"answer": ...}` object instead of prose.
# The decoded continuation omits it; the parser restores it before json.loads.
JSON_TERMINAL_PREFIX = '{"answer":'
STRUCTURED_JSON_TERMINAL_PREFIX = JSON_TERMINAL_PREFIX
BOXED_TERMINAL_PREFIX = r"\boxed{"


def _boxed_payloads(text: str) -> tuple[str, ...]:
    """Return every complete ``\\boxed{...}`` payload in decode order."""
    payloads: list[str] = []
    opener = r"\boxed{"
    for match in re.finditer(re.escape(opener), text):
        depth = 1
        for index in range(match.end(), len(text)):
            character = text[index]
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
            if depth == 0:
                payloads.append(text[match.end() : index].strip())
                break
    return tuple(payloads)


def _extract_boxed(text: str) -> str:
    """Return the only boxed payload, or the legacy final payload when ambiguous."""
    payloads = _boxed_payloads(text)
    return payloads[-1] if payloads else ""


def _extract_boxed_rationale(text: str) -> str:
    """Return the model reasoning preceding the final answer box."""
    box_index = text.rfind(r"\boxed{")
    if box_index < 0:
        return ""
    return text[:box_index].replace("</think>", "").strip()


def _extract_answer_line(text: str) -> str:
    """Return the last `answer: <...>` line payload (0717-style), or '' if absent."""
    matches = re.findall(r"(?im)^\s*answer\s*:\s*(.+?)\s*$", text)
    return matches[-1].strip() if matches else ""


def _extract_json_answer(text: str) -> str:
    """Pull `answer` out of the 0717-style `{"answer": "..."}` object.

    Mirrors agents/diagnosis.py::_parse_action in 0717: drop any leaked think
    block and code fences, parse the first JSON object, then fall back to the
    raw `"answer": "..."` value when the object is truncated or trailed by junk.
    Greedy decode reaches a self-terminating `}`, so this is what tames the
    Thinking model without grammar or sampling.
    """
    tail = re.split(r"</think>", text, maxsplit=1)[-1]
    tail = re.sub(r"```(?:json)?", "", tail)
    match = re.search(r"\{.*\}", tail, re.DOTALL)
    if match is not None:
        try:
            data = json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            data = None
        if isinstance(data, dict):
            answer = data.get("answer")
            if isinstance(answer, str) and answer.strip():
                return answer.strip()
    partial = re.search(r'"answer"\s*:\s*"([^"]*)"', tail)
    if partial is not None and partial.group(1).strip():
        return partial.group(1).strip()
    return ""


def _extract_json_object(text: str) -> dict[str, object]:
    """Parse the terminal `{...}` object so rationale/confidence can be recovered.

    Drops leaked think blocks / code fences. `generate_terminal_json` teacher-forces
    a leading `{` as the prompt prefix, so the decoded continuation usually omits it —
    restore it before parsing (mirrors LatentJsonClient). Uses raw_decode so trailing
    junk after the object does not defeat the parse. Returns {} when unparseable, so
    the caller falls back to answer-only extraction.
    """
    stripped = re.sub(r"```(?:json)?", "", text).strip()
    # Do NOT split on </think> and keep only the tail: the Thinking model often
    # emits the COMPLETE envelope inside the think block and a truncated copy after
    # </think>, so the tail is the broken one. Restore the teacher-forced leading
    # "{" and take the FIRST complete object (raw_decode stops at its close), which
    # is the good one wherever it sits.
    if stripped and not stripped.startswith("{"):
        stripped = "{" + stripped
    decoder = json.JSONDecoder()
    for start, character in enumerate(stripped):
        if character != "{":
            continue
        try:
            data, _ = decoder.raw_decode(stripped[start:])
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, dict) and "answer" in data:
            return data
    return {}


def _extract_tagged(text: str) -> tuple[str, str]:
    """Pull (answer, rationale) from the `<answer>…</answer><rationale>…</rationale>`
    tag envelope.

    generate_terminal_json teacher-forces a leading "<answer>" prefix that the
    decoded continuation omits, so restore it. The Thinking model may re-emit the
    pair after a leaked </think>; re.search takes the FIRST complete match of each
    tag. Tolerates a truncated answer close by falling back to the text before the
    rationale tag.
    """
    # Whitespace/newline-tolerant delimiters: the Thinking model sometimes breaks the
    # close tag as "</ answer>" or "</\n\n\nanswer>", which a strict "</answer>" regex
    # misses -> the fallback then drags junk into the answer.
    open_a = r"<\s*answer\s*>"
    close_a = r"<\s*/\s*answer\s*>"
    open_r = r"<\s*rationale\s*>"
    close_r = r"<\s*/\s*rationale\s*>"
    body = text
    if re.search(open_a, body) is None and re.search(close_a, body) is not None:
        body = "<answer>" + body
    answer_match = re.search(open_a + r"(.*?)" + close_a, body, re.DOTALL)
    if answer_match is not None:
        answer = answer_match.group(1).strip()
    else:
        # Truncated / missing close: take the head before the rationale tag.
        head = re.split(open_r, body, maxsplit=1)[0]
        answer = re.sub(open_a + "|" + close_a, "", head).strip()
    rationale_match = re.search(open_r + r"(.*?)" + close_r, body, re.DOTALL)
    if rationale_match is not None:
        rationale = rationale_match.group(1).strip()
    else:
        rationale_open = re.search(open_r + r"(.*)", body, re.DOTALL)
        rationale = rationale_open.group(1).strip() if rationale_open is not None else ""
    return answer, rationale


def _normalize_tag_repair_output(continuation: str) -> str:
    """Keep a model-emitted answer envelope intact during boxed fallback."""
    stripped = continuation.strip()
    answer_start = stripped.find("<answer>")
    answer_end = stripped.find("</answer>", answer_start + len("<answer>"))
    if answer_start >= 0 and answer_end >= 0:
        return stripped[answer_start : answer_end + len("</answer>")]
    return stripped if stripped.startswith("<answer>") else "<answer>" + stripped


def _normalize_choice(answer: str, choices: tuple[str, ...]) -> str:
    """Map a decoded answer onto its exact CHOICE text when choices are given.

    Accepts an exact (case-insensitive) match, a bare letter (`A`, `B.`), or a
    unique substring hit, so a slightly verbose readout still scores. Returns the
    answer unchanged when there is no confident choice match.
    """
    if not choices or not answer:
        return answer
    stripped = answer.strip()
    for choice in choices:
        if stripped.lower() == choice.strip().lower():
            return choice
    letter = re.match(r"^\s*([A-Za-z])[).:]?\s*$", stripped)
    if letter is not None:
        index = ord(letter.group(1).upper()) - ord("A")
        if 0 <= index < len(choices):
            return choices[index]
    numbered_choice = re.match(r"^\s*(?:choice\s+)?(\d+)\s*[).:]?\s*$", stripped, re.I)
    if numbered_choice is not None:
        index = int(numbered_choice.group(1)) - 1
        if 0 <= index < len(choices):
            return choices[index]
    hits = [choice for choice in choices if choice.strip().lower() in stripped.lower()]
    if len(hits) == 1:
        return hits[0]
    return answer


class LatentAnswererClient(Protocol):
    """Schema-free terminal readout capability over a carried latent state."""

    def generate_final_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        final_tokens: int,
        terminal_prefix: str | None,
        json_schema: str | None = None,
    ) -> RoleCall: ...


def latent_answerer_prompt(
    *,
    question: str,
    choices: tuple[str, ...],
    decision_basis: str = "",
    cited_patch_ids: tuple[str, ...] = (),
    include_rationale: bool = False,
    protocol: AnswererProtocol = AnswererProtocol.TAG,
) -> str:
    """Use the legacy prompt unless the compact prompt2 set is enabled."""
    if prompt2_enabled():
        return prompt2_latent_answerer_prompt(
            question=question,
            choices=choices,
            decision_basis=decision_basis,
            cited_patch_ids=cited_patch_ids,
            include_rationale=include_rationale,
            protocol=protocol,
        )
    _choice_fmt = os.environ.get("VLMAS_CHOICE_FORMAT", "").strip()
    if choices and _choice_fmt == "letters":
        # 0915 A/B: 0717 / single-baseline rendering ("Choices:" + lettered lines).
        def _letter(i: int) -> str:
            return chr(65 + i) if i < 26 else chr(65 + i // 26 - 1) + chr(65 + i % 26)

        choice_lines = "Choices:\n" + "\n".join(
            f"{_letter(index)}. {choice}" for index, choice in enumerate(choices)
        )
    elif choices and _choice_fmt == "list":
        # 0915 A/B: hand the dataset's list over verbatim instead of 30 CHOICE lines.
        import json as _json

        choice_lines = "Choice: " + _json.dumps(list(choices), ensure_ascii=False)
    else:
        choice_lines = "\n".join(
            f"CHOICE {index}: {choice}"
            for index, choice in enumerate(choices, start=1)
        )
    if choices:
        choice_rule = (
            "For this multiple-choice question, put only the exact choice text in "
            "the answer field.\n"
            if _choice_fmt == "letters"
            else "When Choices are provided, the answer must be exactly one of the "
            "given choice texts.\n"
        )
        # 0916 VLMAS_ANSWERER_GRADE_RULE=1 (off = unchanged): when the choices ARE the grade
        # values 0..N (PANDA ISUP), the CHOICE line numbers are offset from the values by one.
        # State the value range so the answer is a grade, not a line index.
        if (os.environ.get("VLMAS_ANSWERER_GRADE_RULE", "").strip() == "1"
                and [str(c).strip() for c in choices] == [str(i) for i in range(len(choices))]):
            choice_rule = (
                "Choose exactly one grade value from "
                + ", ".join(str(i) for i in range(len(choices)))
                + ". Put that value itself in the answer field, not the CHOICE line number.\n"
            )
        answer_placeholder = "<exact choice text>"
    else:
        choice_rule = ""
        answer_placeholder = "short answer phrase"
    if protocol is AnswererProtocol.BOXED:
        tag_block = (
            "Reason step by step, then put the single selected CHOICE — its exact "
            "choice text or letter — inside \\boxed{...}.\n"
            "The final answer must appear exactly once inside \\boxed{...}."
            if choices
            else "Reason step by step, then put the concise final answer inside "
            "\\boxed{...}.\nThe final answer must appear exactly once inside "
            "\\boxed{...}."
        )
    elif protocol in (AnswererProtocol.JSON, AnswererProtocol.STRUCTURED_JSON):
        if protocol is AnswererProtocol.STRUCTURED_JSON:
            tag_block = (
                "Reason step by step internally, then output ONLY one complete JSON "
                "object and nothing else, using this exact schema:\n"
                f'{{"answer": "{answer_placeholder}", '
                '"rationale": "brief evidence-grounded explanation", "confidence": 0.0}'
            )
        elif include_rationale:
            tag_block = (
                "Reason step by step internally, then output ONLY a single-line JSON "
                "object and nothing else:\n"
                f'{{"answer": "{answer_placeholder}", "rationale": "text grounded in '
                'the cited patches"}'
            )
        else:
            tag_block = (
                "Reason step by step internally, then output ONLY a single-line JSON "
                "object and nothing else:\n"
                f'{{"answer": "{answer_placeholder}"}}'
            )
    elif include_rationale:
        tag_block = (
            "Reason step by step internally, then output ONLY the final answer in "
            "exactly these two tags and nothing else:\n"
            f"<answer>{answer_placeholder}</answer>\n"
            "<rationale>text grounded in the cited patches</rationale>"
        )
    else:
        tag_block = (
            "Reason step by step internally, then output ONLY the final answer in "
            "exactly this tag and nothing else:\n"
            f"<answer>{answer_placeholder}</answer>"
        )
    _ = (decision_basis, cited_patch_ids)
    return (
        f"Question: {question}\n{choice_lines}\n"
        "Use only the pathology evidence available in this turn; it may contain "
        "irrelevant detail, so use only what helps answer. "
        "Do not request more evidence.\n"
        f"{'' if protocol is AnswererProtocol.BOXED else choice_rule}"
        f"{tag_block}"
    )


@final
class LatentAnswererAgent:
    """Run one upstream-style Judger call and build the evaluator artifact."""

    def __init__(
        self,
        client: LatentAnswererClient,
        *,
        final_tokens: int = FINAL_ANSWER_TOKEN_BUDGET,
        include_rationale: bool = False,
        protocol: AnswererProtocol = AnswererProtocol.TAG,
        canonical_open_options: bool = False,
    ) -> None:
        self._client = client
        self._final_tokens = final_tokens
        self._include_rationale = include_rationale
        self._protocol = protocol
        # Opt-in (default off = byte-identical): for OPEN questions (no dataset
        # choices) supply the dataset's declared canonical labels as choices so the
        # prompt constrains + the answer normalizes onto them (e.g. histological_type
        # -> IDC/ILC/Medullary/Other instead of free-text "Adenocarcinoma").
        self._canonical_open_options = canonical_open_options

    def _refeed_evidence_crops(
        self, *, question: str, memory: EvidenceMemory
    ) -> None:
        """Opt-in: append the evidence crops as images before the terminal decode."""
        try:
            refeed_count = int(os.environ.get("VLMAS_ANSWERER_REFEED_K", "0"))
        except ValueError:
            refeed_count = 0
        if refeed_count <= 0:
            return
        append_only = getattr(self._client, "append_only", None)
        if append_only is None:
            print("[Refeed] client has no append_only; skipped", flush=True)
            return
        patches = [
            patch for round_ in memory.rounds for patch in round_.patches
        ][:refeed_count]
        if not patches:
            print("[Refeed] no evidence patches available; skipped", flush=True)
            return
        try:
            max_side = int(os.environ.get("VLMAS_ANSWERER_REFEED_MAXSIDE", "448"))
        except ValueError:
            max_side = 448
        # A failed append still grows the shared DynamicCache in place, so the
        # crop count must be fitted BEFORE the one and only append attempt:
        # estimate per-crop visual tokens from the resize target and shrink K
        # (and then the resize side) until the whole turn fits the context
        # budget with a safety margin.
        state = getattr(self._client, "state", None)
        cache_length = int(getattr(state, "cache_length", 0) or 0)
        context_limit = int(getattr(self._client, "_max_model_len", 8192) or 8192)
        prompt_budget = 260  # chat template + refeed instruction + question stem
        margin = 128
        available = context_limit - cache_length - prompt_budget - margin
        chosen_side = 0
        for side in (max_side, 392, 336, 280, 224):
            if side > max_side:
                continue
            # GPU-measured: the processor's smart resize inflates real visual
            # tokens to ~3.2x the naive (side/28)^2/4 count (8 crops at 448
            # cost 1849 cache tokens, ~200/crop). Estimate with that factor.
            per_crop = ((side // 28) * (side // 28) // 4) * 16 // 5 + 10
            fit = max(0, available // per_crop)
            if fit >= min(len(patches), 2) or fit >= len(patches):
                patches = patches[: min(len(patches), fit)]
                chosen_side = side
                break
        if chosen_side == 0 or not patches:
            print(
                f"[Refeed] skipped (cache {cache_length}/{context_limit} leaves "
                "no room for crops)",
                flush=True,
            )
            return
        max_side = chosen_side
        images: list[Image.Image] = []
        for patch in patches:
            with Image.open(patch.image_path) as source:
                image = source.convert("RGB")
            if max_side > 0 and max(image.size) > max_side:
                scale = max_side / max(image.size)
                image = image.resize(
                    (
                        max(1, round(image.width * scale)),
                        max(1, round(image.height * scale)),
                    )
                )
            image.info["latent_patch_id"] = str(patch.patch_id)
            image.info["latent_parent_patch_id"] = str(patch.source_x5_anchor)
            image.info["latent_magnification"] = int(patch.magnification.value)
            image.info["latent_box"] = (
                int(patch.box.x), int(patch.box.y),
                int(patch.box.width), int(patch.box.height),
            )
            images.append(image)
        prompt = (
            "These images are the exact evidence patches selected earlier for this "
            "slide (overview x5 patches first, then x20 detail patches). Re-examine "
            "them directly — they are the visual evidence for the question below. "
            f"Question stem: {question}"
        )
        # Exactly ONE append attempt: a failed append still grows the shared
        # cache in place, so never retry. The fit computation above should make
        # overflow unreachable; if the estimate is ever wrong, log and accept
        # the unmodified terminal decode.
        try:
            record = append_only(
                role="answerer_refeed",
                images=tuple(images),
                image_labels=tuple(str(patch.patch_id) for patch in patches),
                system_prompt=active_judger_system(),
                user_prompt=prompt,
            )
        except PipelineFailure as failure:
            print(f"[Refeed] OVER-BUDGET despite fit ({failure.detail})", flush=True)
            return
        new_length = int(
            getattr(getattr(self._client, "state", None), "cache_length", 0) or 0
        )
        print(
            f"[Refeed] appended {len(images)} crops (max_side={max_side}, "
            f"cache {cache_length}→{new_length}/{context_limit}) "
            f"in {record.elapsed_seconds:.1f}s",
            flush=True,
        )

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
        """Decode one Judger response and read the answer from its tag envelope."""
        _ = (thumbnail, evidence_board, reasoner_reports)
        # Crop re-feed experiment (opt-in via VLMAS_ANSWERER_REFEED_K, default
        # 0 = off = byte-identical): re-append the selected evidence crops as
        # REAL images in one closed user turn right before the terminal decode,
        # so the answer conditions on a fresh input-level visual channel instead
        # of only the carried (non-visual) latent trace. Uses the same
        # append_only machinery as the Reasoner; no engine/backbone changes.
        self._refeed_evidence_crops(question=question, memory=memory)
        cited_ids = self._cited_ids(verifier_decision)
        # Off by default: only substitute canonical labels for genuinely OPEN
        # questions (dataset gave no choices) when the opt-in flag is set. MCQ
        # questions (choices present) are untouched.
        effective_choices = choices
        if self._canonical_open_options and not choices:
            effective_choices = canonical_open_answer_options(question)
        # 0915 diagnostic arm: build the Answerer prompt WITHOUT the CHOICE list
        # (open-question form). Scoring/snapping still sees no choices, so the
        # arm is for first-token probing only.
        if os.environ.get("VLMAS_ANSWERER_DROP_CHOICES", "").strip() == "1":
            effective_choices = ()
        # C2 VER decoding reads the candidate list from the engine
        # (VLMAS_ANSWERER_VER=1); harmless attribute otherwise.
        _eng = getattr(getattr(self._client, "_backend", None), "_engine", None)
        if _eng is not None:
            _eng._ver_candidates = tuple(effective_choices)
        record = self._client.generate_final_text(
            system_prompt=active_judger_system(),
            user_prompt=latent_answerer_prompt(
                question=question,
                choices=effective_choices,
                include_rationale=self._include_rationale,
                protocol=self._protocol,
            ),
            final_tokens=self._final_tokens,
            terminal_prefix=self._terminal_prefix(),
            # 0915 VLMAS_TERMINAL_GRAMMAR=1: xgrammar-constrained readout (0902 grammar_json
            # port). answer is an enum of the exact choice texts, so garbage is impossible.
            json_schema=(
                latent_answerer_json_schema(answer_options=tuple(effective_choices or ()))
                if os.environ.get("VLMAS_TERMINAL_GRAMMAR", "").strip() == "1"
                else None
            ),
        )
        raw = record.final_outputs[0].strip()
        if self._protocol in (
            AnswererProtocol.JSON,
            AnswererProtocol.STRUCTURED_JSON,
        ):
            # generate_final_text teacher-forces JSON_TERMINAL_PREFIX, which the decoded
            # continuation omits. Restore it so the flat object parses, unless the
            # Thinking model already re-emitted a complete object after a leaked
            # </think> (then it starts with "{" and needs no repair).
            # VLMAS_TERMINAL_NO_FORCE=1 (0915): nothing was teacher-forced, so the raw
            # output is the model's own text; never prepend the opener.
            if not raw.lstrip().startswith("{") and os.environ.get(
                "VLMAS_TERMINAL_NO_FORCE", ""
            ).strip() != "1":
                prefix = (
                    STRUCTURED_JSON_TERMINAL_PREFIX
                    if self._protocol is AnswererProtocol.STRUCTURED_JSON
                    else JSON_TERMINAL_PREFIX
                )
                raw = prefix + raw
        # Primary: the <answer>/<rationale> tag envelope (teacher-forced prefix).
        # Fall back to legacy JSON object / `answer:` line / \boxed{} / post-</think>
        # tail so an off-format readout still yields an answer. Finally snap onto the
        # exact CHOICE text so a slightly verbose readout still scores.
        tail = re.split(r"</think>", raw, maxsplit=1)[-1].strip()
        tagged_answer, tagged_rationale = _extract_tagged(raw)
        parsed = _extract_json_object(raw)
        parsed_answer = parsed.get("answer")
        if self._protocol is AnswererProtocol.BOXED:
            boxed_payloads = _boxed_payloads(raw)
            answer_text = boxed_payloads[0] if len(boxed_payloads) == 1 else ""
            normalized_boxed_answer = _normalize_choice(answer_text, effective_choices)
            has_valid_choice = (
                not effective_choices or normalized_boxed_answer in effective_choices
            )
            if (
                not answer_text
                or len(answer_text) > MAX_BOXED_ANSWER_CHARACTERS
                or len(answer_text.split()) > MAX_BOXED_ANSWER_WORDS
                or not has_valid_choice
            ):
                original_raw = raw
                repair_outputs = [raw]
                repair_elapsed = record.elapsed_seconds
                repair_instructions = (
                    (
                        "The preceding reasoning did not finish its answer. Reconsider "
                        "the same question and evidence, then output only the exact "
                        "answer and the closing brace."
                    ),
                    (
                        "Return only one exact answer for the question above. If choices "
                        "are listed, copy exactly one choice. End immediately with the "
                        "closing brace."
                    ),
                )
                base_prompt = latent_answerer_prompt(
                    question=question,
                    choices=effective_choices,
                    decision_basis=decision_basis.value,
                    cited_patch_ids=cited_ids,
                    include_rationale=self._include_rationale,
                    protocol=self._protocol,
                )
                for repair_index, repair_instruction in enumerate(
                    repair_instructions[:MAX_TERMINAL_REPAIRS]
                ):
                    repair_record = self._client.generate_final_text(
                        system_prompt=active_judger_system(),
                        user_prompt=f"{base_prompt}\n\n{repair_instruction}",
                        final_tokens=min(self._final_tokens, 128),
                        terminal_prefix=BOXED_TERMINAL_PREFIX,
                    )
                    continuation = repair_record.final_outputs[0].strip()
                    repaired_raw = (
                        continuation
                        if continuation.startswith(BOXED_TERMINAL_PREFIX)
                        else BOXED_TERMINAL_PREFIX + continuation
                    )
                    repair_outputs.append(repaired_raw)
                    repair_elapsed += repair_record.elapsed_seconds
                    answer_text = _extract_boxed(repaired_raw)
                    if (
                        answer_text
                        and len(answer_text) <= MAX_BOXED_ANSWER_CHARACTERS
                        and len(answer_text.split()) <= MAX_BOXED_ANSWER_WORDS
                    ):
                        raw = original_raw + "\n" + repaired_raw
                        record = record.model_copy(
                            update={
                                "final_outputs": tuple(repair_outputs),
                                "format_repairs": repair_index + 1,
                                "physical_calls": record.physical_calls
                                + repair_index
                                + 1,
                                "elapsed_seconds": repair_elapsed,
                            }
                        )
                        break
                else:
                    if MAX_TERMINAL_REPAIRS == 0:
                        raise PipelineFailure(
                            code=FailureCode.ANSWER_CONTRACT,
                            stage="answerer",
                            detail=(
                                "boxed Judger response did not contain one concise "
                                "answer payload"
                            ),
                        )
                    tag_prompt = latent_answerer_prompt(
                        question=question,
                        choices=effective_choices,
                        decision_basis=decision_basis.value,
                        cited_patch_ids=cited_ids,
                        include_rationale=False,
                        protocol=AnswererProtocol.TAG,
                    )
                    if effective_choices:
                        fallback_instruction = (
                            f"Return only the number 1 through {len(effective_choices)} "
                            "for the single best CHOICE."
                        )
                        fallback_prefix = "CHOICE "
                        fallback_tokens = 8
                    else:
                        fallback_instruction = (
                            "Return only the exact answer and the closing </answer> tag."
                        )
                        fallback_prefix = "<answer>"
                        fallback_tokens = 128
                    tag_record = self._client.generate_final_text(
                        system_prompt=active_judger_system(),
                        user_prompt=f"{tag_prompt}\n\n{fallback_instruction}",
                        final_tokens=min(self._final_tokens, fallback_tokens),
                        terminal_prefix=fallback_prefix,
                    )
                    continuation = tag_record.final_outputs[0].strip()
                    if effective_choices:
                        tagged_raw = fallback_prefix + continuation
                        tag_answer = continuation
                    else:
                        tagged_raw = _normalize_tag_repair_output(continuation)
                        tag_answer, _ = _extract_tagged(tagged_raw)
                    normalized_tag_answer = _normalize_choice(
                        tag_answer,
                        effective_choices,
                    )
                    tag_is_valid = bool(normalized_tag_answer) and (
                        len(normalized_tag_answer) <= MAX_BOXED_ANSWER_CHARACTERS
                        and len(normalized_tag_answer.split()) <= MAX_BOXED_ANSWER_WORDS
                    )
                    if effective_choices:
                        tag_is_valid = tag_is_valid and (
                            normalized_tag_answer in effective_choices
                        )
                    if not tag_is_valid:
                        repair_outputs.append(tagged_raw)
                        output_preview = tuple(
                            output[:160] for output in repair_outputs
                        )
                        raise PipelineFailure(
                            code=FailureCode.ANSWER_CONTRACT,
                            stage="answerer",
                            detail=(
                                "boxed Judger response did not contain one concise "
                                "answer payload after three terminal repairs; "
                                f"output_preview={output_preview!r}"
                            ),
                        )
                    answer_text = normalized_tag_answer
                    repair_outputs.append(tagged_raw)
                    repair_elapsed += tag_record.elapsed_seconds
                    record = record.model_copy(
                        update={
                            "final_outputs": tuple(repair_outputs),
                            "format_repairs": 3,
                            "physical_calls": record.physical_calls + 3,
                            "elapsed_seconds": repair_elapsed,
                        }
                    )
        elif self._protocol in (
            AnswererProtocol.JSON,
            AnswererProtocol.STRUCTURED_JSON,
        ):
            # Read the answer from the parsed object first. _extract_tagged's no-tag
            # fallback returns the whole JSON string as the "tag body", so it must
            # NOT feed answer_text here — the parsed `answer` field is authoritative.
            answer_text = (
                (parsed_answer.strip() if isinstance(parsed_answer, str) else "")
                or _extract_json_answer(raw)
                or _extract_answer_line(raw)
                # free generation (VLMAS_TERMINAL_NO_FORCE=1) may answer in plain text
                # with no object at all; keep it so _normalize_choice can snap it.
                or (
                    tail
                    if os.environ.get("VLMAS_TERMINAL_NO_FORCE", "").strip() == "1"
                    else ""
                )
            )
        else:
            answer_text = (
                tagged_answer
                or (parsed_answer.strip() if isinstance(parsed_answer, str) else "")
                or _extract_json_answer(raw)
                or _extract_answer_line(raw)
                or _extract_boxed(raw)
                or tail
            )
        answer_text = _normalize_choice(answer_text, effective_choices)
        if (
            effective_choices
            and answer_text
            and answer_text not in effective_choices
            and os.environ.get("VLMAS_CHOICE_SNAP", "").strip() == "nearest"
        ):
            # 0915 (0731 select_choice_prediction): always commit to the nearest choice.
            import difflib as _difflib

            _norm = lambda t: re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()  # noqa: E731
            _scored = [
                (_difflib.SequenceMatcher(None, _norm(answer_text), _norm(c)).ratio(), c)
                for c in effective_choices
            ]
            _best = max(_scored, key=lambda item: item[0])
            print(
                f"[Answerer] nearest-choice snap {answer_text[:40]!r} -> {_best[1]!r} "
                f"(ratio {_best[0]:.2f})",
                flush=True,
            )
            answer_text = _best[1]
        if not answer_text and not _digit_repair_enabled():
            # 0915: no fabricated re-decode. Keep the raw so the empty readout is
            # inspectable, and let the case score as wrong.
            print(
                f"[Answerer] EMPTY terminal answer (raw={raw[:80]!r}); "
                "digit repair disabled -> recorded as no-answer",
                flush=True,
            )
            answer_text = NO_ANSWER_SENTINEL
        if not answer_text:
            if MAX_TERMINAL_REPAIRS == 0:
                raise PipelineFailure(
                    code=FailureCode.ANSWER_CONTRACT,
                    stage="answerer",
                    detail="structured Judger returned an empty answer",
                )
            repair_prompt = latent_answerer_prompt(
                question=question,
                choices=effective_choices,
                decision_basis=decision_basis.value,
                cited_patch_ids=cited_ids,
                include_rationale=False,
                protocol=AnswererProtocol.TAG,
            )
            repair_instruction = (
                f"Return only the number 1 through {len(effective_choices)} for the "
                "single best CHOICE."
                if effective_choices
                else "Return only the exact answer inside an <answer> tag."
            )
            repair_record = self._client.generate_final_text(
                system_prompt=active_judger_system(),
                user_prompt=f"{repair_prompt}\n\n{repair_instruction}",
                final_tokens=min(self._final_tokens, 128),
                terminal_prefix="CHOICE " if effective_choices else "<answer>",
            )
            repair_raw = repair_record.final_outputs[0].strip()
            repair_answer = (
                repair_raw
                if effective_choices
                else _extract_tagged(
                    repair_raw
                    if repair_raw.startswith("<answer>")
                    else "<answer>" + repair_raw
                )[0]
            )
            answer_text = _normalize_choice(repair_answer, effective_choices)
            record = record.model_copy(
                update={
                    "final_outputs": (*record.final_outputs, repair_raw),
                    "format_repairs": record.format_repairs + 1,
                    "physical_calls": record.physical_calls + 1,
                    "elapsed_seconds": record.elapsed_seconds
                    + repair_record.elapsed_seconds,
                }
            )
            if not answer_text:
                raise PipelineFailure(
                    code=FailureCode.ANSWER_CONTRACT,
                    stage="answerer",
                    detail=(
                        "structured Judger returned an empty answer after terminal repair"
                    ),
                )
        # Carry the model's own rationale from the tag (or legacy JSON) envelope;
        # fall back to a fixed note only when the decode omitted it (keeps
        # AnswerOutput's min_length contract).
        parsed_rationale = parsed.get("rationale")
        boxed_rationale = (
            _extract_boxed_rationale(raw)
            if self._protocol is AnswererProtocol.BOXED
            else ""
        )
        rationale = tagged_rationale or boxed_rationale or (
            parsed_rationale if isinstance(parsed_rationale, str) else ""
        )
        if not (isinstance(rationale, str) and rationale.strip()):
            rationale = "Judger answer read from the carried latent-state tag envelope."
        parsed_confidence = parsed.get("confidence")
        confidence = (
            float(parsed_confidence)
            if isinstance(parsed_confidence, (int, float))
            and not isinstance(parsed_confidence, bool)
            and 0.0 <= float(parsed_confidence) <= 1.0
            else 0.0
        )
        return ParsedRoleCall(
            value=AnswerOutput(
                decision_basis=decision_basis,
                answer=answer_text,
                evidence_patches=cited_ids,
                rationale=rationale.strip(),
                confidence=confidence,
            ),
            record=record,
        )

    def _terminal_prefix(self) -> str | None:
        """Teacher-forced assistant opener for the active protocol.

        TAG commits to the `<answer>` envelope and JSON to the `{"answer":` object.
        BOXED closes Qwen's thinking boundary before the 0808 free-text readout,
        then parses the model-emitted final `\\boxed{...}` answer.
        """
        match self._protocol:
            case AnswererProtocol.TAG:
                return "<answer>"
            case AnswererProtocol.JSON:
                return JSON_TERMINAL_PREFIX
            case AnswererProtocol.STRUCTURED_JSON:
                return STRUCTURED_JSON_TERMINAL_PREFIX
            case AnswererProtocol.BOXED:
                return ""
            case _ as unreachable:
                assert_never(unreachable)

    @staticmethod
    def _cited_ids(decision: AnswerEvidence) -> tuple[PatchId, ...]:
        match decision:
            case PassDecision():
                return decision.supporting_patches
            case InsufficientDecision():
                return decision.best_available_patches
            case PlannedEvidence():
                return decision.supporting_patches
            case NewRegionDecision() | ZoomDetailDecision():
                raise PipelineFailure(
                    code=FailureCode.ANSWER_CONTRACT,
                    stage="answerer",
                    detail="navigation decisions cannot be answered",
                )
            case _ as unreachable:
                assert_never(unreachable)
