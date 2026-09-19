#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Run the official LatentMAS Single reasoning contract on WSI-VQA.

# ─── How to run ───
# CUDA_VISIBLE_DEVICES=0 bash thinking/run_thinking.sh

The July-30 adapter path is retained so historical launch commands continue to
work. The stock WSI loader/evaluator is imported unchanged; this file replaces
only prompt construction, thinking control, and the one-pass output bridge.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Final, Protocol, runtime_checkable

PROJECT_ROOT: Final = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.single_v7.thinking.output_contract import extract_single_response
from models.single_v7.thinking.single_runner_patch import (
    TraceWriter,
    install_runner_patch,
)
from vision_text_mas.latent_answerer import (
    AnswererProtocol,
    LATENT_JUDGER_SYSTEM,
    latent_answerer_prompt,
)
from vision_text_mas.prompts import canonical_open_answer_options

HERE: Final = Path(__file__).resolve().parent
REPO: Final = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
ADAPTER_SOURCE: Final = REPO / "baselines" / "inference" / "qwen3-vl-4b-inference.py"
ADAPTER_BYTECODE: Final = (
    REPO / "baselines" / "inference" / "__pycache__" / "qwen3-vl-4b-inference.cpython-312.pyc"
)


SYSTEM_PROMPT: Final = LATENT_JUDGER_SYSTEM
LEGACY_RESPONSE_FORMAT: Final = (
    "Always respond in this exact format:\n"
    "answer: <exact choice text or short phrase>\n"
    "explanation: <one short sentence>"
)
STRUCTURED_JSON_RESPONSE_FORMAT: Final = (
    "Reason step by step internally, then output ONLY one complete JSON object "
    "and nothing else, using this exact schema:\n"
    '{"answer": "<exact choice text or short phrase>", '
    '"rationale": "brief evidence-grounded explanation", "confidence": 0.0}'
)
REASONER_INTERNAL_GUIDANCE: Final = (
    "Before producing the final answer, reason internally as a pathology "
    "observer: assess tissue architecture, cellularity, cytology, stroma, "
    "necrosis, boundaries, artifacts, and their relevance to the question. "
    "Use only visual evidence in the supplied whole-slide thumbnail. Do not "
    "expose a patch report or intermediate reasoning; use it only to ground "
    "the final answer and rationale."
)


@dataclass(frozen=True, slots=True)
class AdapterImportError(RuntimeError):
    """Raised when the stock 0730-compatible adapter cannot be imported."""

    path: Path

    def __str__(self) -> str:
        return f"cannot import stock adapter: {self.path}"


@dataclass(frozen=True, slots=True)
class InvalidMaxModelLenError(ValueError):
    """The configured model context length is not positive."""

    value: int

    def __str__(self) -> str:
        return f"MAX_MODEL_LEN must be positive, got {self.value}"


@runtime_checkable
class VllmAdapterModule(Protocol):
    """Adapter capabilities required by both optional vLLM runners."""

    SYSTEM_PROMPT: str

    def image_to_data_url(self, image_path: str | Path) -> str: ...

    def clean_generation(self, text: str) -> str: ...

    def main(self) -> None: ...


def build_wsi_single_prompt(question: str, choices: list[str]) -> str:
    """Use Reasoner-style internal analysis with the shared Answerer contract."""
    protocol = (
        AnswererProtocol.STRUCTURED_JSON
        if os.environ.get("SINGLE_ANSWERER_PROTOCOL", "structured_json").strip()
        == "structured_json"
        else AnswererProtocol.TAG
    )
    answerer_prompt = latent_answerer_prompt(
        question=question,
        choices=tuple(choices),
        include_rationale=True,
        protocol=protocol,
    )
    if os.environ.get("SINGLE_REASONER_STYLE", "").lower() in {"1", "true", "yes"}:
        return f"{REASONER_INTERNAL_GUIDANCE}\n\n{answerer_prompt}"
    return answerer_prompt


def effective_single_choices(question: str, choices: list[str]) -> list[str]:
    """Add declared labels only for eligible OPEN fields when explicitly enabled."""
    canonical_open = os.environ.get("SINGLE_CANONICAL_OPEN_OPTIONS", "1").lower()
    if choices or canonical_open in {"0", "false", "no"}:
        return choices
    return list(canonical_open_answer_options(question))


def _final_response_format() -> str:
    """Select the opt-in terminal schema without changing legacy Single runs."""
    protocol = os.environ.get("SINGLE_ANSWERER_PROTOCOL", "structured_json").strip()
    if protocol == "structured_json":
        return STRUCTURED_JSON_RESPONSE_FORMAT
    if protocol == "official_free_text":
        return (
            "Reason step by step and conclude with the single best final answer. "
            "Do not emit JSON and do not request another model call."
        )
    return LEGACY_RESPONSE_FORMAT


def extract_single_answer(raw_output: str) -> str:
    """Extract Single's answer from VL-MAS labels with legacy-box fallback."""
    answer, _ = extract_single_response(raw_output)
    return answer


def load_adapter() -> ModuleType:
    """Import the stock adapter from its real path so repo-root logic is intact."""
    adapter = ADAPTER_SOURCE if ADAPTER_SOURCE.is_file() else ADAPTER_BYTECODE
    if not adapter.is_file():
        raise SystemExit(f"stock adapter not found: {ADAPTER_SOURCE}")
    spec = importlib.util.spec_from_file_location("qwen3vl_stock_adapter", adapter)
    if spec is None or spec.loader is None:
        raise AdapterImportError(adapter)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _argv_value(flag: str) -> str | None:
    for index, item in enumerate(sys.argv):
        if item == flag and index + 1 < len(sys.argv):
            return sys.argv[index + 1]
        if item.startswith(f"{flag}="):
            return item.split("=", 1)[1]
    return None


def resolve_sink_path() -> Path:
    """Place the trace beside stock predictions unless explicitly overridden."""
    explicit = os.environ.get("THINKING_JSONL", "").strip()
    if explicit:
        return Path(explicit)
    answers = _argv_value("--answers-file")
    if answers is None:
        raise SystemExit("--answers-file missing; cannot place thinking.jsonl")
    return Path(answers).parent / "thinking.jsonl"


def runner_name_for(llm_url: str | None) -> str:
    """Select the stock-compatible runner from the configured transport."""
    return "OpenAICompatibleVLRunner" if llm_url else "QwenVLRunner"


class TraceSink:
    """Mutable append-only trace writer that flushes after every WSI question."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("a", encoding="utf-8")
        self.n = 0
        self.n_truncated = 0

    def write(self, record: dict[str, str | int | bool | None]) -> None:
        _ = self.stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.stream.flush()
        self.n += 1
        self.n_truncated += int(bool(record["truncated"]))

    def close(self) -> None:
        self.stream.close()
        print(
            f"[single] trace={self.path} rows={self.n} "
            f"missing_box={self.n_truncated}"
        )


def install_patches(
    adapter,
    sink: TraceWriter,
    *,
    direct_final_only: bool = False,
    enable_thinking: bool = True,
    runner_name: str = "QwenVLRunner",
    max_model_len: int | None = None,
    enable_finalization: bool = True,
) -> None:
    """Install official prompt and one sampled generation on the stock adapter."""
    install_runner_patch(
        adapter,
        sink,
        system_prompt=SYSTEM_PROMPT,
        prompt_builder=build_wsi_single_prompt,
        choice_resolver=effective_single_choices,
        direct_final_only=direct_final_only,
        enable_thinking=enable_thinking,
        runner_name=runner_name,
        max_model_len=max_model_len,
        enable_finalization=enable_finalization,
    )


def main() -> None:
    """Run the stock WSI pipeline with LatentMAS Single generation patches."""
    adapter = load_adapter()
    runner_name = runner_name_for(_argv_value("--llm-url"))
    if runner_name == "OpenAICompatibleVLRunner":
        if not isinstance(adapter, VllmAdapterModule):
            raise AdapterImportError(ADAPTER_SOURCE)
        if os.environ.get("SINGLE_DIRECT_VLLM", "") == "1":
            from models.single_v7.thinking.single_native_vllm_runner import (
                build_native_inprocess_vllm_runner,
            )

            setattr(
                adapter,
                "OpenAICompatibleVLRunner",
                build_native_inprocess_vllm_runner(adapter),
            )
        else:
            from models.single_v7.thinking.single_vllm_runner import (
                build_native_thinking_vllm_runner,
            )

            setattr(
                adapter,
                "OpenAICompatibleVLRunner",
                build_native_thinking_vllm_runner(adapter),
            )
    sink = TraceSink(resolve_sink_path())
    direct_final_only = os.environ.get("SINGLE_DIRECT_FINAL_ONLY", "").lower() in {
        "1",
        "true",
        "yes",
    }
    enable_thinking = os.environ.get("SINGLE_NATIVE_THINKING", "true").lower() in {
        "1",
        "true",
        "yes",
    }
    official_one_call = os.environ.get("SINGLE_OFFICIAL_ONE_CALL", "").lower() in {
        "1",
        "true",
        "yes",
    }
    raw_max_model_len = os.environ.get("MAX_MODEL_LEN", "").strip()
    max_model_len = int(raw_max_model_len) if raw_max_model_len else None
    if max_model_len is not None and max_model_len < 1:
        raise InvalidMaxModelLenError(max_model_len)
    install_patches(
        adapter,
        sink,
        direct_final_only=direct_final_only,
        enable_thinking=enable_thinking,
        runner_name=runner_name,
        max_model_len=max_model_len,
        enable_finalization=not official_one_call,
    )
    print("[single] LatentMAS prompt + one-pass sampled reasoning enabled")
    try:
        adapter.main()
    finally:
        sink.close()


if __name__ == "__main__":
    main()
