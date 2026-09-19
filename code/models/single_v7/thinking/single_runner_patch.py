"""Patch the stock WSI adapter with the Single v7 output contract."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from models.single_v7.thinking.output_contract import (
    extract_last_boxed_answer,
    extract_single_response,
    parse_single_response,
    requires_finalization,
    to_stock_labeled_output,
)


class TraceWriter(Protocol):
    """Minimal trace boundary used by production and in-memory tests."""

    n: int

    def write(self, record: dict[str, str | int | bool | None]) -> None: ...


@dataclass(frozen=True, slots=True)
class StrictContextLimitError(RuntimeError):
    """The requested completion cannot fit within the physical context."""

    input_length: int
    request_length: int
    max_model_len: int

    def __str__(self) -> str:
        return (
            "strict context limit exceeded: "
            f"input={self.input_length}, request={self.request_length}, "
            f"max_model_len={self.max_model_len}"
        )


def _install_context_guard(runner_class, max_model_len: int) -> None:
    """Reject a local-HF generation before it exceeds the shared physical cap."""
    for method_name in ("_inputs_with_qwen_utils", "_inputs_fallback"):
        stock_inputs = getattr(runner_class, method_name, None)
        if stock_inputs is None:
            continue

        def guarded_inputs(self, image_path, prompt, _stock_inputs=stock_inputs):
            inputs = _stock_inputs(self, image_path, prompt)
            input_ids = inputs.get("input_ids")
            input_length = 0 if input_ids is None else int(input_ids.shape[-1])
            request_length = int(self.args.max_new_tokens)
            if input_length + request_length > max_model_len:
                raise StrictContextLimitError(
                    input_length=input_length,
                    request_length=request_length,
                    max_model_len=max_model_len,
                )
            return inputs

        setattr(runner_class, method_name, guarded_inputs)


def install_runner_patch(
    adapter,
    sink: TraceWriter,
    *,
    system_prompt: str,
    prompt_builder: Callable[[str, list[str]], str],
    choice_resolver: Callable[[str, list[str]], list[str]] | None = None,
    direct_final_only: bool = False,
    enable_thinking: bool = True,
    runner_name: str = "QwenVLRunner",
    max_model_len: int | None = None,
    enable_finalization: bool = True,
) -> None:
    """Install the Single prompt and bounded finalization on a stock runner."""
    runner_class = getattr(adapter, runner_name)
    stock_generate = runner_class.generate
    stock_generate_batch = getattr(runner_class, "generate_batch", None)
    stock_close_thinking = runner_class._close_thinking
    choices_for_prompt: dict[str, list[str]] = {}
    adapter.SYSTEM_PROMPT = system_prompt
    if max_model_len is not None:
        _install_context_guard(runner_class, max_model_len)

    def format_prompt(
        sample,
        force_choice_answer: bool = True,
        mcq_style: str = "letter",
        prompt_style: str = "wsi",
    ) -> str:
        del force_choice_answer, mcq_style, prompt_style
        raw_choices = sample.get("Choice") or []
        choices = raw_choices if isinstance(raw_choices, list) else [raw_choices]
        question = str(sample.get("Question", ""))
        raw_choices = [str(choice) for choice in choices]
        effective_choices = (
            choice_resolver(question, raw_choices)
            if choice_resolver is not None
            else raw_choices
        )
        prompt = prompt_builder(question, effective_choices)
        choices_for_prompt[prompt] = effective_choices
        return prompt

    adapter.format_prompt = format_prompt

    def keep_thinking_open(_self, text: str) -> str:
        return text

    if enable_thinking and not direct_final_only:
        runner_class._close_thinking = keep_thinking_open

    def reset_direct_hf_seed(runner) -> None:
        """Make each local sampled generation invariant to worker assignment."""
        reset = getattr(adapter, "set_seed", None)
        seed = getattr(getattr(runner, "args", None), "seed", None)
        if runner_name == "QwenVLRunner" and callable(reset) and seed is not None:
            reset(int(seed))

    def generate(self, image_path, prompt: str) -> tuple[str, int]:
        primary_limit = int(
            getattr(getattr(self, "args", None), "max_new_tokens", 4_096)
        )
        reset_direct_hf_seed(self)
        generation_output, generation_tokens = stock_generate(self, image_path, prompt)
        primary_output = "" if direct_final_only else generation_output
        primary_tokens = 0 if direct_final_only else generation_tokens
        final_output = generation_output
        finalization_tokens = generation_tokens if direct_final_only else 0
        finalization_attempted = direct_final_only or (
            enable_finalization
            and requires_finalization(
                generation_output,
                generation_tokens,
                primary_limit,
            )
        )
        if finalization_attempted and not direct_final_only:
            self._close_thinking = stock_close_thinking.__get__(self, runner_class)
            raw_finalization_limit = os.environ.get(
                "SINGLE_FINALIZATION_MAX_NEW_TOKENS", ""
            ).strip()
            finalization_limit = int(raw_finalization_limit or "512")
            self.args.max_new_tokens = min(finalization_limit, primary_limit)
            try:
                reset_direct_hf_seed(self)
                candidate_output, finalization_tokens = stock_generate(
                    self,
                    image_path,
                    prompt,
                )
            finally:
                self.args.max_new_tokens = primary_limit
                del self._close_thinking
            if parse_single_response(candidate_output).explicit:
                final_output = candidate_output
        if finalization_attempted and not parse_single_response(final_output).explicit:
            final_output = "answer: unknown\nexplanation:"

        output_tokens = primary_tokens + finalization_tokens
        boxed = extract_last_boxed_answer(final_output)
        answer, explanation = extract_single_response(final_output)
        choices = choices_for_prompt.get(prompt, [])
        if choices:
            answer, _ = adapter.select_choice_prediction(
                answer, choices, mcq_style="text"
            )
        labeled = to_stock_labeled_output(answer, explanation)
        tokenizer = getattr(getattr(self, "processor", None), "tokenizer", None)
        explicit = parse_single_response(final_output).explicit
        sink.write(
            {
                "seq": sink.n,
                "image": str(image_path),
                "prompt": prompt,
                "thinking": primary_output,
                "raw_output": primary_output,
                "final_output": final_output,
                "boxed_answer": boxed,
                "visible": labeled,
                "truncated": not explicit,
                "finalization_attempted": finalization_attempted,
                "finalization_failed": finalization_attempted and not explicit,
                "num_output_tokens": output_tokens,
                "num_thinking_tokens": primary_tokens,
                "num_finalization_tokens": finalization_tokens,
                "num_visible_tokens": adapter.count_text_tokens(tokenizer, labeled),
            }
        )
        return labeled, output_tokens

    runner_class.generate = generate

    if stock_generate_batch is not None:
        def generate_batch(self, image_paths, prompts):
            primary_limit = int(
                getattr(getattr(self, "args", None), "max_new_tokens", 4_096)
            )
            reset_direct_hf_seed(self)
            primary_results = stock_generate_batch(self, image_paths, prompts)
            final_outputs = [output for output, _ in primary_results]
            finalization_tokens = [0] * len(primary_results)
            needs_finalization = [
                index
                for index, (output, tokens) in enumerate(primary_results)
                if enable_finalization
                and requires_finalization(output, tokens, primary_limit)
            ]
            if needs_finalization:
                self._close_thinking = stock_close_thinking.__get__(self, runner_class)
                raw_limit = os.environ.get(
                    "SINGLE_FINALIZATION_MAX_NEW_TOKENS", ""
                ).strip()
                self.args.max_new_tokens = min(int(raw_limit or "512"), primary_limit)
                try:
                    reset_direct_hf_seed(self)
                    finalized = stock_generate_batch(
                        self,
                        [image_paths[index] for index in needs_finalization],
                        [prompts[index] for index in needs_finalization],
                    )
                finally:
                    self.args.max_new_tokens = primary_limit
                    del self._close_thinking
                for index, (candidate, tokens) in zip(
                    needs_finalization, finalized, strict=True
                ):
                    finalization_tokens[index] = tokens
                    if parse_single_response(candidate).explicit:
                        final_outputs[index] = candidate

            results = []
            tokenizer = getattr(getattr(self, "processor", None), "tokenizer", None)
            for index, ((primary_output, primary_tokens), final_output) in enumerate(
                zip(primary_results, final_outputs, strict=True)
            ):
                parsed = parse_single_response(final_output)
                if not parsed.explicit:
                    parsed_output = "answer: unknown\nexplanation:"
                    answer, explanation = "unknown", ""
                else:
                    parsed_output = final_output
                    answer, explanation = parsed.answer, parsed.explanation
                choices = choices_for_prompt.get(prompts[index], [])
                if choices:
                    answer, _ = adapter.select_choice_prediction(
                        answer, choices, mcq_style="text"
                    )
                labeled = to_stock_labeled_output(answer, explanation)
                sink.write(
                    {
                        "seq": sink.n,
                        "image": str(image_paths[index]),
                        "prompt": prompts[index],
                        "thinking": primary_output,
                        "raw_output": primary_output,
                        "final_output": parsed_output,
                        "boxed_answer": extract_last_boxed_answer(parsed_output),
                        "visible": labeled,
                        "truncated": not parsed.explicit,
                        "finalization_attempted": index in needs_finalization,
                        "finalization_failed": index in needs_finalization and not parsed.explicit,
                        "num_output_tokens": primary_tokens + finalization_tokens[index],
                        "num_thinking_tokens": primary_tokens,
                        "num_finalization_tokens": finalization_tokens[index],
                        "num_visible_tokens": adapter.count_text_tokens(tokenizer, labeled),
                    }
                )
                results.append((labeled, primary_tokens + finalization_tokens[index]))
            return results

        runner_class.generate_batch = generate_batch

    raw_limit = os.environ.get("THINKING_MAX_ITEMS", "").strip()
    if raw_limit:
        stock_loader = adapter.load_pseudo_json
        limit = int(raw_limit)

        def load_pseudo_json(path):
            samples = stock_loader(path)
            print(f"[single] limiting {len(samples)} -> {min(limit, len(samples))}")
            return samples[:limit]

        adapter.load_pseudo_json = load_pseudo_json
