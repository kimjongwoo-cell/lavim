"""Terminal sampled decode on a latent Qwen cache at an assistant boundary."""

from __future__ import annotations

import re
from typing import Protocol, TypeVar

import torch
from transformers import StoppingCriteria, StoppingCriteriaList


CacheT = TypeVar("CacheT")


class _StopOnDecodedText(StoppingCriteria):
    """Halt generation once the decoded continuation contains a stop string.

    Whitespace-tolerant on the answer close tag so a broken "</ answer>" /
    "</\\nanswer>" still stops the decode at the source instead of rambling on.
    """

    def __init__(
        self,
        *,
        tokenizer: object,
        prompt_len: int,
        stops: tuple[str, ...],
        balanced_box: bool = False,
    ) -> None:
        self._tokenizer = tokenizer
        self._prompt_len = prompt_len
        self._stops = tuple(stops)
        self._balanced_box = balanced_box
        self._tolerant = re.compile(r"<\s*/\s*answer\s*>")

    def __call__(
        self,
        input_ids: torch.Tensor,
        scores: torch.Tensor | None = None,
        **kwargs: object,
    ) -> bool:
        text = self._tokenizer.decode(
            input_ids[0][self._prompt_len :],
            skip_special_tokens=True,
        )
        if self._balanced_box:
            # ``\\boxed{`` is teacher-forced and therefore absent from ``text``.
            # Start at depth one so a nested JSON/LaTex brace cannot terminate the
            # payload before the forced outer box itself has closed.
            depth = 1
            for character in text:
                if character == "{":
                    depth += 1
                elif character == "}":
                    depth -= 1
                    if depth == 0:
                        return True
            return False
        if any(stop in text for stop in self._stops):
            return True
        return "</answer>" in {stop.strip() for stop in self._stops} and bool(
            self._tolerant.search(text)
        )


class TerminalTokenizer(Protocol):
    """Tokenizer operations required for one terminal cached decode."""

    pad_token_id: int | None
    eos_token_id: int | None

    def __call__(
        self,
        text: str,
        *,
        return_tensors: str,
        add_special_tokens: bool,
    ) -> dict[str, torch.Tensor]: ...

    def decode(self, token_ids: torch.Tensor, *, skip_special_tokens: bool) -> str: ...


class TerminalProcessor(Protocol):
    """Chat-template surface exposed by Qwen's processor."""

    tokenizer: TerminalTokenizer

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str: ...


class TerminalModel(Protocol):
    """Generation subset needed after the latent cache is prepared."""

    device: torch.device

    def generate(self, **inputs: torch.Tensor | int | float | bool) -> torch.Tensor: ...


class TerminalVlm(Protocol):
    """MRoPE delta holder used by the underlying Qwen generation method."""

    rope_deltas: torch.Tensor | None


class TerminalBackbone(Protocol[CacheT]):
    """Minimal public Qwen backbone surface for terminal assistant-prefixed decode."""

    processor: TerminalProcessor
    model: TerminalModel
    vlm: TerminalVlm
    device: str
    mrope_pos: bool

    def _kv_len(self, cache: CacheT) -> int: ...

    def _text_positions(self, n: int, start: int) -> torch.Tensor: ...
def _assistant_prompt(
    processor: TerminalProcessor,
    *,
    system_prompt: str,
    user_prompt: str,
    json_prefix: str | None,
) -> str:
    """Render Qwen's assistant turn before adding a JSON prefix to that turn."""
    if not system_prompt and not user_prompt:
        # Latent appends close Qwen's textual think turn before M latent states
        # are written to the cache. Repeating ``</think>`` after that cache makes
        # the terminal continuation off-distribution; append only the JSON prefix.
        return "" if json_prefix is None else json_prefix
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if prompt.endswith("<think>\n"):
        prompt += "</think>\n\n"
    return prompt if json_prefix is None else prompt + json_prefix


def generate_terminal_json(
    *,
    backbone: TerminalBackbone[CacheT],
    cache: CacheT,
    position_cursor: int,
    system_prompt: str,
    user_prompt: str,
    json_prefix: str | None,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool = True,
    min_new_tokens: int = 0,
    force_single_digit: bool = False,
    stop_strings: tuple[str, ...] = (),
    stop_at_balanced_box: bool = False,
    no_repeat_ngram_size: int = 0,
    repetition_penalty: float = 1.0,
) -> str:
    """Decode a terminal readout from a carried KV state without re-entering text.

    do_sample=False makes the readout greedy (deterministic, no sampling drift that
    breaks the tag/JSON envelope). stop_strings halts generation as soon as the
    envelope closes (e.g. "</answer>"), so the raw output is just the answer instead
    of the model rambling / re-emitting </think> and repeated answers to the budget.
    no_repeat_ngram_size > 0 blocks a greedy readout from looping a phrase to the
    budget when it never reaches the closing token (e.g. a rambling JSON rationale
    that omits "}"); 0 keeps the decode byte-identical for callers that don't need it.
    """
    prompt = _assistant_prompt(
        backbone.processor,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        json_prefix=json_prefix,
    )
    input_ids = backbone.processor.tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"].to(backbone.device)
    attention_mask = torch.ones_like(input_ids, device=backbone.device)
    past_len = backbone._kv_len(cache)
    position_ids = None
    if backbone.mrope_pos and hasattr(backbone, "_text_positions"):
        # Qwen3-VL's generation contract uses 3-D MRoPE position_ids.  The
        # text-only Qwen cache_position kwarg is rejected by the VL wrapper.
        position_ids = backbone._text_positions(
            input_ids.shape[1],
            start=position_cursor if position_cursor is not None else past_len,
        )
    if past_len > 0:
        past_mask = torch.ones(
            (attention_mask.shape[0], past_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
        if backbone.mrope_pos:
            backbone.vlm.rope_deltas = torch.tensor(
                [[position_cursor - past_len]],
                dtype=torch.long,
                device=backbone.device,
            )
    generation_kwargs: dict[str, object] = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=cache,
        do_sample=do_sample,
        max_new_tokens=max_new_tokens,
        pad_token_id=(
            backbone.processor.tokenizer.pad_token_id
            or backbone.processor.tokenizer.eos_token_id
        ),
        use_cache=True,
    )
    if do_sample:
        # temperature/top_p only apply to sampling; passing them under greedy just
        # triggers transformers warnings.
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
    if min_new_tokens > 0:
        generation_kwargs["min_new_tokens"] = min_new_tokens
    if force_single_digit:
        digit_token_ids: list[int] = []
        for digit in range(1, 10):
            encoded = backbone.processor.tokenizer(
                str(digit),
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"][0]
            if int(encoded.numel()) == 1:
                digit_token_ids.append(int(encoded.item()))
        eos_token_id = backbone.processor.tokenizer.eos_token_id
        if not digit_token_ids or eos_token_id is None:
            raise RuntimeError("terminal choice decode requires digit and EOS tokens")
        prompt_length = int(input_ids.shape[1])

        def choice_token_constraint(
            batch_id: int,
            generated_ids: torch.Tensor,
        ) -> list[int]:
            _ = batch_id
            if int(generated_ids.shape[-1]) == prompt_length:
                return digit_token_ids
            return [int(eos_token_id)]

        generation_kwargs["prefix_allowed_tokens_fn"] = choice_token_constraint
    if no_repeat_ngram_size > 0:
        generation_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
    if repetition_penalty != 1.0:
        generation_kwargs["repetition_penalty"] = repetition_penalty
    if stop_strings:
        generation_kwargs["stopping_criteria"] = StoppingCriteriaList(
            [
                _StopOnDecodedText(
                    tokenizer=backbone.processor.tokenizer,
                    prompt_len=int(input_ids.shape[1]),
                    stops=stop_strings,
                    balanced_box=stop_at_balanced_box,
                )
            ]
        )
    if position_ids is not None:
        generation_kwargs["position_ids"] = position_ids
    generated = backbone.model.generate(**generation_kwargs)
    new_ids = generated[:, input_ids.shape[1] :]
    return backbone.processor.tokenizer.decode(
        new_ids[0],
        skip_special_tokens=True,
    ).strip()
