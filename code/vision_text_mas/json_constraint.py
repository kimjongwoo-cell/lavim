"""Transformers-compatible XGrammar JSON constraint adapter."""

from __future__ import annotations

import torch
import xgrammar
from transformers import LogitsProcessor


def token_id_from_tensor(token: torch.Tensor) -> int:
    """Convert a Transformers scalar token tensor to the XGrammar API type."""
    return int(token.item())


class XGrammarJsonLogitsProcessor(LogitsProcessor):
    """Mask each next-token distribution to a compiled JSON schema."""

    def __init__(self, grammar: xgrammar.CompiledGrammar) -> None:
        self._grammars = [grammar]
        self._matchers: list[xgrammar.GrammarMatcher] = []
        self._vocab_size = grammar.tokenizer_info.vocab_size
        self._stop_token_ids = tuple(grammar.tokenizer_info.stop_token_ids)
        self._token_bitmask: torch.Tensor | None = None
        self._prefilled = False
        self._batch_size = 0

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Accept the sampled token and mask the following token distribution."""
        if not self._matchers:
            self._batch_size = input_ids.shape[0]
            grammars = self._grammars * self._batch_size
            self._matchers = [xgrammar.GrammarMatcher(grammar) for grammar in grammars]
            self._token_bitmask = xgrammar.allocate_token_bitmask(
                self._batch_size,
                self._vocab_size,
            )
        if input_ids.shape[0] != self._batch_size:
            raise RuntimeError("JSON constraint batch size changed during generation")
        if self._prefilled:
            for row, matcher in enumerate(self._matchers):
                if not matcher.is_terminated():
                    token_id = token_id_from_tensor(input_ids[row, -1])
                    accepted = matcher.accept_token(token_id)
                    if not accepted:
                        raise RuntimeError(
                            f"generated token violated the JSON grammar: id={token_id}"
                        )
        else:
            self._prefilled = True
        assert self._token_bitmask is not None
        for row, matcher in enumerate(self._matchers):
            if not matcher.is_terminated():
                matcher.fill_next_token_bitmask(self._token_bitmask, row)
        xgrammar.apply_token_bitmask_inplace(
            scores,
            self._token_bitmask.to(scores.device),
        )
        for row, matcher in enumerate(self._matchers):
            if not matcher.is_terminated():
                scores[row, list(self._stop_token_ids)] = -float("inf")
        return scores


def build_json_logits_processor(
    *,
    tokenizer: object,
    json_schema: str,
    model_vocab_size: int | None = None,
) -> LogitsProcessor:
    """Compile one strict JSON-schema grammar for a Transformers tokenizer."""
    tokenizer_info = xgrammar.TokenizerInfo.from_huggingface(
        tokenizer,
        vocab_size=model_vocab_size,
    )
    grammar = xgrammar.GrammarCompiler(tokenizer_info).compile_json_schema(json_schema)
    return XGrammarJsonLogitsProcessor(grammar)
