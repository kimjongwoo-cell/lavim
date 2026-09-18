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
    import os as _ttos

    _tt = _ttos.environ.get("VLMAS_THINK_TAGS", "").strip()
    if _tt == "strip" and prompt.endswith("<think>\n"):
        prompt = prompt[: -len("<think>\n")]
    elif _tt == "open":
        pass
    elif prompt.endswith("<think>\n"):
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
    logits_processor: object | None = None,
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
    if logits_processor is not None:
        # Schema-constrained decode (e.g. VLMAS_NAV_GRAMMAR): mask each step to
        # the compiled JSON grammar. None keeps the decode byte-identical.
        from transformers import LogitsProcessorList

        generation_kwargs["logits_processor"] = LogitsProcessorList(
            [logits_processor]
        )
    if no_repeat_ngram_size > 0:
        generation_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
    if repetition_penalty != 1.0:
        generation_kwargs["repetition_penalty"] = repetition_penalty
    # 0915 first-token probe (VLMAS_TERMINAL_FIRSTTOK_PROBE=<jsonl>, terminal
    # Answerer call only): capture the raw first-step scores without changing them.
    import os as _pos
    _probe_path = _pos.environ.get("VLMAS_TERMINAL_FIRSTTOK_PROBE", "").strip()
    _probe_cap: dict[str, torch.Tensor] = {}
    if _probe_path and getattr(backbone, "_route_terminal_call", False):
        from transformers import LogitsProcessor, LogitsProcessorList

        class _FirstTokCapture(LogitsProcessor):
            def __call__(self, input_ids, scores):  # noqa: D401
                if "first" not in _probe_cap:
                    _probe_cap["first"] = scores[0].detach().float().cpu()
                return scores

        _lst = generation_kwargs.get("logits_processor")
        if _lst is None:
            _lst = LogitsProcessorList([])
        _lst.append(_FirstTokCapture())
        generation_kwargs["logits_processor"] = _lst
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
    # Visual-Grounded Receiver Prefill (C2 spec 0907, env VLMAS_VG_PREFILL=1):
    # during the Answerer PROMPT PREFILL each prompt token may read only
    # J_t = R* (compressed visual cols) ∪ causal prompt prefix — the latent
    # history is masked OUT of the prefill sources, so the prompt K/V become
    # visual-conditioned receiver states. Decode steps (q_len == 1) run with
    # the standard FULL mask, so answer generation uses the whole cache.
    import os as _os
    # consume-once: the engine arms this for exactly ONE terminal decode; any
    # other decode sharing this function (navigator JSON, decode-on-clone)
    # must never inherit stale columns from a previous case's cache.
    vg_cols = getattr(backbone, "_vg_prefill_cols", None)
    backbone._vg_prefill_cols = None
    if (
        _os.environ.get("VLMAS_VG_PREFILL", "") == "1"
        and isinstance(vg_cols, torch.Tensor)
        and vg_cols.numel() > 0
        and past_len > 0
        and int(vg_cols.max()) < past_len
        and int(input_ids.shape[1]) > 1
    ):
        # Two-phase VG prefill using the proven 2-D allowed-vector mask (the
        # relay precedent): phase 1 prefills the first T-1 prompt tokens with
        # past sources restricted to R* (latent history masked out), so their
        # K/V become visual-conditioned receiver states; phase 2 hands the
        # LAST prompt token + generation to the standard full-mask generate.
        prompt_len = int(input_ids.shape[1])
        head_ids = input_ids[:, :-1]
        allowed = torch.zeros(
            past_len, dtype=attention_mask.dtype, device=backbone.device)
        allowed[vg_cols.to(device=backbone.device, dtype=torch.long)] = 1
        phase1_mask = torch.cat([
            allowed,
            torch.ones(prompt_len - 1, dtype=attention_mask.dtype,
                       device=backbone.device),
        ]).unsqueeze(0)
        # phase 1 runs through the INNER language model with inputs_embeds and
        # a 2-D allowed-vector mask — the exact call shape the visual relay
        # already validated on GPU (the VLM wrapper's mask path device-asserts).
        head_embeds = backbone.model.get_input_embeddings()(head_ids)
        phase1_kwargs: dict[str, object] = dict(
            inputs_embeds=head_embeds,
            attention_mask=phase1_mask,
            past_key_values=cache,
            use_cache=True,
        )
        if position_ids is not None:
            phase1_kwargs["position_ids"] = position_ids[..., :-1]
        with torch.no_grad():
            phase1 = backbone.lm(**phase1_kwargs)
        cache = phase1.past_key_values
        print(
            f"[VGPrefill] prompt={prompt_len} past={past_len} "
            f"vis={int(vg_cols.numel())} (latent history masked in prefill)",
            flush=True)
        generation_kwargs["input_ids"] = input_ids[:, -1:]
        generation_kwargs["past_key_values"] = cache
        generation_kwargs["attention_mask"] = torch.ones(
            (1, past_len + prompt_len), dtype=attention_mask.dtype,
            device=backbone.device)
        if position_ids is not None:
            generation_kwargs["position_ids"] = position_ids[..., -1:]
        if stop_strings:
            # the stop criterion slices past the FED tokens (now 1), not the
            # original prompt length
            generation_kwargs["stopping_criteria"] = StoppingCriteriaList([
                _StopOnDecodedText(
                    tokenizer=backbone.processor.tokenizer,
                    prompt_len=1,
                    stops=stop_strings,
                    balanced_box=stop_at_balanced_box,
                )
            ])
    # Single-Pass Dual-Address Visual Routing (env VLMAS_KV_ROUTE=1, off =
    # byte-identical): hooks capture the boundary query inside THIS prefill
    # and resolve each observation page's address once; no extra forward.
    _router = None
    if _os.environ.get("VLMAS_KV_ROUTE", "") == "1":
        from memory import route_address as _ra
        _r_start = (position_cursor
                    if (backbone.mrope_pos and position_cursor is not None)
                    else past_len)
        _b_pos = int(_r_start) + int(input_ids.shape[1]) - 1
        _mode = _os.environ.get("VLMAS_KV_ROUTE_MODE", "").strip()
        if _mode == "diag":
            _ddir = _os.environ.get("VLMAS_KV_ROUTE_DIAG", "").strip() or "."
            import pathlib as _pl
            _pl.Path(_ddir).mkdir(parents=True, exist_ok=True)
            _ci = getattr(backbone, "_route_case_index", None)
            _ci = int(_ci) if _ci is not None else len(list(_pl.Path(_ddir).glob("diag_*.json")))
            _router = _ra.DiagCapture(
                backbone, generation_kwargs["past_key_values"], _b_pos,
                str(_pl.Path(_ddir) / f"diag_{_ci:04d}.json"))
            _router.attach()
        elif _mode == "bookkeep":
            # visual bookkeeping only (recorded at the Reasoner boundary); decode stays native
            pass
        elif _mode in ("canonical", "canonical_t", "canonical_tperm", "derope"):
            # Observation-Canonical Visual Addressing: rewrite visual phases
            # BEFORE the prompt prefill (no hooks, no query needed).
            _cols = getattr(backbone, "_route_vis_cols", None)
            _vpos = getattr(backbone, "_route_vis_pos", None)
            _pages = getattr(backbone, "_route_pages", None)
            if (_os.environ.get("VLMAS_KV_ROUTE_TERMINAL_ONLY", "") == "1"
                    and not getattr(backbone, "_route_terminal_call", False)):
                # Bookkeeping is recorded at the Reasoner boundary and never cleared, so
                # without this gate every later decode (the next case planner/navigator,
                # report decodes) would re-rotate stale column indices.
                print("[KVCanon] SKIP: not the Answerer call (terminal-only)", flush=True)
            elif _cols is not None and _vpos is not None and _pages:
                _ra.apply_canonical(
                    backbone, generation_kwargs["past_key_values"],
                    _cols, _vpos, _pages, _b_pos)
            else:
                print("[KVCanon] SKIP: no visual bookkeeping", flush=True)
        else:
            _router = _ra.SinglePassRouter.build(
                backbone, generation_kwargs["past_key_values"], past_len,
                _b_pos)
            if _router is not None:
                _router.attach()
    try:
        generated = backbone.model.generate(**generation_kwargs)
    finally:
        if _router is not None:
            _router.detach()
    # VG phase-2 feeds only the last prompt token, so slice past THAT input,
    # not the full original prompt.
    fed = generation_kwargs["input_ids"]
    new_ids = generated[:, fed.shape[1] :]
    _out_text = backbone.processor.tokenizer.decode(
        new_ids[0],
        skip_special_tokens=True,
    ).strip()
    if _probe_path and "first" in _probe_cap:
        import json as _pjson
        _tok = backbone.processor.tokenizer
        _sc = _probe_cap["first"]
        _pr = torch.softmax(_sc, dim=-1)
        _eos = _tok.eos_token_id
        _top = torch.topk(_sc, 10)
        _rec = {
            "case": getattr(backbone, "_route_case_index", None),
            "past_len": int(past_len),
            "prompt_len": int(input_ids.shape[1]),
            "json_prefix": json_prefix,
            "user_prompt_head": (user_prompt or "")[:90],
            "n_choice_lines": (user_prompt or "").count("\nCHOICE "),
            "eos_id": int(_eos) if _eos is not None else None,
            "eos_prob": float(_pr[_eos]) if _eos is not None else None,
            "eos_rank": int((_sc > _sc[_eos]).sum()) if _eos is not None else None,
            "top10": [
                [_tok.decode([int(i)]), round(float(_pr[i]), 4), round(float(_sc[i]), 2)]
                for i in _top.indices.tolist()
            ],
            "gen_text_head": _out_text[:80],
        }
        try:
            with open(_probe_path, "a") as _f:
                _f.write(_pjson.dumps(_rec, ensure_ascii=False) + "\n")
        except OSError as _e:  # noqa: BLE001
            print(f"[FirstTokProbe] write failed: {_e!r}", flush=True)
    return _out_text


@torch.no_grad()
def score_terminal_continuation(
    *,
    backbone: TerminalBackbone[CacheT],
    cache: CacheT,
    position_cursor: int,
    system_prompt: str,
    user_prompt: str,
    json_prefix: str | None,
    continuation: str,
) -> tuple[float, int]:
    """Teacher-forced sum log-probability of `continuation` after the terminal
    prompt, under the given carried KV state.

    Mirrors generate_terminal_json's prompt/position/rope handling exactly, but
    runs ONE forward over prompt+continuation instead of generating. MUTATES the
    cache it is given (the forward appends the prompt/continuation K/V), so
    callers must pass a disposable copy. Returns (sum_logprob, n_tokens).
    """
    prompt = _assistant_prompt(
        backbone.processor,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        json_prefix=json_prefix,
    )
    tokenizer = backbone.processor.tokenizer
    prompt_ids = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(backbone.device)
    continuation_ids = tokenizer(
        continuation, return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(backbone.device)
    if int(continuation_ids.numel()) == 0:
        return 0.0, 0
    input_ids = torch.cat([prompt_ids, continuation_ids], dim=1)
    attention_mask = torch.ones_like(input_ids, device=backbone.device)
    past_len = backbone._kv_len(cache)
    position_ids = None
    if backbone.mrope_pos and hasattr(backbone, "_text_positions"):
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
    output = backbone.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=cache,
        use_cache=True,
    )
    logits = output.logits[0, prompt_ids.shape[1] - 1 : -1, :].float()
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(
        1, continuation_ids[0].unsqueeze(-1)
    ).squeeze(-1)
    return float(token_log_probs.sum().item()), int(continuation_ids.shape[1])
