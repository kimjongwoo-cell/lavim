"""0915 Answerer budgeted thinking (env-gated, off = byte-identical).

VLMAS_ANSWERER_THINK_BUDGET=K  Terminal Answerer JSON readout in two phases, using only the
                               model's own tokens:
  phase 1: the assistant turn is left OPEN at "<think>\n" and the model writes its own reasoning
           (greedy, on a cache copy) until it emits "</think>" or K tokens.
  phase 2: the real decode prompt becomes "<think>\n" + <that reasoning> + "\n</think>\n\n" +
           the usual opener, so the answer is written after the model's own thought.
No grammar, no logit masking, no extra latent steps.
"""
import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "."
p = os.path.join(root, "vision_text_mas/latent_terminal.py")
s = open(p).read()
old = '''    prompt = _assistant_prompt(
        backbone.processor,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        json_prefix=json_prefix,
    )
    input_ids = backbone.processor.tokenizer(
'''
new = '''    prompt = _assistant_prompt(
        backbone.processor,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        json_prefix=json_prefix,
    )
    # 0915 VLMAS_ANSWERER_THINK_BUDGET: phase-1 call leaves the think block open.
    if getattr(backbone, "_think_phase_active", False) and prompt.endswith("</think>\\n\\n"):
        prompt = prompt[: -len("</think>\\n\\n")]
    import os as _tos
    _think_budget = int(_tos.environ.get("VLMAS_ANSWERER_THINK_BUDGET", "0") or 0)
    _close = "</think>\\n\\n" + (json_prefix or "")
    if (
        _think_budget > 0
        and json_prefix is not None
        and getattr(backbone, "_route_terminal_call", False)
        and not getattr(backbone, "_think_phase_active", False)
        and prompt.endswith("<think>\\n" + _close)
    ):
        from copy import deepcopy as _tdc

        backbone._think_phase_active = True
        _saved_route = backbone._route_terminal_call
        backbone._route_terminal_call = False  # phase 1 is not the probed/hooked readout
        try:
            _think_raw = generate_terminal_json(
                backbone=backbone,
                cache=_tdc(cache),
                position_cursor=position_cursor,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                json_prefix="",
                max_new_tokens=_think_budget,
                temperature=temperature,
                top_p=top_p,
                do_sample=False,
                stop_strings=("</think>",),
            )
        finally:
            backbone._think_phase_active = False
            backbone._route_terminal_call = _saved_route
        _think_text = _think_raw.split("</think>", 1)[0].strip()
        prompt = prompt[: -len(_close)] + _think_text + "\\n" + _close
        print(
            f"[AnswererThink] budget={_think_budget} tokens_text_chars={len(_think_text)} "
            f"closed={'</think>' in _think_raw} head={_think_text[:240]!r}",
            flush=True,
        )
    input_ids = backbone.processor.tokenizer(
'''
assert s.count(old) == 1, ("terminal prompt anchor", s.count(old))
s = s.replace(old, new)
open(p, "w").write(s)
print("terminal: VLMAS_ANSWERER_THINK_BUDGET added")
