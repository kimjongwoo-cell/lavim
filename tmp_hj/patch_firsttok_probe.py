"""0915 diagnostics (print-only, env-gated, off = byte-identical):

VLMAS_TERMINAL_FIRSTTOK_PROBE=<jsonl path>
    In generate_terminal_json, for the terminal Answerer call only, capture the
    first-step logits and append {top10, eos prob/rank, prompt head, output head}.
VLMAS_ANSWERER_DROP_CHOICES=1
    Answerer prompt is built with no CHOICE lines (open-question form) so the
    first-token distribution can be compared with / without the 30-line list.
"""
import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "."

# ---------------------------------------------------------------- terminal probe
p = os.path.join(root, "vision_text_mas/latent_terminal.py")
s = open(p).read()
old = '''    if no_repeat_ngram_size > 0:
        generation_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
    if repetition_penalty != 1.0:
        generation_kwargs["repetition_penalty"] = repetition_penalty
'''
new = '''    if no_repeat_ngram_size > 0:
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
'''
assert s.count(old) == 1, ("terminal anchor 1", s.count(old))
s = s.replace(old, new)

old = '''    fed = generation_kwargs["input_ids"]
    new_ids = generated[:, fed.shape[1] :]
    return backbone.processor.tokenizer.decode(
        new_ids[0],
        skip_special_tokens=True,
    ).strip()
'''
new = '''    fed = generation_kwargs["input_ids"]
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
            "n_choice_lines": (user_prompt or "").count("\\nCHOICE "),
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
                _f.write(_pjson.dumps(_rec, ensure_ascii=False) + "\\n")
        except OSError as _e:  # noqa: BLE001
            print(f"[FirstTokProbe] write failed: {_e!r}", flush=True)
    return _out_text
'''
assert s.count(old) == 1, ("terminal anchor 2", s.count(old))
s = s.replace(old, new)
open(p, "w").write(s)
print("terminal probe patched")

# ---------------------------------------------------------------- drop choices arm
p = os.path.join(root, "vision_text_mas/latent_answerer.py")
s = open(p).read()
old = '''        effective_choices = choices
        if self._canonical_open_options and not choices:
            effective_choices = canonical_open_answer_options(question)
'''
new = '''        effective_choices = choices
        if self._canonical_open_options and not choices:
            effective_choices = canonical_open_answer_options(question)
        # 0915 diagnostic arm: build the Answerer prompt WITHOUT the CHOICE list
        # (open-question form). Scoring/snapping still sees no choices, so the
        # arm is for first-token probing only.
        if os.environ.get("VLMAS_ANSWERER_DROP_CHOICES", "").strip() == "1":
            effective_choices = ()
'''
assert s.count(old) == 1, ("answerer anchor", s.count(old))
s = s.replace(old, new)
open(p, "w").write(s)
print("drop-choices arm patched")
