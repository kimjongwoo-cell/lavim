"""0915: remaining 0717-style differences, env-gated (off = byte-identical).

VLMAS_TERMINAL_EOS_STOP=1     Answerer JSON decode stops only at EOS (no '}' stop string), like
                              0717 generate_on_kv / upstream judger.
VLMAS_CHOICE_FORMAT=letters   Answerer choices rendered as 0717's "Choices:\nA. ...\nB. ..." block
                              with the 0717 instruction sentence.
VLMAS_CHOICE_SNAP=nearest     0731 select_choice_prediction behaviour: when choices exist and the
                              decoded answer matches none, commit to the difflib-nearest choice
                              instead of keeping the raw text.
"""
import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "."

# ------------------------------------------------------------------ engine: EOS-only stop
p = os.path.join(root, "vision_text_mas/latent_qwen_engine.py")
s = open(p).read()
old = '''        if grammar_processor is not None:
            # The grammar emits one complete flat object; its closing brace is
            # the whole-output terminator. The teacher-forced prefix must be
            # dropped — the matcher generates the object from "{" itself.
            stop_strings = ("}",)
'''
new = '''        if grammar_processor is not None:
            # The grammar emits one complete flat object; its closing brace is
            # the whole-output terminator. The teacher-forced prefix must be
            # dropped — the matcher generates the object from "{" itself.
            stop_strings = ("}",)
        if is_terminal_json and os.environ.get("VLMAS_TERMINAL_EOS_STOP", "").strip() == "1":
            # 0915 (0717-style): let the model end its own turn; no '}' stop string.
            stop_strings = ()
'''
assert s.count(old) == 1, ("engine stop anchor", s.count(old))
s = s.replace(old, new)
open(p, "w").write(s)
print("engine: VLMAS_TERMINAL_EOS_STOP added")

# ------------------------------------------------------------------ answerer: letters + snap
p = os.path.join(root, "vision_text_mas/latent_answerer.py")
s = open(p).read()
old = '''    if choices and os.environ.get("VLMAS_CHOICE_FORMAT", "").strip() == "list":
'''
new = '''    _choice_fmt = os.environ.get("VLMAS_CHOICE_FORMAT", "").strip()
    if choices and _choice_fmt == "letters":
        # 0915 A/B: 0717 / single-baseline rendering ("Choices:" + lettered lines).
        def _letter(i: int) -> str:
            return chr(65 + i) if i < 26 else chr(65 + i // 26 - 1) + chr(65 + i % 26)

        choice_lines = "Choices:\\n" + "\\n".join(
            f"{_letter(index)}. {choice}" for index, choice in enumerate(choices)
        )
    elif choices and _choice_fmt == "list":
'''
assert s.count(old) == 1, ("choice fmt anchor", s.count(old))
s = s.replace(old, new)

old = '''    if choices:
        choice_rule = (
            "When Choices are provided, the answer must be exactly one of the "
            "given choice texts.\\n"
        )
        answer_placeholder = "<exact choice text>"
'''
new = '''    if choices:
        choice_rule = (
            "For this multiple-choice question, put only the exact choice text in "
            "the answer field.\\n"
            if _choice_fmt == "letters"
            else "When Choices are provided, the answer must be exactly one of the "
            "given choice texts.\\n"
        )
        answer_placeholder = "<exact choice text>"
'''
assert s.count(old) == 1, ("choice rule anchor", s.count(old))
s = s.replace(old, new)

old = '''        answer_text = _normalize_choice(answer_text, effective_choices)
        if not answer_text and not _digit_repair_enabled():
'''
new = '''        answer_text = _normalize_choice(answer_text, effective_choices)
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
'''
assert s.count(old) == 1, ("snap anchor", s.count(old))
s = s.replace(old, new)
open(p, "w").write(s)
print("answerer: VLMAS_CHOICE_FORMAT=letters + VLMAS_CHOICE_SNAP=nearest added")
