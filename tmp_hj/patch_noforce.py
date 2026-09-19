"""0915 VLMAS_TERMINAL_NO_FORCE opt-in: drop the teacher-forced '{"answer":' opener.

Off (env unset) = byte-identical behaviour. Run from the tree root.
"""
import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "."

p = os.path.join(root, "vision_text_mas/latent_qwen_engine.py")
s = open(p).read()
old = '''        self._backbone._route_terminal_call = bool(is_terminal)
        try:
            with context as realloc_probe, _acc_terminal, _pfr_ctx, _vcut_ctx, _gc2_ctx, \\
                    _psar_ctx as _psar_stats, _lpvh_ctx as _lpvh_stats, _srvw_ctx as _srvw_stats:
                output = generate_terminal_json(
                    backbone=self._backbone,
                    cache=cache,
                    position_cursor=position_cursor,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    json_prefix=(
                        None if grammar_processor is not None else json_prefix
                    ),'''
new = '''        self._backbone._route_terminal_call = bool(is_terminal)
        # VLMAS_TERMINAL_NO_FORCE=1 (0915, off = byte-identical): do NOT teacher-force
        # the '{"answer":' opener. The assistant turn starts empty (0717-style free
        # generation) and the model writes the whole object itself; the '}' stop and
        # every is_terminal-gated hook still key off the original json_prefix.
        _eff_prefix = json_prefix
        if is_terminal_json and os.environ.get("VLMAS_TERMINAL_NO_FORCE", "").strip() == "1":
            _eff_prefix = ""
        try:
            with context as realloc_probe, _acc_terminal, _pfr_ctx, _vcut_ctx, _gc2_ctx, \\
                    _psar_ctx as _psar_stats, _lpvh_ctx as _lpvh_stats, _srvw_ctx as _srvw_stats:
                output = generate_terminal_json(
                    backbone=self._backbone,
                    cache=cache,
                    position_cursor=position_cursor,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    json_prefix=(
                        None if grammar_processor is not None else _eff_prefix
                    ),'''
assert s.count(old) == 1, ("engine anchor", s.count(old))
s = s.replace(old, new)
open(p, "w").write(s)
print("engine patched")

p = os.path.join(root, "vision_text_mas/latent_answerer.py")
s = open(p).read()
old = '''            if not raw.lstrip().startswith("{"):
                prefix = (
                    STRUCTURED_JSON_TERMINAL_PREFIX
                    if self._protocol is AnswererProtocol.STRUCTURED_JSON
                    else JSON_TERMINAL_PREFIX
                )
                raw = prefix + raw'''
new = '''            # VLMAS_TERMINAL_NO_FORCE=1 (0915): nothing was teacher-forced, so the raw
            # output is the model's own text; never prepend the opener.
            if not raw.lstrip().startswith("{") and os.environ.get(
                "VLMAS_TERMINAL_NO_FORCE", ""
            ).strip() != "1":
                prefix = (
                    STRUCTURED_JSON_TERMINAL_PREFIX
                    if self._protocol is AnswererProtocol.STRUCTURED_JSON
                    else JSON_TERMINAL_PREFIX
                )
                raw = prefix + raw'''
assert s.count(old) == 1, ("answerer anchor 1", s.count(old))
s = s.replace(old, new)
old = '''            answer_text = (
                (parsed_answer.strip() if isinstance(parsed_answer, str) else "")
                or _extract_json_answer(raw)
                or _extract_answer_line(raw)
            )
        else:'''
new = '''            answer_text = (
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
        else:'''
assert s.count(old) == 1, ("answerer anchor 2", s.count(old))
s = s.replace(old, new)
open(p, "w").write(s)
print("answerer patched")
