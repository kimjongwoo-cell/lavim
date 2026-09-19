"""0915 LatentMAS-faithful framing steps (env-gated, off = byte-identical).

VLMAS_NO_TURN_CLOSE=1   step 2: do not append </think>/<|im_end|> after a role's latent block
                        (upstream LatentMAS appends nothing; the next role's prompt follows directly).
VLMAS_NO_CONTINUATION=1 step 3: later roles get a full system+user chat turn instead of the
                        "Next-stage role instructions:" continuation user turn (upstream re-renders
                        system+user for every agent).
"""
import os
import re
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "."
p = os.path.join(root, "vision_text_mas/latent_qwen_engine.py")
s = open(p).read()

old = '''                closed_cache, closed_length, closed_cursor = (
                    self._backbone.close_assistant_turn(
                        result["past_key_values"],
                        int(result["pos_cursor"]),
                        close_thinking=keep_thinking_open,
                    )
                )
'''
new = '''                if os.environ.get("VLMAS_NO_TURN_CLOSE", "").strip() == "1":
                    # 0915 step 2 (LatentMAS-faithful): append NOTHING after the latent
                    # block; the next role's prompt is prefilled directly behind it.
                    closed_cache = result["past_key_values"]
                    closed_length = int(full_cache_length)
                    closed_cursor = int(result["pos_cursor"])
                else:
                    closed_cache, closed_length, closed_cursor = (
                        self._backbone.close_assistant_turn(
                            result["past_key_values"],
                            int(result["pos_cursor"]),
                            close_thinking=keep_thinking_open,
                        )
                    )
'''
assert s.count(old) == 1, ("close anchor", s.count(old))
s = s.replace(old, new)

pat = re.compile(
    r'continuation_user_turn=self\._transport_mode\s*\n\s*in \(([^)]*)\),'
)
hits = pat.findall(s)
assert len(hits) == 2, ("continuation anchors", len(hits))
s = pat.sub(
    lambda m: (
        'continuation_user_turn=(\n'
        '                    os.environ.get("VLMAS_NO_CONTINUATION", "").strip() != "1"\n'
        '                    and self._transport_mode in (' + m.group(1) + ')\n'
        '                ),'
    ),
    s,
)
open(p, "w").write(s)
print("engine: step2 (no turn close) + step3 (no continuation) gates added")
