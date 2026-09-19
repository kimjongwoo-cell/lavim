"""0915 Answerer choice rendering A/B (env-gated, off = byte-identical).

VLMAS_CHOICE_FORMAT=list  -> one line  Choice: ["A", "B", ...]  (the dataset's own list form)
unset / anything else     -> the existing  CHOICE 1: A / CHOICE 2: B ... lines
"""
import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "."
p = os.path.join(root, "vision_text_mas/latent_answerer.py")
s = open(p).read()
old = '''    choice_lines = "\\n".join(
        f"CHOICE {index}: {choice}"
        for index, choice in enumerate(choices, start=1)
    )
'''
new = '''    if choices and os.environ.get("VLMAS_CHOICE_FORMAT", "").strip() == "list":
        # 0915 A/B: hand the dataset's list over verbatim instead of 30 CHOICE lines.
        import json as _json

        choice_lines = "Choice: " + _json.dumps(list(choices), ensure_ascii=False)
    else:
        choice_lines = "\\n".join(
            f"CHOICE {index}: {choice}"
            for index, choice in enumerate(choices, start=1)
        )
'''
assert s.count(old) == 1, ("choice_lines anchor", s.count(old))
s = s.replace(old, new)
open(p, "w").write(s)
print("answerer: VLMAS_CHOICE_FORMAT=list added")
