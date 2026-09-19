"""0915 LatentMAS-faithful think-tag handling (env-gated, off = byte-identical).

VLMAS_THINK_TAGS=strip  upstream LatentMAS default for Qwen3-VL-Thinking (--disable_chat_thinking):
                        the template's trailing "<think>\\n" is REMOVED (no think tags at all) for
                        every latent stage and for the final Answerer prompt.
VLMAS_THINK_TAGS=open   upstream LatentMAS --think: "<think>\\n" is left OPEN and never closed, for
                        every latent stage and for the final Answerer prompt (the model thinks,
                        closes </think> itself, then answers).
unset                   current behaviour (append "\\n</think>\\n\\n" when thinking is disabled).
"""
import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "."

p = os.path.join(root, "backbone/qwen3vl.py")
s = open(p).read()
sites = [
    (
        '''            # Thinking model always appends <think>\\n; close it if thinking is disabled
            if not enable_thinking:
                prompt = prompt + "\\n</think>\\n\\n"
''',
        '''            # Thinking model always appends <think>\\n; close it if thinking is disabled
            _tt = os.environ.get("VLMAS_THINK_TAGS", "").strip()
            if _tt == "strip" and prompt.endswith("<think>\\n"):
                prompt = prompt[: -len("<think>\\n")]
            elif _tt == "open":
                pass
            elif not enable_thinking:
                prompt = prompt + "\\n</think>\\n\\n"
''',
    ),
    (
        '''        if not self.instruct and not enable_thinking:
            text = text + "\\n</think>\\n\\n"   # match embed_text convention
''',
        '''        _tt = os.environ.get("VLMAS_THINK_TAGS", "").strip()
        if _tt == "strip" and text.endswith("<think>\\n"):
            text = text[: -len("<think>\\n")]
        elif _tt == "open":
            pass
        elif not self.instruct and not enable_thinking:
            text = text + "\\n</think>\\n\\n"   # match embed_text convention
''',
    ),
    (
        '''        if not self.instruct and not enable_thinking:
            text = text + "\\n</think>\\n\\n"
        inputs = self.processor(text=[text], images=images, return_tensors="pt").to(self.device)
''',
        '''        _tt = os.environ.get("VLMAS_THINK_TAGS", "").strip()
        if _tt == "strip" and text.endswith("<think>\\n"):
            text = text[: -len("<think>\\n")]
        elif _tt == "open":
            pass
        elif not self.instruct and not enable_thinking:
            text = text + "\\n</think>\\n\\n"
        inputs = self.processor(text=[text], images=images, return_tensors="pt").to(self.device)
''',
    ),
]
for i, (old, new) in enumerate(sites):
    assert s.count(old) == 1, (f"backbone site {i}", s.count(old))
    s = s.replace(old, new)
if "\nimport os\n" not in s[:4000]:
    s = "import os\n" + s
open(p, "w").write(s)
print("backbone: 3 latent-stage think-tag sites gated")

p = os.path.join(root, "vision_text_mas/latent_terminal.py")
s = open(p).read()
old = '''    if prompt.endswith("<think>\\n"):
        prompt += "</think>\\n\\n"
    return prompt if json_prefix is None else prompt + json_prefix
'''
new = '''    import os as _ttos

    _tt = _ttos.environ.get("VLMAS_THINK_TAGS", "").strip()
    if _tt == "strip" and prompt.endswith("<think>\\n"):
        prompt = prompt[: -len("<think>\\n")]
    elif _tt == "open":
        pass
    elif prompt.endswith("<think>\\n"):
        prompt += "</think>\\n\\n"
    return prompt if json_prefix is None else prompt + json_prefix
'''
assert s.count(old) == 1, ("terminal assistant prompt", s.count(old))
s = s.replace(old, new)
open(p, "w").write(s)
print("terminal: Answerer think-tag handling gated")
