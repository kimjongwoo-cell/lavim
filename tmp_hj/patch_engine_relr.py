"""0917 RELR: pass the Answerer prompt to the RN-LCR terminal context (mode relr needs a teacher prefill). md5-guarded."""
import hashlib, sys
p = sys.argv[1] if len(sys.argv) > 1 else "vision_text_mas/latent_qwen_engine.py"
src = open(p).read()
md5 = hashlib.md5(src.encode()).hexdigest()[:8]
assert md5 == "005cd7b3", f"engine md5 {md5} (expected 005cd7b3)"
old = '''            _rn_ctx = _rn_t.terminal(self, cache, int(position_cursor), case_index=int(getattr(self, "_rpath_case_index", -1) or 0))
'''
new = '''            _rn_ctx = _rn_t.terminal(self, cache, int(position_cursor), case_index=int(getattr(self, "_rpath_case_index", -1) or 0),
                                     system_prompt=system_prompt, user_prompt=user_prompt,
                                     json_prefix=(None if grammar_processor is not None else _eff_prefix))
'''
assert src.count(old) == 1, src.count(old)
src = src.replace(old, new)
open(p, "w").write(src)
print("patched", p, "->", hashlib.md5(src.encode()).hexdigest()[:8])
