"""0917 Latent Unsilencing (LU, Phase 1) engine wiring for the decode copy tree. md5-guarded; off = byte-identical.

Three edits in vision_text_mas/latent_qwen_engine.py `append()`:
  (1) build the Reasoner capture context next to the SRVW probe,
  (2) enter it together with the LPVH/SRVW/RELH contexts on both image-bearing branches,
  (3) right after `result` exists (before relay extraction / turn close): optimise Z + regenerate the latent KV.
"""
import hashlib, sys
p = sys.argv[1] if len(sys.argv) > 1 else "vision_text_mas/latent_qwen_engine.py"
src = open(p).read()
md5 = hashlib.md5(src.encode()).hexdigest()[:8]
assert md5 == "f35c7e74", f"engine md5 {md5} (expected f35c7e74)"

# (1) capture context (nullcontext unless VLMAS_LU is set and stage == reasoner)
old = '''        _srvw_r = _srvw.reasoner_probe(self._backbone, m=latent_steps, stage=stage)
'''
new = old + '''        # Latent Unsilencing (memory/lu.py, VLMAS_LU): capture the Reasoner prefill's question->visual
        # attention + input embeddings and the latent-step bookkeeping. Off / other stage = nullcontext.
        from memory import lu as _lu
        _lu_r = _lu.reasoner_probe(self._backbone, m=latent_steps, stage=stage, targets=attention_targets)
'''
assert src.count(old) == 1, ("srvw anchor", src.count(old))
src = src.replace(old, new)

# (2) enter it on both image-bearing branches
old = '''            with _lpvh_r, _srvw_r, _relh_r:
'''
new = '''            with _lpvh_r, _srvw_r, _relh_r, _lu_r:
'''
assert src.count(old) == 2, ("with anchor", src.count(old))
src = src.replace(old, new)

# (3) optimise + regenerate right after the Reasoner call returned (before relay / close)
old = '''        if (
            config is not None
            and (
                config.visual_grounded_latent_kv_relay
                or config.visual_contrast_latent_kv_relay
            )
            and stage != "reasoner"
'''
new = '''        if _lu.enabled() and stage == "reasoner":
            # Latent Unsilencing: Z = native latent input embeddings -> Stage I/II -> crop + replay once.
            _lu.apply(self, result, m=latent_steps,
                      case_index=int(getattr(self, "_rpath_case_index", -1) or 0))
''' + old
assert src.count(old) == 1, ("relay anchor", src.count(old))
src = src.replace(old, new)

open(p, "w").write(src)
print("patched", p, "->", hashlib.md5(src.encode()).hexdigest()[:8])
