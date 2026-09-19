"""0917 RN-LCR (C2 #15) engine wiring for the decode copy tree. md5-guarded; off = byte-identical.

Edits in vision_text_mas/latent_qwen_engine.py:
  append():  (1) Reasoner capture probe next to the Text-cut probe, (2) enter it with the other probes,
             (3) record / ground + replay right after the VELH record (after LU),
  terminal:  (4) Step-3 hook context around generate_terminal_json, (5) summary print.
"""
import hashlib, sys
p = sys.argv[1] if len(sys.argv) > 1 else "vision_text_mas/latent_qwen_engine.py"
src = open(p).read()
md5 = hashlib.md5(src.encode()).hexdigest()[:8]
assert md5 == "9ac3bfc1", f"engine md5 {md5} (expected 9ac3bfc1)"
CI = 'int(getattr(self, "_rpath_case_index", -1) or 0)'

old = '''        _tc_r = _tc.stage_probe(self._backbone, stage)
'''
new = old + '''        # RN-LCR (memory/rnlcr.py, VLMAS_RNLCR): question->visual relevance + prefill embeddings for grounding.
        from memory import rnlcr as _rn
        _rn_r = _rn.reasoner_probe(self._backbone, m=latent_steps, stage=stage, targets=attention_targets)
'''
assert src.count(old) == 1, ("tc probe anchor", src.count(old))
src = src.replace(old, new)

old = '''            with _lpvh_r, _srvw_r, _relh_r, _lu_r, _velh_r, _tc_r:
'''
new = '''            with _lpvh_r, _srvw_r, _relh_r, _lu_r, _velh_r, _tc_r, _rn_r:
'''
assert src.count(old) == 2, ("with anchor", src.count(old))
src = src.replace(old, new)

old = '''        if _velh.enabled() and stage == "reasoner":
            _velh.record_reasoner(self, result, m=latent_steps, case_index=''' + CI + ''')
'''
new = old + '''        if _rn.enabled() and stage == "reasoner":
            # RN-LCR: record the latent columns; ground/full also ground the trajectory and replay it once.
            _rn.apply_reasoner(self, result, m=latent_steps, case_index=''' + CI + ''')
'''
assert src.count(old) == 1, ("velh record anchor", src.count(old))
src = src.replace(old, new)

old = '''        _tc_ctx = contextlib.nullcontext()
        _tc_t = None
        if is_terminal and os.environ.get("VLMAS_TEXTCUT", "").strip():
            from memory import textcut as _tc_t
            _tc_ctx = _tc_t.terminal(self, cache, int(position_cursor), case_index=''' + CI + ''')
'''
new = old + '''        _rn_ctx = contextlib.nullcontext()
        _rn_t = None
        if is_terminal and os.environ.get("VLMAS_RNLCR", "").strip():
            from memory import rnlcr as _rn_t
            _rn_ctx = _rn_t.terminal(self, cache, int(position_cursor), case_index=''' + CI + ''')
'''
assert src.count(old) == 1, ("tc terminal anchor", src.count(old))
src = src.replace(old, new)

old = '''                    _velh_ctx as _velh_stats, _tc_ctx as _tc_stats:
'''
new = '''                    _velh_ctx as _velh_stats, _tc_ctx as _tc_stats, _rn_ctx as _rn_stats:
'''
assert src.count(old) == 1, ("terminal with anchor", src.count(old))
src = src.replace(old, new)

old = '''        if _tc_t is not None and _tc_stats is not None:
            print(f"[TextCut] case {getattr(self, '_rpath_case_index', -1)} " + _tc_t.summary(_tc_stats), flush=True)
'''
new = old + '''        if _rn_t is not None and _rn_stats is not None:
            print(f"[RNLCR] case {getattr(self, '_rpath_case_index', -1)} " + _rn_t.summary(_rn_stats), flush=True)
'''
assert src.count(old) == 1, ("tc print anchor", src.count(old))
src = src.replace(old, new)

open(p, "w").write(src)
print("patched", p, "->", hashlib.md5(src.encode()).hexdigest()[:8])
