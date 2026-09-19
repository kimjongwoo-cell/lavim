"""0917 VELH (C2 #14) + Text-cut engine wiring for the decode copy tree. md5-guarded; off = byte-identical.

Edits in vision_text_mas/latent_qwen_engine.py:
  append():  (1) VELH Reasoner capture + Text-cut stage probe next to the LU probe,
             (2) enter them with the other probes on both image-bearing branches,
             (3) VELH record after the Reasoner call returned (after LU),
             (4) Text-cut block bookkeeping after the cumulative turn close.
  terminal:  (5) VELH / Text-cut in-place contexts around generate_terminal_json, (6) summary prints.
"""
import hashlib, sys
p = sys.argv[1] if len(sys.argv) > 1 else "vision_text_mas/latent_qwen_engine.py"
src = open(p).read()
md5 = hashlib.md5(src.encode()).hexdigest()[:8]
assert md5 == "2a872768", f"engine md5 {md5} (expected 2a872768)"

CI = 'int(getattr(self, "_rpath_case_index", -1) or 0)'

# (1) probes
old = '''        _lu_r = _lu.reasoner_probe(self._backbone, m=latent_steps, stage=stage, targets=attention_targets)
'''
new = old + '''        # VELH (memory/velh.py, VLMAS_VELH): anchor query / cos-sin rows / latent bookkeeping of the Reasoner.
        from memory import velh as _velh
        _velh_r = _velh.reasoner_probe(self._backbone, m=latent_steps, stage=stage, targets=attention_targets)
        # Text-cut (memory/textcut.py, VLMAS_TEXTCUT): token ids / positions of every role's prefill rows.
        from memory import textcut as _tc
        if _tc.enabled() and cache is None:
            _tc.reset(self)
        _tc_r = _tc.stage_probe(self._backbone, stage)
'''
assert src.count(old) == 1, ("lu probe anchor", src.count(old))
src = src.replace(old, new)

# (2) with-lists
old = '''            with _lpvh_r, _srvw_r, _relh_r, _lu_r:
'''
new = '''            with _lpvh_r, _srvw_r, _relh_r, _lu_r, _velh_r, _tc_r:
'''
assert src.count(old) == 2, ("with anchor", src.count(old))
src = src.replace(old, new)

# (3) VELH record after LU
old = '''            _lu.apply(self, result, m=latent_steps,
                      case_index=int(getattr(self, "_rpath_case_index", -1) or 0))
'''
new = old + '''        if _velh.enabled() and stage == "reasoner":
            _velh.record_reasoner(self, result, m=latent_steps, case_index=''' + CI + ''')
'''
assert src.count(old) == 1, ("lu apply anchor", src.count(old))
src = src.replace(old, new)

# (4) Text-cut bookkeeping after the cumulative turn close
old = '''                relay_rows = relay_length(boundary_relay)
                append_relay(closed_cache, boundary_relay)
'''
new = '''                if _tc.enabled():
                    _tc.record_stage(
                        self, stage=stage, start=int(previous_cache_length), past_len=int(full_cache_length),
                        latent_steps=int(latent_steps), closed_length=int(closed_length),
                        vis_cols=vision_columns if isinstance(vision_columns, torch.Tensor) else None,
                        close_thinking=bool(keep_thinking_open), pos_cursor=int(result["pos_cursor"]))
''' + old
assert src.count(old) == 1, ("close anchor", src.count(old))
src = src.replace(old, new)

# (5) terminal contexts
old = '''        try:
            with context as realloc_probe, _acc_terminal, _pfr_ctx, _vcut_ctx, _gc2_ctx, \\
'''
new = '''        # VELH (C2 #14) / Text-cut: in-place changes on the Answerer's cache + attention-mass measurement.
        _velh_ctx = contextlib.nullcontext()
        _velh_t = None
        if is_terminal and os.environ.get("VLMAS_VELH", "").strip():
            from memory import velh as _velh_t
            _velh_ctx = _velh_t.terminal(self, cache, int(position_cursor), case_index=''' + CI + ''')
        _tc_ctx = contextlib.nullcontext()
        _tc_t = None
        if is_terminal and os.environ.get("VLMAS_TEXTCUT", "").strip():
            from memory import textcut as _tc_t
            _tc_ctx = _tc_t.terminal(self, cache, int(position_cursor), case_index=''' + CI + ''')
''' + old
assert src.count(old) == 1, ("terminal try anchor", src.count(old))
src = src.replace(old, new)

old = '''                    _relh_ctx as _relh_stats:
                output = generate_terminal_json(
'''
new = '''                    _relh_ctx as _relh_stats, \\
                    _velh_ctx as _velh_stats, _tc_ctx as _tc_stats:
                output = generate_terminal_json(
'''
assert src.count(old) == 1, ("terminal with anchor", src.count(old))
src = src.replace(old, new)

# (6) summaries
old = '''        if _relh_a is not None and _relh_stats is not None:
            print(f"[RELH] case {getattr(self, '_rpath_case_index', -1)} " + _relh_a.summary(_relh_stats), flush=True)
'''
new = old + '''        if _velh_t is not None and _velh_stats is not None:
            print(f"[VELH] case {getattr(self, '_rpath_case_index', -1)} " + _velh_t.summary(_velh_stats), flush=True)
        if _tc_t is not None and _tc_stats is not None:
            print(f"[TextCut] case {getattr(self, '_rpath_case_index', -1)} " + _tc_t.summary(_tc_stats), flush=True)
'''
assert src.count(old) == 1, ("relh print anchor", src.count(old))
src = src.replace(old, new)

open(p, "w").write(src)
print("patched", p, "->", hashlib.md5(src.encode()).hexdigest()[:8])
