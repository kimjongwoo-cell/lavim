"""0916 RELH (C2 13) engine wiring for the decode copy tree. md5-guarded; off = byte-identical."""
import hashlib, shutil, sys
p = sys.argv[1] if len(sys.argv) > 1 else "vision_text_mas/latent_qwen_engine.py"
src = open(p).read()
md5 = hashlib.md5(src.encode()).hexdigest()[:8]
assert md5 == "8af14fc0", f"engine md5 {md5} (expected 8af14fc0)"

# (1) record each sender role's terminal latent column right after its append
old = '''        full_cache_length = int(result["past_len"])
'''
new = '''        full_cache_length = int(result["past_len"])
        # C2 13 RELH (memory/relh.py): remember this role's terminal latent column + coordinate.
        from memory import relh as _relh
        if _relh.enabled():
            _relh.record_endpoint(self._backbone, stage, full_cache_length,
                                  int(result["pos_cursor"]))
'''
assert src.count(old) == 1, ("past_len anchor", src.count(old))
src = src.replace(old, new)

# (2) consumer = Reasoner: re-address the Navigator endpoint during its prefill + latent steps
old = '''        from memory import rspa as _rspa
'''
if old in src:
    raise SystemExit("unexpected rspa import in the decode tree")
old = '''        from memory import prov_gate as _pgate
'''
anchor = None
for cand in ('''        from memory import srvw as _srvw
        _srvw_r = _srvw.reasoner_probe(self._backbone, m=latent_steps, stage=stage)
''',):
    if cand in src:
        anchor = cand
assert anchor is not None, "srvw reasoner probe anchor not found"
new = anchor + '''        # C2 13 RELH consumer hook: the Reasoner reads the Navigator endpoint at its own boundary.
        _relh_r = contextlib.nullcontext()
        if _relh.enabled() and _relh.active_at(stage) and stage == "reasoner":
            _rcfg = _relh.config()
            _reps = _relh.endpoints_for(self._backbone, "reasoner")
            if _reps:
                _relh_r = _relh.install_relh(
                    self._backbone, _reps, int(position_cursor) - 1, rows=_rcfg["rows"],
                    identity=(_rcfg["mode"] == "identity"))
                print(f"[RELH:reasoner] mode={_rcfg['mode']} endpoints={_reps} "
                      f"handoff={int(position_cursor) - 1}", flush=True)
'''
src = src.replace(anchor, new)

old = "            with _lpvh_r, _srvw_r, _ocld_r:\n"
if old in src:
    raise SystemExit("unexpected ocld context in the decode tree")
old = "            with _lpvh_r, _srvw_r:\n"
assert src.count(old) == 2, ("reasoner with anchor", src.count(old))
src = src.replace(old, "            with _lpvh_r, _srvw_r, _relh_r:\n")

# (3) consumer = Answerer: re-address the Reasoner endpoint at the terminal
anchor = '''        # C2 12.8 Latent-Anchored Dual-Frame Read (VLMAS_LADR=1|identity, memory/ladr.py): keep the
'''
new = '''        # C2 13 Role-Endpoint Latent Handoff (VLMAS_RELH, memory/relh.py): the Answerer reads the
        # Reasoner's terminal latent column at its own role boundary coordinate. One cached column,
        # key phase only, no new column, native values.
        _relh_ctx = contextlib.nullcontext()
        _relh_a = None
        if is_terminal and os.environ.get("VLMAS_RELH", "").strip():
            from memory import relh as _relh_a
            _acfg = _relh_a.config()
            _aeps = _relh_a.endpoints_for(self._backbone, "answerer")
            if not _aeps or not _relh_a.active_at("answerer"):
                print(f"[RELH] SKIP: endpoints={_aeps} mode={_acfg['mode']}", flush=True)
            else:
                _relh_ctx = _relh_a.install_relh(
                    self._backbone, _aeps, int(position_cursor) - 1, rows=_acfg["rows"],
                    identity=(_acfg["mode"] == "identity"))
                print(f"[RELH] mode={_acfg['mode']} endpoints={_aeps} "
                      f"handoff={int(position_cursor) - 1} rows={_acfg['rows']}", flush=True)
''' + anchor
assert src.count(anchor) == 1, ("ladr comment anchor", src.count(anchor))
src = src.replace(anchor, new)

old = "                    _lrpsr_ctx as _lrpsr_stats, _psfp_ctx as _psfp_stats, \\\n                    _ladr_ctx as _ladr_stats:"
assert src.count(old) == 1, ("terminal with anchor", src.count(old))
src = src.replace(old, old[:-1] + ", \\\n                    _relh_ctx as _relh_stats:")

old = '''        if _ladr is not None and _ladr_stats is not None:
'''
new = '''        if _relh_a is not None and _relh_stats is not None:
            print(f"[RELH] case {getattr(self, '_rpath_case_index', -1)} " + _relh_a.summary(_relh_stats), flush=True)
''' + old
assert src.count(old) == 1, ("summary anchor", src.count(old))
src = src.replace(old, new)

shutil.copy(p, p + ".bak_0916_relh")
open(p, "w").write(src)
print("engine patched", hashlib.md5(src.encode()).hexdigest()[:8])
