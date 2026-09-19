"""0916 PSFP (C2 12.7) engine wiring for the decode copy tree. md5-guarded; off = byte-identical."""
import hashlib, shutil, sys
p = sys.argv[1] if len(sys.argv) > 1 else "vision_text_mas/latent_qwen_engine.py"
src = open(p).read()
md5 = hashlib.md5(src.encode()).hexdigest()[:8]
assert md5 == "6c579960", f"engine md5 {md5} (expected 6c579960)"

anchor = '''        # C2 Latent-Provenance Visual Handoff (VLMAS_LPVH=1|identity, memory/lpvh.py): compose the
'''
new = '''        # C2 12.7 Physical-Support Fisher Projection (VLMAS_PSFP=1|identity|diag, memory/psfp.py):
        # read the physical-support visual direction at the fixed formation layers, map it and the
        # native visual term to direct logits, and project the decision remainder in the native
        # softmax Fisher metric only when it opposes that direction. Logits only; no hidden state.
        _psfp_ctx = contextlib.nullcontext()
        _psfp = None
        if is_terminal and os.environ.get("VLMAS_PSFP", "").strip():
            from memory import psfp as _psfp
            _fcfg = _psfp.config()
            _fg = _psfp.engine_groups(self, _fcfg["support"])
            if _fg["groups"] is None:
                print(f"[PSFP] SKIP: {_fg['note']}", flush=True)
            else:
                _flayers = _psfp.layer_set(_fcfg["layers"], len(self._backbone.lm.layers))
                _psfp_ctx = _psfp.install_psfp(
                    self._backbone, _fg["cols"].to(self._backbone.device), _fg["groups"],
                    layers=_flayers, rows=_fcfg["rows"], apply=_psfp.active())
                print(f"[PSFP] mode={_fcfg['mode']} support={_fcfg['support']} layers={list(_flayers)} "
                      f"rows={_fcfg['rows']} n_vis={int(_fg['cols'].numel())} "
                      f"supports={len(_fg['groups'])} sizes={[len(g) for g in _fg['groups']]} "
                      f"apply={_psfp.active()} src={_fg['note']}", flush=True)
''' + anchor
assert src.count(anchor) == 1, ("anchor", src.count(anchor))
src = src.replace(anchor, new)

old = "                    _lrpsr_ctx as _lrpsr_stats:"
new = "                    _lrpsr_ctx as _lrpsr_stats, _psfp_ctx as _psfp_stats:"
assert src.count(old) == 1, ("with anchor", src.count(old))
src = src.replace(old, new)

old = '''        if _lrpsr is not None and _lrpsr_stats is not None:
'''
new = '''        if _psfp is not None and _psfp_stats is not None:
            print(f"[PSFP] case {getattr(self, '_rpath_case_index', -1)} " + _psfp.summary(_psfp_stats), flush=True)
''' + old
assert src.count(old) == 1, ("summary anchor", src.count(old))
src = src.replace(old, new)

shutil.copy(p, p + ".bak_0916_psfp")
open(p, "w").write(src)
print("engine patched", hashlib.md5(src.encode()).hexdigest()[:8])
