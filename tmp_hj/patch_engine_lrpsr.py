"""0916 LR-PSR (C2 12.6) engine wiring for the decode copy tree. md5-guarded; off = byte-identical."""
import hashlib, shutil, sys
p = sys.argv[1] if len(sys.argv) > 1 else "vision_text_mas/latent_qwen_engine.py"
src = open(p).read()
md5 = hashlib.md5(src.encode()).hexdigest()[:8]
assert md5 == "5636f516", f"engine md5 {md5} (expected 5636f516)"

anchor = '''        # C2 Latent-Provenance Visual Handoff (VLMAS_LPVH=1|identity, memory/lpvh.py): compose the
'''
new = '''        # C2 12.6 Latent-Referenced Physical-Support Residual Readout (VLMAS_LRPSR=1|identity,
        # memory/lrpsr.py): at the fixed formation layers, keep the native visual component that
        # lies in the span of the Reasoner latent values and rebuild only the complementary
        # component from the per-physical-support reads, with the native norms conserved.
        _lrpsr_ctx = contextlib.nullcontext()
        _lrpsr = None
        if is_terminal and os.environ.get("VLMAS_LRPSR", "").strip():
            from memory import lrpsr as _lrpsr
            _lcfg = _lrpsr.config()
            _lg = _lrpsr.engine_groups(self, _lcfg["support"])
            if _lg["groups"] is None:
                print(f"[LRPSR] SKIP: {_lg['note']}", flush=True)
            else:
                _llayers = _lrpsr.layer_set(_lcfg["layers"], len(self._backbone.lm.layers))
                _lrpsr_ctx = _lrpsr.install_lrpsr(
                    self._backbone, _lg["cols"].to(self._backbone.device),
                    _lg["latent_cols"].to(self._backbone.device), _lg["groups"],
                    layers=_llayers, rows=_lcfg["rows"],
                    identity=(_lcfg["mode"] == "identity"))
                print(f"[LRPSR] mode={_lcfg['mode']} support={_lcfg['support']} layers={list(_llayers)} "
                      f"rows={_lcfg['rows']} n_vis={int(_lg['cols'].numel())} "
                      f"n_latent={int(_lg['latent_cols'].numel())} supports={len(_lg['groups'])} "
                      f"sizes={[len(g) for g in _lg['groups']]} src={_lg['note']}", flush=True)
''' + anchor
assert src.count(anchor) == 1, ("lpvh comment anchor", src.count(anchor))
src = src.replace(anchor, new)

old = "                    _psar_ctx as _psar_stats, _lpvh_ctx as _lpvh_stats, _srvw_ctx as _srvw_stats:"
new = "                    _psar_ctx as _psar_stats, _lpvh_ctx as _lpvh_stats, _srvw_ctx as _srvw_stats, \\\n                    _lrpsr_ctx as _lrpsr_stats:"
assert src.count(old) == 1, ("with anchor", src.count(old))
src = src.replace(old, new)

old = '''        if _psar_stats is not None:
'''
new = '''        if _lrpsr is not None and _lrpsr_stats is not None:
            print(f"[LRPSR] case {getattr(self, '_rpath_case_index', -1)} " + _lrpsr.summary(_lrpsr_stats), flush=True)
''' + old
assert src.count(old) == 1, ("summary anchor", src.count(old))
src = src.replace(old, new)

shutil.copy(p, p + ".bak_0916_lrpsr")
open(p, "w").write(src)
print("engine patched", hashlib.md5(src.encode()).hexdigest()[:8])
