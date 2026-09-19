"""0916 LADR (C2 12.8) wiring for the decode copy tree. md5-guarded; off = byte-identical."""
import hashlib, shutil, sys

# ---------------------------------------------------------------- backbone: record visual MRoPE positions
p = "backbone/qwen3vl.py"
src = open(p).read()
md5 = hashlib.md5(src.encode()).hexdigest()[:8]
assert md5 == "2e524853", f"backbone md5 {md5} (expected 2e524853)"
old = '''        if os.environ.get("VLMAS_KV_RESTAGE", "") == "1":
            self._restage_vis_positions = pos[:, 0, vis_idx].detach().clone()
'''
new = '''        # LADR (C2 12.8, memory/ladr.py) reads the same provenance to build its role-local frame.
        if os.environ.get("VLMAS_KV_RESTAGE", "") == "1" or os.environ.get("VLMAS_LADR", "").strip():
            self._restage_vis_positions = pos[:, 0, vis_idx].detach().clone()
'''
assert src.count(old) == 1, ("backbone anchor", src.count(old))
src = src.replace(old, new)
shutil.copy(p, p + ".bak_0916_ladr")
open(p, "w").write(src)
print("backbone patched", hashlib.md5(src.encode()).hexdigest()[:8])

# ---------------------------------------------------------------- engine: install at the terminal
p = "vision_text_mas/latent_qwen_engine.py"
src = open(p).read()
md5 = hashlib.md5(src.encode()).hexdigest()[:8]
assert md5 == "cb6d8ad4", f"engine md5 {md5} (expected cb6d8ad4)"
anchor = '''        # C2 Latent-Provenance Visual Handoff (VLMAS_LPVH=1|identity, memory/lpvh.py): compose the
'''
new = '''        # C2 12.8 Latent-Anchored Dual-Frame Read (VLMAS_LADR=1|identity, memory/ladr.py): keep the
        # visual columns' acquisition MRoPE address AND a role-local address re-anchored at the
        # Reasoner boundary, and marginalise the two address scores (log-mean-exp) inside every
        # Answerer attention. Same K/V payload, value read once, no layer choice.
        _ladr_ctx = contextlib.nullcontext()
        _ladr = None
        if is_terminal and os.environ.get("VLMAS_LADR", "").strip():
            from memory import ladr as _ladr
            _dcfg = _ladr.config()
            _dg = _ladr.engine_state(self)
            if _dg["cols"] is None:
                print(f"[LADR] SKIP: {_dg['note']}", flush=True)
            else:
                _ladr_ctx = _ladr.install_ladr(
                    self._backbone, _dg["cols"].to(self._backbone.device), _dg["positions"],
                    position_cursor, rows=_dcfg["rows"],
                    identity=(_dcfg["mode"] == "identity"))
                print(f"[LADR] mode={_dcfg['mode']} rows={_dcfg['rows']} "
                      f"n_vis={int(_dg['cols'].numel())} cursor={position_cursor} "
                      f"src={_dg['note']}", flush=True)
''' + anchor
assert src.count(anchor) == 1, ("engine anchor", src.count(anchor))
src = src.replace(anchor, new)

old = "                    _lrpsr_ctx as _lrpsr_stats, _psfp_ctx as _psfp_stats:"
new = "                    _lrpsr_ctx as _lrpsr_stats, _psfp_ctx as _psfp_stats, \\\n                    _ladr_ctx as _ladr_stats:"
assert src.count(old) == 1, ("with anchor", src.count(old))
src = src.replace(old, new)

old = '''        if _psfp is not None and _psfp_stats is not None:
'''
new = '''        if _ladr is not None and _ladr_stats is not None:
            print(f"[LADR] case {getattr(self, '_rpath_case_index', -1)} " + _ladr.summary(_ladr_stats), flush=True)
''' + old
assert src.count(old) == 1, ("summary anchor", src.count(old))
src = src.replace(old, new)
shutil.copy(p, p + ".bak_0916_ladr")
open(p, "w").write(src)
print("engine patched", hashlib.md5(src.encode()).hexdigest()[:8])
