import hashlib, os, shutil, sys
P = "/home/users/whddn12316/wsi_latent_0915_decode_hj/vision_text_mas/latent_qwen_engine.py"
src = open(P).read()
md5 = hashlib.md5(src.encode()).hexdigest()
if not md5.startswith("160c9e18"):
    sys.exit(f"engine md5 {md5[:8]} != 160c9e18 — someone changed it; abort")

def rep(old, new, count=1):
    global src
    n = src.count(old)
    if n != count:
        sys.exit(f"expected {count} occurrence(s), found {n}: {old[:80]!r}")
    src = src.replace(old, new)

rep('''            seed=42 + int(getattr(self, "_rpath_case_index", 0) or 0))
        if not images and cache is not None:''',
'''            seed=42 + int(getattr(self, "_rpath_case_index", 0) or 0))
        # C2 Support-Resolved Visual Write Handoff (memory/srvw.py, VLMAS_SRVW): accumulate, during the
        # Reasoner's latent steps, the value message alpha*V each physical support writes (per layer).
        # Off / other stage = nullcontext.
        from memory import srvw as _srvw
        _srvw_r = _srvw.reasoner_probe(self._backbone, m=latent_steps, stage=stage)
        if not images and cache is not None:''')
rep("            with _lpvh_r:\n", "            with _lpvh_r, _srvw_r:\n", count=2)
rep('''        # Canonical addressing restricted to this Answerer call (VLMAS_KV_ROUTE_TERMINAL_ONLY=1,''',
'''        # C2 Support-Resolved Visual Write Handoff (VLMAS_SRVW=1|identity, memory/srvw.py): at the Answerer
        # boundary row, a_g = [M_g . C_g / |M_g|^2]_+ per layer (Reasoner visual write C_g vs the Answerer's
        # native support message M_g); the visual mass is then re-weighted by a between physical supports
        # (rho_V and within-support ratios conserved). Off = no hooks. Not combined with LPVH (same hook site).
        _srvw_ctx = contextlib.nullcontext()
        _srvw_state = None
        if is_terminal and os.environ.get("VLMAS_SRVW", "").strip():
            from memory import srvw as _srvw
            _scfg = _srvw.config()
            _srvw_state = _srvw.engine_state(self)
            if os.environ.get("VLMAS_LPVH", "").strip():
                _srvw_state = {"ok": False, "note": "VLMAS_LPVH also set (same self_attn post-hook site)"}
            if not _srvw_state["ok"]:
                print(f"[SRVW] SKIP: {_srvw_state['note']}", flush=True)
                _srvw_state = None
            else:
                _slo, _shi = _srvw.layer_window(_scfg["layers"], len(self._backbone.lm.layers))
                _srvw_ctx = _srvw.install_srvw(
                    self._backbone, _srvw_state["C"], _srvw_state["vis"], _srvw_state["gid"], _srvw_state["sizes"],
                    layers=(_slo, _shi), boundary=_scfg["boundary"], identity=(_scfg["mode"] == "identity"),
                    log=_scfg["log"])
                print(f"[SRVW] mode={_scfg['mode']} boundary={_scfg['boundary']} layers={_slo}-{_shi} "
                      f"n_vis={int(_srvw_state['vis'].numel())} supports={len(_srvw_state['groups'])} "
                      f"sizes={_srvw_state['sizes'].tolist()} src={_srvw_state['note']}", flush=True)
        # Canonical addressing restricted to this Answerer call (VLMAS_KV_ROUTE_TERMINAL_ONLY=1,''')
rep("                    _psar_ctx as _psar_stats, _lpvh_ctx as _lpvh_stats:\n",
    "                    _psar_ctx as _psar_stats, _lpvh_ctx as _lpvh_stats, _srvw_ctx as _srvw_stats:\n")
rep('''            print(f"[LPVH] case {getattr(self, '_rpath_case_index', -1)} " + _lpvh.summary(_lpvh_stats, _lpvh_state), flush=True)
''',
'''            print(f"[LPVH] case {getattr(self, '_rpath_case_index', -1)} " + _lpvh.summary(_lpvh_stats, _lpvh_state), flush=True)
        if _srvw_state is not None and _srvw_stats is not None:
            print(f"[SRVW] case {getattr(self, '_rpath_case_index', -1)} " + _srvw.summary(_srvw_stats, _srvw_state), flush=True)
''')
compile(src, P, "exec")
shutil.copy2(P, P + ".bak_0915_srvw")
tmp = P + ".tmp_srvw"
open(tmp, "w").write(src)
cur = hashlib.md5(open(P).read().encode()).hexdigest()
if not cur.startswith("160c9e18"):
    os.remove(tmp); sys.exit("engine changed during patch; abort")
os.replace(tmp, P)
print("patched", hashlib.md5(src.encode()).hexdigest()[:8])
