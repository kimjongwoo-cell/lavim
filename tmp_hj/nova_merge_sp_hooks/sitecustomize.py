"""[09-19 COPY of tmp_hj/nova_lgv_latplan_hooks, candidate E; ONLY change: with VLMAS_NOVA_MERGE=1 the dropped
tokens are folded by nova_lgv.merge_dropped_sp, i.e. only into a kept token within Chebyshev radius
VLMAS_NOVA_MERGE_R (default 2) on the crop token grid. Original hook dir untouched.]
NOVA Final (Early-LGV L2) + latplan hook. Same chain as tmp_hj/nova_rho_latplan_hooks (lpvh_ntrs_hooks + latplan_hooks),
then with VLMAS_NOVA_LGV=1: memory.secld_select.evidence -> nova_lgv.evidence_lgv and
memory.nova_select.select_vision_features wrapped to capture vision block-2 hidden. memory/ files untouched."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE_HOOKS = os.path.join(os.path.dirname(_HERE), "lpvh_ntrs_hooks")
if _BASE_HOOKS not in sys.path:
    sys.path.insert(1, _BASE_HOOKS)
_f = os.path.join(os.path.dirname(_HERE), "latplan_hooks", "sitecustomize.py")
exec(compile(open(_f).read(), _f, "exec"), {"__file__": _f, "__name__": "sitecustomize_latplan"})

if os.environ.get("VLMAS_NOVA_LGV", "") == "1":
    for v in ("VLMAS_NOVA_RHO", "VLMAS_NOVA_V2", "VLMAS_NOVA_V3"):
        if os.environ.get(v, "") == "1":
            raise RuntimeError(f"VLMAS_NOVA_LGV is exclusive with {v}")
    import importlib
    import importlib.abc
    import importlib.machinery
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)

    def _patch_secld(module):
        import nova_lgv
        module.evidence = nova_lgv.evidence_lgv
        print("[NOVALGV] patched memory.secld_select.evidence -> nova_lgv.evidence_lgv (Early-LGV L2, per-query mean-normalised s)", flush=True)

    def _patch_nova(module):
        import nova_lgv
        orig = module.select_vision_features
        do_merge = os.environ.get("VLMAS_NOVA_MERGE", "") == "1"
        def select_vision_features(bb, pixel_values, grid_thw):
            nova_lgv.capture(bb)
            nova_lgv.STORE["h2"] = None; nova_lgv.STORE["v6"] = None
            if not do_merge:
                return orig(bb, pixel_values, grid_thw)
            cap = {}
            gif = bb.vlm.get_image_features
            def gif_cap(*a, **k):
                r = gif(*a, **k); cap["r"] = r; return r
            bb.vlm.get_image_features = gif_cap
            try:
                kept, keep = orig(bb, pixel_values, grid_thw)
            finally:
                del bb.vlm.get_image_features
            full = cap["r"].pooler_output
            import torch as _t
            full = _t.cat(list(full), 0) if isinstance(full, (list, tuple)) else full
            unit = int(bb.visual.spatial_merge_unit)
            mrg = int(bb.visual.spatial_merge_size)
            grids = [(int(h) // mrg, int(w) // mrg) for t, h, w in _t.as_tensor(grid_thw).tolist() for _ in range(int(t))]
            return nova_lgv.merge_dropped_sp(full, keep, grids), keep
        module.select_vision_features = select_vision_features
        print(f"[NOVALGV] wrapped memory.nova_select.select_vision_features (block-2 capture, spatial merge={do_merge})", flush=True)

    _TARGETS = {"memory.secld_select": _patch_secld, "memory.nova_select": _patch_nova}

    class _LgvFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname not in _TARGETS:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                return spec
            orig = spec.loader.exec_module
            def exec_module(module, _fn=_TARGETS[fullname], _orig=orig):
                _orig(module)
                _fn(module)
            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _LgvFinder())
