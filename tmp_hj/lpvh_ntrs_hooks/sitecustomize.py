"""Import hooks for the NTRS + LPVH runs (sender_relay_exp/lpvh_ntrs_eff.sh). Original files untouched.

VLMAS_NAV3_X5_CLAMP=1   tmp_hj/nav3_x5_clamp (x5 side = min(4096, root w, root h), md5-guarded)
VLMAS_C1_META_BRIDGE=1  tmp_hj/lpvh_ntrs_hooks/c1_meta_bridge.py (C1 keep mask -> _visual_meta)
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PATCHES = {}
if os.environ.get("VLMAS_NAV3_X5_CLAMP", "") == "1":
    sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "nav3_x5_clamp"))
    _PATCHES["vision_text_mas.onepass_navigation_roots"] = ("nav3_x5_clamp", "patch")
if os.environ.get("VLMAS_C1_META_BRIDGE", "") == "1":
    _PATCHES["memory.qasc_select"] = ("c1_meta_bridge", "patch")

if _PATCHES:
    import importlib
    import importlib.abc
    import importlib.machinery

    class _PatchFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname not in _PATCHES:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                return spec
            original_exec = spec.loader.exec_module
            mod_name, fn = _PATCHES[fullname]

            def exec_module(module):
                original_exec(module)
                getattr(importlib.import_module(mod_name), fn)(module)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _PatchFinder())
