"""NOVA v2 import hooks. Chains tmp_hj/lpvh_ntrs_hooks/sitecustomize.py (x5 clamp, C1META bridge) unchanged,
then, when VLMAS_NOVA_V2=1, patches memory.secld_select.evidence -> memory.nova_v2_select.evidence_v2.
No existing file is modified."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE_HOOKS = os.path.join(os.path.dirname(_HERE), "lpvh_ntrs_hooks")
if _BASE_HOOKS not in sys.path:
    sys.path.insert(1, _BASE_HOOKS)
_base_file = os.path.join(_BASE_HOOKS, "sitecustomize.py")
_g = {"__file__": _base_file, "__name__": "sitecustomize_lpvh_ntrs"}
exec(compile(open(_base_file).read(), _base_file, "exec"), _g)

if os.environ.get("VLMAS_NOVA_V2", "") == "1":
    import importlib
    import importlib.abc
    import importlib.machinery

    class _NovaV2Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != "memory.secld_select":
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                return spec
            original_exec = spec.loader.exec_module

            def exec_module(module):
                original_exec(module)
                importlib.import_module("memory.nova_v2_select").patch(module)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _NovaV2Finder())
