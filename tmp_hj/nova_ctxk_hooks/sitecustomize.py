"""09-19 C1 kernel-side neighbourhood arms (this session). Chains tmp_hj/latplan_hooks (which itself
uses lpvh_ntrs_hooks) and then patches memory.secld_select at import time with ctxk.patch, so that both
secld_select.select and nova_select's logdet branch go through the modified kernel. memory/ untouched,
and with VLMAS_CTXK unset (or =off) nothing is patched at all (byte-identical behaviour to the base arm).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE_HOOKS = os.path.join(os.path.dirname(_HERE), "lpvh_ntrs_hooks")
if _BASE_HOOKS not in sys.path:
    sys.path.insert(1, _BASE_HOOKS)
_f = os.path.join(os.path.dirname(_HERE), "latplan_hooks", "sitecustomize.py")
exec(compile(open(_f).read(), _f, "exec"), {"__file__": _f, "__name__": "sitecustomize_latplan"})

if os.environ.get("VLMAS_CTXK", "off").strip().lower() in ("rbf", "ctx"):
    import importlib.abc
    import importlib.machinery

    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)

    class _CtxkFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != "memory.secld_select":
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                return spec
            orig = spec.loader.exec_module

            def exec_module(module, _orig=orig):
                _orig(module)
                import ctxk
                ctxk.patch(module)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _CtxkFinder())
