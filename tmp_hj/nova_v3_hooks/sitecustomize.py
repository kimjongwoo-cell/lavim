"""NOVA v3 import hooks. Chains tmp_hj/nova_dump_hooks (-> nova_v2_hooks -> lpvh_ntrs_hooks; dump only if
VLMAS_NOVA_DUMP_DIR, v2 only if VLMAS_NOVA_V2=1), then with VLMAS_NOVA_V3=1 patches
memory.secld_select.evidence -> memory.nova_v3_select.evidence_v3. No existing file is modified."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_f = os.path.join(os.path.dirname(_HERE), "nova_dump_hooks", "sitecustomize.py")
exec(compile(open(_f).read(), _f, "exec"), {"__file__": _f, "__name__": "sitecustomize_nova_dump"})

if os.environ.get("VLMAS_NOVA_V3", "") == "1":
    if os.environ.get("VLMAS_NOVA_V2", "") == "1":
        raise RuntimeError("VLMAS_NOVA_V2 and VLMAS_NOVA_V3 are mutually exclusive")
    import importlib
    import importlib.abc
    import importlib.machinery

    class _NovaV3Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != "memory.secld_select":
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                return spec
            orig = spec.loader.exec_module

            def exec_module(module):
                orig(module)
                importlib.import_module("memory.nova_v3_select").patch(module)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _NovaV3Finder())
