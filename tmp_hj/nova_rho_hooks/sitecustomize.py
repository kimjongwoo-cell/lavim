"""NOVA rho (canonical draft) import hook. Chains tmp_hj/nova_dump_hooks (-> nova_v2_hooks -> lpvh_ntrs_hooks),
then with VLMAS_NOVA_RHO=1 patches memory.secld_select.evidence -> memory.nova_rho_select.evidence_rho."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_f = os.path.join(os.path.dirname(_HERE), "nova_dump_hooks", "sitecustomize.py")
exec(compile(open(_f).read(), _f, "exec"), {"__file__": _f, "__name__": "sitecustomize_nova_dump"})

if os.environ.get("VLMAS_NOVA_RHO", "") == "1":
    if os.environ.get("VLMAS_NOVA_V2", "") == "1" or os.environ.get("VLMAS_NOVA_V3", "") == "1":
        raise RuntimeError("VLMAS_NOVA_RHO is exclusive with VLMAS_NOVA_V2 / V3")
    import importlib
    import importlib.abc
    import importlib.machinery

    class _RhoFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != "memory.secld_select":
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                return spec
            orig = spec.loader.exec_module

            def exec_module(module):
                orig(module)
                importlib.import_module("memory.nova_rho_select").patch(module)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _RhoFinder())
