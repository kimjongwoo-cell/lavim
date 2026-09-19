"""NOVA rho + coherence hook. Chains tmp_hj/nova_dump_hooks (-> nova_v2_hooks -> lpvh_ntrs_hooks); with VLMAS_NOVA_COH=1
patches memory.secld_select.select/evidence via memory.nova_coh_select.patch. Exclusive with VLMAS_NOVA_RHO/V2/V3."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_f = os.path.join(os.path.dirname(_HERE), "nova_dump_hooks", "sitecustomize.py")
exec(compile(open(_f).read(), _f, "exec"), {"__file__": _f, "__name__": "sitecustomize_nova_dump"})

if os.environ.get("VLMAS_NOVA_COH", "") == "1":
    for k in ("VLMAS_NOVA_RHO", "VLMAS_NOVA_V2", "VLMAS_NOVA_V3"):
        if os.environ.get(k, "") == "1":
            raise RuntimeError(f"VLMAS_NOVA_COH is exclusive with {k}")
    import importlib
    import importlib.abc
    import importlib.machinery

    class _CohFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != "memory.secld_select":
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                return spec
            orig = spec.loader.exec_module

            def exec_module(module):
                orig(module)
                importlib.import_module("memory.nova_coh_select").patch(module)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _CohFinder())
