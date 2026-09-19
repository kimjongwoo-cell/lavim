"""Import hook for nav3_x5_clamp (only when VLMAS_NAV3_X5_CLAMP=1). See nav3_x5_clamp.py."""
import os
import sys

if os.environ.get("VLMAS_NAV3_X5_CLAMP", "") == "1":
    import importlib.abc
    import importlib.machinery

    _TARGET = "vision_text_mas.onepass_navigation_roots"

    class _ClampFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != _TARGET:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                return spec
            original_exec = spec.loader.exec_module

            def exec_module(module):
                original_exec(module)
                import nav3_x5_clamp
                nav3_x5_clamp.patch(module)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _ClampFinder())
