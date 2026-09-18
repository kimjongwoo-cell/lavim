"""Latent-only Planner -> Navigator hand-off (opt-in VLMAS_PLAN_LATENT_ONLY=1). No existing file is modified.

Chains tmp_hj/lpvh_ntrs_hooks (x5 clamp, C1 meta bridge). With the flag:
  * vision_text_mas.latent_onepass.LatentRoleClient.decode_plan_targets -> returns None without decoding, so the
    Planner keeps fixed_patch_plan's placeholder targets and emits no text (VLMAS_NAV2=1 no longer triggers a clone decode);
  * vision_text_mas.onepass_navigation_nav4._prompt -> the same prompt with the two target lines
    ("x5 overview target:", "x20 detail target:") removed, so the Navigator receives the plan only through the shared
    latent KV (the "Validated evidence plan" line is already swapped for the latent-handoff note upstream).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_f = os.path.join(os.path.dirname(_HERE), "lpvh_ntrs_hooks", "sitecustomize.py")
exec(compile(open(_f).read(), _f, "exec"), {"__file__": _f, "__name__": "sitecustomize_lpvh_ntrs"})

if os.environ.get("VLMAS_PLAN_LATENT_ONLY", "") == "1":
    import importlib.abc
    import importlib.machinery

    TARGET_PREFIXES = ("x5 overview target:", "x20 detail target:")

    def _patch_onepass(module):
        cls = module.LatentRoleClient

        def decode_plan_targets(self, question):
            return None

        cls.decode_plan_targets = decode_plan_targets
        print("[LATPLAN] patched LatentRoleClient.decode_plan_targets -> None (no Planner text decode)", flush=True)

    def _patch_nav4(module):
        orig = module._prompt

        def _prompt(plan, **kw):
            text = orig(plan, **kw)
            return "".join(l for l in text.splitlines(keepends=True) if not l.startswith(TARGET_PREFIXES))

        _prompt.__wrapped__ = orig
        module._prompt = _prompt
        print("[LATPLAN] patched onepass_navigation_nav4._prompt (target lines removed)", flush=True)

    _P = {"vision_text_mas.latent_onepass": _patch_onepass, "vision_text_mas.onepass_navigation_nav4": _patch_nav4}

    class _LatPlanFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname not in _P:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                return spec
            orig_exec = spec.loader.exec_module

            def exec_module(module):
                orig_exec(module)
                _P[fullname](module)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _LatPlanFinder())
