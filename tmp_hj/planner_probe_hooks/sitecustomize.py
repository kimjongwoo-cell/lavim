"""0919 Planner-turn probe on the latplan base (why do base answers break?). Original files untouched.

Chains tmp_hj/latplan_hooks (the latplan base arm). Two opt-in changes to the Planner turn only:
  VLMAS_PLANNER_LATENT_STEPS=N  the evidence_planner role runs N latent steps instead of --latent-steps
                                (Navigator / Reasoner keep theirs). A = 0.
  VLMAS_PLANNER_BLANK_THUMB=1   the Planner sees a white image of the same size instead of the slide thumbnail
                                (the first role must carry an image). B.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_f = os.path.join(os.path.dirname(_HERE), "latplan_hooks", "sitecustomize.py")
exec(compile(open(_f).read(), _f, "exec"), {"__file__": _f, "__name__": "sitecustomize_latplan"})

_STEPS = os.environ.get("VLMAS_PLANNER_LATENT_STEPS", "").strip()
_BLANK = os.environ.get("VLMAS_PLANNER_BLANK_THUMB", "") == "1"
if _STEPS or _BLANK:
    import importlib.abc
    import importlib.machinery

    def _wrap(cls):
        orig = cls.append_only

        def append_only(self, *, role, images, image_labels, system_prompt, user_prompt):
            if role != "evidence_planner":
                return orig(self, role=role, images=images, image_labels=image_labels,
                            system_prompt=system_prompt, user_prompt=user_prompt)
            if _BLANK:
                from PIL import Image
                images = tuple(Image.new("RGB", im.size, (255, 255, 255)) for im in images)
            saved = self._latent_steps
            if _STEPS:
                self._latent_steps = int(_STEPS)
            try:
                rec = orig(self, role=role, images=images, image_labels=image_labels,
                           system_prompt=system_prompt, user_prompt=user_prompt)
            finally:
                self._latent_steps = saved
            print(f"[PPROBE] planner latent_steps={_STEPS or saved} blank_thumb={_BLANK} "
                  f"img={[im.size for im in images]}", flush=True)
            return rec

        cls.append_only = append_only

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != "vision_text_mas.latent_onepass":
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                return spec
            orig = spec.loader.exec_module

            def exec_module(module):
                orig(module)
                _wrap(module.LatentRoleClient)
                print(f"[PPROBE] patched LatentRoleClient.append_only (steps={_STEPS or '-'}, blank={_BLANK})", flush=True)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
