"""Import hook for the GLVR diagnostic (tmp_hj/glvr).  Original files untouched.

Chains the latplan hooks first (which themselves chain lpvh_ntrs_hooks), then, when
VLMAS_GLVR is set, patches two modules:

  memory.rnlcr.apply_reasoner  -> install the replay capture while the grounding
                                  replay runs, so each grounded latent's conditional
                                  visual read u_{V,k} is recorded without an extra forward.
  memory.alsi_diag.run         -> run glvr_diag.run at the Answerer boundary
                                  (provenance masses, trajectory stats, lambda sweep).

VLMAS_GLVR_SKIP_ALSI=1 skips the original ALSI measurement afterwards.
"""
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

# --- chain the latplan hooks (Planner text path removal) -------------------------------
for _sibling in ("latplan_hooks", "lpvh_ntrs_hooks"):
    _path = os.path.join(os.path.dirname(_HERE), _sibling, "sitecustomize.py")
    if os.path.exists(_path):
        try:
            _spec = importlib.util.spec_from_file_location(f"_glvr_chain_{_sibling}", _path)
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
        except Exception as _exc:  # noqa: BLE001
            print(f"[GLVR] chain {_sibling} FAILED: {_exc!r}", flush=True)
        break

if os.environ.get("VLMAS_GLVR", "").strip():
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)

    def _patch_rnlcr(module):
        original = module.apply_reasoner

        def apply_reasoner(engine, result, **kwargs):
            import traceback
            try:
                import glvr_diag as GD
                bb = engine._backbone
                m = int(kwargs.get("m") or 0)
                rec = GD.on_reasoner(engine, result, m=m,
                                     case_index=int(kwargs.get("case_index", -1)))
                vis = (rec or {}).get("vis")
                if vis is not None and vis.numel():
                    with GD.capture_replay(bb, m, vis, int(result.get("past_len") or 0)) as cap:
                        out = original(engine, result, **kwargs)
                    GD.finish_reasoner(bb, cap)
                    return out
            except Exception as exc:  # noqa: BLE001 - never break the case
                print(f"[GLVR] reasoner capture FAILED: {exc!r}", flush=True)
                traceback.print_exc()
            return original(engine, result, **kwargs)

        module.apply_reasoner = apply_reasoner
        module._glvr_patched = True
        print("[GLVR] patched memory.rnlcr.apply_reasoner (replay capture)", flush=True)

    def _patch_alsi(module):
        original_run = module.run

        def run(engine, **kwargs):
            import traceback
            try:
                importlib.import_module("glvr_diag").run(engine, **kwargs)
            except Exception as exc:  # noqa: BLE001
                print(f"[GLVR] FAILED case {kwargs.get('case_index')}: {exc!r}", flush=True)
                traceback.print_exc()
            if os.environ.get("VLMAS_GLVR_SKIP_ALSI", "") == "1":
                return None
            return original_run(engine, **kwargs)

        module.run = run
        module._glvr_patched = True
        print("[GLVR] patched memory.alsi_diag.run (terminal-row diagnostic)", flush=True)

    _TARGETS = {"memory.rnlcr": _patch_rnlcr, "memory.alsi_diag": _patch_alsi}

    class _GlvrFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            patch = _TARGETS.get(fullname)
            if patch is None:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                return spec
            original_exec = spec.loader.exec_module

            def exec_module(module):
                original_exec(module)
                try:
                    patch(module)
                except Exception as exc:  # noqa: BLE001
                    print(f"[GLVR] patch {fullname} FAILED: {exc!r}", flush=True)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _GlvrFinder())
