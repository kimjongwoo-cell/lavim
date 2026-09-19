"""0919 selector-only comparison: PLIP picks from the SAME nav4 candidate pool the Navigator sees. Original files untouched.

Chains tmp_hj/ext_noplan_hooks (VLMAS_NO_PLANNER=1 skips the Planner; VLMAS_EXT_PATCHES is left unset, so no external
jpgs). With VLMAS_PLIPSEL=1, run_nav4_navigation (in onepass_navigation_nav4 and in onepass_navigator, which imports the
name) is replaced by:

  x5_cands, x20_cands = nav2_global_candidates(slide, root_grid)      # identical pool to the nav4 Navigator
  score every candidate crop image by PLIP cosine(image, question)   # same PLIP checkpoint / wrapper as HJ plip_precompute
  keep the top scale.overview_patch_count x5 and top scale.detail_patch_count x20 (4 + 8)
  build the patches with nav2_patch                                   # identical crop objects / scale labels to nav4

No Navigator VLM call is made and nothing is appended to the latent state, so the Reasoner and Answerer see only the 12
crops. The pair arm navsel runs the real Navigator on the same pool with --no-navigator-kv (its turn is not kept either),
so the two arms differ only in which 12 candidates are chosen.

  VLMAS_PLIPSEL=1            enable
  VLMAS_PLIP_LIB / _CKPT     dir holding plip_v58.py / PLIP checkpoint dir
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_f = os.path.join(os.path.dirname(_HERE), "ext_noplan_hooks", "sitecustomize.py")
exec(compile(open(_f).read(), _f, "exec"), {"__file__": _f, "__name__": "sitecustomize_ext_noplan"})

if os.environ.get("VLMAS_PLIPSEL", "") == "1":
    import importlib.abc
    import importlib.machinery

    _MODEL = []

    def _plip():
        if not _MODEL:
            lib = os.environ["VLMAS_PLIP_LIB"]
            if lib not in sys.path:
                sys.path.insert(0, lib)
            from plip_v58 import PLIP
            _MODEL.append(PLIP(os.environ["VLMAS_PLIP_CKPT"]))
            print(f"[PLIPSEL] loaded PLIP on {_MODEL[0].device}", flush=True)
        return _MODEL[0]

    def plipsel_run_nav4_navigation(slide, *, client, root_grid, evidence_plan, artifact_root, round_number,
                                    save_navigation_pngs=True):
        import numpy as np
        from vision_text_mas.contracts import PatchId
        from vision_text_mas.navigation_contracts import SelectionTrace
        from vision_text_mas.onepass_navigation_nav2 import nav2_global_candidates, nav2_patch

        x5c, x20c = nav2_global_candidates(slide, root_grid=root_grid)
        scale = evidence_plan.scale_plan
        k5 = min(scale.overview_patch_count, len(x5c))
        k20 = min(scale.detail_patch_count, len(x20c))
        question = evidence_plan.question_focus
        model = _plip()
        q = model.encode_text([question], batch_size=1)[0]
        q = q / np.linalg.norm(q)

        def top(cands, k):
            if k <= 0:
                return [], []
            emb = model.encode_images([c.image for c in cands], batch_size=32)
            emb = emb / np.linalg.norm(emb, axis=-1, keepdims=True)
            sims = emb @ q
            order = [int(i) for i in np.argsort(-sims, kind="stable")[:k]]
            return order, [round(float(sims[i]), 4) for i in order]

        o5, s5 = top(x5c, k5)
        o20, s20 = top(x20c, k20)
        chosen = (*(x5c[i] for i in o5), *(x20c[i] for i in o20))
        anchor_ids: dict[str, PatchId] = {}
        patches = tuple(
            nav2_patch(obs, rank=rank, round_number=round_number, anchor_ids=anchor_ids, artifact_root=artifact_root)
            for rank, obs in enumerate(chosen, start=1)
        )
        ids5 = tuple(i + 1 for i in o5)
        trace = SelectionTrace(label=f"round-{round_number}-plipsel-nav4-pool",
                               ranked_ids=ids5[:10] or (1,), skipped_low_tissue_ids=(), selected_ids=ids5[:5])
        print(f"[PLIPSEL] pool x5={len(x5c)} x20={len(x20c)} pick x5={[i + 1 for i in o5]} {s5} "
              f"x20={[i + 1 for i in o20]} {s20}", flush=True)
        return patches, (), trace

    class _NavFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname not in ("vision_text_mas.onepass_navigator", "vision_text_mas.onepass_navigation_nav4"):
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                return spec
            orig = spec.loader.exec_module

            def exec_module(module):
                orig(module)
                module.run_nav4_navigation = plipsel_run_nav4_navigation
                print(f"[PLIPSEL] patched {fullname}.run_nav4_navigation", flush=True)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _NavFinder())
