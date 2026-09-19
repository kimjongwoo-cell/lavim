"""0919 external patch bank: feed WSI-VQA PLIP question-ranked patches straight to the Reasoner. Original files untouched.

Chains tmp_hj/latplan_hooks (Planner latent-only hand-off; itself chains lpvh_ntrs_hooks). With VLMAS_EXT_PATCHES=<ranking json>
the nav4 Navigator is replaced: instead of one visual Navigator call, run_nav4_navigation returns the top-K patches of the
precomputed PLIP ranking for this question (HJ plip_precompute, key "<slide>||q<dataset_index>"), so the Reasoner and the
Answerer see exactly those images and nothing else changes.

  VLMAS_EXT_PATCHES    ranking json ({"meta":..., "rankings": {"<slide>||q<i>": {"slide_id","slide_name","ranked":[[x,y,s],..],"fov"}}})
  VLMAS_EXT_PATCH_ROOT dir with <slide_name>/<x>_<y>.jpg (CLAM level-2 512 px patches, Level-0 coords)
  VLMAS_EXT_K          patches per question (default 12 = nav4 budget; slides with fewer patches give all they have)

The dataset index is read from engine._rpath_case_index / _rpath_slide_id (set per case in latent_onepass_cli.py) and the slide id
there must equal the ranking's slide_id, otherwise the case fails loudly. Patches are labelled x5 (the magnification enum
has no 2.5x; one 512 px patch covers 8192 px of Level 0).
"""
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_f = os.path.join(os.path.dirname(_HERE), "latplan_hooks", "sitecustomize.py")
exec(compile(open(_f).read(), _f, "exec"), {"__file__": _f, "__name__": "sitecustomize_latplan"})

_RANK = os.environ.get("VLMAS_EXT_PATCHES", "").strip()
if _RANK:
    import importlib.abc
    import importlib.machinery
    import json
    from pathlib import Path

    _ROOT = Path(os.environ["VLMAS_EXT_PATCH_ROOT"])
    _K = int(os.environ.get("VLMAS_EXT_K", "12"))
    _RANKINGS = json.load(open(_RANK))["rankings"]
    _CASE_DIR = re.compile(r"^(\d+)_(.+)$")

    def _case_of(client, artifact_root):
        # latent_onepass_cli sets engine._rpath_case_index / _rpath_slide_id per case (artifact_root is the attempt root).
        seen, stack = set(), [client]
        while stack:
            obj = stack.pop()
            if obj is None or id(obj) in seen:
                continue
            seen.add(id(obj))
            idx, sid = getattr(obj, "_rpath_case_index", None), getattr(obj, "_rpath_slide_id", None)
            if idx is not None and sid:
                return int(idx), str(sid)
            stack += [getattr(obj, a, None) for a in ("_backend", "_engine", "_client", "engine", "backend")]
        for part in reversed(Path(artifact_root).parts):
            m = _CASE_DIR.match(part)
            if m:
                return int(m.group(1)), m.group(2)
        raise RuntimeError(f"[EXTPATCH] case index not found (client {type(client).__name__}, root {artifact_root})")

    def ext_run_nav4_navigation(slide, *, client, root_grid, evidence_plan, artifact_root, round_number,
                                save_navigation_pngs=True):
        from PIL import Image
        from vision_text_mas.contracts import (CandidateId, NavigationAction, PatchId, PatchMetadata,
                                               RootCellId)
        from vision_text_mas.geometry import Box, Magnification
        from vision_text_mas.navigation_contracts import SelectionTrace
        from vision_text_mas.navigation_render import save_image

        idx, sid = _case_of(client, artifact_root)
        case_dir = Path(artifact_root) / f"{idx:03d}_{sid}"
        out_dir = case_dir if case_dir.is_dir() else Path(artifact_root)
        rec = _RANKINGS.get(f"{sid}||q{idx}")
        if rec is None or rec["slide_id"] != sid:
            raise RuntimeError(f"[EXTPATCH] no ranking for {sid}||q{idx}")
        top = rec["ranked"][:_K]
        fov = int(rec.get("fov", 8192))
        patches = []
        for rank, (x, y, _score) in enumerate(top, start=1):
            with Image.open(_ROOT / rec["slide_name"] / f"{x}_{y}.jpg") as src:
                img = src.convert("RGB")
            pid = PatchId(f"R{round_number}-P{rank}")
            path = save_image(img, out_dir / f"round_{round_number}" / f"ext_{pid}.png")
            patches.append(PatchMetadata(
                patch_id=pid, round_number=round_number, rank=rank, action=NavigationAction.INITIAL,
                magnification=Magnification.X5, box=Box(x=int(x), y=int(y), width=fov, height=fov),
                root_id=RootCellId(rank), source_x5_anchor=pid, candidate_id=CandidateId(rank),
                tissue_fraction=1.0, image_path=path))
        n = len(patches)
        trace = SelectionTrace(label=f"round-{round_number}-ext-plip-top{_K}",
                               ranked_ids=tuple(range(1, min(n, 10) + 1)), skipped_low_tissue_ids=(),
                               selected_ids=tuple(range(1, min(n, 5) + 1)))
        print(f"[EXTPATCH] idx={idx} slide={sid} k={n}/{rec.get('n', len(rec['ranked']))} "
              f"top_score={top[0][2]:.5f} size={img.size}", flush=True)
        return tuple(patches), (), trace

    class _NavFinder(importlib.abc.MetaPathFinder):
        # onepass_navigator imports run_nav4_navigation by name, so patch the name there (and at the source).
        def find_spec(self, fullname, path, target=None):
            if fullname not in ("vision_text_mas.onepass_navigator", "vision_text_mas.onepass_navigation_nav4"):
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                return spec
            orig = spec.loader.exec_module

            def exec_module(module):
                orig(module)
                module.run_nav4_navigation = ext_run_nav4_navigation
                print(f"[EXTPATCH] patched {fullname}.run_nav4_navigation (K={_K})", flush=True)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _NavFinder())
