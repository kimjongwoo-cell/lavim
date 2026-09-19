"""Opt-in clamp of the nav3 x5 candidate side for narrow slides — without editing vision_text_mas.

Problem (09-15, PANDA 11/196): vision_text_mas/onepass_navigation_roots._best_x5_child lays 4x6 exact 4096 px squares
inside each root cell; when the slide's short side is < 4096 px (needle-core strips such as 2816 x 27648)
geometry.edge_anchored_grid raises ValueError, every root returns None and the case fails with
FAILED_INSUFFICIENT_CANDIDATES ("no tissue-safe x5 candidate remains after local replacement").

Fix (only when active): side = min(4096, root.width, root.height). Everything else is a verbatim copy of the original
function (md5 a36830f5 of onepass_navigation_roots.py). Slides whose root cells are >= 4096 px on both sides give
exactly the same boxes. x20 fields are the 4x4 split of the (smaller) x5 box, so they shrink with it.

Activation (both required, nothing else changes):
    PYTHONPATH=<repo>/tmp_hj/nav3_x5_clamp:<repo>   (sitecustomize.py here installs an import hook)
    VLMAS_NAV3_X5_CLAMP=1
If the original file's md5 differs, the hook does NOT patch and prints a warning.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

TARGET = "vision_text_mas.onepass_navigation_roots"
EXPECTED_MD5 = "a36830f507eccf96325092727c54c23e"
_LOGGED = {"n": 0}


def file_md5(path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def make_clamped(m):
    """Verbatim copy of m._best_x5_child with the x5 side clamped to the root cell."""

    def _best_x5_child(slide, *, root, root_id, used_boxes, artifact_root, round_number, save_navigation_pngs):
        side = min(m.EXACT_X5_SIDE, int(root.width), int(root.height))
        if side < m.EXACT_X5_SIDE and _LOGGED["n"] < 20:
            _LOGGED["n"] += 1
            print(f"[NAV3-X5CLAMP] root {root_id} {root.width}x{root.height} -> x5 side {side}", flush=True)
        try:
            boxes = m.edge_anchored_grid(root, side=side, rows=4, cols=6)
        except ValueError:
            return None
        fractions = m.tissue_fractions(slide, boxes)
        eligible = m.eligible_tissue_ids(
            fractions,
            minimum_fraction=m.MINIMUM_TISSUE_FRACTION,
        )
        if save_navigation_pngs:
            _ = m.save_image(
                m.numbered_grid(
                    m.render_box(slide, root, max_side=768),
                    parent=root,
                    boxes=boxes,
                    eligible_ids=eligible,
                ),
                artifact_root / f"round_{round_number}_root_{root_id}_children.png",
            )
        candidates = tuple(
            candidate_id
            for candidate_id in eligible
            if boxes[candidate_id - 1] not in used_boxes
        )
        if not candidates:
            return None
        chosen = min(
            candidates,
            key=lambda candidate_id: (
                -fractions[candidate_id - 1],
                m._squared_distance(boxes[candidate_id - 1], root),
                candidate_id,
            ),
        )
        return boxes[chosen - 1], chosen, fractions[chosen - 1]

    _best_x5_child.__qualname__ = "_best_x5_child[nav3_x5_clamp]"
    return _best_x5_child


def patch(module) -> bool:
    got = file_md5(module.__file__)
    if got != EXPECTED_MD5:
        print(f"[NAV3-X5CLAMP] NOT active: {module.__file__} md5 {got[:8]} != expected {EXPECTED_MD5[:8]}", flush=True)
        return False
    module._nav3_x5_clamp_original = module._best_x5_child
    module._best_x5_child = make_clamped(module)
    print("[NAV3-X5CLAMP] active: x5 side = min(4096, root w, root h)", flush=True)
    return True
