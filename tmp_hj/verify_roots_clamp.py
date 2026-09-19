"""Verify the in-place x5 clamp in vision_text_mas/onepass_navigation_roots.py against the pre-edit file
(.bak_0915_x5clamp) and the opt-in hook copy (tmp_hj/nav3_x5_clamp.make_clamped)."""
import importlib.util, sys, tempfile
from pathlib import Path
REPO = Path("/home/users/whddn12316/wsi_latent_0915_decode_hj")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "tmp_hj/nav3_x5_clamp"))
from PIL import Image
import vision_text_mas.onepass_navigation_roots as new
import importlib.machinery
_ldr = importlib.machinery.SourceFileLoader("roots_bak", str(REPO / "vision_text_mas/onepass_navigation_roots.py.bak_0915_x5clamp"))
spec = importlib.util.spec_from_loader("roots_bak", _ldr)
old = importlib.util.module_from_spec(spec); sys.modules["roots_bak"] = old; spec.loader.exec_module(old)
import nav3_x5_clamp as nc
from vision_text_mas.cli import _open_slide
from vision_text_mas.dataset import load_cases
from vision_text_mas.errors import PipelineFailure
DATA = Path("/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda")
tmp = Path(tempfile.mkdtemp()); PASS = FAIL = 0
def check(name, cond, extra=""):
    global PASS, FAIL
    PASS, FAIL = PASS + bool(cond), FAIL + (not cond)
    print(("ok   " if cond else "FAIL ") + name, extra if not cond else "")
def four_x5(mod, slide, fn=None):
    grid = mod.prepare_root_grid(slide, thumbnail=Image.new("RGB", (8, 8)))
    saved = mod._best_x5_child
    if fn is not None:
        mod._best_x5_child = fn
    used, boxes = frozenset(), []
    try:
        req = sorted(grid.eligible_ids)[0]
        for rank in range(1, 5):
            p = mod._materialize_x5(slide, grid=grid, requested_root_id=req, rank=rank, used_boxes=used,
                                    artifact_root=tmp, round_number=1, save_navigation_pngs=False)
            used = frozenset((*used, p.box)); boxes.append((p.box.x, p.box.y, p.box.width, p.box.height, round(p.tissue_fraction, 4)))
        return boxes, None
    except PipelineFailure as exc:
        return boxes, str(exc)
    finally:
        mod._best_x5_child = saved
clamped_old = nc.make_clamped(old)
normal = [("panda", i) for i in (0, 1, 2, 3, 4)] + [("gtex", 0), ("gtex", 1), ("tcga", 0), ("tcga_slidebench", 0), ("tcga_expert_vqa", 0)]
for ds, i in normal:
    s = _open_slide(load_cases(DATA / f"{ds}.json", slide_root=DATA / "slides", indices=(i,))[0])
    a, ea = four_x5(old, s); b, eb = four_x5(new, s)
    check(f"normal {ds}#{i} {s.dimensions}: new original == pre-edit original (x5 boxes)", ea is None and eb is None and a == b)
for i in (9, 11, 46, 55, 73, 89, 109, 115, 122, 172, 184):
    s = _open_slide(load_cases(DATA / "panda.json", slide_root=DATA / "slides", indices=(i,))[0])
    a, ea = four_x5(old, s); b, eb = four_x5(new, s); c, ec = four_x5(old, s, clamped_old)
    check(f"narrow panda#{i} {s.dimensions}: pre-edit fails, new == hook clamp", ea is not None and eb is None and b == c, f"{b[:1]} vs {c[:1]}")
print(f"{PASS} passed, {FAIL} failed")
