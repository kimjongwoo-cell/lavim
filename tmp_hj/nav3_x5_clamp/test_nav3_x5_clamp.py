"""CPU tests for tmp_hj/nav3_x5_clamp. Run from the repo root:
    python tmp_hj/nav3_x5_clamp/test_nav3_x5_clamp.py
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path("/home/users/whddn12316/wsi_latent_0915_decode_hj")
HERE = Path(__file__).resolve().parent
PY = sys.executable
PASS = 0


def check(name, cond, detail=""):
    global PASS
    if not cond:
        raise AssertionError(f"{name} {detail}")
    PASS += 1
    print("ok", name, detail)


def run(code, env_extra, pythonpath):
    env = {**os.environ, "PYTHONPATH": pythonpath, "CUDA_VISIBLE_DEVICES": "", **env_extra}
    env.pop("VLMAS_NAV3_X5_CLAMP", None) if "VLMAS_NAV3_X5_CLAMP" not in env_extra else None
    r = subprocess.run([PY, "-c", code], cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
    return r.returncode, r.stdout, r.stderr


probe = "import vision_text_mas.onepass_navigation_roots as m; print('QUAL', m._best_x5_child.__qualname__)"
rc, out, err = run(probe, {}, f"{HERE}:{REPO}")
check("no env -> original function", rc == 0 and "QUAL _best_x5_child\n" in out and "NAV3-X5CLAMP" not in out, out + err[-300:])
rc, out, err = run(probe, {"VLMAS_NAV3_X5_CLAMP": "1"}, str(REPO))
check("env without PYTHONPATH -> original function", rc == 0 and "QUAL _best_x5_child\n" in out, out + err[-300:])
rc, out, err = run(probe, {"VLMAS_NAV3_X5_CLAMP": "1"}, f"{HERE}:{REPO}")
check("env + PYTHONPATH -> patched", rc == 0 and "[NAV3-X5CLAMP] active" in out and "nav3_x5_clamp" in out, out + err[-300:])

# md5 guard
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
import nav3_x5_clamp as nc  # noqa: E402
import types  # noqa: E402

with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
    fh.write("x = 1\n")
fake = types.SimpleNamespace(__file__=fh.name, _best_x5_child="orig")
check("md5 mismatch -> no patch", nc.patch(fake) is False and fake._best_x5_child == "orig")

# behaviour on real PANDA slides (original vs clamped, same process)
from PIL import Image  # noqa: E402

import vision_text_mas.onepass_navigation_roots as m  # noqa: E402
from vision_text_mas.cli import _open_slide  # noqa: E402
from vision_text_mas.dataset import load_cases  # noqa: E402
from vision_text_mas.errors import PipelineFailure  # noqa: E402

DATA = Path("/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda")
orig = m._best_x5_child
clamped = nc.make_clamped(m)
tmp = Path(tempfile.mkdtemp())


def four_x5(slide, fn):
    grid = m.prepare_root_grid(slide, thumbnail=Image.new("RGB", (8, 8)))
    m._best_x5_child = fn
    used = frozenset()
    boxes = []
    try:
        requested = sorted(grid.eligible_ids)[0]
        for rank in range(1, 5):
            p = m._materialize_x5(slide, grid=grid, requested_root_id=requested, rank=rank, used_boxes=used,
                                  artifact_root=tmp, round_number=1, save_navigation_pngs=False)
            used = frozenset((*used, p.box))
            boxes.append((p.box.x, p.box.y, p.box.width, p.box.height, round(p.tissue_fraction, 4)))
        return boxes, None
    except PipelineFailure as exc:
        return boxes, str(exc)
    finally:
        m._best_x5_child = orig


for i in (0, 1, 2, 3, 4):
    s = _open_slide(load_cases(DATA / "panda.json", slide_root=DATA / "slides", indices=(i,))[0])
    a, ea = four_x5(s, orig)
    b, eb = four_x5(s, clamped)
    check(f"normal slide {i}: identical x5 boxes", ea is None and eb is None and a == b, f"{a[:1]}")
for i in (9, 11, 46, 55, 73, 89, 109, 115, 122, 172, 184):
    s = _open_slide(load_cases(DATA / "panda.json", slide_root=DATA / "slides", indices=(i,))[0])
    a, ea = four_x5(s, orig)
    b, eb = four_x5(s, clamped)
    check(f"narrow slide {i} {s.dimensions}: original fails, clamp gives 4 distinct tissue-safe x5",
          ea is not None and eb is None and len(set(x[:4] for x in b)) == 4 and min(x[4] for x in b) >= m.MINIMUM_TISSUE_FRACTION,
          f"side {b[0][2] if b else None} fractions {[x[4] for x in b]}")
print(f"{PASS}/{PASS} passed")
