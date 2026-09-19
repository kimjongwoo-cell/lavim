"""Unit tests for sender_relay_exp/vit_sal_stats.py (run: python tests_hj_test_vit_sal.py)."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent / "sender_relay_exp"))
import vit_sal_stats as S  # noqa: E402

fails = 0
count = 0


def check(name, cond):
    global fails, count
    count += 1
    if not cond:
        fails += 1
        print("FAIL", name)


check("ranks no ties", np.allclose(S.average_ranks([3, 1, 2]), [2, 0, 1]))
check("ranks ties", np.allclose(S.average_ranks([1, 1, 2, 0]), [1.5, 1.5, 3, 0]))
check("spearman +1", abs(S.spearman([1, 2, 3, 4], [10, 20, 30, 45]) - 1) < 1e-12)
check("spearman -1", abs(S.spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1) < 1e-12)
check("spearman monotone nonlinear", abs(S.spearman([1, 2, 3, 4], [1, 8, 27, 64]) - 1) < 1e-12)
check("spearman constant None", S.spearman([1, 1, 1], [1, 2, 3]) is None)
check("spearman short None", S.spearman([1, 2], [1, 2]) is None)
# ties: matches Pearson of average ranks computed by hand
x = [1, 1, 2, 3]; y = [1, 2, 2, 3]
rx = np.array([0.5, 0.5, 2, 3]); ry = np.array([0, 1.5, 1.5, 3])
check("spearman ties", abs(S.spearman(x, y) - np.corrcoef(rx, ry)[0, 1]) < 1e-12)
try:
    S.spearman([1, 2, 3], [1, 2]); check("length mismatch raises", False)
except ValueError:
    check("length mismatch raises", True)

rng = np.random.default_rng(0)
z = rng.normal(size=4000)
a = z + 0.3 * rng.normal(size=4000)
b = z + 0.3 * rng.normal(size=4000)
check("confounded raw high", S.spearman(a, b) > 0.8)
check("partial removes confound", abs(S.partial_spearman(a, b, z)) < 0.1)
c = rng.normal(size=4000)
d = c + 0.2 * rng.normal(size=4000)
check("partial keeps direct link", S.partial_spearman(c, d, z) > 0.9)
check("partial constant z = spearman", abs(S.partial_spearman(c, d, np.ones(4000)) - S.spearman(c, d)) < 1e-12)

g = S.grid_coords(16, 16)
check("idx raster", g["idx"][17] == 17 and g["row"][17] == 1 and g["col"][17] == 1)
check("dist symmetric", np.isclose(g["dist"][0], g["dist"][255]) and np.isclose(g["dist"][15], g["dist"][240]))
check("dist centre min", g["dist"].min() == g["dist"][7 * 16 + 7] and np.isclose(g["dist"].min(), np.hypot(.5, .5)))
g2 = S.grid_coords(3, 5)
check("dist non-square centre", np.isclose(g2["dist"][1 * 5 + 2], 0.0))

spans = S.crop_spans([[1, 32, 32], [1, 32, 24]])
check("spans", spans == [(0, 256, 16, 16), (256, 448, 16, 12)])
vals = np.concatenate([np.full(256, 1 / 256), np.full(192, 2 / 192)])
check("crop_map uniform=1 a", np.allclose(S.crop_map(vals, spans[0]), 1))
check("crop_map uniform=1 b", np.allclose(S.crop_map(vals, spans[1]), 1))

check("ratio uniform 1", np.isclose(S.centre_border_ratio(np.ones((16, 16))), 1))
m = np.ones((16, 16)); m[4:12, 4:12] = 3
check("ratio centre 3", np.isclose(S.centre_border_ratio(m), 3))

base = rng.normal(size=(1, 64))
maps = np.concatenate([base + 0.01 * rng.normal(size=(1, 64)) for _ in range(6)])
groups = ["c0", "c0", "c1", "c1", "c2", "c2"]
r = S.loo_template_r(maps, groups)
check("loo shared template high", all(v > 0.99 for v in r))
maps2 = rng.normal(size=(6, 64))
check("loo random low", np.mean([abs(v) for v in S.loo_template_r(maps2, groups)]) < 0.3)
check("loo single group None", S.loo_template_r(maps2[:2], ["a", "a"]) == [None, None])
# the leave-out must exclude the whole case: case c0 = two identical maps, others random
maps3 = rng.normal(size=(6, 64)); maps3[1] = maps3[0]
r3 = S.loo_template_r(maps3, groups)
check("loo excludes same case", abs(r3[0]) < 0.5)

img = Image.new("RGB", (64, 32), (255, 255, 255))
img.paste(Image.new("RGB", (32, 32), (200, 40, 120)), (0, 0))
tb = S.tissue_blocks(img, 1, 2)
check("tissue left stained right white", np.allclose(tb, [1.0, 0.0]))
tb2 = S.tissue_blocks(img.resize((128, 64)), 1, 2)
check("tissue resized input", np.allclose(tb2, [1.0, 0.0], atol=0.05))

s = S.summarize([0.1, -0.2, None, 0.3])
check("summarize", s["n"] == 3 and np.isclose(s["median"], 0.1) and np.isclose(s["frac_neg"], 1 / 3))
check("summarize empty", S.summarize([None]) == {"n": 0})

print(f"{count - fails}/{count} passed")
sys.exit(1 if fails else 0)
