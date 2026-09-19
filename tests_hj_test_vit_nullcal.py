"""Unit tests for sender_relay_exp/vit_nullcal.py pure helpers (numpy only). Run: python3 tests_hj_test_vit_nullcal.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "sender_relay_exp"))

import numpy as np  # noqa: E402

import vit_nullcal as nc  # noqa: E402

PASS = 0


def check(name, cond):
    global PASS
    if not cond:
        raise AssertionError(name)
    PASS += 1
    print("ok", name)


rng = np.random.default_rng(1)

# calibrate: exact division when eps = 0, broadcasting
a = rng.random((3, 5)) + 0.1
p = rng.random(5) + 0.1
check("calibrate division", np.allclose(nc.calibrate(a, p[None], 0.0), a / p))
check("calibrate eps", np.allclose(nc.calibrate(a, p, 0.5), (a + 0.5) / (p + 0.5)))

# loco_means: brute force
maps = rng.random((6, 4))
groups = ["x", "x", "y", "z", "z", "z"]
lm = nc.loco_means(maps, groups)
ok = True
for i, g in enumerate(groups):
    other = [j for j, h in enumerate(groups) if h != g]
    ok &= np.allclose(lm[i], maps[other].mean(axis=0))
check("loco brute force", ok)
check("loco single group nan", np.isnan(nc.loco_means(maps[:2], ["a", "a"])).all())

# position_eta2: pure positional pattern -> 1, pure noise -> ~ small
pattern = rng.normal(size=16)
check("eta2 pure position", abs(nc.position_eta2(np.tile(pattern, (50, 1))) - 1.0) < 1e-9)
noise = rng.normal(size=(400, 16))
check("eta2 noise small", nc.position_eta2(noise) < 0.05)
check("eta2 per-crop level removed", abs(nc.position_eta2(np.tile(pattern, (50, 1)) + rng.normal(size=(50, 1)) * 10) - 1.0) < 1e-9)

# dividing a multiplicative template out removes position (eta2 on log)
tmpl = np.exp(rng.normal(size=16))
content = np.exp(rng.normal(size=(300, 16)) * 0.3)
biased = content * tmpl
check("eta2 biased high", nc.position_eta2(np.log(biased)) > 0.7)
check("eta2 after exact calibration low", nc.position_eta2(np.log(nc.calibrate(biased, tmpl, 0.0))) < 0.05)

# mean_percentile
v = np.arange(10, dtype=float)
check("percentile top", nc.mean_percentile(v, np.array([9])) == 1.0)
check("percentile bottom", nc.mean_percentile(v, np.array([0])) == 0.0)
check("percentile mid", abs(nc.mean_percentile(v, np.array([0, 9])) - 0.5) < 1e-12)

# border mask
bm = nc.border_mask(16).reshape(16, 16)
check("border count", bm.sum() == 60 and bm[0].all() and not bm[1, 1])

# crop_metrics on constructed map: top cells = tissue cells
coords = nc.grid_coords(16, 16)
tissue = np.zeros(256)
tissue[:64] = 1.0
v = np.zeros(256)
v[:64] = 2.0
raw = v.copy()
m = nc.crop_metrics(v + rng.random(256) * 1e-3, raw, tissue, np.ones(256) + rng.random(256), np.arange(8), coords, nc.border_mask(16))
check("metrics tissue diff", abs(m["top25_tissue_minus_bottom"] - 1.0) < 1e-12)
check("metrics tissue minus crop", abs(m["top25_tissue_minus_crop"] - 0.75) < 1e-12)
check("metrics overlap raw", m["top25_overlap_raw"] == 1.0)
check("metrics hot in top", m["hot8_in_top25"] == 1.0)

print(f"{PASS}/{PASS} passed")
