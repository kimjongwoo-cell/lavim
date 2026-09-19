"""Unit tests for the pure helpers of sender_relay_exp/vit_sink_probe.py. Run: python tests_hj_test_vit_sink.py"""
import sys

import numpy as np

sys.path.insert(0, "/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp")
import vit_sink_probe as V  # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"PASS {name}")
    else:
        FAIL += 1
        print(f"FAIL {name} {detail}")


rng = np.random.default_rng(0)
g = 16
content = rng.random((g, g))                      # tissue pattern in slide coordinates
positional = np.zeros((g, g)); positional[2, 15] = 50; positional[13, 0] = 40   # fixed sink cells
k = 3
# box shifted +k cells in x: content moves -k cells in the image
content_shift = np.zeros((g, g)); content_shift[:, : g - k] = content[:, k:]; content_shift[:, g - k:] = rng.random((g, k))
f, c = V.shift_alignment(content.ravel(), content_shift.ravel(), g, k, "x")
check("content-driven map aligns with content", c is not None and c > 0.99 and (f is None or f < 0.5), f"{f} {c}")
pos0 = (positional + 0.01 * content).ravel()
posk = (positional + 0.01 * content_shift).ravel()
f2, c2 = V.shift_alignment(pos0, posk, g, k, "x")
check("rank correlation is dominated by the bulk (documented weakness)", c2 > f2, f"{f2} {c2}")
hot = np.array([2 * g + 15, 13 * g + 0, 13 * g + 3])
hp = V.hot_fixed_vs_moved(pos0, posk, hot, g, k, "x")
check("positional sink: fixed >> moved", hp["hot_fixed_mean"] > 10 * max(hp["hot_moved_mean"], 1e-9), str(hp))
check("positional sink: top1 stays", hp["top1_fixed"] > hp["top1_moved"], str(hp))
hx = V.hot_fixed_vs_moved(pos0, posk, np.array([13 * g + 3, 13 * g + 0]), g, k, "x")
check("moved cell that is itself hot is skipped", hx["hot_moved_mean"] is None, str(hx))
cont0 = content.copy(); cont0[5, 9] = 30.0
contk = np.zeros((g, g)); contk[:, : g - k] = cont0[:, k:]; contk[:, g - k:] = rng.random((g, k))
hc = V.hot_fixed_vs_moved(cont0.ravel(), contk.ravel(), np.array([5 * g + 9]), g, k, "x")
check("content peak: moved >> fixed", hc["hot_moved_mean"] > 10 * hc["hot_fixed_mean"], str(hc))
check("content peak: top1 moves", hc["top1_moved"] > hc["top1_fixed"], str(hc))
content_y = np.zeros((g, g)); content_y[: g - k, :] = content[k:, :]; content_y[g - k:, :] = rng.random((k, g))
fy, cy = V.shift_alignment(content.ravel(), content_y.ravel(), g, k, "y")
check("y-axis content alignment", cy > 0.99, f"{fy} {cy}")

t = np.arange(256, dtype=float)
r8 = V.resample_relative(t, 8).reshape(8, 8)
check("resample relative corners", r8[0, 0] == t.reshape(16, 16)[1, 1] and r8[-1, -1] == t.reshape(16, 16)[15, 15], str(r8[0, 0]))
check("resample identity at 16", np.array_equal(V.resample_relative(t, 16), t))
m24 = np.zeros((24, 24)); m24[:16, :16] = t.reshape(16, 16)
check("absolute overlap uses shared top-left", abs(V.absolute_overlap(t, m24.ravel(), 24) - 1.0) < 1e-12)
check("hot cells order", V.hot_cells(np.array([0.1, 5, 3, 9]), 2).tolist() == [3, 1])
mb = np.arange(24 * 4, dtype=float).reshape(24, 4)
rep = V.with_mean_block(mb)
check("block report keys", set(rep) == {"mean", "0", "3", "12", "23"} and np.allclose(rep["mean"], mb.mean(0)))
print(f"\n{PASS}/{PASS + FAIL} passed")
sys.exit(1 if FAIL else 0)
