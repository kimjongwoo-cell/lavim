"""SEC-LD (C1 #19) unit tests. python3 tests_hj_test_secld.py

Pixel layout against the real HF processor and a PIL reference, the Gaussian against cv2 with
replicate border, the support-wise log-det greedy against brute force and against the single-support
QASC greedy, and select() bookkeeping (no quota, empty crops allowed, optional minimum).
"""
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from memory import secld_select as S  # noqa: E402
from memory.qasc_select import logdet_greedy  # noqa: E402

PASS = FAIL = 0
torch.manual_seed(0)
np.random.seed(0)
MODEL = "/home/users/whddn12316/models/Qwen3-VL-4B-Thinking"


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print("FAIL", name, extra)


# ------------------------------------------------------------------ config
os.environ.pop("VLMAS_SECLD", None)
check("off", not S.enabled())
os.environ["VLMAS_SECLD"] = "1"
check("on", S.enabled())
cfg = S.config()
check("defaults", cfg["tau_p"] == 215.0 and cfg["tau_c"] == 0.9 and cfg["sigma"] == 2.0
      and cfg["keep"] is None and cfg["min_per_support"] == 0, cfg)

# ------------------------------------------------------------------ pixel layout vs HF processor
from PIL import Image  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

proc = AutoProcessor.from_pretrained(MODEL)
ip = proc.image_processor
check("processor geometry", ip.patch_size == 16 and ip.merge_size == 2 and ip.temporal_patch_size == 2
      and list(ip.image_mean) == [0.5] * 3 and list(ip.image_std) == [0.5] * 3)


def ref_cells(im: Image.Image, tau_p=215.0, cell=32):
    """glass_fat_5ds.py rule: PIL 'L' luminance, per 32 px cell, fraction of gray > tau_p."""
    g = np.asarray(im.convert("L"), dtype=np.float64)
    gh, gw = g.shape[0] // cell, g.shape[1] // cell
    wm = (g > tau_p).astype(np.float64)
    return wm[:gh * cell, :gw * cell].reshape(gh, cell, gw, cell).mean(axis=(1, 3)).ravel()


def ref_cells_float(arr: np.ndarray, tau_p=215.0, cell=32):
    x = arr.astype(np.float64)
    g = x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114
    gh, gw = g.shape[0] // cell, g.shape[1] // cell
    return (g > tau_p).astype(np.float64)[:gh * cell, :gw * cell].reshape(gh, cell, gw, cell).mean(axis=(1, 3)).ravel()


# structured image: white block top-left 4x8 cells, a half-bright cell, pink elsewhere
arr = np.zeros((512, 512, 3), dtype=np.uint8)
arr[..., :] = (200, 150, 180)                    # gray ~168
arr[0:128, 0:256] = 255                           # cells rows 0..3, cols 0..7 fully bright
arr[160:176, 288:320] = 255                       # half of cell (5, 9)
arr[400:512, 448:512] = (230, 230, 230)           # gray 230 > 215 -> bright, non-white
im1 = Image.fromarray(arr)
# random image with a second (non-square) crop size to exercise multi-image concatenation
arr2 = np.random.randint(0, 256, size=(256, 384, 3), dtype=np.uint8)
im2 = Image.fromarray(arr2)
inputs = proc.image_processor(images=[im1, im2], return_tensors="pt")
pv, thw = inputs["pixel_values"], inputs["image_grid_thw"]
check("grid 16x16 + 8x12", thw.tolist() == [[1, 32, 32], [1, 16, 24]], thw.tolist())
r = S.cell_bright_ratio(pv, thw)
check("cells count", int(r.numel()) == 256 + 96)
ref1 = np.concatenate([ref_cells_float(arr), ref_cells_float(arr2)])
check("r ~= float-luminance reference (rounding at the threshold only)", float(np.abs(r.numpy() - ref1).max()) <= 8 / 1024,
      float(np.abs(r.numpy() - ref1).max()))
refL = np.concatenate([ref_cells(im1), ref_cells(im2)])
check("r == PIL 'L' reference (audit rule, exact)", float(np.abs(r.numpy() - refL).max()) < 1e-6,
      float(np.abs(r.numpy() - refL).max()))
r1 = r[:256].reshape(16, 16)
check("white block cells == 1", bool((r1[:4, :8] == 1).all()))
check("half cell == 0.5", abs(float(r1[5, 9]) - 0.5) < 1e-6, float(r1[5, 9]))
check("pink cells == 0", bool((r1[6:12, :] == 0).all()))
check("gray-230 block bright", bool((r1[13:16, 14:16] == 1).all()))
# bf16 pixel_values (the backbone casts pv to model dtype) must give the same cells here
r_bf = S.cell_bright_ratio(pv.to(torch.bfloat16), thw)
check("bf16 pixel_values: structured image identical", float((r_bf - r)[:256].abs().max()) < 1e-6)
check("bf16 pixel_values: random image within a few px per cell", float((r_bf - r).abs().max()) <= 8 / 1024,
      float((r_bf - r).abs().max()))
# wrong grid must raise
try:
    S.cell_bright_ratio(pv, torch.tensor([[1, 32, 32]]))
    check("grid mismatch raises", False)
except ValueError:
    check("grid mismatch raises", True)

# ------------------------------------------------------------------ Gaussian vs cv2 replicate
import cv2  # noqa: E402

for sigma in (1.5, 2.0, 4.0):
    m = (torch.rand(16 * 16 + 8 * 12) > 0.6).float()
    b = S.occupancy(m, [(16, 16), (8, 12)], sigma)
    ref = np.concatenate([
        cv2.GaussianBlur(m[:256].reshape(16, 16).numpy(), (0, 0), sigma, borderType=cv2.BORDER_REPLICATE).ravel(),
        cv2.GaussianBlur(m[256:].reshape(8, 12).numpy(), (0, 0), sigma, borderType=cv2.BORDER_REPLICATE).ravel(),
    ])
    check(f"gaussian sigma={sigma} == cv2 replicate", float(np.abs(b.numpy() - ref).max()) < 1e-5,
          float(np.abs(b.numpy() - ref).max()))
check("kernel taps sigma=2 -> 17", S.gaussian_kernel(2.0).shape == (17, 17))
check("kernel normalised", abs(float(S.gaussian_kernel(2.0).sum()) - 1.0) < 1e-6)
allw = S.occupancy(torch.ones(256), [(16, 16)], 2.0)
check("all-bright crop -> b == 1 everywhere (replicate, no white prior needed)", float((allw - 1).abs().max()) < 1e-6)
allt = S.occupancy(torch.zeros(256), [(16, 16)], 2.0)
check("all-tissue crop -> b == 0 at the border too", float(allt.abs().max()) < 1e-6)
ev = S.evidence(pv, thw, cfg)
check("evidence shapes", all(int(ev[k].numel()) == 352 for k in ("r", "m", "b", "e")))
check("e = 1 - b", float((ev["e"] - (1 - ev["b"])).abs().max()) < 1e-6)
e1 = ev["e"][:256].reshape(16, 16)
check("deep inside white block: low demand", float(e1[0, 0]) < 0.25, float(e1[0, 0]))
check("deep tissue: high demand", float(e1[9, 3]) > 0.95, float(e1[9, 3]))
check("white/tissue edge: in between", 0.25 < float(e1[4, 4]) < 0.75, float(e1[4, 4]))


# ------------------------------------------------------------------ greedy vs brute force
def F_sum(q, support, S_idx):
    tot = 0.0
    for g in support.unique().tolist():
        rows = [j for j in S_idx if int(support[j]) == g]
        if rows:
            Q = q[rows].double()
            tot += float(torch.logdet(torch.eye(Q.shape[0], dtype=torch.float64) + Q @ Q.T))
    return tot


def brute_greedy(q, support, budget):
    S_idx, gains = [], []
    for _ in range(budget):
        base = F_sum(q, support, S_idx)
        best, bj = -1.0, -1
        for j in range(q.shape[0]):
            if j in S_idx:
                continue
            v = F_sum(q, support, S_idx + [j]) - base
            if v > best + 1e-12:
                best, bj = v, j
        S_idx.append(bj)
        gains.append(best)
    return S_idx, gains


d, N = 6, 14
support = torch.tensor([0] * 5 + [1] * 4 + [2] * 5)
z = torch.nn.functional.normalize(torch.randn(N, d, dtype=torch.float64), dim=-1)
e = torch.rand(N, dtype=torch.float64)
e[3] = 0.0                                              # a zero-demand token
q = e.sqrt().unsqueeze(1) * z
order, gains, d2s = S.support_logdet_greedy(q, support, 9)
bo, bg = brute_greedy(q, support, 9)
check("greedy order == brute force", order.tolist() == bo, (order.tolist(), bo))
check("greedy gains == brute force", max(abs(a - b) for a, b in zip(gains, bg)) < 1e-9)
# gain identity log(1 + e z^T M^{-1} z) for the 5th pick
Sg = [j for j in order[:4].tolist() if int(support[j]) == int(support[order[4]])]
M = torch.eye(d, dtype=torch.float64) + sum(float(e[j]) * torch.outer(z[j], z[j]) for j in Sg) if Sg else torch.eye(d, dtype=torch.float64)
j5 = int(order[4])
ident = math.log(1.0 + float(e[j5]) * float(z[j5] @ torch.linalg.solve(M, z[j5])))
check("gain == log(1 + e z^T M_g^{-1} z)", abs(ident - gains[4]) < 1e-9, (ident, gains[4]))
check("novelty from d2", abs((d2s[4] - 1.0) / float(e[j5]) - float(z[j5] @ torch.linalg.solve(M, z[j5]))) < 1e-9)
# full budget: zero-demand token comes last with zero gain
order_all, gains_all, _ = S.support_logdet_greedy(q, support, N)
check("zero-demand token picked last", int(order_all[-1]) == 3 and abs(gains_all[-1]) < 1e-12)
# single support == QASC greedy (same kernel, no mask)
sup1 = torch.zeros(N, dtype=torch.long)
o1, g1, _ = S.support_logdet_greedy(q, sup1, 8)
o2, g2 = logdet_greedy(q, 8)
check("single support == qasc logdet_greedy", o1.tolist() == o2.tolist() and max(abs(a - b) for a, b in zip(g1, g2)) < 1e-9)
# supports are independent: same-direction tokens in different crops are NOT redundant
zz = torch.nn.functional.normalize(torch.randn(1, d, dtype=torch.float64), dim=-1).repeat(4, 1)
qq = zz * math.sqrt(0.9)
o3, g3, _ = S.support_logdet_greedy(qq, torch.tensor([0, 0, 1, 1]), 4)
check("cross-support repeat keeps full gain", abs(g3[0] - g3[1]) < 1e-9 and g3[2] < g3[1] - 1e-6, g3)
# forced minimum
o4, g4, _ = S.support_logdet_greedy(q, support, 3, first_per_support=1)
check("min 1 per support -> one from each", sorted(int(support[j]) for j in o4.tolist()) == [0, 1, 2])

# ------------------------------------------------------------------ select() bookkeeping
states = torch.randn(352, 32)
info = S.select(pv, thw, states, [256, 96], 88, cfg)
check("keep == budget", int(info["keep"].sum()) == 88 and info["budget"] == 88)
check("per_support sums", sum(info["per_support"]) == 88 and len(info["per_support"]) == 2)
check("order matches keep", bool(info["keep"][info["order"]].all()))
check("summary text", "erank_overlap=" in S.summary(info) and "empty_crops=" in S.summary(info))
check("stats finite", all(math.isfinite(info[k]) for k in ("e_sel_mean", "gain_first", "gain_last", "nov_first", "nov_last")))
# an all-bright crop next to a tissue crop gets zero tokens when the budget is small (no quota)
arr_w = np.full((512, 512, 3), 255, dtype=np.uint8)
arr_t = np.random.randint(40, 180, size=(512, 512, 3), dtype=np.uint8)
inp2 = proc.image_processor(images=[Image.fromarray(arr_w), Image.fromarray(arr_t)], return_tensors="pt")
st2 = torch.randn(512, 32)
info2 = S.select(inp2["pixel_values"], inp2["image_grid_thw"], st2, [256, 256], 64, cfg)
check("all-white crop gets 0 tokens (no quota)", info2["per_support"][0] == 0 and info2["empty_crops"] == 1, info2["per_support"])
check("tissue crop demand ~1", info2["e_sel_min"] > 0.99, info2["e_sel_min"])
cfg_min = dict(cfg, min_per_support=1)
info3 = S.select(inp2["pixel_values"], inp2["image_grid_thw"], st2, [256, 256], 64, cfg_min)
check("VLMAS_SECLD_MIN=1 forces one token in the white crop", info3["per_support"][0] == 1, info3["per_support"])
try:
    S.select(pv, thw, states[:100], [256, 96], 10, cfg)
    check("state/count mismatch raises", False)
except ValueError:
    check("state/count mismatch raises", True)
# budget > N clamps
info4 = S.select(pv, thw, states, [256, 96], 999, cfg)
check("budget > N clamps", int(info4["keep"].sum()) == 352)

# ------------------------------------------------------------------ realistic size timing (CPU)
big = torch.randn(3072, 1024)
sup = torch.repeat_interleave(torch.arange(12), 256)
qb = torch.rand(3072).sqrt().unsqueeze(1) * torch.nn.functional.normalize(big, dim=-1)
t0 = time.time()
S.support_logdet_greedy(qb.float(), sup, 768)
t_768 = time.time() - t0
t0 = time.time()
S.support_logdet_greedy(qb.float(), sup, 1536)
t_1536 = time.time() - t0
print(f"[timing] CPU N=3072 d=1024: B=768 {t_768:.1f}s, B=1536 {t_1536:.1f}s (threads={torch.get_num_threads()})")

print(f"SECLD tests {PASS}/{PASS + FAIL}")
sys.exit(1 if FAIL else 0)
