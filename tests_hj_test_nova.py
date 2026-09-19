"""NOVA (C1 #21) unit tests. python3 tests_hj_test_nova.py

Adapter over memory/secld_select (whose own 45 tests cover pixels, Gaussian, greedy): dispatch,
budget, no-quota / minimum, the three modes, fp32 pixel override, survivor bookkeeping. CPU, no model.
"""
import os
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ["VLMAS_C1_SELECT"] = "nova"
for k in list(os.environ):
    if k.startswith("VLMAS_NOVA_"):
        del os.environ[k]
from memory import nova_select as NV  # noqa: E402
from memory import secld_select as S  # noqa: E402
from memory import qasc_select as Q  # noqa: E402

PASS = FAIL = 0
torch.manual_seed(0)


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print("FAIL", name, extra)


# ------------------------------------------------------------------ synthetic crops
# two crops, grid (1, 8, 8) each -> 64 patches -> 16 merged tokens (4 x 4) per crop; patch vector
# [3, T=2, 16, 16]; normalised x = pixel/255*2-1 (mean/std 0.5).
P, T = 16, 2
GRID = torch.tensor([[1, 8, 8], [1, 8, 8]])


def crop_pixels(level: int):
    x = torch.full((64, 3 * T * P * P), float(level) / 255.0 * 2.0 - 1.0)
    return x


def pv_of(levels):
    return torch.cat([crop_pixels(v) for v in levels], dim=0)


WHITE, DARK = 255, 102
pv = pv_of([WHITE, DARK])                     # crop 0 = glass, crop 1 = tissue
N, D = 32, 24
feats = torch.randn(N, D)
counts = [16, 16]


class _VLM:
    def get_image_features(self, pixel_values, grid_thw, return_dict=True):
        return types.SimpleNamespace(pooler_output=[feats[:16], feats[16:]])


def make_bb():
    return types.SimpleNamespace(vlm=_VLM(), visual=types.SimpleNamespace(spatial_merge_unit=4),
                                 _prune_image_magnifications=(20, 5))


def setenv(**kw):
    for k in list(os.environ):
        if k.startswith("VLMAS_NOVA_"):
            del os.environ[k]
    for k, v in kw.items():
        os.environ["VLMAS_NOVA_" + k] = str(v)


# ------------------------------------------------------------------ config / budget
setenv()
cfg = NV.config()
check("cfg defaults", cfg["tau_p"] == 215.0 and cfg["tau_c"] == 0.9 and cfg["sigma"] == 2.0
      and cfg["keep"] == 0.25 and cfg["min_per_support"] == 0 and cfg["mode"] == "nova")
setenv(MODE="bogus")
try:
    NV.config()
    check("bad mode raises", False)
except ValueError:
    check("bad mode raises", True)
setenv()
check("budget round", NV.budget_for(32, 0.25) == 8 and NV.budget_for(32, 0.5) == 16
      and NV.budget_for(3072, 0.25) == 768 and NV.budget_for(0, 0.25) == 0 and NV.budget_for(1, 0.1) == 1)
check("enabled", NV.enabled())

# ------------------------------------------------------------------ mode nova, no quota
bb = make_bb()
out, keep = NV.select_vision_features(bb, pv, GRID)
sup = torch.repeat_interleave(torch.arange(2), torch.tensor(counts))
per = [int(keep[sup == g].sum()) for g in range(2)]
check("nova budget 8", int(keep.sum()) == 8, per)
check("nova glass crop empty (no quota)", per == [0, 8], per)
check("nova returns feats[keep]", torch.equal(out, feats[keep]))
check("survivor group ids", torch.equal(bb._prefill_prune_survivor_group_ids, torch.nonzero(keep, as_tuple=True)[0]))
check("survivor image ids", torch.equal(bb._prefill_prune_survivor_image_ids, sup[keep]))
check("survivor scores = e", bb._prefill_prune_survivor_scores.shape == (8,)
      and bool((bb._prefill_prune_survivor_scores > 0.99).all()))
# identical to calling SEC-LD's select directly on the same inputs
ref = S.select(pv, GRID, feats, counts, 8, {**cfg})
check("nova == secld.select", torch.equal(ref["keep"], keep))

# ------------------------------------------------------------------ minimum fallback
setenv(MIN=1)
bb = make_bb()
_, keep1 = NV.select_vision_features(bb, pv, GRID)
per1 = [int(keep1[sup == g].sum()) for g in range(2)]
check("min 1 -> glass crop keeps exactly 1", per1 == [1, 7], per1)
setenv()

# ------------------------------------------------------------------ mode logdet (e == 1)
setenv(MODE="logdet")
bb = make_bb()
_, keep_ld = NV.select_vision_features(bb, pv, GRID)
z = F.normalize(feats.float(), dim=-1)
order_ref, _, _ = S.support_logdet_greedy(z, sup, 8)
ref_keep = torch.zeros(N, dtype=torch.bool)
ref_keep[order_ref] = True
check("logdet == support greedy on z", torch.equal(keep_ld, ref_keep))
check("logdet ignores glass (glass crop non-empty)", int(keep_ld[:16].sum()) > 0, int(keep_ld[:16].sum()))
info_ld = NV.select_tokens(pv, GRID, feats, counts, 8, NV.config())
check("logdet info keys", info_ld["mode"] == "logdet" and len(info_ld["gains"]) == 8
      and info_ld["nov_min"] == info_ld["nov_min"])
setenv()

# ------------------------------------------------------------------ mode etopk
setenv(MODE="etopk")
bb = make_bb()
_, keep_tk = NV.select_vision_features(bb, pv, GRID)
check("etopk = first 8 of the tissue crop (ties -> lowest index)",
      torch.equal(torch.nonzero(keep_tk, as_tuple=True)[0], torch.arange(16, 24)))
info_tk = NV.select_tokens(pv, GRID, feats, counts, 8, NV.config())
check("etopk diagnostics (e ties -> topk overlap < 1 is legal)", 0.5 <= info_tk["erank_overlap"] <= 1.0 and info_tk["gains"] == [])
setenv(MODE="etopk", MIN=1)
info_tk1 = NV.select_tokens(pv, GRID, feats, counts, 8, NV.config())
check("etopk min 1", info_tk1["per_support"] == [1, 7], info_tk1["per_support"])
setenv()

# ------------------------------------------------------------------ fp32 override / fallback
bb = make_bb()
bb._secld_pixel_values_fp32 = pv_of([DARK, WHITE])          # swapped: crop 0 tissue, crop 1 glass
_, keep_sw = NV.select_vision_features(bb, pv, GRID)
per_sw = [int(keep_sw[sup == g].sum()) for g in range(2)]
check("fp32 backbone pixels used when shapes match", per_sw == [8, 0], per_sw)
bb = make_bb()
bb._secld_pixel_values_fp32 = pv_of([DARK, WHITE])[:64]     # wrong shape -> fallback to model input
_, keep_fb = NV.select_vision_features(bb, pv, GRID)
per_fb = [int(keep_fb[sup == g].sum()) for g in range(2)]
check("shape mismatch -> fallback to pixel_values", per_fb == [0, 8], per_fb)
bb = make_bb()
_, keep_bf = NV.select_vision_features(bb, pv.to(torch.bfloat16), GRID)
per_bf = [int(keep_bf[sup == g].sum()) for g in range(2)]
check("bf16 pixel input works", per_bf == [0, 8], per_bf)

# ------------------------------------------------------------------ mixed crop: e follows the null map
half = crop_pixels(DARK)
half[:32] = crop_pixels(WHITE)[:32]                          # merge groups 0..7 = top two token rows white
pv_mix = torch.cat([half, crop_pixels(DARK)], dim=0)
info_mix = NV.select_tokens(pv_mix, GRID, feats, counts, 8, NV.config())
check("mixed crop e_sel > e_all", info_mix["e_sel_mean"] > info_mix["e_all_mean"],
      (info_mix["e_sel_mean"], info_mix["e_all_mean"]))
check("mixed crop m_sel < m_all", info_mix["m_sel_frac"] < info_mix["m_all_frac"])

# ------------------------------------------------------------------ dispatch
sentinel = ("nova", "called")
orig = NV.select_vision_features
NV.select_vision_features = lambda bb, pv_, thw_: sentinel
try:
    check("qasc dispatch -> nova", Q.select_vision_features(None, None, None) is sentinel)
finally:
    NV.select_vision_features = orig
os.environ["VLMAS_C1_SELECT"] = "ntrs"
check("not enabled for ntrs", not NV.enabled())
os.environ["VLMAS_C1_SELECT"] = "nova"

# ------------------------------------------------------------------ realistic size timing
big_counts = [256] * 12
big_pv = torch.cat([crop_pixels(WHITE if i % 4 == 0 else DARK).repeat(16, 1) for i in range(12)], dim=0)  # 12 crops x 1024 patches
big_grid = torch.tensor([[1, 32, 32]] * 12)
big_feats = torch.randn(3072, 256)
import time
t0 = time.time()
info_big = NV.select_tokens(big_pv, big_grid, big_feats, big_counts, 768, NV.config())
check("3072/768 runs", int(info_big["keep"].sum()) == 768 and info_big["empty_crops"] == 3, info_big["per_support"])
print(f"select 3072 -> 768 (d=256, CPU) {time.time() - t0:.2f}s empty_crops={info_big['empty_crops']}")

print(f"PASS {PASS} FAIL {FAIL}")
sys.exit(1 if FAIL else 0)
