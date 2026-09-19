"""Unit tests: Directed Cross-Scale Submodular Visual KV Selection (VLMAS_WSI_CONSOL_MODE=dcs)."""
import itertools
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

from memory.dcs_select import (
    dcs_affinity, dcs_greedy, dcs_keep_mask, debiased_salience,
    facility_value, physical_kernel)
from memory.wsi_memory import apply_parent_shuffle, token_footprints

# coarse crop C (x5, 4x4 grid, 800px box -> 200px cells);
# fine crop F (x20, 4x4 grid) = top-left 200x200 of C -> 50px cells
counts = (16, 16)
grids = ((4, 4), (4, 4))
boxes = ((0, 0, 800, 800), (0, 0, 200, 200))
mags = (5, 20)


def _mags_t(m=mags, c=counts):
    return torch.repeat_interleave(torch.tensor([float(x) for x in m]), torch.tensor(c))


def test_directed_phys_asymmetry():
    rects, _ = token_footprints(boxes, grids)
    kd = physical_kernel(rects, "directed")
    ks = physical_kernel(rects, "symmetric")
    fine0, coarse0 = 16, 0            # F token 0 lies inside C token 0
    assert abs(float(kd[fine0, coarse0]) - 1.0) < 1e-6
    assert abs(float(kd[coarse0, fine0]) - 1.0 / 16) < 1e-6
    assert abs(float(ks[fine0, coarse0]) - 1.0 / 16) < 1e-6
    assert abs(float(ks[coarse0, fine0]) - 1.0 / 16) < 1e-6
    # F token outside C token 5 -> no relation
    assert float(kd[fine0, 5]) == 0.0


def test_affinity_rule_same_vs_cross():
    torch.manual_seed(0)
    z = torch.randn(32, 8)
    rects, _ = token_footprints(boxes, grids)
    m = _mags_t()
    a = dcs_affinity(z, m, rects, "directed", "merger")
    zn = torch.nn.functional.normalize(z, dim=-1)
    ksem = (zn @ zn.T).clamp(min=0)
    kd = physical_kernel(rects, "directed")
    # same scale -> k_sem only (even without physical overlap)
    assert torch.allclose(a[1, 2], ksem[1, 2])
    assert torch.allclose(a[17, 30], ksem[17, 30])
    # cross scale -> k_sem * k_phys(i<-j)
    assert torch.allclose(a[16, 0], ksem[16, 0] * kd[16, 0])
    assert torch.allclose(a[0, 16], ksem[0, 16] * kd[0, 16])
    assert float(a[16, 5]) == 0.0
    assert torch.allclose(torch.diagonal(a), torch.ones(32))
    # spatial-only: a = k_phys for all pairs (same-scale non-overlap -> 0)
    a0 = dcs_affinity(None, m, rects, "directed", "none")
    assert float(a0[1, 2]) == 0.0 and abs(float(a0[16, 0]) - 1.0) < 1e-6


def test_debiased_salience():
    # 3 crops, 2x2 grids; position 0 is high in every crop (common bias)
    u = torch.tensor([9., 1., 1., 1.,
                      9., 1., 5., 1.,
                      9., 1., 1., 1.])
    s, info = debiased_salience(u, (4, 4, 4), ((2, 2), (2, 2), (2, 2)), eps=1e-6)
    assert torch.all(s >= 0)
    assert float(s[0]) == 0.0 and float(s[4]) == 0.0 and float(s[8]) == 0.0
    assert float(s[6]) > 1.5                           # log 5 - log 1
    assert info["b_fallback_crops"] == 0 and info["b_groups"] == 1


def test_debiased_unique_grid_fallback():
    u = torch.tensor([1., 2., 3., 4., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    s, info = debiased_salience(u, (4, 16), ((2, 2), (4, 4)))
    assert info["b_fallback_crops"] == 2 and s.shape[0] == 20


def test_budget_exact_and_identity():
    torch.manual_seed(1)
    z = torch.randn(32, 8)
    rel = torch.rand(32)
    for B in (1, 3, 8, 16, 31):
        keep, info = dcs_keep_mask(z, rel, counts, grids, mags, boxes, B, return_info=True)
        assert int(keep.sum()) == B, (B, int(keep.sum()))
        assert sum(info["per_crop"]) == B
    keep = dcs_keep_mask(z, rel, counts, grids, mags, boxes, 40)
    assert bool(keep.all())


def test_greedy_matches_oracle_F_and_first_pick():
    torch.manual_seed(2)
    z = torch.randn(32, 8)
    rel = torch.rand(32)
    rects, _ = token_footprints(boxes, grids)
    m = _mags_t()
    a = dcs_affinity(z, m, rects, "directed", "merger")
    s, _ = debiased_salience(rel, counts, grids)
    keep, _ = dcs_greedy(a, s, 6)
    k2, info = dcs_keep_mask(z, rel, counts, grids, mags, boxes, 6, return_info=True)
    assert torch.equal(keep.cpu(), k2.cpu())
    assert abs(info["F"] - facility_value(a, s, keep)) < 1e-3
    first = int((s[:, None] * a).sum(0).argmax())
    k1, _ = dcs_greedy(a, s, 1)
    assert bool(k1[first])


def test_monotone_submodular():
    torch.manual_seed(3)
    random.seed(3)
    z = torch.randn(32, 8)
    rects, _ = token_footprints(boxes, grids)
    a = dcs_affinity(z, _mags_t(), rects, "directed", "merger")
    s = torch.rand(32)
    for _ in range(200):
        T = set(random.sample(range(32), 8))
        S = set(random.sample(sorted(T), 4))
        v = random.choice([x for x in range(32) if x not in T])
        def val(X):
            k = torch.zeros(32, dtype=torch.bool); k[list(X)] = True
            return facility_value(a, s, k)
        gS = val(S | {v}) - val(S)
        gT = val(T | {v}) - val(T)
        assert gS >= gT - 1e-5 and gT >= -1e-6


def test_directed_vs_symmetric_differs():
    # salience only on fine tokens; spatial-only affinity. Directed: one coarse
    # token fully represents its 16 fine tokens -> it is the first pick.
    # Symmetric (IoU 1/16): the coarse token is worth much less.
    rel = torch.zeros(32)
    rel[16:] = 1.0
    rel[20] = 2.0                     # breaks the symmetric tie (1 vs 17/16)
    kd, _ = dcs_keep_mask(None, rel, counts, grids, mags, boxes, 1,
                          aff="directed", z_mode="none", sal="raw", return_info=True)
    ks, _ = dcs_keep_mask(None, rel, counts, grids, mags, boxes, 1,
                          aff="symmetric", z_mode="none", sal="raw", return_info=True)
    assert bool(kd[0]) and not bool(ks[0])


def test_no_hard_precedence():
    # all demand on two fine tokens, semantic kernel orthogonal to coarse:
    # greedy keeps fine tokens without any parent token (allowed by design).
    z = torch.zeros(32, 4)
    z[:16, 0] = 1.0                   # coarse tokens
    z[16:, 1] = 1.0                   # fine tokens (orthogonal -> a_ij = 0 cross-scale)
    rel = torch.zeros(32)
    rel[16] = 1.0; rel[21] = 1.0
    keep, info = dcs_keep_mask(z, rel, counts, grids, mags, boxes, 1,
                               sal="raw", return_info=True)
    assert info["per_crop"][1] >= 1 and info["per_crop"][0] == 0


def test_no_boxes_no_cross_scale():
    torch.manual_seed(4)
    z = torch.randn(32, 8)
    rects, _ = token_footprints((None, None), grids)
    a = dcs_affinity(z, _mags_t(), rects, "directed", "merger")
    assert float(a[:16, 16:].abs().sum()) == 0.0 and float(a[16:, :16].abs().sum()) == 0.0


def test_saturation_fills_to_budget():
    rel = torch.zeros(32)
    rel[3] = 1.0
    keep, info = dcs_keep_mask(None, rel, counts, grids, mags, boxes, 10,
                               z_mode="none", sal="raw", return_info=True)
    assert int(keep.sum()) == 10 and info["saturated_at"] == 1


def test_shuffled_parent_changes_affinity():
    # two coarse crops, one fine child under the first
    c3 = (16, 16, 16)
    g3 = ((4, 4), (4, 4), (4, 4))
    b3 = ((0, 0, 800, 800), (2000, 0, 800, 800), (0, 0, 200, 200))
    m3 = (5, 5, 20)
    sh = apply_parent_shuffle(b3, m3)
    assert sh[2] != b3[2]
    torch.manual_seed(5)
    z = torch.randn(48, 8)
    mt = _mags_t(m3, c3)
    a_true = dcs_affinity(z, mt, token_footprints(b3, g3)[0], "directed", "merger")
    a_shuf = dcs_affinity(z, mt, token_footprints(sh, g3)[0], "directed", "merger")
    assert float(a_true[32:, :16].sum()) > 0 and float(a_shuf[32:, :16].sum()) == 0
    assert float(a_shuf[32:, 16:32].sum()) > 0


def test_uniform_fallback_without_q2v():
    torch.manual_seed(6)
    z = torch.randn(32, 8)
    keep, info = dcs_keep_mask(z, None, counts, grids, mags, boxes, 5, return_info=True)
    assert info["sal"].startswith("uniform") and int(keep.sum()) == 5


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_")]
    ok = 0
    for name, fn in tests:
        fn()
        ok += 1
        print(f"PASS {name}")
    print(f"{ok}/{len(tests)} passed")
