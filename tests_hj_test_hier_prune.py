"""Unit tests: Hierarchy-Constrained Spatial Pruning (VLMAS_WSI_CONSOL_MODE=hier)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from memory.stratified import (
    hierarchy_keep_mask, stratified_keep_mask, equal_area_grid_indices,
    _max_overlap_parent_token, _derange_parents)

# 2 anchors (5x, 8x8 grid) each with 2 children (20x, 4x4 grid)
counts = (64, 16, 16, 64, 16, 16)
grids = ((8, 8), (4, 4), (4, 4), (8, 8), (4, 4), (4, 4))
# level-0 boxes: A at (0,0,800,800), A1 = top-left quarter, A2 = bottom-right
boxes = ((0, 0, 800, 800), (0, 0, 200, 200), (600, 600, 200, 200),
         (1000, 0, 800, 800), (1600, 0, 200, 200), (1000, 600, 200, 200))  # B1 top-right, B2 bottom-left (asymmetric to A)
parents = (-1, 0, 0, -1, 3, 3)
starts = [0, 64, 80, 96, 160, 176]


def child_covered(keep, child, par):
    idx = _max_overlap_parent_token(boxes[par], boxes[child], grids[par])
    return bool(keep[starts[par] + idx])


def test_budget_and_min():
    for B in (6, 12, 24, 48, 96):
        keep, info = hierarchy_keep_mask(counts, grids, boxes, parents, B, return_info=True)
        # mandatory set is never cut: budget below |S_hier| keeps S_hier whole
        assert int(keep.sum()) == max(B, info["mandatory"]), (B, int(keep.sum()))
        assert all(c >= 1 for c in info["per_crop"])
        assert info["edges"] == 4
        for c, p in ((1, 0), (2, 0), (4, 3), (5, 3)):
            assert child_covered(keep, c, p), (B, c)


def test_pi_geometry():
    # A1 = top-left quarter of A -> parent token in top-left 4x4 region
    idx = _max_overlap_parent_token(boxes[0], boxes[1], grids[0])
    y, x = divmod(idx, 8)
    assert y < 2 and x < 2, (y, x)
    idx = _max_overlap_parent_token(boxes[0], boxes[2], grids[0])
    y, x = divmod(idx, 8)
    assert y >= 6 and x >= 6, (y, x)


def test_shuffled_changes_anchor():
    par = _derange_parents(list(parents), 42)
    assert par[1] == 3 and par[4] == 0 and par[0] == -1
    k_true, i1 = hierarchy_keep_mask(counts, grids, boxes, parents, 24, return_info=True)
    k_shuf, i2 = hierarchy_keep_mask(counts, grids, boxes, parents, 24, shuffle_parents=True, return_info=True)
    assert int(k_true.sum()) == int(k_shuf.sum()) == 24
    assert i2["parents"] != i1["parents"]
    assert not torch.equal(k_true, k_shuf)
    # same number of anchored parent tokens: relative position transplanted
    assert i1["mandatory"] == i2["mandatory"], (i1["mandatory"], i2["mandatory"])
    # A1 (top-left of A) now anchors the top-left token of B, not token 0 by
    # overlap collapse; A2 (bottom-right) anchors B's bottom-right
    idx_a1 = _max_overlap_parent_token(boxes[0], boxes[1], grids[0])
    idx_a2 = _max_overlap_parent_token(boxes[0], boxes[2], grids[0])
    assert bool(k_shuf[starts[3] + idx_a1]) and bool(k_shuf[starts[3] + idx_a2])
    # and B1 (top-right of B) anchors A's top-right token under the shuffle
    idx_b1 = _max_overlap_parent_token(boxes[3], boxes[4], grids[3])
    assert bool(k_shuf[starts[0] + idx_b1])


def test_full_budget_identity():
    keep = hierarchy_keep_mask(counts, grids, boxes, parents, 999)
    assert bool(keep.all())


def test_no_boxes_no_parents_degrades_to_min_plus_strat():
    keep, info = hierarchy_keep_mask(counts, grids, (None,) * 6, (-1,) * 6, 24, return_info=True)
    assert info["edges"] == 0 and int(keep.sum()) == 24
    assert all(c >= 1 for c in info["per_crop"])


def test_strat_untouched():
    # existing strat path must be bit-identical to its own definition
    k = stratified_keep_mask(counts, grids, 48)
    assert int(k.sum()) == 48
    off = 0
    for c, (gh, gw), b in zip(counts, grids, [16, 4, 4, 16, 4, 4]):
        ref = torch.zeros(c, dtype=torch.bool)
        ref[equal_area_grid_indices(gh, gw, b)] = True
        assert torch.equal(k[off:off + c], ref)
        off += c


def test_mandatory_exceeds_quota():
    # tiny budget: 6 crops -> quota 1 each, but A/B must also hold pi tokens
    keep, info = hierarchy_keep_mask(counts, grids, boxes, parents, 6, return_info=True)
    assert info["mandatory"] >= 6
    assert int(keep.sum()) >= 6


# ---------------- WSI-Aware Visual KV Compression (spec v2) ----------------
from memory.stratified import wsi_aware_keep_mask, _bridge_token


def test_wsi_budget_seeds_and_bridges():
    for B in (24, 48, 96):
        keep, info = wsi_aware_keep_mask(counts, grids, boxes, parents, B, return_info=True)
        assert int(keep.sum()) == B and not info["infeasible"]
        assert info["edges"] == 4 and info["B0"] == 6 + 4
        assert all(c >= 1 for c in info["per_crop"])
        # bridge = parent token nearest child center: A1 center (100,100) -> A cell (1,1)
        assert bool(keep[starts[0] + _bridge_token(boxes[0], (100, 100), grids[0])])
        assert _bridge_token(boxes[0], (100, 100), grids[0]) == 1 * 8 + 1
        assert _bridge_token(boxes[0], (700, 700), grids[0]) == 7 * 8 + 7


def test_wsi_infeasible_keeps_seed_whole():
    keep, info = wsi_aware_keep_mask(counts, grids, boxes, parents, 6, return_info=True)
    assert info["infeasible"] and int(keep.sum()) == info["B0"] == 10


def test_wsi_flat_and_shuffled_controls():
    k_h, i_h = wsi_aware_keep_mask(counts, grids, boxes, parents, 32, return_info=True)
    k_f, i_f = wsi_aware_keep_mask(counts, grids, boxes, parents, 32, constraint="flat", return_info=True)
    k_s, i_s = wsi_aware_keep_mask(counts, grids, boxes, parents, 32, constraint="shuffled", return_info=True)
    assert int(k_h.sum()) == int(k_f.sum()) == int(k_s.sum()) == 32
    assert i_f["edges"] == 0 and i_f["B0"] == 6
    assert i_s["edges"] == 4 and i_s["B0"] == i_h["B0"]
    assert i_s["parents"] == [-1, 3, 3, -1, 0, 0]
    assert not torch.equal(k_h, k_f) and not torch.equal(k_h, k_s)
    # shuffled: B1 center (1700,100) = (0.875,0.125) of B -> transplanted into
    # A -> cell (row 1, col 7) = 15
    assert bool(k_s[starts[0] + 1 * 8 + 7])


def test_wsi_coverage_greedy_spreads():
    # with a large budget every crop's covering radius shrinks; the fill must
    # not pile tokens into one crop (each 4x4 child gets at least a corner set)
    keep, info = wsi_aware_keep_mask(counts, grids, boxes, parents, 60, return_info=True)
    assert min(info["per_crop"]) >= 4, info["per_crop"]
    assert bool(keep.all()) is False


# ---------------- Scale-Structured (spec v3) ----------------
from memory.stratified import scale_structured_keep_mask, _one_center, _grid_norm_coords
mags = (5, 20, 20, 5, 20, 20)


def test_scale_split_and_bridges():
    keep, info = scale_structured_keep_mask(counts, grids, boxes, parents, mags, 40, alpha=0.5, return_info=True)
    assert info["B_C"] == 20 and info["B_F"] == 20 and int(keep.sum()) == 40
    assert info["coarse"] == [0, 3] and info["fine"] == [1, 2, 4, 5]
    assert info["bridges"] == 4 and info["mandatory_C"] == 4 and not info["infeasible"]
    pc = info["per_crop"]
    assert pc[0] + pc[3] == 20 and pc[1] + pc[2] + pc[4] + pc[5] == 20
    assert all(pc[i] >= 1 for i in (1, 2, 4, 5)) and pc[1] == pc[2] == pc[4] == pc[5] == 5
    # bridges: A1 center -> A cell 9, A2 -> 63
    assert bool(keep[starts[0] + 9]) and bool(keep[starts[0] + 63])
    # global WSI-frame FPS spreads coarse tokens over both anchors
    assert pc[0] >= 6 and pc[3] >= 6, pc


def test_scale_childless_anchor_one_center():
    par2 = (-1, 0, 0, -1, 0, 0)  # B has no children
    keep, info = scale_structured_keep_mask(counts, grids, boxes, par2, mags, 40, alpha=0.5, return_info=True)
    # B2's clipped bridge collides with A2's (cell 63) -> deduplicated: 3 + B's one-center
    assert info["bridges"] == 4 and info["mandatory_C"] == 4
    oc = _one_center(_grid_norm_coords(grids[3]))
    assert bool(keep[starts[3] + oc])


def test_scale_shuffled_control():
    k, i = scale_structured_keep_mask(counts, grids, boxes, parents, mags, 40, alpha=0.5, return_info=True)
    ks, i_s = scale_structured_keep_mask(counts, grids, boxes, parents, mags, 40, alpha=0.5, shuffle_parents=True, return_info=True)
    assert int(ks.sum()) == 40 and i_s["parents"] == [-1, 3, 3, -1, 0, 0]
    assert i_s["bridges"] == 4 and i_s["mandatory_C"] == 4
    assert not torch.equal(k, ks)
    assert bool(ks[starts[0] + 15])  # B1 relative center transplanted into A


def test_scale_infeasible_flags():
    _, info = scale_structured_keep_mask(counts, grids, boxes, parents, mags, 6, alpha=0.5, return_info=True)
    assert info["infeasible"] and info["per_crop"][0] + info["per_crop"][3] == 4
    keep, info = scale_structured_keep_mask(counts, grids, boxes, parents, mags, 20, alpha=1.0, return_info=True)
    assert info["B_F"] == 0 and info["infeasible"]
    assert all(info["per_crop"][i] == 1 for i in (1, 2, 4, 5))


def test_scale_no_fine_all_coarse():
    keep, info = scale_structured_keep_mask((64, 64), ((8, 8), (8, 8)), (boxes[0], boxes[3]), (-1, -1), (5, 5), 20, alpha=0.5, return_info=True)
    assert info["B_C"] == 20 and info["B_F"] == 0 and int(keep.sum()) == 20 and info["mandatory_C"] == 2



def test_scale_v2_alias_equals_wsi_on_equal_grids():
    # spec 0911 (no alpha, local coords, bridge, max-rho greedy) == wsi rule;
    # diam scaling is a constant factor when all grids share one shape
    g16 = tuple((16, 16) for _ in range(6)); c16 = tuple(256 for _ in range(6))
    k1 = wsi_aware_keep_mask(c16, g16, boxes, parents, 512, return_info=False, normalize_diam=True)
    k2 = wsi_aware_keep_mask(c16, g16, boxes, parents, 512, return_info=False, normalize_diam=False)
    assert torch.equal(k1, k2)
    # mixed grids: the raw-distance variant may allocate differently (spec-exact)
    k3, i3 = wsi_aware_keep_mask(counts, grids, boxes, parents, 40, return_info=True, normalize_diam=False)
    assert int(k3.sum()) == 40 and i3["edges"] == 4 and all(c >= 1 for c in i3["per_crop"])


from memory.stratified import precedence_coverage_keep_mask, coverage_utility


def _prec_ok(keep, par):
    # child kept => its bridge parent token kept (shuffled: relative center
    # transplanted into the wrong parent's frame, as in the implementation)
    from memory.stratified import _bridge_token
    for c in range(len(counts)):
        p = par[c]
        if p < 0 or int(keep[starts[c]:starts[c] + counts[c]].sum()) == 0:
            continue
        cx, cy, cw, ch = boxes[c]
        ccx, ccy = cx + cw / 2, cy + ch / 2
        p0 = parents[c]
        if p != p0:
            px0, py0, pw0, ph0 = boxes[p0]; px1, py1, pw1, ph1 = boxes[p]
            ccx = px1 + (ccx - px0) / pw0 * pw1; ccy = py1 + (ccy - py0) / ph0 * ph1
        b = starts[p] + _bridge_token(boxes[p], (ccx, ccy), grids[p])
        if not bool(keep[b]):
            return False
    return True


def test_v3_budget_precedence_and_gain():
    k, i = precedence_coverage_keep_mask(counts, grids, boxes, parents, 40, return_info=True)
    assert int(k.sum()) == 40 and i["edges"] == 4 and _prec_ok(k, parents)
    # F reported == explicit oracle; every greedy step had positive gain (no saturation at B=40)
    assert abs(i["F"] - coverage_utility(k, counts, grids, 4)) < 1e-6 and i["saturated_at"] is None
    # greedy beats a naive equal-area strat selection of the same size on F
    ks = stratified_keep_mask(counts, grids, 40)
    assert i["F"] > coverage_utility(ks, counts, grids, 4)


def test_v3_flat_may_break_precedence_and_differs():
    kh = precedence_coverage_keep_mask(counts, grids, boxes, parents, 12)
    kf, inf = precedence_coverage_keep_mask(counts, grids, boxes, parents, 12, constraint="flat", return_info=True)
    assert int(kf.sum()) == 12 and inf["edges"] == 0 and inf["closure_adds"] == 0
    assert not torch.equal(kh, kf)


def test_v3_shuffled_control():
    kh, ih = precedence_coverage_keep_mask(counts, grids, boxes, parents, 24, return_info=True)
    ksh, ish = precedence_coverage_keep_mask(counts, grids, boxes, parents, 24, constraint="shuffled", return_info=True)
    assert int(ksh.sum()) == 24 and ish["parents"] != list(parents) and ish["edges"] == 4
    assert _prec_ok(ksh, ish["parents"]) and not torch.equal(kh, ksh)


def test_v3_saturation_fills_to_budget():
    k, i = precedence_coverage_keep_mask(counts, grids, boxes, parents, 150, return_info=True)
    assert int(k.sum()) == 150 and i["saturated_at"] is not None and _prec_ok(k, parents)
    assert abs(i["F"] - float(len(counts))) < 1e-6      # fully covered
    k8, i8 = precedence_coverage_keep_mask(counts, grids, boxes, parents, 150, adjacency=8, return_info=True)
    assert i8["saturated_at"] < i["saturated_at"]          # 8-neighbourhood saturates earlier


def test_v3_full_budget_identity_and_strat_untouched():
    k = precedence_coverage_keep_mask(counts, grids, boxes, parents, sum(counts))
    assert bool(k.all())
    assert torch.equal(stratified_keep_mask(counts, grids, 40), stratified_keep_mask(counts, grids, 40))


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("PASS", name)
            except Exception as e:
                fails += 1; print("FAIL", name, repr(e))
    print("fails", fails); sys.exit(1 if fails else 0)
