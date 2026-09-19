"""Unit tests for memory/qpris_select.py (Q-PRIS, Notion sections 2-5).
Run: python tests_hj_test_qpris.py"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memory.pcris_select import logdet_greedy  # noqa: E402
from memory.qpris_select import (null_calibrated_residuals, qpris_keep_mask,  # noqa: E402
                                 question_basis, question_relevance)

D = 8


def e(i, scale=1.0):
    v = torch.zeros(D, dtype=torch.float64)
    v[i] = scale
    return v


def _case(fine_dir):
    """obs0 = span{e0,e1} (parent), obs1 = span{e0,e1} (alternative parent),
    obs2 = one fine token (parent 0) pointing halfway into `fine_dir`."""
    z = torch.stack([
        e(0), e(1),                                   # obs0
        e(0), e(1),                                   # obs1  (same span as obs0)
        (e(0) + fine_dir) / (2 ** 0.5),               # obs2
    ])
    return z, [2, 2, 1], (-1, -1, 0)


def test_root_tokens_keep_the_unit_state():
    z, counts, par = _case(e(2))
    R, _ = null_calibrated_residuals(z, counts, par)
    assert torch.allclose(R[:4].norm(dim=1), torch.ones(4, dtype=torch.float64), atol=1e-12)


def test_null_calibration_makes_generic_shrinkage_unit_norm():
    """Every candidate parent spans the same subspace, so the true-parent residual is
    exactly the generic one: ||r~|| must come out 1, not the shrunk 1/sqrt(2)."""
    z, counts, par = _case(e(2))
    par = (-1, -1, 0)
    z2 = torch.cat([z, ((e(0) + e(3)) / (2 ** 0.5))[None]])   # obs3, parent 1
    counts2, par2 = counts + [1], (-1, -1, 0, 1)
    R_raw, _ = null_calibrated_residuals(z2, counts2, par2, nullcal=False)
    R_cal, info = null_calibrated_residuals(z2, counts2, par2, nullcal=True)
    assert info["nullcal_obs"] == 2
    assert abs(float(R_raw[4].norm()) - 0.5 ** 0.5) < 1e-9      # shrunk by the projection
    assert abs(float(R_cal[4].norm()) - 1.0) < 1e-6             # calibrated back to 1


def test_parent_that_really_explains_the_fine_state_gives_a_small_residual():
    """True parent spans the fine direction, the alternative does not -> r~ ~ 0."""
    z = torch.stack([
        e(0), e(2),                                   # obs0 = span{e0,e2}  (true parent)
        e(0), e(1),                                   # obs1 = span{e0,e1}  (alternative)
        (e(0) + e(2)) / (2 ** 0.5),                   # obs2 fine, parent 0
        (e(0) + e(3)) / (2 ** 0.5),                   # obs3 fine, parent 1
    ])
    R, _ = null_calibrated_residuals(z, [2, 2, 1, 1], (-1, -1, 0, 1))
    assert float(R[4].norm()) < 1e-6                  # explained by its true parent
    assert float(R[5].norm()) > 0.9                   # not explained


def test_question_relevance_is_the_projection_on_the_question_span():
    z, counts, par = _case(e(2))
    R, _ = null_calibrated_residuals(z, counts, par, nullcal=False)
    q_in = question_relevance(R, question_basis(torch.stack([e(2)])))
    q_out = question_relevance(R, question_basis(torch.stack([e(3)])))
    assert abs(float(q_in[4]) - float(R[4] @ R[4])) < 1e-12     # residual lies in the span
    assert float(q_out[4]) < 1e-12                              # orthogonal to the span


def test_missing_question_falls_back_to_unweighted_logdet():
    z, counts, par = _case(e(2))
    R, _ = null_calibrated_residuals(z, counts, par)
    keep_ref, _, _ = logdet_greedy(R, 3)
    keep_none = qpris_keep_mask(z, None, counts, par, 3, q_mode="none")
    keep_missing = qpris_keep_mask(z, None, counts, par, 3, q_mode="true")
    assert torch.equal(keep_ref, keep_none) and torch.equal(keep_ref, keep_missing)


def test_objective_value_matches_the_weighted_logdet_formula():
    z, counts, par = _case(e(2))
    H_Q = torch.stack([e(2), e(0)])
    keep, info = qpris_keep_mask(z, H_Q, counts, par, 3, return_info=True)
    R, _ = null_calibrated_residuals(z, counts, par)
    qi = question_relevance(R, question_basis(H_Q))
    S = keep.nonzero().flatten()
    M = torch.eye(D, dtype=torch.float64)
    for i in S.tolist():
        M = M + qi[i] * torch.outer(R[i], R[i])
    assert abs(info["F"] - float(torch.logdet(M))) < 1e-3
    assert int(keep.sum()) == 3 == info["B"]


def test_the_question_changes_which_tokens_are_kept():
    z = torch.stack([
        e(0), e(1),                                   # obs0 parent
        e(0), e(1),                                   # obs1 alternative parent
        (e(0) + e(2)) / (2 ** 0.5),                   # obs2 fine -> residual along e2
        (e(0) + e(3)) / (2 ** 0.5),                   # obs3 fine -> residual along e3
    ])
    counts, par = [2, 2, 1, 1], (-1, -1, 0, 1)
    k2 = qpris_keep_mask(z, torch.stack([e(2)]), counts, par, 1)
    k3 = qpris_keep_mask(z, torch.stack([e(3)]), counts, par, 1)
    assert int(k2.nonzero()) == 4 and int(k3.nonzero()) == 5


def test_budget_is_a_hard_cardinality_cap():
    torch.manual_seed(0)
    z = torch.randn(24, D, dtype=torch.float64)
    counts, par = [8, 8, 4, 4], (-1, -1, 0, 1)
    for b in (0, 1, 7, 24, 99):
        keep = qpris_keep_mask(z, torch.randn(3, D, dtype=torch.float64), counts, par, b)
        assert int(keep.sum()) == min(b, 24)


def test_provenance_controls_change_the_residuals():
    torch.manual_seed(1)
    z = torch.randn(20, D, dtype=torch.float64)
    counts, par = [6, 6, 4, 4], (-1, -1, 0, 1)
    R_true, _ = null_calibrated_residuals(z, counts, par, cond="true")
    R_none, i_none = null_calibrated_residuals(z, counts, par, cond="none")
    R_wrong, _ = null_calibrated_residuals(z, counts, par, cond="wrong")
    assert torch.allclose(R_none.norm(dim=1), torch.ones(20, dtype=torch.float64), atol=1e-12)
    assert i_none["nullcal_obs"] == 0
    assert not torch.allclose(R_true, R_wrong)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"{len(tests)}/{len(tests)} passed")
