"""Unit tests for memory/obscal_diag.py (observation-level receiver calibration probe).
Run: python tests_hj_test_obscal.py"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memory.obscal_diag import reach_from, rmsnorm_jvp  # noqa: E402


def _rmsnorm(h, w, eps=1e-6):
    return w * h / torch.sqrt((h * h).mean() + eps)


def test_rmsnorm_jvp_matches_finite_difference():
    torch.manual_seed(0)
    h = torch.randn(64, dtype=torch.float64) * 3
    m = torch.randn(64, dtype=torch.float64) * 0.1
    w = torch.rand(64, dtype=torch.float64) + 0.5
    analytic = rmsnorm_jvp(h, m, w)
    e = 1e-6
    fd = (_rmsnorm(h + e * m, w) - _rmsnorm(h - e * m, w)) / (2 * e)
    assert torch.allclose(analytic, fd, atol=1e-6), float((analytic - fd).abs().max())


def test_rmsnorm_jvp_is_linear_in_the_direction():
    torch.manual_seed(1)
    h = torch.randn(32, dtype=torch.float64)
    a, b = torch.randn(32, dtype=torch.float64), torch.randn(32, dtype=torch.float64)
    w = torch.ones(32, dtype=torch.float64)
    lhs = rmsnorm_jvp(h, 2 * a + 3 * b, w)
    rhs = 2 * rmsnorm_jvp(h, a, w) + 3 * rmsnorm_jvp(h, b, w)
    assert torch.allclose(lhs, rhs, atol=1e-10)


def test_reach_from_basic():
    z = torch.tensor([2.0, 1.0, 0.5], dtype=torch.float64)     # winner = 0, gaps 0/1/1.5
    d = torch.tensor([0.0, 0.8, 0.0], dtype=torch.float64)     # pushes challenger 1 only
    r = reach_from(d, z, a=0)
    assert r["n_push"] == 1 and abs(r["v_star"] - 0.8) < 1e-12
    assert abs(r["g_star"] - 1.0) < 1e-12 and abs(r["R"] - 0.8) < 1e-9
    assert abs(r["v_max"] - 0.8) < 1e-12


def test_reach_from_no_positive_push():
    z = torch.tensor([2.0, 1.0], dtype=torch.float64)
    d = torch.tensor([1.0, 0.0], dtype=torch.float64)          # winner pushed up
    r = reach_from(d, z, a=0)
    assert r["n_push"] == 0 and r["R"] is None and r["v_max"] is None


def test_reach_from_picks_best_ratio_not_biggest_push():
    z = torch.tensor([5.0, 1.0, 4.0], dtype=torch.float64)     # gaps 0 / 4 / 1
    d = torch.tensor([0.0, 2.0, 0.6], dtype=torch.float64)
    r = reach_from(d, z, a=0)
    assert abs(r["v_max"] - 2.0) < 1e-12                        # biggest absolute push
    assert abs(r["R"] - 0.6) < 1e-9 and abs(r["g_star"] - 1.0) < 1e-12


def test_influence_is_additive_over_observations():
    """d_{o} = J m_{o} is linear, so the per-observation influences must sum to the
    influence of the total visual write — the partition cannot change the total."""
    torch.manual_seed(2)
    h = torch.randn(48, dtype=torch.float64)
    w = torch.rand(48, dtype=torch.float64) + 0.5
    W = torch.randn(20, 48, dtype=torch.float64)
    ms = [torch.randn(48, dtype=torch.float64) * 0.05 for _ in range(4)]
    parts = [W @ rmsnorm_jvp(h, m, w) for m in ms]
    total = W @ rmsnorm_jvp(h, sum(ms), w)
    assert torch.allclose(sum(parts), total, atol=1e-10)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"{len(tests)}/{len(tests)} passed")
