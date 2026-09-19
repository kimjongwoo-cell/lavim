"""Unit tests for memory/lens_diag.py (layer-wise logit lens at the Answerer).
Run: python tests_hj_test_lens.py"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memory.lens_diag import decision_layer, half_margin_layer, lens_readout  # noqa: E402


class _Norm:
    """Stand-in for the model's final RMSNorm."""
    def __init__(self, w):
        self.weight = w
        self.variance_epsilon = 1e-6


def test_decision_layer_is_the_first_of_the_final_true_run():
    assert decision_layer([False, True, False, True, True, True]) == 3
    assert decision_layer([True, True, True]) == 0
    assert decision_layer([True, True, False]) is None
    assert decision_layer([False, False]) is None


def test_decision_layer_ignores_an_early_flicker():
    """A layer that briefly agrees and then loses it is not the decision layer."""
    assert decision_layer([True, False, False, True, True]) == 3


def test_half_margin_layer_needs_a_positive_final_margin():
    assert half_margin_layer([0.0, 1.0, 3.0, 4.0]) == 2        # half of 4.0 is 2.0
    assert half_margin_layer([5.0]) == 0
    assert half_margin_layer([1.0, -2.0]) is None              # final margin not positive
    assert half_margin_layer([]) is None


def test_lens_readout_reports_a_locked_answer():
    W = torch.eye(4)
    w = torch.ones(4)
    h = torch.tensor([0.0, 3.0, 0.0, 1.0])                     # token 1 wins
    r = lens_readout(h, _Norm(w), W, answer_id=1)
    assert r["lock"] and r["argmax"] == 1 and r["rank"] == 1
    assert r["g"] > 0 and 0.0 < r["p"] <= 1.0


def test_lens_readout_reports_a_losing_answer_with_a_negative_margin():
    W = torch.eye(4)
    h = torch.tensor([0.0, 3.0, 0.0, 1.0])
    r = lens_readout(h, _Norm(torch.ones(4)), W, answer_id=3)   # token 3 is behind token 1
    assert not r["lock"] and r["argmax"] == 1 and r["rank"] == 2
    assert r["g"] < 0


def test_margin_is_invariant_to_the_residual_scale():
    """RMSNorm removes the scale, so doubling h must not change g — that is what makes
    g comparable across layers (and to RMVR's g)."""
    W = torch.randn(6, 5, generator=torch.Generator().manual_seed(0))
    h = torch.randn(5, generator=torch.Generator().manual_seed(1))
    n = _Norm(torch.ones(5))
    a = int((W.double() @ (h.double() / torch.sqrt((h.double() ** 2).mean() + 1e-6))).argmax())
    r1 = lens_readout(h, n, W, a)
    r2 = lens_readout(h * 7.0, n, W, a)
    assert abs(r1["g"] - r2["g"]) < 1e-3 and r1["argmax"] == r2["argmax"]


def test_readout_uses_the_norm_weight():
    W = torch.eye(3)
    h = torch.tensor([1.0, 0.9, 0.0])
    flat = lens_readout(h, _Norm(torch.ones(3)), W, answer_id=0)
    tilt = lens_readout(h, _Norm(torch.tensor([1.0, 2.0, 1.0])), W, answer_id=0)
    assert flat["argmax"] == 0 and tilt["argmax"] == 1        # the weight flips the winner


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"{len(tests)}/{len(tests)} passed")
