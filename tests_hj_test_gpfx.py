"""Unit tests for memory/gpfx_diag.py (how much of the answer margin is the JSON prefix).
Run: python tests_hj_test_gpfx.py"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memory.gpfx_diag import ARMS, margins  # noqa: E402


def test_margin_is_positive_when_the_token_wins():
    logits = torch.tensor([[0.0, 5.0, 1.0]])
    m = margins(logits, [1])[0]
    assert m["keep"] and abs(m["g"] - 4.0) < 1e-9 and m["argmax"] == 1


def test_margin_is_negative_when_the_token_loses():
    logits = torch.tensor([[0.0, 5.0, 1.0]])
    m = margins(logits, [2])[0]
    assert not m["keep"] and abs(m["g"] + 4.0) < 1e-9 and m["argmax"] == 1


def test_margin_is_zero_on_an_exact_tie():
    logits = torch.tensor([[2.0, 2.0]])
    m = margins(logits, [0])[0]
    assert m["keep"] and abs(m["g"]) < 1e-12


def test_margin_matches_rmvr_units():
    """g is a logit difference, so it equals the log-odds gap — the same quantity RMVR
    compares the visual push against."""
    logits = torch.tensor([[1.0, 3.5]])
    m = margins(logits, [1])[0]
    lp = torch.log_softmax(logits[0].double(), dim=-1)
    assert abs(m["g"] - float(lp[1] - lp[0])) < 1e-9
    assert abs(m["lp"] - float(lp[1])) < 1e-4


def test_every_row_is_scored_independently():
    logits = torch.tensor([[0.0, 9.0], [4.0, 0.0], [1.0, 1.5]])
    out = margins(logits, [1, 0, 0])
    assert [x["keep"] for x in out] == [True, True, False]
    assert abs(out[0]["g"] - 9.0) < 1e-9 and abs(out[2]["g"] + 0.5) < 1e-9


def test_arms_are_the_four_prompt_conditions():
    assert ARMS == ("full", "noprefix", "bare", "naked")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"{len(tests)}/{len(tests)} passed")
