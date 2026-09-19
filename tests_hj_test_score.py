"""Unit tests for memory/score_select.py (SCoRe Alg.1 arm). Run: python tests_hj_test_score.py"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memory.score_select import score_keep_mask  # noqa: E402


def reference(X, w, k, alpha):
    """The offline numpy implementation used for the SCoRe Fig.2 reproduction."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    wa = np.power(np.clip(w, 1e-12, None), alpha)
    s = int(np.argmax(w))
    S = [s]
    dmin = 1.0 - Xn @ Xn[s]
    for _ in range(k - 1):
        comp = dmin * wa
        comp[S] = -np.inf
        i = int(np.argmax(comp))
        S.append(i)
        dmin = np.minimum(dmin, 1.0 - Xn @ Xn[i])
    return np.array(sorted(S))


def test_matches_offline_reference():
    rng = np.random.default_rng(0)
    for _ in range(5):
        n, d, k = 120, 16, 30
        X = rng.normal(size=(n, d)).astype(np.float32)
        w = rng.random(n).astype(np.float32)
        want = reference(X, w, k, 0.8)
        keep = score_keep_mask(torch.tensor(X), torch.tensor(w), [n], k, alpha=0.8)
        got = np.flatnonzero(keep.numpy())
        assert np.array_equal(got, want), (got[:10], want[:10])


def test_budget_edges():
    X = torch.randn(20, 8)
    w = torch.rand(20)
    assert score_keep_mask(X, w, [20], 0).sum() == 0
    assert score_keep_mask(X, w, [20], 20).all()
    assert score_keep_mask(X, w, [20], 25).all()        # clipped to n
    assert int(score_keep_mask(X, w, [20], 7).sum()) == 7


def test_uniform_relevance_is_plain_kcenter():
    X = torch.randn(30, 4)
    keep, info = score_keep_mask(X, None, [30], 5, return_info=True)
    assert info["rel"] == "uniform" and info["seed_idx"] == 0 and int(keep.sum()) == 5
    keep2 = score_keep_mask(X, torch.ones(30), [30], 5)
    assert torch.equal(keep, keep2)


def test_seed_is_salience_argmax_and_counts_per_crop():
    X = torch.randn(12, 3)
    w = torch.tensor([0.1] * 12)
    w[7] = 1.0
    keep, info = score_keep_mask(X, w, [4, 4, 4], 6, return_info=True)
    assert info["seed_idx"] == 7 and keep[7]
    assert sum(info["per_crop"]) == 6 and len(info["per_crop"]) == 3


def test_covers_both_clusters_where_topk_does_not():
    # cluster A (indices 0-9) carries all the salience, cluster B (10-19) none
    torch.manual_seed(0)
    a = torch.randn(10, 8) * 0.01 + torch.tensor([5.0] + [0.0] * 7)
    b = torch.randn(10, 8) * 0.01 + torch.tensor([0.0, 5.0] + [0.0] * 6)
    X = torch.cat([a, b])
    w = torch.cat([torch.rand(10) + 1.0, torch.full((10,), 0.01)])
    keep = score_keep_mask(X, w, [20], 6, alpha=0.8)
    idx = set(int(i) for i in torch.nonzero(keep).flatten())
    assert any(i >= 10 for i in idx), "SCoRe must reach the un-salient cluster"
    topk = set(int(i) for i in torch.topk(w, 6).indices)
    assert all(i < 10 for i in topk), "top-k stays inside the salient cluster"


def test_ties_go_to_lower_index():
    X = torch.eye(6)
    w = torch.ones(6)
    keep = score_keep_mask(X, w, [6], 3)
    assert [int(i) for i in torch.nonzero(keep).flatten()][0] == 0


def test_relevance_length_mismatch_raises():
    try:
        score_keep_mask(torch.randn(10, 4), torch.rand(9), [10], 3)
    except ValueError:
        return
    raise AssertionError("expected ValueError on relevance/token_counts mismatch")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"{len(tests)}/{len(tests)} passed")
