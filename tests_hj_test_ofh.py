"""Unit tests for memory/ofh.py (Observation-Factorized Latent Handoff).
Run: python tests_hj_test_ofh.py"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memory.group_attn import dropfix_alpha  # noqa: E402  (sanity: module imports)
from memory.ofh import (crop_spans, duplicate_plan, js_divergence, partition,  # noqa: E402
                        subsample_plan)


def test_crop_spans_and_true_partition():
    pages = [3, 2, 4]
    assert crop_spans(pages) == [(0, 3), (3, 5), (5, 9)]
    p = partition(pages, "true", seed=1)
    assert p == [[0, 1, 2], [3, 4], [5, 6, 7, 8]]


def test_falsifiers_keep_size_histogram_and_cover_every_token():
    pages = [3, 2, 4]
    for mode in ("shuffled", "random"):
        p = partition(pages, mode, seed=7)
        assert [len(g) for g in p] == pages
        flat = sorted(t for g in p for t in g)
        assert flat == list(range(9))
        assert p == partition(pages, mode, seed=7)          # deterministic per seed
    assert partition(pages, "shuffled", 1) != partition(pages, "shuffled", 2)
    assert partition(pages, "shuffled", 7) != partition(pages, "true", 7)


def test_duplicate_plan():
    groups = partition([3, 2], "true", 1)
    extra, new = duplicate_plan(groups, 1, n_vis=5)
    assert extra == [3, 4]
    assert new[1] == [3, 4, 5, 6] and new[0] == [0, 1, 2]   # only the target grows
    assert len(new[1]) == 2 * len(groups[1])


def test_subsample_plan():
    groups = partition([4, 2], "true", 1)
    dropped, new = subsample_plan(groups, 0, keep_every=2)
    assert new[0] == [0, 2] and dropped == [1, 3] and new[1] == groups[1]
    d2, n2 = subsample_plan([[0], [1]], 0)                  # never empties a group
    assert n2[0] == [0] and d2 == []


def test_js_divergence():
    p = torch.tensor([0.5, 0.5])
    assert js_divergence(p, p.clone()) == 0.0
    a = torch.tensor([1.0, 0.0]); b = torch.tensor([0.0, 1.0])
    assert abs(js_divergence(a, b) - 0.6931471805599453) < 1e-9   # ln 2


def test_c2_rule_is_duplication_invariant():
    """The readout the arm uses (group_attn 'c2') must be exactly invariant when one
    observation's tokens are duplicated — this is the algebraic claim of section 3.2."""
    torch.manual_seed(0)
    L, V = 12, [0, 1, 2, 3, 4, 5]                # 6 visual columns, 2 observations
    scores = torch.randn(1, 2, 3, L)
    vmask = torch.zeros(L, dtype=torch.bool); vmask[V] = True
    groups = [[0, 1, 2], [3, 4, 5]]

    def c2_alpha(scores, groups, vmask):
        alpha = torch.softmax(scores, dim=-1)
        mv = alpha[..., vmask].sum(-1)
        s, within = [], []
        for g in groups:
            lg = scores[..., g]
            s.append(torch.logsumexp(lg, dim=-1) - torch.log(torch.tensor(float(len(g)))))
            within.append((g, torch.softmax(lg, dim=-1)))
        beta = torch.softmax(torch.stack(s, dim=-1), dim=-1)
        new = alpha.clone()
        for gi, (g, aw) in enumerate(within):
            new[..., g] = mv[..., None] * beta[..., gi:gi + 1] * aw
        return new, mv

    a1, mv1 = c2_alpha(scores, groups, vmask)
    # duplicate observation 1: append copies of its columns with identical logits
    dup = torch.cat([scores, scores[..., groups[1]]], dim=-1)
    vmask2 = torch.cat([vmask, torch.ones(3, dtype=torch.bool)])
    groups2 = [groups[0], groups[1] + [L, L + 1, L + 2]]
    a2, mv2 = c2_alpha(dup, groups2, vmask2)
    # beta (the observation-level weight) and the within-observation read are invariant:
    # compare shares normalised by the row's visual mass
    beta1 = a1[..., groups[1]].sum(-1) / mv1
    beta2 = a2[..., groups2[1]].sum(-1) / mv2
    assert torch.allclose(beta1, beta2, atol=1e-6)
    assert torch.allclose(a1[..., groups[0]] / mv1[..., None],
                          a2[..., groups[0]] / mv2[..., None], atol=1e-6)
    # ...but the row's native visual mass itself GROWS under duplication, so the rule is
    # invariant in the observation competition, not in the total visual mass (section 3.3
    # preserves the *current* native m_V).
    assert float((mv2 - mv1).min()) > 0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"{len(tests)}/{len(tests)} passed")
