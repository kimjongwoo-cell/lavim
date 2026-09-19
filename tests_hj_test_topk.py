"""Unit tests: plain score top-k visual KV pruning (VLMAS_WSI_CONSOL_MODE=topk).

The arm the accuracy question asks about: keep the B visual tokens with the highest
question->vision salience (VLMAS_WSI_CONSOL_SC=1 relevance), no coverage, no provenance.
per_crop=True keeps a proportional share inside every crop instead (stratified top-k).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

from memory.topk_select import salience_topk_keep_mask


def test_keeps_exactly_the_highest_scoring_tokens():
    rel = torch.tensor([0.1, 0.9, 0.5, 0.7, 0.2, 0.3])
    keep = salience_topk_keep_mask(rel, (3, 3), 3)
    assert keep.dtype == torch.bool and int(keep.sum()) == 3
    assert [bool(v) for v in keep] == [False, True, True, True, False, False]   # 0.9, 0.7, 0.5


def test_exact_budget_and_budget_ge_n():
    rel = torch.rand(20, generator=torch.Generator().manual_seed(0))
    for b in (1, 7, 19):
        assert int(salience_topk_keep_mask(rel, (10, 10), b).sum()) == b
    assert bool(salience_topk_keep_mask(rel, (10, 10), 99).all())


def test_ties_break_by_lowest_index_deterministically():
    rel = torch.tensor([0.5, 0.5, 0.5, 0.5])
    keep = salience_topk_keep_mask(rel, (4,), 2)
    assert [bool(v) for v in keep] == [True, True, False, False]
    assert torch.equal(keep, salience_topk_keep_mask(rel, (4,), 2))


def test_uniform_relevance_falls_back_to_first_tokens():
    keep = salience_topk_keep_mask(None, (4, 4), 4)
    assert int(keep.sum()) == 4 and [bool(v) for v in keep][:4] == [True] * 4


def test_per_crop_keeps_a_share_inside_every_crop():
    # crop 0 holds the four highest scores; global top-k would empty crop 1
    rel = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])
    glob = salience_topk_keep_mask(rel, (4, 4), 4)
    assert int(glob[4:].sum()) == 0
    per = salience_topk_keep_mask(rel, (4, 4), 4, per_crop=True)
    assert int(per.sum()) == 4 and int(per[:4].sum()) == 2 and int(per[4:].sum()) == 2
    assert [bool(v) for v in per] == [True, True, False, False, True, True, False, False]


def test_relevance_length_mismatch_is_rejected():
    try:
        salience_topk_keep_mask(torch.rand(5), (3, 3), 2)
    except ValueError:
        return
    raise AssertionError("expected ValueError on relevance/token_counts mismatch")


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_")]
    ok = 0
    for name, fn in tests:
        try:
            fn()
            ok += 1
            print(f"PASS {name}")
        except Exception as e:
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(tests)} passed")
