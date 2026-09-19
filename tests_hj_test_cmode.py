"""Unit tests: C2 cross-dataset common-mode diagnosis plumbing (VLMAS_CMODE).

Spec = Notion Experiment Log page "C2 구조 진단: Cross-Dataset Common-Mode / Residual
Decomposition (5셋×20)", section 3: the decision row is the position where the tokenized
candidate continuations FIRST diverge; the shared prefix (fixed JSON prefix and any common
answer prefix) lies before it. Sections 4-6 need d_i = h_F - h_0A at that row, the candidate
token ids at that row, and the row's logits under both arms.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory.cmode_diag import decision_row_tokens, divergence_position, row_record


def test_common_prefix_then_first_divergence():
    # '"Colon"' and '"Heart"' share the opening quote token, then diverge
    ids = [[9, 101, 5], [9, 102, 5], [9, 103, 7]]
    assert divergence_position(ids) == 1


def test_no_shared_prefix_gives_position_zero():
    assert divergence_position([[11, 2], [12, 2], [13, 9]]) == 0


def test_longer_shared_prefix_across_all_candidates():
    ids = [[4, 4, 4, 1], [4, 4, 4, 2], [4, 4, 4, 3]]
    assert divergence_position(ids) == 3


def test_candidate_that_is_a_prefix_of_another_diverges_at_its_end():
    # "Lung" vs "Lung cancer": the shorter one ends where the longer continues
    ids = [[7, 8], [7, 8, 9]]
    assert divergence_position(ids) == 2


def test_single_candidate_has_no_divergence():
    assert divergence_position([[3, 4, 5]]) is None


def test_identical_candidates_have_no_divergence():
    assert divergence_position([[3, 4], [3, 4]]) is None


def test_decision_row_tokens_are_per_candidate_ids_at_the_divergence():
    ids = [[9, 101, 5], [9, 102, 5], [9, 103, 7]]
    k, toks = decision_row_tokens(ids)
    assert k == 1 and toks == [101, 102, 103]
    k2, toks2 = decision_row_tokens([[7, 8], [7, 8, 9]])
    assert k2 == 2 and toks2 == [None, 9]        # exhausted candidate -> None


def test_row_record_shape_and_json_roundtrip():
    rec = row_record(
        case_index=3,
        cands=["Colon", "Heart"],
        candidate_ids=[[9, 101, 5], [9, 102, 5]],
        hidden_full=[0.5, -0.25, 1.0],
        hidden_zeroa=[0.25, -0.25, 0.5],
        logits_full=[1.5, 0.5],
        logits_zeroa=[1.0, 0.75],
        extra={"attn": "eager"},
    )
    assert rec["case"] == 3 and rec["cands"] == ["Colon", "Heart"]
    assert rec["k"] == 1 and rec["tok"] == [101, 102]
    assert rec["d"] == [0.25, 0.0, 0.5]                      # d = h_full - h_zeroA
    assert rec["h_full"] == [0.5, -0.25, 1.0] and rec["h_zeroa"] == [0.25, -0.25, 0.5]
    assert rec["logit_full"] == [1.5, 0.5] and rec["logit_zeroa"] == [1.0, 0.75]
    assert rec["attn"] == "eager"
    assert rec["cand_ids"] == [[9, 101, 5], [9, 102, 5]] and rec["n_distinct"] == 2
    assert json.loads(json.dumps(rec))["k"] == 1


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
