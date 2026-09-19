"""Unit tests for memory/rmvr_diag.py (Open/Closed Receiver-Margin Visual Reach).
Run: python tests_hj_test_rmvr.py"""
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memory.rmvr_diag import (answer_span, gt_thresholds, node_reach,  # noqa: E402
                              summarize, vocab_reach)


def test_no_visual_effect():
    z0 = [2.0, 1.0, 0.5]
    r = node_reach(z0, list(z0))
    assert r["a"] == 0 and r["aF"] == 0 and r["dz"] == [0.0, 0.0, 0.0]
    assert r["n_push"] == 0 and r["R_n"] is None and r["lam_n"] is None
    assert r["pred_flip"] is False and r["flip"] is False


def test_push_below_margin():
    r = node_reach([2.0, 1.0, 0.5], [2.0, 1.8, 0.5])       # dz = +0.8 on the challenger
    assert r["g"][1] == 1.0 and abs(r["v"][1] - 0.8) < 1e-12
    assert abs(r["R_n"] - 0.8) < 1e-9 and abs(r["lam_n"] - 1.25) < 1e-9
    assert r["pred_flip"] is False and r["flip"] is False


def test_push_above_margin_and_tie_at_lambda():
    z0, zF = [2.0, 1.0, 0.5], [2.0, 2.2, 0.5]              # dz = +1.2 > margin 1.0
    r = node_reach(z0, zF)
    assert abs(r["R_n"] - 1.2) < 1e-9 and r["pred_flip"] and r["flip"] and r["aF"] == 1
    tied = [z0[i] + r["lam_n"] * r["dz"][i] for i in range(3)]
    assert abs(tied[0] - tied[1]) < 1e-9                    # lam_n is exactly the tie gain


def test_winner_pushed_up_keeps_winner():
    r = node_reach([2.0, 1.0], [3.0, 1.5])                  # both up, winner more
    assert r["v"][1] < 0 and r["n_push"] == 0 and r["R_n"] is None
    assert r["pred_flip"] is False and r["flip"] is False


def test_gate_identity_random():
    rng = random.Random(0)
    for _ in range(500):
        k = rng.randint(2, 6)
        z0 = [rng.uniform(-5, 5) for _ in range(k)]
        zF = [z + rng.uniform(-3, 3) for z in z0]
        r = node_reach(z0, zF)
        assert r["pred_flip"] == r["flip"], (z0, zF, r["R_n"])


def test_vocab_reach_matches_list_version():
    rng = random.Random(1)
    for _ in range(200):
        k = rng.randint(2, 12)
        z0 = [rng.uniform(-5, 5) for _ in range(k)]
        zF = [z + rng.uniform(-3, 3) for z in z0]
        a = node_reach(z0, zF)
        b = vocab_reach(torch.tensor(z0, dtype=torch.float64), torch.tensor(zF, dtype=torch.float64))
        assert a["a"] == b["a"] and a["aF"] == b["aF"] and a["flip"] == b["flip"]
        assert a["n_push"] == b["n_push"] and a["pred_flip"] == b["pred_flip"]
        if a["R_n"] is None:
            assert b["R"] is None
        else:
            assert abs(a["R_n"] - b["R"]) < 1e-9
            assert b["v_max"] >= b["v_star"] - 1e-12        # v_max is the largest push
            assert b["g_star"] >= 0 and b["g_at_v_max"] >= 0


def test_vocab_reach_numerator_denominator():
    # challenger 1 has the biggest push, challenger 2 the best reach (smaller gap)
    z0 = torch.tensor([5.0, 1.0, 4.0], dtype=torch.float64)
    zF = torch.tensor([5.0, 3.0, 4.6], dtype=torch.float64)
    b = vocab_reach(z0, zF)
    assert b["c_v_max"] == 1 and abs(b["v_max"] - 2.0) < 1e-9 and abs(b["g_at_v_max"] - 4.0) < 1e-9
    assert b["c_star"] == 2 and abs(b["R"] - 0.6) < 1e-9 and abs(b["g_star"] - 1.0) < 1e-9
    assert b["pred_flip"] is False and b["flip"] is False


def test_gt_repair_and_break():
    t = gt_thresholds([2.0, 1.0, 0.5], [2.0, 1.6, 0.5], gold=1)
    assert t["gold_is_winner"] is False and t["lam_break"] is None
    assert abs(t["lam_repair"] - (1.0 / 0.6)) < 1e-9
    tb = gt_thresholds([2.0, 1.0], [2.0, 1.4], gold=0)
    assert tb["gold_is_winner"] is True and abs(tb["lam_break"] - (1.0 / 0.4)) < 1e-9
    assert gt_thresholds([2.0, 1.0], [2.5, 1.0], gold=1)["lam_repair"] is None
    assert gt_thresholds([2.0, 1.0], [2.0, 1.4], gold=None)["gold_is_winner"] is None


def test_summarize_node_and_step_forms():
    z0 = [2.0, 1.0]
    rs = [node_reach(z0, [2.0, 1.8]), node_reach(z0, [2.0, 2.5]), node_reach(z0, list(z0))]
    s = summarize([{**r, "R": r["R_n"]} for r in rs])
    assert s["n"] == 3 and s["n_push"] == 2
    assert abs(s["R_max"] - 1.5) < 1e-9 and abs(s["R_med"] - 1.15) < 1e-9
    assert abs(s["F_reach"] - 1 / 3) < 1e-9
    assert s["flips"] == 1 and s["pred_flips"] == 1 and s["gate_ok"] is True
    steps = [vocab_reach(torch.tensor(z0, dtype=torch.float64), torch.tensor([2.0, 2.5], dtype=torch.float64))]
    assert summarize(steps)["flips"] == 1 and summarize(steps)["gate_ok"] is True
    broken = dict(steps[0]); broken["pred_flip"] = False
    assert summarize([broken])["gate_ok"] is False


def test_answer_span():
    text = '{"answer": "Colon", "rationale": "glands"}'
    # fake offsets: one token per character run
    offsets = [(i, i + 1) for i in range(len(text))]
    idx, span = answer_span(text, offsets)
    assert span is not None and text[span[0]:span[1]] == "Colon"
    assert [text[i] for i in idx] == list("Colon")
    assert answer_span('{"rationale": "x"}', offsets) == ([], None)
    esc = '{"answer": "a\\"b", "x": 1}'
    _idx2, span2 = answer_span(esc, [(i, i + 1) for i in range(len(esc))])
    assert esc[span2[0]:span2[1]] == 'a\\"b'


def test_field_spans_with_open_prefix():
    from memory.rmvr_diag import field_spans
    # structured_json: the prompt ends with '{"answer": ' and the model emits the quote
    text = '"Skin", "rationale": "layered tissue"}'
    off = [(i, i + 1) for i in range(len(text))]
    f = field_spans(text, off, json_prefix='{"answer": ')
    assert text[f["answer"]["span"][0]:f["answer"]["span"][1]] == "Skin"
    assert [text[i] for i in f["answer"]["idx"]] == list("Skin")
    assert text[f["rationale"]["span"][0]:f["rationale"]["span"][1]] == "layered tissue"
    # prefix that already contains the opening quote
    text2 = 'Skin", "rationale": "x"}'
    off2 = [(i, i + 1) for i in range(len(text2))]
    f2 = field_spans(text2, off2, json_prefix='{"answer": "')
    assert text2[f2["answer"]["span"][0]:f2["answer"]["span"][1]] == "Skin"
    # without the prefix hint the leading value is invisible (the old failure mode)
    assert "answer" not in field_spans(text, off)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"{len(tests)}/{len(tests)} passed")
