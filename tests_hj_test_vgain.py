"""Unit tests for memory/vgain_diag.py (run: python tests_hj_test_vgain.py)."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory import vgain_diag as vg  # noqa: E402

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"PASS {name}")
    else:
        FAIL += 1
        print(f"FAIL {name} {detail}")


# ------------------------------------------------------------------ pure helpers
check("grid default", vg.parse_grid(None) == (0.0, 0.5, 1.0, 1.5, 2.0, 4.0))
check("grid dedup/spaces", vg.parse_grid(" 1, 2,1 ,0") == (1.0, 2.0, 0.0))
plan = vg.arm_plan(vg.parse_grid(None), ["V", "W", "N"])
names = [a["name"] for a in plan]
check("plan ident first", names[0] == "ident")
check("plan size 16", len(plan) == 16, names)
check("plan V excludes 1", "V1" not in names and "V0" in names and "V4" in names)
check("plan W excludes 0", "W0" not in names and "W1" in names)
check("plan N excludes 1", "N1" not in names and "N0" in names)
n2 = [a for a in plan if a["name"] == "N2"][0]
check("plan N scales gn", n2["gv"] == 1.0 and n2["gn"] == 2.0)
v05 = [a for a in plan if a["name"] == "V0.5"][0]
check("plan V scales gv", v05["gv"] == 0.5 and v05["gn"] == 1.0)
check("plan families filter", [a["name"] for a in vg.arm_plan((0.0, 2.0), ["V"])] == ["ident", "V0", "V2"])

w = vg.value_weights(6, torch.tensor([1, 4]), 3.0, 0.5)
check("weights", w.tolist() == [0.5, 3.0, 0.5, 0.5, 3.0, 0.5])

bank = []
check("donor none on empty", vg.pick_donor(bank, "s1") is None)
vg.push_donor(bank, {"slide": "s1", "case": 0, "v": None})
check("donor same slide refused", vg.pick_donor(bank, "s1") is None)
vg.push_donor(bank, {"slide": "s2", "case": 1, "v": None})
check("donor newest other slide", vg.pick_donor(bank, "s1")["case"] == 1)
check("donor skips same slide", vg.pick_donor(bank, "s2")["case"] == 0)
vg.push_donor(bank, {"slide": "s1", "case": 2, "v": None})
check("donor one entry per slide", [e["case"] for e in bank] == [1, 2])
vg.push_donor(bank, {"slide": "s3", "case": 3, "v": None})
check("donor keep 2", [e["slide"] for e in bank] == ["s1", "s3"])

check("lead space quote", vg.candidate_lead(' "3", "rationale": "x"}') == ' "')
check("lead no space", vg.candidate_lead('"3"}') == '"')
check("lead fallback", vg.candidate_lead("garbage") == ' "')
check("leads stripped quote -> both", vg.candidate_leads('"Colon", "r": "x"}') == ['"', ' "'])
check("leads explicit space kept", vg.candidate_leads(' "Colon"}') == [' "'])
check("tail comma", vg.candidate_tail('"Colon", "rationale": "x"}') == '",')
check("tail brace", vg.candidate_tail(' "3"}') == '"}')
check("tail bare quote", vg.candidate_tail('"3"') == '"')
check("tail fallback", vg.candidate_tail("garbage") == '"')
_l = vg.logsumexp_by([0.0, 0.0, -1.0], [0, 0, 1], 3)
check("logsumexp_by", abs(_l[0] - 0.693147) < 1e-5 and _l[1] == -1.0 and _l[2] is None, _l)

try:
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained("/home/users/whddn12316/models/Qwen3-VL-4B-Thinking")
    _d = lambda ids: [_tok.decode([i]) for i in ids]
    _v = vg.candidate_variants(_tok, '"', "Spleen", '",')
    check("variants spleen: sep + merged", [_d(x) for x in _v] == [['"', 'S', 'ple', 'en', '",'], ['"S', 'ple', 'en', '",']], _v)
    _v = vg.candidate_variants(_tok, '"', "Colon", '",')
    check("variants colon: one form", [_d(x) for x in _v] == [['"', 'Colon', '",']], _v)
    _nat = '"Spleen", "rationale": "x"}'
    _have = list(_tok(_nat, add_special_tokens=False)["input_ids"])
    check("retokenized decode matches a variant",
          any(_have[:len(w)] == w for w in vg.candidate_variants(_tok, vg.candidate_lead(_nat), "Spleen", vg.candidate_tail(_nat))))
    _sp = vg.candidate_variants(_tok, ' "', "Colon", '",')
    check("space lead is its own token", _d(_sp[0])[:2] == [' "', 'Colon'], [_d(x) for x in _sp])
    check("variants decode to the string", all(_tok.decode(x) == ' "Small Intestine",'
                                               for x in vg.candidate_variants(_tok, ' "', "Small Intestine", '",')))
except OSError as exc:  # tokenizer not on this machine
    check("tokenizer load", False, repr(exc))

try:
    check("parse json", vg.parse_answer(' "Skin", "rationale": "r", "confidence": 0.9}', ("Skin", "Colon")) == "Skin")
    check("parse letter snaps", vg.parse_answer(' "B", "rationale": "r"}', ("Skin", "Colon")) == "Colon")
    check("parse broken json falls back", vg.parse_answer(' "Colon", "rationale": "unterminated', ("Skin", "Colon")) == "Colon")
except ImportError as exc:  # pipeline package missing in a bare checkout
    check("parse import", False, repr(exc))


# ------------------------------------------------------------------ math on one attention layer
def tiny_backbone():
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLTextAttention,
        Qwen3VLTextRotaryEmbedding,
    )
    cfg = Qwen3VLTextConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
                            head_dim=16, intermediate_size=128, num_hidden_layers=1,
                            rope_scaling={"rope_type": "default", "mrope_section": [3, 3, 2]},
                            vocab_size=50)
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    attn = Qwen3VLTextAttention(cfg, layer_idx=0).eval()
    rope = Qwen3VLTextRotaryEmbedding(cfg)
    layer = SimpleNamespace(self_attn=attn)
    bb = SimpleNamespace(lm=SimpleNamespace(layers=[layer]), device=torch.device("cpu"))
    return bb, attn, rope, cfg


def causal(past, T):
    L = past + T
    m = torch.full((1, 1, T, L), float("-inf"))
    for r in range(T):
        m[0, 0, r, : past + r + 1] = 0.0
    return m


def run_attn(attn, rope, hidden, past_len, cache):
    from transformers.cache_utils import DynamicCache  # noqa: F401
    T = hidden.shape[1]
    pos = torch.arange(past_len, past_len + T)[None, None, :].expand(3, 1, T)
    pe = rope(hidden, pos)
    out = attn(hidden_states=hidden, position_embeddings=pe, attention_mask=causal(past_len, T),
               past_key_values=cache)
    return out[0] if isinstance(out, tuple) else out


def prefilled(attn, rope, ctx):
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()
    with torch.no_grad():
        run_attn(attn, rope, ctx, 0, cache)
    return cache


try:
    from copy import deepcopy

    bb, attn, rope, cfg = tiny_backbone()
    torch.manual_seed(1)
    ctx = torch.randn(1, 12, 64)
    rows = torch.randn(1, 3, 64)
    vis = torch.tensor([2, 3, 4, 5, 6])
    base = prefilled(attn, rope, ctx)

    with torch.no_grad():
        native = run_attn(attn, rope, rows, 12, deepcopy(base))
        with vg.install_gain(bb, vis, 1.0, 1.0) as st:
            ident = run_attn(attn, rope, rows, 12, deepcopy(base))
    check("gamma=1 equals native eager", torch.allclose(ident, native, atol=1e-5),
          float((ident - native).abs().max()))
    check("hook counted rows", st["calls"] == 1 and st["rows"] == 3)

    # manual decomposition: o = o_N + o_V with native alpha, output projection is linear w/o bias
    def manual(gv, gn, donor_v=None):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, repeat_kv
        c = deepcopy(base)
        with torch.no_grad():
            run_attn(attn, rope, rows, 12, c)       # appends the 3 rows' K/V
            K = repeat_kv(c.layers[0].keys, 2)
            V = repeat_kv(c.layers[0].values, 2).clone()
            if donor_v is not None:
                V[:, :, vis, :] = repeat_kv(donor_v, 2)
            T = 3
            pos = torch.arange(12, 15)[None, None, :].expand(3, 1, T)
            cos, sin = rope(rows, pos)
            q = attn.q_norm(attn.q_proj(rows).view(1, T, -1, 16)).transpose(1, 2)
            q, _ = apply_rotary_pos_emb(q, torch.zeros_like(q), cos, sin)
            s = q @ K.transpose(-2, -1) * attn.scaling + causal(12, T)
            a = torch.softmax(s, -1)
            m = torch.zeros(15, dtype=torch.bool); m[vis] = True
            oV = (a * m.float()) @ V
            oN = (a * (~m).float()) @ V
            return attn.o_proj((gn * oN + gv * oV).transpose(1, 2).reshape(1, T, -1))

    for gv, gn in ((0.0, 1.0), (0.5, 1.0), (2.0, 1.0), (4.0, 1.0), (1.0, 0.0), (1.0, 3.0)):
        with torch.no_grad(), vg.install_gain(bb, vis, gv, gn):
            got = run_attn(attn, rope, rows, 12, deepcopy(base))
        want = manual(gv, gn)
        check(f"o_N*{gn:g} + o_V*{gv:g}", torch.allclose(got, want, atol=1e-5),
              float((got - want).abs().max()))

    # gamma_V = 0 equals memory/visual_cut 'zero'
    from memory import visual_cut as vcut
    old = {k: os.environ.get(k) for k in ("VLMAS_VCUT", "VLMAS_VCUT_STAGE", "VLMAS_VCUT_LAYERS")}
    os.environ["VLMAS_VCUT"] = "zero"; os.environ["VLMAS_VCUT_STAGE"] = "A"; os.environ.pop("VLMAS_VCUT_LAYERS", None)
    with torch.no_grad(), vcut.install_visual_cut(bb, vis, stage="A"):
        cut = run_attn(attn, rope, rows, 12, deepcopy(base))
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    with torch.no_grad(), vg.install_gain(bb, vis, 0.0, 1.0):
        v0 = run_attn(attn, rope, rows, 12, deepcopy(base))
    check("V0 == visual_cut zero", torch.allclose(v0, cut, atol=1e-6), float((v0 - cut).abs().max()))

    # donor: visual V rows swapped, K untouched
    donor_v = torch.randn(1, 2, int(vis.numel()), 16)
    for gv in (1.0, 2.0):
        with torch.no_grad(), vg.install_gain(bb, vis, gv, 1.0, donor=[donor_v]):
            got = run_attn(attn, rope, rows, 12, deepcopy(base))
        want = manual(gv, 1.0, donor_v=donor_v)
        check(f"donor gamma {gv:g}", torch.allclose(got, want, atol=1e-5), float((got - want).abs().max()))
    with torch.no_grad(), vg.install_gain(bb, vis, 0.0, 1.0, donor=[donor_v]):
        w0 = run_attn(attn, rope, rows, 12, deepcopy(base))
    check("W0 == V0 (content removed either way)", torch.allclose(w0, v0, atol=1e-6))

    # single decode row (T=1) path, no causal mask branch
    one = rows[:, :1]
    with torch.no_grad():
        nat1 = run_attn(attn, rope, one, 12, deepcopy(base))
        with vg.install_gain(bb, vis, 1.0, 1.0):
            id1 = run_attn(attn, rope, one, 12, deepcopy(base))
    check("T=1 gamma=1 equals native", torch.allclose(id1, nat1, atol=1e-5))
    # hooks removed after the context
    with torch.no_grad():
        again = run_attn(attn, rope, rows, 12, deepcopy(base))
    check("hooks removed", torch.equal(again, native))
except Exception as exc:  # noqa: BLE001
    import traceback
    traceback.print_exc()
    check("attention math block", False, repr(exc))

print(f"\n{PASS}/{PASS + FAIL} passed")
sys.exit(1 if FAIL else 0)
