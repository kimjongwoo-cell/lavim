"""RELH (C2 13) unit tests. python3 tests_hj_test_relh.py

Endpoint bookkeeping, boundary mapping and the single-column rephase are checked against a hand
written reference on a stub decoder with the model's real rotary table. Identity (p_handoff =
p_endpoint) must reproduce native attention, and the cache must not grow or change.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memory import relh  # noqa: E402
from memory.restage import rerotate_keys  # noqa: E402

PASS = FAIL = 0
torch.manual_seed(0)


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print("FAIL", name, extra)


def near(a, b, tol=1e-9):
    return float((a - b).abs().max()) < tol


# ------------------------------------------------------------------ config / modes
os.environ.pop("VLMAS_RELH", None)
check("off", not relh.enabled() and relh.mode() == "")
for m, want in (("1", True), ("identity", True), ("answerer", True), ("reasoner", True),
                ("bogus", False)):
    os.environ["VLMAS_RELH"] = m
    check(f"mode {m}", relh.enabled() == want)
os.environ["VLMAS_RELH"] = "answerer"
check("answerer only", relh.active_at("answerer") and not relh.active_at("reasoner"))
os.environ["VLMAS_RELH"] = "reasoner"
check("reasoner only", relh.active_at("reasoner") and not relh.active_at("answerer"))
os.environ["VLMAS_RELH"] = "1"
check("both", relh.active_at("reasoner") and relh.active_at("answerer"))
check("rows default", relh.config()["rows"] == "gen")


# ------------------------------------------------------------------ endpoint bookkeeping
class BBStub:
    pass


bb0 = BBStub()
relh.record_endpoint(bb0, "planner", 100, 50)
check("planner not recorded", not getattr(bb0, "_relh_endpoints", {}))
relh.record_endpoint(bb0, "navigator", 100, 50)
relh.record_endpoint(bb0, "reasoner", 260, 130)
eps = bb0._relh_endpoints
check("endpoint col/pos", eps["navigator"] == {"col": 99, "pos": 49}
      and eps["reasoner"] == {"col": 259, "pos": 129}, eps)
check("reasoner consumer sees navigator",
      relh.endpoints_for(bb0, "reasoner") == [{"sender": "navigator", "col": 99, "pos": 49}])
check("answerer consumer sees reasoner",
      relh.endpoints_for(bb0, "answerer") == [{"sender": "reasoner", "col": 259, "pos": 129}])
check("navigator consumer sees nothing", relh.endpoints_for(bb0, "navigator") == [])

# ------------------------------------------------------------------ stub decoder + real rotary
from transformers.models.qwen3_vl.modeling_qwen3_vl import (  # noqa: E402
    Qwen3VLTextConfig, Qwen3VLTextRotaryEmbedding, apply_rotary_pos_emb)

dm, H, Hd, L = 32, 4, 16, 14
cfg = Qwen3VLTextConfig(hidden_size=dm, num_attention_heads=H, num_key_value_heads=H,
                        head_dim=Hd, intermediate_size=2 * dm, num_hidden_layers=1,
                        max_position_embeddings=512,
                        rope_scaling={"rope_type": "default", "mrope_section": [4, 2, 2]})


class Attn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(dm, H * Hd, bias=False)
        self.o_proj = torch.nn.Linear(H * Hd, dm, bias=False)
        self.q_norm = torch.nn.Identity()
        self.head_dim, self.num_key_value_groups, self.scaling = Hd, 1, Hd ** -0.5

    def forward(self, hidden_states=None, position_embeddings=None, attention_mask=None,
                past_key_values=None, **kw):
        k, v = past_key_values.layers[0].keys, past_key_values.layers[0].values
        T = hidden_states.shape[1]
        cos, sin = position_embeddings
        q = self.q_proj(hidden_states).view(1, T, -1, Hd).transpose(1, 2)
        q, _ = apply_rotary_pos_emb(q, torch.zeros_like(q), cos, sin)
        s = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        o = torch.matmul(torch.softmax(s, -1), v)
        return (self.o_proj(o.transpose(1, 2).reshape(1, T, -1).to(self.o_proj.weight.dtype)),)


class Layer(torch.nn.Module):
    def __init__(self, a):
        super().__init__()
        self.self_attn = a


class LM(torch.nn.Module):
    def __init__(self, layers, rot):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.rotary_emb = rot


class BB:
    def __init__(self, lm):
        self.lm, self.device = lm, torch.device("cpu")


class CL:
    def __init__(self, k, v):
        self.keys, self.values = k, v


class Cache:
    def __init__(self, k, v):
        self.layers = [CL(k, v)]


attn = Attn().double()
rot = Qwen3VLTextRotaryEmbedding(cfg).double()
bb = BB(LM([Layer(attn)], rot))
keys = torch.randn(1, H, L, Hd, dtype=torch.float64)
values = torch.randn(1, H, L, Hd, dtype=torch.float64)
cache = Cache(keys, values)
keys_before = keys.clone()
values_before = values.clone()
hidden = torch.randn(1, 1, dm, dtype=torch.float64)
pe = rot(keys, torch.tensor([[[L - 1]]] * 3, dtype=torch.long))
native = attn(hidden_states=hidden, position_embeddings=pe, attention_mask=None,
              past_key_values=cache)[0]
ep = [{"sender": "reasoner", "col": 6, "pos": 6}]
handoff = L - 1

with relh.install_relh(bb, ep, handoff, identity=True) as st_id:
    out_id = attn(hidden_states=hidden, position_embeddings=pe, attention_mask=None,
                  past_key_values=cache)[0]
check("identity reproduces native", near(out_id, native, 1e-8),
      float((out_id - native).abs().max()))
check("identity shift 0", st_id["shift"] == [0] and st_id["calls"] == 1, st_id)

with relh.install_relh(bb, ep, handoff) as st:
    out = attn(hidden_states=hidden, position_embeddings=pe, attention_mask=None,
               past_key_values=cache)[0]
check("rebinding moves the row", not near(out, native, 1e-9))
check("shift recorded", st["shift"] == [handoff - 6], st["shift"])
check("cache untouched", near(keys, keys_before) and near(values, values_before))
check("cardinality unchanged", int(cache.layers[0].keys.shape[2]) == L)

# independent reference: rephase that one column's key, replace only its score
old_pos = torch.tensor([[6]] * 3, dtype=torch.long)
new_pos = torch.tensor([[handoff]] * 3, dtype=torch.long)
cos_o, sin_o = rot(keys, old_pos.unsqueeze(1))
cos_n, sin_n = rot(keys, new_pos.unsqueeze(1))
cos_d = cos_n * cos_o + sin_n * sin_o
sin_d = sin_n * cos_o - cos_n * sin_o
k_bar = rerotate_keys(keys[:, :, 6:7, :], cos_d, sin_d)
q = attn.q_proj(hidden).view(1, 1, -1, Hd).transpose(1, 2)
q, _ = apply_rotary_pos_emb(q, torch.zeros_like(q), pe[0], pe[1])
s = torch.matmul(q, keys.transpose(-2, -1)) * attn.scaling
s_ref = s.clone()
s_ref[..., 6:7] = torch.matmul(q, k_bar.transpose(-2, -1)) * attn.scaling
o_ref = torch.matmul(torch.softmax(s_ref, -1), values)
ref_row = attn.o_proj(o_ref.transpose(1, 2).reshape(1, 1, -1))
check("hook == independent formula", near(out, ref_row, 1e-9),
      float((out - ref_row).abs().max()))
check("only one score changed",
      int((s_ref - s).abs().gt(1e-12).sum()) == int(s.shape[1]) * 1, int((s_ref - s).abs().gt(1e-12).sum()))
# two endpoints at once
ep2 = [{"sender": "navigator", "col": 3, "pos": 3}, {"sender": "reasoner", "col": 6, "pos": 6}]
with relh.install_relh(bb, ep2, handoff) as st2:
    out2 = attn(hidden_states=hidden, position_embeddings=pe, attention_mask=None,
                past_key_values=cache)[0]
check("two endpoints", st2["shift"] == [handoff - 3, handoff - 6] and not near(out2, out, 1e-9),
      st2["shift"])
# guards
with relh.install_relh(bb, [], handoff) as st3:
    out3 = attn(hidden_states=hidden, position_embeddings=pe, attention_mask=None,
                past_key_values=cache)[0]
check("no endpoints -> native", near(out3, native) and st3["calls"] == 0)
with relh.install_relh(bb, [{"sender": "reasoner", "col": L + 5, "pos": 3}], handoff) as st4:
    out4 = attn(hidden_states=hidden, position_embeddings=pe, attention_mask=None,
                past_key_values=cache)[0]
check("column past the cache -> native + skip", near(out4, native) and st4["skipped"] == 1, st4)
check("summary text", "endpoints=" in relh.summary(st) and "w_relh=" in relh.summary(st))

print(f"RELH tests {PASS}/{PASS + FAIL}")
sys.exit(1 if FAIL else 0)
