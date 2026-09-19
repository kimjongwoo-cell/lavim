"""LADR (C2 12.8) unit tests. python3 tests_hj_test_ladr.py

The rigid shift, the frame marginalisation and the rotation reuse are checked against hand
written references; the hook is checked on a stub decoder layer with a real rotary table, both
for the identity case (p_role = p_acq must reproduce native attention exactly) and against an
independent recomputation of the dual-frame read.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memory import ladr  # noqa: E402
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


# ------------------------------------------------------------------ config
os.environ.pop("VLMAS_LADR", None)
check("off", not ladr.enabled())
for m, want in (("1", True), ("identity", True), ("bogus", False)):
    os.environ["VLMAS_LADR"] = m
    check(f"mode {m}", ladr.enabled() == want)
os.environ["VLMAS_LADR"] = "1"
check("config rows default", ladr.config()["rows"] == "gen")

# ------------------------------------------------------------------ rigid shift
pos = torch.stack([torch.tensor([10, 11, 12, 13]),
                   torch.tensor([3, 3, 4, 4]),
                   torch.tensor([7, 8, 7, 8])])
new = ladr.role_positions(pos, 100)
check("shift lands min at cursor", int(new.min()) == 100)
check("rigid: all differences preserved", near((new - new[:, :1]).float(), (pos - pos[:, :1]).float()))
check("same shift for every token (per axis, backbone _reanchor convention)",
      len({tuple((new[:, j] - pos[:, j]).tolist()) for j in range(pos.shape[1])}) == 1)
check("per-axis shift value", (new[:, 0] - pos[:, 0]).tolist() == [90, 97, 93],
      (new[:, 0] - pos[:, 0]).tolist())
check("each axis min lands on cursor", [int(x) for x in new.min(dim=1).values] == [100, 100, 100])

# ------------------------------------------------------------------ marginalisation
a = torch.randn(5, dtype=torch.float64)
b = torch.randn(5, dtype=torch.float64)
m = ladr.marginalize(a, b)
ref = ((a.exp() + b.exp()) / 2).log()
check("marginalize == log mean exp", near(m, ref, 1e-10))
check("5.1 identity when frames agree", near(ladr.marginalize(a, a), a, 1e-12))
check("symmetric", near(ladr.marginalize(a, b), ladr.marginalize(b, a), 1e-12))
big = torch.tensor([800.0, -800.0], dtype=torch.float64)
check("stable at large scores", torch.isfinite(ladr.marginalize(big, big + 1)).all())
check("bounded by max", bool((m <= torch.maximum(a, b) + 1e-12).all())
      and bool((m >= torch.minimum(a, b) - 1e-12).all()))

# ------------------------------------------------------------------ stub decoder + real rotary
from transformers.models.qwen3_vl.modeling_qwen3_vl import (  # noqa: E402
    Qwen3VLTextConfig, Qwen3VLTextRotaryEmbedding, apply_rotary_pos_emb, repeat_kv)

dm, H, Hd, L, nvis = 32, 4, 16, 14, 5
cfg = Qwen3VLTextConfig(hidden_size=dm, num_attention_heads=H, num_key_value_heads=H,
                        head_dim=Hd, intermediate_size=2 * dm, num_hidden_layers=1,
                        max_position_embeddings=512, rope_scaling={"rope_type": "default",
                                                                   "mrope_section": [4, 2, 2]})


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
lm = LM([Layer(attn)], rot)
bb = BB(lm)
keys = torch.randn(1, H, L, Hd, dtype=torch.float64)
values = torch.randn(1, H, L, Hd, dtype=torch.float64)
cache = Cache(keys, values)
hidden = torch.randn(1, 1, dm, dtype=torch.float64)
q_pos = torch.tensor([[[L - 1]]] * 3, dtype=torch.long)     # [3,1,1]
pe = rot(keys, q_pos)
cols = torch.arange(3, 3 + nvis)
vis_pos = torch.stack([torch.arange(3, 3 + nvis),
                       torch.tensor([1, 1, 2, 2, 3]),
                       torch.tensor([0, 1, 0, 1, 2])])

native = attn(hidden_states=hidden, position_embeddings=pe, attention_mask=None,
              past_key_values=cache)[0]

# identity: p_role = p_acq -> exactly native (page §5.1)
with ladr.install_ladr(bb, cols, vis_pos, cursor=int(vis_pos.min()), identity=True) as st_id:
    out_id = attn(hidden_states=hidden, position_embeddings=pe, attention_mask=None,
                  past_key_values=cache)[0]
# the delta rotation tables are built in fp32 (restage convention), so a float64 stub differs
# by ~1e-9; on the bf16 model this is far below one ULP
check("identity reproduces native", near(out_id, native, 1e-8),
      float((out_id - native).abs().max()))
check("identity stats", st_id["calls"] == 1 and st_id["n_vis"] == nvis
      and st_id["shift"] == [0, 0, 0], st_id)

# real dual-frame read at a later cursor
cursor = L + 4
new_pos_ref = ladr.role_positions(vis_pos, cursor)
with ladr.install_ladr(bb, cols, vis_pos, cursor=cursor) as st:
    out = attn(hidden_states=hidden, position_embeddings=pe, attention_mask=None,
               past_key_values=cache)[0]
check("dual frame moves the row", not near(out, native, 1e-9))
check("shift recorded per axis",
      st["shift"] == [int(x) for x in (new_pos_ref[:, 0] - vis_pos[:, 0])], st["shift"])

# independent recomputation
new_pos = ladr.role_positions(vis_pos, cursor)
cos_o, sin_o = rot(keys, vis_pos.unsqueeze(1))
cos_n, sin_n = rot(keys, new_pos.unsqueeze(1))
cos_d = cos_n * cos_o + sin_n * sin_o
sin_d = sin_n * cos_o - cos_n * sin_o
k_role = rerotate_keys(keys.index_select(2, cols), cos_d, sin_d)
q = attn.q_proj(hidden).view(1, 1, -1, Hd).transpose(1, 2)
q, _ = apply_rotary_pos_emb(q, torch.zeros_like(q), pe[0], pe[1])
s_all = torch.matmul(q, keys.transpose(-2, -1)) * attn.scaling
s_role = torch.matmul(q, k_role.transpose(-2, -1)) * attn.scaling
s_ref = s_all.clone()
s_ref[..., cols] = ladr.marginalize(s_all[..., cols], s_role)
o_ref = torch.matmul(torch.softmax(s_ref, -1), values)
ref_row = attn.o_proj(o_ref.transpose(1, 2).reshape(1, 1, -1))
check("hook == independent dual-frame formula", near(out, ref_row, 1e-9),
      float((out - ref_row).abs().max()))
# 5.2: the value bank is read once -> the attention weights still sum to 1
w = torch.softmax(s_ref, -1)
check("5.2 weights sum to 1 (value read once)", near(w.sum(-1), torch.ones_like(w.sum(-1)), 1e-12))
# marginalised visual score never exceeds the better of the two frames
check("score bounded by frames",
      bool((s_ref[..., cols] <= torch.maximum(s_all[..., cols], s_role) + 1e-12).all()))
check("summary text", "role_preferred" in ladr.summary(st) and "shift" in ladr.summary(st))

# guards
bad = ladr.engine_state(type("E", (), {"_backbone": type("B", (), {})(),
                                       "_rpath_visual_cols": None})())
check("engine_state no columns", bad["cols"] is None and "no visual columns" in bad["note"])
bad2 = ladr.engine_state(type("E", (), {
    "_backbone": type("B", (), {"_restage_vis_positions": torch.zeros(3, 2, dtype=torch.long)})(),
    "_rpath_visual_cols": torch.arange(5)})())
check("engine_state position mismatch", bad2["cols"] is None and "mismatch" in bad2["note"])


# ------------------------------------------------------------------ MC-LADR (mass conserved)
with ladr.install_ladr(bb, cols, vis_pos, cursor=cursor, mass_conserved=True) as st_mc:
    out_mc = attn(hidden_states=hidden, position_embeddings=pe, attention_mask=None,
                  past_key_values=cache)[0]
check("mc flag recorded", st_mc["mass_conserved"] is True)
check("mc differs from full-softmax LADR", not near(out_mc, out, 1e-9))
check("mc differs from native", not near(out_mc, native, 1e-9))
# independent reference: native mass rho_V kept, visual-internal distribution from s_tilde
nat_alpha = torch.softmax(s_all, -1)
rho = nat_alpha[..., cols].sum(-1, keepdim=True)
pi = torch.softmax(ladr.marginalize(s_all[..., cols], s_role), -1)
alpha_mc = nat_alpha.clone()
alpha_mc[..., cols] = rho * pi
o_mc = torch.matmul(alpha_mc, values)
ref_mc = attn.o_proj(o_mc.transpose(1, 2).reshape(1, 1, -1))
check("mc == independent formula", near(out_mc, ref_mc, 1e-9),
      float((out_mc - ref_mc).abs().max()))
check("§5.4 visual mass conserved per head",
      near(alpha_mc[..., cols].sum(-1), nat_alpha[..., cols].sum(-1), 1e-12))
nonvis = torch.tensor([j for j in range(L) if j not in set(cols.tolist())])
check("non-visual weights untouched",
      near(alpha_mc[..., nonvis], nat_alpha[..., nonvis], 1e-12))
check("mc weights still sum to 1", near(alpha_mc.sum(-1), torch.ones_like(alpha_mc.sum(-1)), 1e-12))
# identity: same frames -> pi == native conditional -> exactly native
with ladr.install_ladr(bb, cols, vis_pos, cursor=int(vis_pos.min()), identity=True,
                       mass_conserved=True) as st_mci:
    out_mci = attn(hidden_states=hidden, position_embeddings=pe, attention_mask=None,
                   past_key_values=cache)[0]
check("mc identity reproduces native", near(out_mci, native, 1e-8),
      float((out_mci - native).abs().max()))

print(f"LADR tests {PASS}/{PASS + FAIL}")
sys.exit(1 if FAIL else 0)
