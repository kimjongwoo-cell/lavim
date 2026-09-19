"""LR-PSR (C2 12.6) unit tests. python3 tests_hj_test_lrpsr.py

Pure math is checked against an independent implementation (explicit projector from a QR basis,
per-support softmax written out by hand) and against the page's three conservation identities.
The hook is checked on a stub attention module whose o_proj / q_proj / cache are plain tensors.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memory import lrpsr  # noqa: E402

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


# ------------------------------------------------------------------ config / layer set
for spec, want in (("24,27,30", (24, 27, 30)), ("24-26", (24, 25, 26)), ("30", (30,)),
                   ("24, 27 ,30", (24, 27, 30))):
    check(f"layer_set {spec}", lrpsr.layer_set(spec, 36) == want, lrpsr.layer_set(spec, 36))
check("layer_set clips", lrpsr.layer_set("30,40", 36) == (30,))
try:
    lrpsr.layer_set("99", 36)
    check("layer_set empty raises", False)
except ValueError:
    check("layer_set empty raises", True)
os.environ.pop("VLMAS_LRPSR", None)
check("off", not lrpsr.enabled())
os.environ["VLMAS_LRPSR"] = "1"
check("on", lrpsr.enabled() and lrpsr.config()["layers"] == "24,27,30"
      and lrpsr.config()["support"] == "obs" and lrpsr.config()["rows"] == "gen")
os.environ["VLMAS_LRPSR"] = "identity"
check("identity mode", lrpsr.enabled() and lrpsr.config()["mode"] == "identity")
os.environ["VLMAS_LRPSR"] = "bogus"
check("bogus off", not lrpsr.enabled())

# ------------------------------------------------------------------ orth / project
d, T = 16, 5
C = torch.randn(d, T, dtype=torch.float64)
U = lrpsr.orth(C)
check("orth shape", U.shape == (d, T), U.shape)
check("orth orthonormal", near(U.transpose(0, 1) @ U, torch.eye(T, dtype=U.dtype), 1e-5))
Q_qr, _ = torch.linalg.qr(C.to(torch.float32))
P_svd = U @ U.transpose(0, 1)
check("orth projector == QR projector", near(P_svd, Q_qr @ Q_qr.transpose(0, 1), 1e-4))
Cr = C[:, :3] @ torch.randn(3, T, dtype=torch.float64)          # rank 3
check("orth numerical rank", lrpsr.orth(Cr).shape[-1] == 3, lrpsr.orth(Cr).shape)
check("orth empty", lrpsr.orth(torch.zeros(d, 0)).shape == (d, 0))
x = torch.randn(4, d, dtype=torch.float32)
Px = lrpsr.project(U, x)
check("project idempotent", near(lrpsr.project(U, Px), Px, 1e-4))
check("project complement orthogonal", float(((x - Px) @ U).abs().max()) < 1e-4)
check("project empty basis = 0", near(lrpsr.project(torch.zeros(d, 0), x), torch.zeros_like(x)))

# ------------------------------------------------------------------ support_messages
H, Ts, nv, D = 2, 3, 7, 4
scores = torch.randn(1, H, Ts, nv, dtype=torch.float64)
Vv = torch.randn(1, H, nv, D, dtype=torch.float64)
groups = [[0, 3, 5], [1, 2], [4, 6]]
ms = lrpsr.support_messages(scores, Vv, groups)
ref = []
for g in groups:
    a = torch.softmax(scores[..., g], dim=-1)
    ref.append(torch.einsum("bhtj,bhjd->bhtd", a, Vv[:, :, g, :]))
check("support_messages count", len(ms) == 3)
check("support_messages == manual", all(near(m, r) for m, r in zip(ms, ref)))
check("per-support softmax sums to 1",
      near(torch.softmax(scores[..., groups[0]], -1).sum(-1),
           torch.ones(1, H, Ts, dtype=torch.float64)))

# ------------------------------------------------------------------ readout + conservation
dm = 24
U = lrpsr.orth(torch.randn(dm, 4))
m_V = torch.randn(1, 5, dm)
m_g = [torch.randn(1, 5, dm) for _ in range(3)]
m_t, diag = lrpsr.readout(m_V, m_g, U)
P = U @ U.transpose(0, 1)
check("5.1 P component conserved", near(m_t @ P, m_V @ P, 1e-4))
perp_t = m_t - m_t @ P
perp_v = m_V - m_V @ P
check("5.2 complement energy conserved", near(perp_t.norm(dim=-1), perp_v.norm(dim=-1), 1e-4))
check("5.3 total norm conserved", near(m_t.norm(dim=-1), m_V.norm(dim=-1), 1e-4))
# independent recomputation of the complement direction
R = torch.stack([g - g @ P for g in m_g], 0).mean(0)
manual = m_V @ P + R * (perp_v.norm(dim=-1, keepdim=True) / R.norm(dim=-1, keepdim=True))
check("readout == manual", near(m_t, manual, 1e-4))
check("readout direction = support mean complement",
      float(torch.nn.functional.cosine_similarity(perp_t, R, dim=-1).min()) > 1 - 1e-5)
# single support: complement direction comes from that support alone
m_t1, _ = lrpsr.readout(m_V, [m_g[0]], U)
r1 = m_g[0] - m_g[0] @ P
check("single support", near(m_t1, m_V @ P + r1 * (perp_v.norm(dim=-1, keepdim=True)
                                                   / r1.norm(dim=-1, keepdim=True)), 1e-4))
# empty basis: P = 0, so the whole visual message is replaced by the norm-matched support mean
m_t0, _ = lrpsr.readout(m_V, m_g, torch.zeros(dm, 0))
Rm = torch.stack(m_g, 0).mean(0)
check("empty basis", near(m_t0, Rm * (m_V.norm(dim=-1, keepdim=True)
                                      / Rm.norm(dim=-1, keepdim=True)), 1e-4))
check("empty basis keeps norm", near(m_t0.norm(dim=-1), m_V.norm(dim=-1), 1e-4))
# degenerate: supports whose complements cancel -> identity fallback
m_td, diagd = lrpsr.readout(m_V, [m_g[0], -m_g[0]], U)
check("degenerate fallback", near(m_td, m_V, 1e-5) and diagd["degenerate"])
# a support message inside the reference span leaves only the fallback direction
inside = (torch.randn(1, 5, 4) @ U.transpose(0, 1))
m_ti, diagi = lrpsr.readout(m_V, [inside], U)
check("support fully inside span -> fallback", near(m_ti, m_V, 1e-4) and diagi["degenerate"])

# ------------------------------------------------------------------ hook on a stub module
class Stub(torch.nn.Module):
    """Minimal self_attn: q_proj/q_norm/o_proj + module.scaling, head_dim, num_key_value_groups."""

    def __init__(self, dm, H, Hd):
        super().__init__()
        self.q_proj = torch.nn.Linear(dm, H * Hd, bias=False)
        self.o_proj = torch.nn.Linear(H * Hd, dm, bias=False)
        self.q_norm = torch.nn.Identity()
        self.head_dim = Hd
        self.num_key_value_groups = 1
        self.scaling = Hd ** -0.5

    def forward(self, hidden_states=None, position_embeddings=None, attention_mask=None,
                past_key_values=None, **kw):
        keys, values = past_key_values.layers[0].keys, past_key_values.layers[0].values
        T = hidden_states.shape[1]
        q = self.q_proj(hidden_states).view(1, T, -1, self.head_dim).transpose(1, 2)
        s = torch.matmul(q, keys.transpose(-2, -1)) * self.scaling
        L = keys.shape[2]
        past = L - T
        colm = torch.arange(L)[None, None, None, :]
        rowm = (past + torch.arange(T))[None, None, :, None]
        s = s.masked_fill(colm > rowm, float("-inf"))
        o = torch.matmul(torch.softmax(s, -1), values)
        return (self.o_proj(o.transpose(1, 2).reshape(1, T, -1)),)


class Layer(torch.nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.self_attn = attn


class LM(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)


class BB:
    def __init__(self, lm):
        self.lm = lm
        self.device = torch.device("cpu")


class CacheLayer:
    def __init__(self, k, v):
        self.keys, self.values = k, v


class Cache:
    def __init__(self, k, v):
        self.layers = [CacheLayer(k, v)]


dm, H, Hd, L, Tq = 32, 4, 8, 20, 3
attn = Stub(dm, H, Hd).double()
bb = BB(LM([Layer(attn)]))
hidden = torch.randn(1, Tq, dm, dtype=torch.float64)
keys = torch.randn(1, H, L, Hd, dtype=torch.float64)
values = torch.randn(1, H, L, Hd, dtype=torch.float64)
cache = Cache(keys, values)
ones = torch.ones(1, Tq, Hd, dtype=torch.float64)
pe = (ones.clone(), torch.zeros_like(ones))   # cos=1, sin=0 -> rotary is the identity
vis_cols = torch.arange(2, 14)
lat_cols = torch.arange(14, 18)
groups = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]


def run(**kw):
    with lrpsr.install_lrpsr(bb, vis_cols, lat_cols, groups, layers=(0,), **kw) as st:
        out = attn(hidden_states=hidden, position_embeddings=pe, attention_mask=None,
                   past_key_values=cache)
    return out[0], st


native, _ = run(identity=True)   # hook installed but identity -> recompute path
with torch.no_grad():
    plain = attn(hidden_states=hidden, position_embeddings=pe, attention_mask=None,
                 past_key_values=cache)[0]
# the hook recomputes attention in fp32 (as PSAR does), so a float64 stub differs by ~1e-8
check("identity reproduces native", near(native, plain, 1e-6), float((native - plain).abs().max()))
out, st = run()
check("hook touched only the last row", near(out[:, :-1, :], plain[:, :-1, :]))
check("hook changed the last row", float((out[:, -1] - plain[:, -1]).abs().max()) > 1e-6)
check("stats", st["calls"] == 1 and st["rows"] == 1 and st["n_groups"] == 3
      and st["rank_sum"] == 4, st)
out_all, st_all = run(rows="all")
check("rows=all touches every row", st_all["rows"] == Tq
      and float((out_all[:, 0] - plain[:, 0]).abs().max()) > 1e-9)

# independent reference for the touched row: recompute the whole formula by hand
with torch.no_grad():
    q = attn.q_proj(hidden).view(1, Tq, H, Hd).transpose(1, 2)
    s = torch.matmul(q, keys.transpose(-2, -1)) * attn.scaling
    colm = torch.arange(L)[None, None, None, :]
    rowm = (L - Tq + torch.arange(Tq))[None, None, :, None]
    s = s.masked_fill(colm > rowm, float("-inf"))
    a = torch.softmax(s, -1)[:, :, -1:, :]
    o = torch.matmul(a, values)
    o_v = torch.matmul(a[..., vis_cols], values[:, :, vis_cols, :])
    W = attn.o_proj

    def wo(x):
        return W(x.transpose(1, 2).reshape(1, x.shape[2], -1))

    m_V, m_NV = wo(o_v), wo(o - o_v)
    m_g = []
    for g in groups:
        gg = vis_cols[torch.tensor(g)]
        ag = torch.softmax(s[:, :, -1:, gg], -1)
        m_g.append(wo(torch.matmul(ag, values[:, :, gg, :])))
    Cm = W(values[:, :, lat_cols, :].permute(0, 2, 1, 3)
           .reshape(1, int(lat_cols.numel()), -1))[0].transpose(0, 1)
    Um = lrpsr.orth(Cm)
    Pm = Um @ Um.transpose(0, 1)
    perp = m_V - m_V @ Pm.double()
    Rm = torch.stack([g - g @ Pm.double() for g in m_g], 0).mean(0)
    ref_row = (m_NV + m_V @ Pm.double()
               + Rm * (perp.norm(dim=-1, keepdim=True) / Rm.norm(dim=-1, keepdim=True)))
check("hook == independent formula", near(out[:, -1:, :], ref_row, 1e-4),
      float((out[:, -1:, :] - ref_row).abs().max()))
# conservation of the visual part, measured through the hook output
check("hook conserves total visual norm",
      near((out[:, -1:, :] - m_NV).norm(dim=-1), m_V.norm(dim=-1), 1e-4))
check("hook conserves P component",
      near((out[:, -1:, :] - m_NV) @ Pm.double(), m_V @ Pm.double(), 1e-4))
check("summary text", "mean_rank" in lrpsr.summary(st) and "supports=3" in lrpsr.summary(st))

print(f"LR-PSR tests {PASS}/{PASS + FAIL}")
sys.exit(1 if FAIL else 0)
