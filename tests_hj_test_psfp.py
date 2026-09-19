"""PSFP (C2 12.7) unit tests. python3 tests_hj_test_psfp.py

Fisher inner product and the closed-form projection are checked against an explicit
F(p) = diag(p) - p p^T matrix and against a small constrained-QP solve; the structural
properties of page §6 (no gain constant, scale invariance, constant-logit invariance) are
checked numerically. The hooks are checked on a stub decoder (attn + final RMSNorm + lm_head).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memory import psfp  # noqa: E402

PASS = FAIL = 0
torch.manual_seed(0)


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print("FAIL", name, extra)


def near(a, b, tol=1e-8):
    return float((torch.as_tensor(a) - torch.as_tensor(b)).abs().max()) < tol


# ---------------------------------------------------------------- config
os.environ.pop("VLMAS_PSFP", None)
check("off", not psfp.enabled() and not psfp.active())
for m, en, ac in (("1", True, True), ("identity", True, False), ("diag", True, False),
                  ("bogus", False, False)):
    os.environ["VLMAS_PSFP"] = m
    check(f"mode {m}", psfp.enabled() == en and psfp.active() == ac)
os.environ["VLMAS_PSFP"] = "1"
check("config defaults", psfp.config()["layers"] == "24,27,30"
      and psfp.config()["rows"] == "gen" and psfp.config()["support"] == "obs")
check("layer_set", psfp.layer_set("24,27,30", 36) == (24, 27, 30))

# ---------------------------------------------------------------- fisher_inner
V = 9
z = torch.randn(V, dtype=torch.float64)
p = torch.softmax(z, -1)
F = torch.diag(p) - torch.outer(p, p)
a, b = torch.randn(V, dtype=torch.float64), torch.randn(V, dtype=torch.float64)
check("fisher_inner == a^T F b", near(psfp.fisher_inner(p, a, b), a @ F @ b))
check("fisher symmetric", near(psfp.fisher_inner(p, a, b), psfp.fisher_inner(p, b, a)))
ones = torch.ones(V, dtype=torch.float64)
check("fisher kills all-ones", abs(float(psfp.fisher_inner(p, ones, b))) < 1e-12)
check("fisher psd on self", float(psfp.fisher_inner(p, a, a)) > 0)

# ---------------------------------------------------------------- projection
e_V = torch.randn(V, dtype=torch.float64) * 0.3
s_V = torch.randn(V, dtype=torch.float64)
z_t, d = psfp.correct_logits(z, e_V, s_V)
r = z - e_V
rs = float(psfp.fisher_inner(p, r, s_V))
if rs >= 0:
    check("no conflict -> identity", near(z_t, z) and not d["activated"] and d["k"] == 0.0)
else:
    check("conflict -> moved", not near(z_t, z) and d["activated"])
# force both branches
for sign in (+1, -1):
    sv = s_V * sign
    zt, dd = psfp.correct_logits(z, e_V, sv)
    rstar = zt - e_V
    c = float(psfp.fisher_inner(p, rstar, sv))
    check(f"constraint satisfied sign={sign}", c > -1e-9, c)
    if dd["activated"]:
        # KKT of the QP: the constraint is tight and the gradient is -mu F s with mu >= 0
        tight = float(rstar @ F @ sv)
        grad = F @ (rstar - r)
        # L = 1/2 (u-r)'F(u-r) - mu <u,s>_F  =>  F(r*-r) = mu F s, mu >= 0
        mu = float(grad @ sv) / float(sv @ F @ sv)
        check(f"KKT tight sign={sign}", abs(tight) < 1e-9, tight)
        check(f"KKT multiplier sign={sign}", mu > 0, mu)
        check(f"KKT gradient direction sign={sign}",
              near(grad, mu * (F @ sv), 1e-9), float((grad - mu * (F @ sv)).abs().max()))
        # any feasible point is at least as far from r as the closed form
        for _ in range(200):
            u = r + torch.randn(V, dtype=torch.float64) * 0.3
            if float(u @ F @ sv) >= 0:
                d0 = float((rstar - r) @ F @ (rstar - r))
                d1 = float((u - r) @ F @ (u - r))
                check(f"closed form is minimal sign={sign}", d0 <= d1 + 1e-12, (d0, d1))
                break
        delta = zt - z
        cosv = torch.nn.functional.cosine_similarity(delta, sv, dim=-1)
        check(f"correction is parallel to s_V sign={sign}", abs(abs(float(cosv)) - 1) < 1e-9)

# §6.1/6.2 scale invariance, §6.3 constant shift invariance
zt1, _ = psfp.correct_logits(z, e_V, -s_V)
zt2, _ = psfp.correct_logits(z, e_V, -s_V * 7.3)
check("6.2 scale invariance", near(zt1, zt2, 1e-7))
shift = 3.5
zt3, _ = psfp.correct_logits(z + shift, e_V, -s_V)
check("6.3 constant logit shift invariance", near(zt3 - shift, zt1, 1e-7))
zt4, _ = psfp.correct_logits(z, e_V, -s_V + 2.0 * torch.ones(V, dtype=torch.float64))
check("6.3 constant shift of s_V invariance", near(zt4, zt1, 1e-6))
# degenerate s_V (constant vector -> zero Fisher norm): no correction, no NaN
ztc, dc = psfp.correct_logits(z, e_V, torch.ones(V, dtype=torch.float64))
check("degenerate s_V safe", torch.isfinite(ztc).all() and near(ztc, z, 1e-6), dc)

# ---------------------------------------------------------------- hooks on a stub decoder
class Attn(torch.nn.Module):
    def __init__(self, dm, H, Hd):
        super().__init__()
        self.q_proj = torch.nn.Linear(dm, H * Hd, bias=False)
        self.o_proj = torch.nn.Linear(H * Hd, dm, bias=False)
        self.q_norm = torch.nn.Identity()
        self.head_dim, self.num_key_value_groups, self.scaling = Hd, 1, Hd ** -0.5

    def forward(self, hidden_states=None, position_embeddings=None, attention_mask=None,
                past_key_values=None, **kw):
        k, v = past_key_values.layers[self.li].keys, past_key_values.layers[self.li].values
        T = hidden_states.shape[1]
        q = self.q_proj(hidden_states).view(1, T, -1, self.head_dim).transpose(1, 2)
        o = torch.matmul(torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scaling, -1), v)
        return (self.o_proj(o.transpose(1, 2).reshape(1, T, -1)),)


class Layer(torch.nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.self_attn = attn


class Norm(torch.nn.Module):
    def __init__(self, dm):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.rand(dm, dtype=torch.float64) + 0.5)
        self.variance_epsilon = 1e-6

    def forward(self, hidden_states):
        v = hidden_states.pow(2).mean(-1, keepdim=True)
        return self.weight * hidden_states / (v + self.variance_epsilon).sqrt()


class LM(torch.nn.Module):
    def __init__(self, layers, dm):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.norm = Norm(dm)


class Outer(torch.nn.Module):
    def __init__(self, dm, Vv):
        super().__init__()
        self.lm_head = torch.nn.Linear(dm, Vv, bias=False)


class BB:
    def __init__(self, lm, outer):
        self.lm, self.model, self.device = lm, outer, torch.device("cpu")


class CL:
    def __init__(self, k, v):
        self.keys, self.values = k, v


class Cache:
    def __init__(self, kv):
        self.layers = kv


dm, H, Hd, L, Vv = 16, 2, 8, 12, 7
attns = []
for li in range(3):
    a_ = Attn(dm, H, Hd).double()
    a_.li = li
    attns.append(a_)
lm = LM([Layer(a) for a in attns], dm).double()
outer = Outer(dm, Vv).double()
bb = BB(lm, outer)
kv = [CL(torch.randn(1, H, L, Hd, dtype=torch.float64),
         torch.randn(1, H, L, Hd, dtype=torch.float64)) for _ in range(3)]
cache = Cache(kv)
hidden = torch.randn(1, 1, dm, dtype=torch.float64)
ones_pe = (torch.ones(1, 1, Hd, dtype=torch.float64), torch.zeros(1, 1, Hd, dtype=torch.float64))
vis = torch.arange(2, 10)
groups = [[0, 1, 2], [3, 4], [5, 6, 7]]


def forward_once():
    h = hidden
    for a in attns:
        h = a(hidden_states=h, position_embeddings=ones_pe, attention_mask=None,
              past_key_values=cache)[0]
    return outer.lm_head(lm.norm(h))


native = forward_once()
with psfp.install_psfp(bb, vis, groups, layers=(0, 1, 2), apply=False) as st:
    measured = forward_once()
check("apply=False leaves logits native", near(measured, native))
check("measured one step", st["steps"] == 1 and st["skipped"] == 0 and st["n_groups"] == 3, st)
with psfp.install_psfp(bb, vis, groups, layers=(0, 1, 2), apply=True) as st2:
    corrected = forward_once()
moved = not near(corrected, native, 1e-12)
check("apply=True consistent with activation", moved == (st2["activated"] == 1), (moved, st2))

# independent reference for the hook path: recompute m_V, S, rms, Phi, projection by hand
h = hidden
mV = S = None
for a in attns:
    k, v = cache.layers[a.li].keys, cache.layers[a.li].values
    q = a.q_proj(h).view(1, 1, -1, Hd).transpose(1, 2)
    sc = torch.matmul(q, k.transpose(-2, -1)) * a.scaling
    al = torch.softmax(sc, -1)
    o_v = torch.matmul(al[..., vis], v[:, :, vis, :])

    def wo(x, a=a):
        return a.o_proj(x.transpose(1, 2).reshape(1, 1, -1))[0, 0]

    msum = 0
    for g in groups:
        gg = vis[torch.tensor(g)]
        ag = torch.softmax(sc[..., gg], -1)
        msum = msum + torch.matmul(ag, v[:, :, gg, :])
    mV = wo(o_v) if mV is None else mV + wo(o_v)
    S = wo(msum) if S is None else S + wo(msum)
    h = a(hidden_states=h, position_embeddings=ones_pe, attention_mask=None,
          past_key_values=cache)[0]
rms = (h[0, -1].pow(2).mean() + lm.norm.variance_epsilon).sqrt()
Phi = lambda c: outer.lm_head.weight @ (lm.norm.weight * c / rms)  # noqa: E731
z_ref = native[0, -1].double()
ref, dref = psfp.correct_logits(z_ref, Phi(mV), Phi(S))
check("hook == independent formula", near(corrected[0, -1].double(), ref, 1e-8),
      float((corrected[0, -1].double() - ref).abs().max()))
check("diag matches", dref["activated"] == (st2["activated"] == 1))

print(f"PSFP tests {PASS}/{PASS + FAIL}")
sys.exit(1 if FAIL else 0)
