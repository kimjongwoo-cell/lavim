"""NOVA v2 unit tests (CPU, no model). python tests_hj_test_nova_v2.py"""
import math
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
for k in list(os.environ):
    if k.startswith("VLMAS_NOVA"):
        del os.environ[k]
os.environ["VLMAS_C1_SELECT"] = "nova"
from memory import nova_v2_select as V2  # noqa: E402
from memory import secld_select as S  # noqa: E402
from memory import nova_select as NV  # noqa: E402

PASS = FAIL = 0
torch.manual_seed(0)


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print("FAIL", name, extra)


def make_pv(rgb_cells, gh, gw, patch=16, temporal=2, merge=2):
    """rgb_cells [gh*gw, 3] uint8 levels -> pixel_values in HF layout for one crop of gh x gw merged cells."""
    H, W = gh * merge, gw * merge
    img = torch.zeros(3, H * patch, W * patch)
    for i in range(gh):
        for j in range(gw):
            img[:, i * merge * patch:(i + 1) * merge * patch, j * merge * patch:(j + 1) * merge * patch] = \
                rgb_cells[i * gw + j].float()[:, None, None]
    x = (img / 255.0 - 0.5) / 0.5
    rows = []
    for mi in range(gh):
        for mj in range(gw):
            for a in range(merge):
                for b in range(merge):
                    pi, pj = mi * merge + a, mj * merge + b
                    p = x[:, pi * patch:(pi + 1) * patch, pj * patch:(pj + 1) * patch]
                    rows.append(p[:, None].expand(3, temporal, patch, patch).reshape(-1))
    return torch.stack(rows), torch.tensor([[1, H, W]])


# 1. optical density: white is exactly 0, darker is larger
cells = torch.tensor([[255, 255, 255], [200, 120, 180], [0, 0, 0]])
pv, thw = make_pv(cells, 1, 3)
u = V2.cell_od(pv, thw)
check("od shape", tuple(u.shape) == (3, 3))
check("od white == 0", float(u[0].abs().max()) == 0.0, u[0])
check("od black == log256", abs(float(u[2, 0]) - math.log(256.0)) < 1e-9)
check("od monotone", float(u[1].sum()) > 0 and float(u[2].sum()) > float(u[1].sum()))

# 2. EM recovers a synthetic mixture with the background mean pinned at 0
nb, nt = 600, 1400
ub = torch.randn(nb, 3, dtype=torch.float64) * 0.02
ut = torch.tensor([0.3, 0.9, 0.6], dtype=torch.float64) + torch.randn(nt, 3, dtype=torch.float64) * 0.12
uu = torch.cat([ub, ut])
fit = V2.fit_null_em(uu)
check("em pi_b", abs(fit["pi_b"] - nb / (nb + nt)) < 0.02, fit["pi_b"])
check("em mu_t", float((fit["mu_t"] - torch.tensor([0.3, 0.9, 0.6], dtype=torch.float64)).abs().max()) < 0.03, fit["mu_t"])
check("em w on B", float(fit["w"][:nb].mean()) > 0.97)
check("em w on T", float(fit["w"][nb:].mean()) < 0.03)
check("em w range", float(fit["w"].min()) >= 0 and float(fit["w"].max()) <= 1)
check("em not degenerate", not fit["degenerate"])
# loglik does not decrease vs the start-only posterior (EM monotone): run 1 iteration vs full
f1 = V2.fit_null_em(uu, max_iter=1)
check("em loglik >= 1-iter", fit["loglik"] >= f1["loglik"] - 1e-6, (fit["loglik"], f1["loglik"]))
# no threshold: scaling the tissue away still gives a smooth posterior
check("em tiny N degenerate", V2.fit_null_em(uu[:3])["degenerate"])

# 3. local null: constant w -> b == w (in-crop normalisation, no border bias)
w = torch.full((5 * 7,), 0.4, dtype=torch.float64)
b = V2.local_null(w, [(5, 7)])
check("heat constant", float((b - 0.4).abs().max()) < 1e-12)
# isolated background cell inside tissue gets a small b, extended background stays near 1
w = torch.zeros(9 * 9, dtype=torch.float64); w[4 * 9 + 4] = 1.0
b = V2.local_null(w, [(9, 9)])
check("isolated cell diluted", float(b[4 * 9 + 4]) < 0.2, float(b[4 * 9 + 4]))
w = torch.ones(9 * 9, dtype=torch.float64); w[4 * 9 + 4] = 0.0
b = V2.local_null(w, [(9, 9)])
check("hole in glass still high", float(b[4 * 9 + 4]) > 0.8, float(b[4 * 9 + 4]))
# crops do not leak into each other
w = torch.cat([torch.ones(16, dtype=torch.float64), torch.zeros(16, dtype=torch.float64)])
b = V2.local_null(w, [(4, 4), (4, 4)])
check("no cross-crop leak", float(b[:16].min()) == 1.0 and float(b[16:].max()) == 0.0)
# matches a dense K_ij computed from the page formula (exact when the crop fits inside radius 4)
def dense_b(w, gh, gw):
    p = torch.tensor([[i, j] for i in range(gh) for j in range(gw)], dtype=torch.float64)
    K = torch.exp(-((p[:, None] - p[None]) ** 2).sum(-1) / 2.0)
    return (K / K.sum(dim=1, keepdim=True)) @ w
w = torch.rand(25, dtype=torch.float64)
check("heat == dense page K 5x5", float((V2.local_null(w, [(5, 5)]) - dense_b(w, 5, 5)).abs().max()) < 1e-12)
w = torch.rand(4 * 12, dtype=torch.float64)
err = float((V2.local_null(w, [(4, 12)]) - dense_b(w, 4, 12)).abs().max())
check("heat 4-radius truncation small", err < 1e-3, err)
check("heat kernel radius 4", tuple(V2.heat_kernel(1.0).shape) == (9, 9))

# 4. evidence_v2 contract == secld_select.evidence keys
g = 8
rgb = torch.full((g * g, 3), 255)
rgb[: g * g // 2] = torch.tensor([150, 60, 140])
rgb[: g * g // 2] += torch.randint(-20, 20, (g * g // 2, 3))
rgb = rgb.clamp(0, 255)
pv, thw = make_pv(rgb, g, g)
ev1 = S.evidence(pv, thw, S.config())
ev2 = V2.evidence_v2(pv, thw, {"tau_p": 215.0, "tau_c": 0.9})
check("keys superset", set(ev1) <= set(ev2), set(ev1) - set(ev2))
check("e shape", tuple(ev2["e"].shape) == tuple(ev1["e"].shape))
check("m identical to v1 (diagnostic)", torch.equal(ev1["m"], ev2["m"]))
check("e = 1 - b", float((ev2["e"] - (1 - ev2["b"])).abs().max()) < 1e-6)
check("glass half low e", float(ev2["e"][-g:].mean()) < 0.2, float(ev2["e"][-g:].mean()))
check("tissue half high e", float(ev2["e"][:g].mean()) > 0.8, float(ev2["e"][:g].mean()))

# 5. patch: off -> no swap; on -> NOVA select uses v2 evidence
orig = S.evidence
V2.patch(S)
check("flag off no patch", S.evidence is orig)
os.environ["VLMAS_NOVA_V2"] = "1"
V2.patch(S)
check("flag on patched", S.evidence is V2.evidence_v2)
V2.patch(S)
check("idempotent", S.evidence is V2.evidence_v2 and V2._V1_EVIDENCE is orig)
feats = torch.randn(g * g, 32)
cfg = NV.config()
info = NV.select_tokens(pv, thw, feats, [g * g], NV.budget_for(g * g, 0.25), cfg)
check("select budget", int(info["keep"].sum()) == 16)
check("select uses v2 e", torch.allclose(info["e"], ev2["e"]))
check("select avoids glass", int(info["keep"][g * g // 2:].sum()) <= 2, int(info["keep"][g * g // 2:].sum()))
for mode in ("logdet", "etopk"):
    cfg2 = dict(cfg, mode=mode)
    inf = NV.select_tokens(pv, thw, feats, [g * g], 16, cfg2)
    check(f"mode {mode} runs on v2", int(inf["keep"].sum()) == 16 and torch.allclose(inf["e"], ev2["e"]))
S.evidence = orig

print(f"{PASS}/{PASS + FAIL} passed")
sys.exit(1 if FAIL else 0)
