"""NOVA v3 unit tests (CPU). python tests_hj_test_nova_v3.py"""
import math, os, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
for k in list(os.environ):
    if k.startswith("VLMAS_NOVA"):
        del os.environ[k]
from memory import nova_v3_select as V3
from memory import secld_select as S
PASS = FAIL = 0
torch.manual_seed(0)
def check(name, cond, extra=""):
    global PASS, FAIL
    if cond: PASS += 1
    else: FAIL += 1; print("FAIL", name, extra)

g = torch.Generator().manual_seed(0)
def blob(n, mu, sd): return torch.tensor(mu, dtype=torch.float64) + torch.randn(n, 3, generator=g, dtype=torch.float64) * sd
# 1. no glass: pale + dark tissue -> gamma ~ 0
u0 = torch.cat([blob(150, [0.25, 0.7, 0.5], 0.08), blob(106, [0.6, 1.4, 0.9], 0.12)]).clamp_min(0)
gam0, r0, info0 = V3.support_gamma(u0, 0.07)
check("no glass gamma ~0", gam0 < 0.05, gam0)
# 2. glass present (OD ~0.05) + tissue -> gamma ~ 1, r high on glass cells
u1 = torch.cat([blob(120, [0.05, 0.05, 0.05], 0.01), blob(136, [0.3, 0.9, 0.6], 0.1)]).clamp_min(0)
gam1, r1, info1 = V3.support_gamma(u1, 0.07)
check("glass gamma ~1", gam1 > 0.95, gam1)
check("glass r high", float(r1[:120].mean()) > 0.9 and float(r1[120:].mean()) < 0.1, (float(r1[:120].mean()), float(r1[120:].mean())))
# 3. tau sensitivity direction: very wide tau lets pale tissue become background evidence
gam_w, _, _ = V3.support_gamma(u0, 1.0)
check("wide tau raises gamma on no-glass", gam_w >= gam0)
# 4. EM sanity: 2 separated blobs recovered
fit = V3.fit_gmm(u1, 2)
check("em weights", abs(sorted(fit["weights"].tolist())[0] - 120/256) < 0.03, fit["weights"])
check("em resp rows sum 1", torch.allclose(fit["resp"].sum(0), torch.ones(256, dtype=torch.float64)))
# 5. tiny support -> gamma 0 (no H1 fit)
gs, rs, _ = V3.support_gamma(u1[:12], 0.07)
check("tiny support gamma 0", gs == 0.0 and float(rs.abs().sum()) == 0.0)
# 6. evidence_v3 contract on synthetic pixels (one 8x8-cell crop: left half glass, right half tissue)
def make_pv(rgb_cells, gh, gw, P=16, T=2, M=2):
    H, W = gh*M, gw*M
    img = torch.zeros(3, H*P, W*P)
    for i in range(gh):
        for j in range(gw):
            img[:, i*M*P:(i+1)*M*P, j*M*P:(j+1)*M*P] = rgb_cells[i*gw+j].float()[:, None, None]
    x = (img/255.0 - 0.5)/0.5; rows = []
    for mi in range(gh):
        for mj in range(gw):
            for a in range(M):
                for b in range(M):
                    pi, pj = mi*M+a, mj*M+b
                    rows.append(x[:, pi*P:(pi+1)*P, pj*P:(pj+1)*P][:, None].expand(3, T, P, P).reshape(-1))
    return torch.stack(rows), torch.tensor([[1, H, W]])
gh = gw = 8
rgb = torch.zeros(64, 3)
for i in range(gh):
    for j in range(gw):
        rgb[i*gw+j] = torch.tensor([244., 243., 245.]) if j < 4 else torch.tensor([190., 90., 170.])
rgb += torch.randint(-6, 6, (64, 3)).float(); rgb = rgb.clamp(0, 255)
pv, thw = make_pv(rgb, gh, gw)
ev = V3.evidence_v3(pv, thw, S.config())
check("keys", {"r", "m", "b", "e", "grids", "w", "gamma"} <= set(ev))
check("glass support detected", ev["gamma"][0] > 0.5, ev["gamma"])
check("glass side low e", float(ev["e"].reshape(8, 8)[:, :3].mean()) < 0.3, float(ev["e"].reshape(8, 8)[:, :3].mean()))
check("tissue side high e", float(ev["e"].reshape(8, 8)[:, 5:].mean()) > 0.7, float(ev["e"].reshape(8, 8)[:, 5:].mean()))
rgb2 = torch.tensor([190., 90., 170.]).repeat(64, 1) + torch.randint(-20, 20, (64, 3)).float()
pv2, _ = make_pv(rgb2.clamp(0, 255), gh, gw)
ev2 = V3.evidence_v3(pv2, thw, S.config())
check("all tissue: e ~1", float(ev2["e"].min()) > 0.9, float(ev2["e"].min()))
# 7. patch flag
orig = S.evidence
V3.patch(S); check("flag off no patch", S.evidence is orig)
os.environ["VLMAS_NOVA_V3"] = "1"; V3.patch(S); check("flag on patched", S.evidence is V3.evidence_v3)
S.evidence = orig
print(f"{PASS}/{PASS+FAIL} passed"); sys.exit(1 if FAIL else 0)
