"""NOVA rho (canonical draft) unit tests. python tests_hj_test_nova_rho.py"""
import os, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
for k in list(os.environ):
    if k.startswith("VLMAS_NOVA"):
        del os.environ[k]
from memory import nova_rho_select as R
from memory import secld_select as S
PASS = FAIL = 0
def check(name, cond, extra=""):
    global PASS, FAIL
    if cond: PASS += 1
    else: FAIL += 1; print("FAIL", name, extra)

# 1. in-crop normalised kernel: constant field is preserved, no padding bias
rho = torch.full((6 * 7,), 0.4)
check("constant field", float((R.local_occupancy(rho, [(6, 7)], 2.0) - 0.4).abs().max()) < 1e-6)
# 2. crops do not leak into each other
rho = torch.cat([torch.ones(25), torch.zeros(25)])
b = R.local_occupancy(rho, [(5, 5), (5, 5)], 2.0)
check("no cross-crop leak", float(b[:25].min()) > 0.999 and float(b[25:].max()) < 1e-6)
# 3. isolated bright cell is diluted; wide white region stays high
rho = torch.zeros(121); rho[60] = 1.0
check("isolated diluted", float(R.local_occupancy(rho, [(11, 11)], 2.0)[60]) < 0.15)
rho = torch.ones(121); rho[60] = 0.0
check("hole stays high", float(R.local_occupancy(rho, [(11, 11)], 2.0)[60]) > 0.85)
# 4. continuous: half-white cells contribute half of full-white ones
r_half = torch.full((49,), 0.5); r_full = torch.ones(49)
check("linear in rho", abs(float(R.local_occupancy(r_half, [(7, 7)], 2.0).mean()) - 0.5 * float(R.local_occupancy(r_full, [(7, 7)], 2.0).mean())) < 1e-6)
# 5. evidence contract on real-ish pixels (glass half / tissue half)
def make_pv(rgb_cells, gh, gw, P=16, T=2, M=2):
    H, W = gh * M, gw * M
    img = torch.zeros(3, H * P, W * P)
    for i in range(gh):
        for j in range(gw):
            img[:, i*M*P:(i+1)*M*P, j*M*P:(j+1)*M*P] = rgb_cells[i*gw+j].float()[:, None, None]
    x = (img / 255.0 - 0.5) / 0.5
    rows = []
    for mi in range(gh):
        for mj in range(gw):
            for a in range(M):
                for b_ in range(M):
                    p = x[:, (mi*M+a)*P:(mi*M+a+1)*P, (mj*M+b_)*P:(mj*M+b_+1)*P]
                    rows.append(p[:, None].expand(3, T, P, P).reshape(-1))
    return torch.stack(rows), torch.tensor([[1, H, W]])
gh = gw = 8
rgb = torch.zeros(64, 3)
for i in range(gh):
    for j in range(gw):
        rgb[i*gw+j] = torch.tensor([248., 247., 249.]) if j < 4 else torch.tensor([185., 85., 165.])
pv, thw = make_pv(rgb, gh, gw)
ev = R.evidence_rho(pv, thw, S.config())
check("keys", {"r", "m", "b", "e", "grids"} <= set(ev))
check("e = 1 - b", float((ev["e"] - (1 - ev["b"])).abs().max()) < 1e-6)
check("glass side low e", float(ev["e"].reshape(8, 8)[:, :3].mean()) < 0.25, float(ev["e"].reshape(8, 8)[:, :3].mean()))
check("tissue side high e", float(ev["e"].reshape(8, 8)[:, 5:].mean()) > 0.75, float(ev["e"].reshape(8, 8)[:, 5:].mean()))
# 6. no glass anywhere -> e == 1 (background absence = zero field)
rgb2 = torch.tensor([185., 85., 165.]).repeat(64, 1)
ev2 = R.evidence_rho(make_pv(rgb2, gh, gw)[0], thw, S.config())
check("no glass: e==1", float(ev2["e"].min()) > 0.999, float(ev2["e"].min()))
# 7. patch flag
orig = S.evidence
R.patch(S); check("flag off no patch", S.evidence is orig)
os.environ["VLMAS_NOVA_RHO"] = "1"; R.patch(S); check("flag on patched", S.evidence is R.evidence_rho)
S.evidence = orig
print(f"{PASS}/{PASS+FAIL} passed"); sys.exit(1 if FAIL else 0)
