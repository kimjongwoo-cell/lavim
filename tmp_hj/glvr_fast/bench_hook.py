import time, torch
torch.manual_seed(0)
dev = "cuda"
H, Hkv, D, Hid, L, base, Nd, m = 32, 8, 128, 2560, 5000, 4600, 3850, 10
G = H // Hkv
qp = torch.nn.Linear(Hid, H * D, bias=False, device=dev, dtype=torch.bfloat16)
op = torch.nn.Linear(H * D, Hid, bias=False, device=dev, dtype=torch.bfloat16)
qn = torch.nn.RMSNorm(D, device=dev, dtype=torch.bfloat16)
K = torch.randn(Hkv, L, D, device=dev, dtype=torch.bfloat16)
V = torch.randn(Hkv, L, D, device=dev, dtype=torch.bfloat16)
don = torch.randperm(base, device=dev)[:Nd].sort().values
lat = torch.arange(4000, 4010, device=dev)
Kp = K[:, :base].float(); Vd = V.index_select(1, don).float(); U = torch.randn(H, m, D, device=dev)

def rot(q, cos, sin):
    x1, x2 = q[..., :D // 2], q[..., D // 2:]
    return q * cos + torch.cat([-x2, x1], -1) * sin

def step(R, fused=False):
    h = torch.randn(1, R, Hid, device=dev, dtype=torch.bfloat16)
    cos = torch.randn(1, 1, R, D, device=dev, dtype=torch.bfloat16); sin = cos
    q = rot(qn(qp(h).view(1, R, H, D)).transpose(1, 2), cos, sin)
    qg = q[0].float().reshape(Hkv, G * R, D)
    s = torch.cat([qg @ Kp.transpose(-2, -1), qg @ K[:, base:].float().transpose(-2, -1)], -1).view(H, R, L) * 0.088
    A = torch.softmax(s, -1)
    Ad = A.index_select(-1, don)
    rho = Ad.sum(-1, keepdim=True)
    mB = torch.einsum("kgrn,knd->kgrd", Ad.view(Hkv, G, R, -1), Vd).reshape(H, R, D)
    al = A.index_select(-1, lat); g = al / al.sum(-1, keepdim=True)
    u = torch.einsum("hrm,hmd->hrd", g, U)
    d = 0.75 * (rho * u - mB)
    return op(d.permute(1, 0, 2).reshape(1, R, -1).to(torch.bfloat16))

for R, n in ((1, 300), (350, 20)):
    for _ in range(5): step(R)
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n): step(R)
    torch.cuda.synchronize(); print(f"R={R}: {(time.time()-t)/n*1e3:.2f} ms/hook")
