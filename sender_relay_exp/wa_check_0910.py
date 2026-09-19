import torch, json, glob, os
from safetensors import safe_open
M="/home/users/whddn12316/models/Qwen3-VL-4B-Thinking"
idx=json.load(open(f"{M}/model.safetensors.index.json"))["weight_map"]
names=[n for n in idx if n.endswith("embed_tokens.weight")]; head=[n for n in idx if n.endswith("lm_head.weight")]
print("embed:",names,"lm_head:",head)
def load(n):
    with safe_open(f"{M}/{idx[n]}","pt") as f: return f.get_tensor(n).float()
E=load(names[0]); U=load(head[0]) if head else E
print("tied:", (not head), E.shape)
gram=U.T@U; reg=1e-5*torch.eye(gram.shape[0]); W=torch.linalg.solve(gram+reg, U.T@E)
I=torch.eye(W.shape[0]); print("||W_a-I||_F/||I||_F =", float((W-I).norm()/I.norm()))
sv=torch.linalg.svdvals(W); print("singular values min/median/max:", float(sv.min()), float(sv.median()), float(sv.max()))
g=torch.Generator().manual_seed(0); D=torch.randn(64,W.shape[0],generator=g)
rho=(D@W).norm(dim=1)/D.norm(dim=1); print("rho on random directions min/mean/max:", float(rho.min()), float(rho.mean()), float(rho.max()))
