import sys, numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "tmp_hj")
from nova_coh_offline_utils import crops_px
fn, npz, out, title = sys.argv[1:5]
d = torch.load(fn); a = np.load(npz)
cps = crops_px(d["pixel_values"].float().numpy(), d["grid_thw"].tolist())
names = [("원본", None), ("b (ρ Gaussian, 흰=배경 판정)", "b"), ("c (이웃 일관성 순위)", "c"), ("v1 남김", "keep_v1"), ("ρ 남김", "keep_rho"), ("ρ+c 남김", "keep_rho_coh")]
sel = list(range(len(cps)))[:8]
fig, ax = plt.subplots(len(names), len(sel), figsize=(2.1 * len(sel), 2.2 * len(names)))
off = 0; offs = []
for img, gh, gw in cps: offs.append(off); off += gh * gw
for j, ci in enumerate(sel):
    img, gh, gw = cps[ci]; o = offs[ci]; S = 32
    for i, (nm, key) in enumerate(names):
        A = ax[i, j]; A.set_xticks([]); A.set_yticks([])
        if key is None: A.imshow(img.astype(np.uint8))
        elif key in ("b", "c"): A.imshow(a[key][o:o + gh * gw].reshape(gh, gw), vmin=0, vmax=1, cmap="magma")
        else:
            k = a[key][o:o + gh * gw].reshape(gh, gw)
            mask = np.kron(~k, np.ones((S, S)))[:, :, None]
            A.imshow((img * (1 - 0.75 * mask) + 40 * mask).astype(np.uint8)); A.set_title(f"{int(k.sum())}", fontsize=8)
        if j == 0: A.set_ylabel(nm, fontsize=8)
fig.suptitle(title, fontsize=11); fig.tight_layout(); fig.savefig(out, dpi=70)
