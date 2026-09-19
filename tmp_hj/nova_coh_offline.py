"""Offline: NOVA rho vs rho + representation neighbourhood coherence on dumped Reasoner prefills (no existing file modified).
coh_i = || sum_j K_ij z_j ||  (z = L2-normalised merged vision feature, K = in-crop normalised Gaussian sigma=2 cells);
glass tokens have content-free, mutually inconsistent embeddings -> small coh. c_i = rank of coh_i in the prefill (0..1).
e_rho = 1 - b_rho ; e_coh = 1 - b_rho * (1 - c_i). Proxy labels from white-pixel connected components (glass: component
> 40k px touching the crop border; fat/lumen: component < 8k px not touching the border), token white fraction >= 0.9."""
import glob, json, os, sys
import numpy as np, torch
from scipy import ndimage
sys.path.insert(0, os.getcwd())
from memory import secld_select as sec, nova_rho_select as rho_mod
from transformers import Qwen3VLForConditionalGeneration

MODEL = "/home/users/whddn12316/models/Qwen3-VL-4B-Thinking"
P, M, T = 16, 2, 2
OUT = sys.argv[1]
FILES = sys.argv[2:]
os.makedirs(OUT, exist_ok=True)
model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda().eval()


def crops_px(pv, thw):
    out, off = [], 0
    for t, h, w in thw:
        n = t * h * w
        x = pv[off:off + n].reshape(n, 3, T, P, P)[:, :, 0]
        gh, gw = h // M, w // M
        x = x.reshape(gh, gw, M, M, 3, P, P).transpose(0, 2, 5, 1, 3, 6, 4).reshape(gh * M * P, gw * M * P, 3)
        out.append((np.clip((x * 0.5 + 0.5) * 255, 0, 255), gh, gw))
        off += n
    return out


def proxy_labels(img, gh, gw):
    gray = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    wht = gray > 215
    lab, n = ndimage.label(wht)
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    glass_c = np.zeros(n + 1, bool); fat_c = np.zeros(n + 1, bool)
    for k in range(1, n + 1):
        if sizes[k - 1] > 40000 and k in border: glass_c[k] = True
        if sizes[k - 1] < 8000 and k not in border: fat_c[k] = True
    S = M * P
    cell = lambda a: a.reshape(gh, S, gw, S).transpose(0, 2, 1, 3).reshape(gh * gw, S * S)
    wf = cell(wht).mean(1)
    L = cell(lab)
    g = np.array([glass_c[r[r > 0]].mean() if (r > 0).any() else 0 for r in L])
    f = np.array([fat_c[r[r > 0]].mean() if (r > 0).any() else 0 for r in L])
    lbl = np.full(gh * gw, -1)
    lbl[(wf >= 0.9) & (g > 0.5)] = 0
    lbl[(wf >= 0.9) & (f > 0.5)] = 1
    return lbl


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    a = np.concatenate([pos, neg]); r = a.argsort().argsort() + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


rows = []
for fi, fn in enumerate(FILES):
    d = torch.load(fn)
    pv, thw = d["pixel_values"].float(), d["grid_thw"]
    with torch.no_grad():
        feats = model.model.get_image_features(pv.cuda().to(torch.bfloat16), thw.cuda(), return_dict=True).pooler_output
    feats = torch.cat(list(feats), 0) if isinstance(feats, (list, tuple)) else feats
    feats = feats.float()
    cps = crops_px(pv.numpy(), thw.tolist())
    grids = [(gh, gw) for _, gh, gw in cps]
    counts = [gh * gw for gh, gw in grids]
    z = torch.nn.functional.normalize(feats, dim=-1)
    coh, off = [], 0
    for gh, gw in grids:
        zz = z[off:off + gh * gw].T.reshape(-1, gh, gw)
        Mv = rho_mod.local_occupancy(zz.reshape(-1, gh * gw).T.contiguous().reshape(-1), [(gh, gw)] * 0 or [(gh, gw)], 2.0) if False else None
        # in-crop normalised Gaussian on each channel
        sig = 2.0; rad = int(round(4 * sig)); ax = torch.arange(-rad, rad + 1, dtype=torch.float32, device=z.device)
        g1 = torch.exp(-ax ** 2 / (2 * sig * sig)); k = (g1[:, None] * g1[None, :])[None, None]
        num = torch.nn.functional.conv2d(zz[:, None], k, padding=rad)[:, 0]
        den = torch.nn.functional.conv2d(torch.ones(1, 1, gh, gw, device=z.device), k, padding=rad)[0, 0]
        coh.append((num / den).norm(dim=0).reshape(-1))
        off += gh * gw
    coh = torch.cat(coh).cpu()
    c = (coh.argsort().argsort().float() / (len(coh) - 1))
    cfg = sec.config(); cfg["tau_p"] = 215.0; cfg["sigma"] = 2.0
    ev = rho_mod.evidence_rho(pv, thw, cfg)
    b = ev["b"].float(); e_rho = 1 - b; e_coh = 1 - b * (1 - c)
    lbl = np.concatenate([proxy_labels(img, gh, gw) for img, gh, gw in cps])
    B = int(round(0.25 * len(c)))
    res = {"file": fn, "n": len(c), "glass": int((lbl == 0).sum()), "fat": int((lbl == 1).sum()),
           "auc_coh_fat_gt_glass": auc(coh.numpy()[lbl == 1], coh.numpy()[lbl == 0]),
           "auc_e_rho_fat_gt_glass": auc(e_rho.numpy()[lbl == 1], e_rho.numpy()[lbl == 0])}
    keeps = {"v1": d["keep"].numpy()}
    for name, e in (("rho", e_rho), ("rho_coh", e_coh)):
        q = e.clamp_min(0).sqrt()[:, None].cuda() * z
        sup = torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(counts)).cuda()
        order, _, _ = sec.support_logdet_greedy(q, sup, B, first_per_support=int(cfg.get("min_per_support", 0)))
        kk = np.zeros(len(c), bool); kk[order.cpu().numpy()] = True; keeps[name] = kk
    for name, kk in keeps.items():
        res[f"{name}_kept_glass"] = int((kk & (lbl == 0)).sum()); res[f"{name}_kept_fat"] = int((kk & (lbl == 1)).sum())
        res[f"{name}_kept_white09"] = int((kk & (lbl >= 0)).sum())
    rows.append(res); print(json.dumps(res), flush=True)
    np.savez_compressed(f"{OUT}/{fi:02d}.npz", coh=coh.numpy(), c=c.numpy(), b=b.numpy(), lbl=lbl,
                        **{f"keep_{k}": v for k, v in keeps.items()}, grids=np.array(grids))
json.dump(rows, open(f"{OUT}/summary.json", "w"), indent=1)
