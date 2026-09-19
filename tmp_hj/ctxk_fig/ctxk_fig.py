"""09-19 [pruning 흰 픽셀 대체 후보] figure: actual 25% keep masks of each candidate on real run crops.
Copy of tmp_hj/nova_vnorm_viz/variants_fig.py (other session, untouched) with this session's kernel-side candidates.
Selection = memory.secld_select.support_logdet_greedy (or the ctxk Hadamard copy) exactly as in the runs; the
glass / fat / tissue cell labels (fatlabel.cell_labels) are for evaluation only, never a method input.
usage: python ctxk_fig.py   -> tmp_hj/ctxk_fig/<case>.png + retention.json"""
import contextlib, glob, io, json, os, sys, types
import numpy as np, torch, torch.nn.functional as F
ROOT = "/home/users/whddn12316/wsi_latent_0915_decode_hj"
sys.path.insert(0, ROOT); sys.path.insert(0, f"{ROOT}/tmp_hj/nova_vnorm_viz"); sys.path.insert(0, f"{ROOT}/tmp_hj/nova_ctxk_hooks")
import openslide
from pathlib import Path
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from vision_text_mas.navigation_render import render_box
from vision_text_mas.dicom_slide import DicomSlide
from memory import secld_select as SEC, nova_rho_select as RHOM
from fatlabel import cell_labels
import ctxk

DATA = "/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda"
MP = "/home/users/whddn12316/models/Qwen3-VL-4B-Thinking"; dev = "cuda"
model = Qwen3VLForConditionalGeneration.from_pretrained(MP, torch_dtype=torch.bfloat16).to(dev).eval()
proc = AutoProcessor.from_pretrained(MP); vis = model.model.visual
cap = {}
def vh(mod, args, kw, out):
    hs = kw.get("hidden_states", args[0] if args else None); L = hs.shape[0]
    cap["v6"] = mod.qkv(hs).reshape(L, 3, mod.num_heads, -1)[:, 2].float().norm(dim=-1).mean(1)
vis.blocks[6].attn.register_forward_hook(vh, with_kwargs=True)

RUN = f"{ROOT}/sender_relay_exp/runs/nav4/nova"
CASES = ["gtex__GTEX-1F75I-1026", "gtex__GTEX-18D9A-2026", "gtex__GTEX-111VG-1226"]
OUTD = f"{ROOT}/tmp_hj/ctxk_fig"; os.makedirs(OUTD, exist_ok=True)
seen = {}
for d in sorted(glob.glob(f"{RUN}/gtex/gpu*_attempt_*/[0-9]*_gtex__*")):
    if os.path.exists(f"{d}/result.json"):
        seen.setdefault(os.path.basename(d).split("_", 1)[1], d)

def greedy(q, support, B):
    o, _, _ = SEC.support_logdet_greedy(q, support, B)
    return o

ret = {}
for cid in CASES:
    if cid not in seen:
        print("missing", cid); continue
    cdir = seen[cid]
    res = json.load(open(f"{cdir}/result.json")); ev = json.load(open(f"{cdir}/round_1/evidence.json"))
    sp = os.path.realpath(f"{DATA}/slides/{res['slide_id']}")
    slide = DicomSlide(Path(sp)) if os.path.isdir(sp) else openslide.OpenSlide(sp)
    imgs = [render_box(slide, types.SimpleNamespace(**p["box"]), max_side=512) for p in ev["patches"]]
    mags = [p["magnification"] for p in ev["patches"]]; C = len(imgs)
    inp = proc.image_processor(images=imgs, return_tensors="pt"); pv, thw = inp["pixel_values"].to(dev), inp["image_grid_thw"].to(dev)
    with torch.no_grad():
        vo = vis(pv.to(torch.bfloat16), grid_thw=thw)
        Fm = vo.pooler_output; Fm = (torch.cat(list(Fm), 0) if isinstance(Fm, (list, tuple)) else Fm).float()
        N = Fm.shape[0]; z = F.normalize(Fm, dim=-1)
        grids = [(16, 16)] * C
        support = torch.arange(C, device=dev).repeat_interleave(256); B = int(round(0.25 * N))
        cfg = {"tau_p": 215.0, "tau_c": 0.9, "sigma": 2.0}
        with contextlib.redirect_stdout(io.StringIO()):
            rho_e = RHOM.evidence_rho(pv.float(), thw, cfg)["e"].to(dev).float()
        v6 = cap["v6"].view(N, 4).mean(1); vhat = (v6 - v6.min()) / (v6.max() - v6.min()).clamp_min(1e-12)
        vn_e = (1.0 - RHOM.local_occupancy((1.0 - vhat).float(), grids, 2.0)).clamp(0, 1)
        rows, cols = ctxk._coords(grids, dev)
        def rbf(s):
            return torch.exp(-((rows[:, None] - rows[None, :]) ** 2 + (cols[:, None] - cols[None, :]) ** 2) / (2 * s * s))
        c_ctx = ctxk.two_scale(z, grids, 2.0, 1.0)[:, z.shape[1]:]                 # w = 1 -> pure pooled context c
        orders = {
            "NOVA rho (canonical, white px)": greedy(z * rho_e.clamp_min(0).sqrt()[:, None], support, B),
            "setting 1: value norm L6 + s2": greedy(z * vn_e.sqrt()[:, None], support, B),
            "LogDet-only (e=1)": greedy(z, support, B),
            "RBF s2 (kernel x grid RBF)": ctxk._greedy_hadamard(z, rbf(2.0), support, B)[0],
            "RBF s4": ctxk._greedy_hadamard(z, rbf(4.0), support, B)[0],
            "ctx w0.6 (z ; pooled c)": greedy(ctxk.two_scale(z, grids, 2.0, 0.6), support, B),
            "center-surround l1 (z ; z-c)": greedy(torch.cat([z, 1.0 * (z - c_ctx)], -1), support, B),
        }
    cls = np.concatenate([(lambda rh, mf, lb: np.where(rh <= 0.1, 2, lb))(*cell_labels(np.asarray(im))) for im in imgs])
    keep, stats = {}, {}
    for k, o in orders.items():
        kk = np.zeros(N, dtype=bool); kk[o.cpu().numpy()] = True; keep[k] = kk.reshape(C, 16, 16)
        stats[k] = {n: (float(kk[cls == v].mean()) if (cls == v).any() else None) for n, v in (("glass", 0), ("fat", 1), ("tissue", 2))}
    ret[cid] = {"gold": res["gold_answer"], "n_cells": {n: int((cls == v).sum()) for n, v in (("glass", 0), ("fat", 1), ("tissue", 2))}, "keep": stats}
    T, LW = 110, 300
    Wd = Image.new("RGB", (LW + C * (T + 3), 30 + (len(keep) + 1) * (T + 3)), "white"); d = ImageDraw.Draw(Wd)
    d.text((4, 4), f"{cid}  gold={res['gold_answer']}  keep 25% (B={B}/{N})  red = kept   (glass/fat/tissue kept fraction per row)", fill="black")
    for j, im in enumerate(imgs):
        Wd.paste(im.resize((T, T)), (LW + j * (T + 3), 30)); d.text((LW + j * (T + 3) + 2, 32), f"x{mags[j]}", fill="blue")
    for i, (k, kk) in enumerate(keep.items()):
        y = 30 + (i + 1) * (T + 3); s = stats[k]
        f = lambda v: "-" if v is None else f"{v:.2f}"
        d.text((4, y + T // 2 - 12), k, fill="black"); d.text((4, y + T // 2 + 2), f"g {f(s['glass'])} f {f(s['fat'])} t {f(s['tissue'])}", fill="gray")
        for j, im in enumerate(imgs):
            a = np.asarray(im.resize((T, T))).astype(np.float32); a = a * 0.35 + 255 * 0.65
            m = np.kron(kk[j], np.ones((T // 16 + 1, T // 16 + 1)))[:T, :T].astype(bool)
            a[m] = a[m] * 0.25 + np.array([150, 20, 40]) * 0.75
            Wd.paste(Image.fromarray(a.astype(np.uint8)), (LW + j * (T + 3), y)); d.text((LW + j * (T + 3) + 2, y + 2), str(int(kk[j].sum())), fill="black")
    Wd.save(f"{OUTD}/{cid}.png"); print("saved", cid, json.dumps(ret[cid]["keep"]), flush=True)
json.dump(ret, open(f"{OUTD}/retention.json", "w"), indent=1)
print("done")
