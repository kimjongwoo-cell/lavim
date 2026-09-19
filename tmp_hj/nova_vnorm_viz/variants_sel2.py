"""C1 salience 변형별 실제 25% 선택 비교 (런 크롭 재현, NOVA support_logdet_greedy 그대로).
평가용 셀 라벨은 fatlabel.cell_labels (방법 입력 아님)."""
import glob, json, os, sys, types, io, contextlib
import numpy as np, torch, torch.nn.functional as F
ROOT = "/home/users/whddn12316/wsi_latent_0915_decode_hj"; sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(__file__))
import openslide
from pathlib import Path
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from vision_text_mas.navigation_render import render_box
from vision_text_mas.dicom_slide import DicomSlide
from memory import secld_select as SEC, nova_rho_select as RHOM
from fatlabel import cell_labels
DATA = "/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda"
MP = "/home/users/whddn12316/models/Qwen3-VL-4B-Thinking"; dev = "cuda"
model = Qwen3VLForConditionalGeneration.from_pretrained(MP, torch_dtype=torch.bfloat16).to(dev).eval()
proc = AutoProcessor.from_pretrained(MP); vis = model.model.visual
cap = {}
vis.blocks[2].register_forward_hook(lambda mo, a, o: cap.__setitem__(2, (o[0] if isinstance(o, tuple) else o).float()))
vis.patch_embed.register_forward_hook(lambda mo, a, o: cap.__setitem__("pe", o.float()))
def vh(mod, args, kw, out):
    hs = kw.get("hidden_states", args[0] if args else None); L = hs.shape[0]
    cap["v6"] = mod.qkv(hs).reshape(L, 3, mod.num_heads, -1)[:, 2].float().norm(dim=-1).mean(1)
vis.blocks[6].attn.register_forward_hook(vh, with_kwargs=True)
def conv_mean(x, C, k):
    g = x.view(C, 1, 16, 16); r = k.shape[-1] // 2
    return (F.conv2d(F.pad(g, (r,) * 4), k) / F.conv2d(F.pad(torch.ones_like(g), (r,) * 4), k)).reshape(-1)
K1 = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=torch.float32, device=dev)[None, None]
def gk(s):
    r = max(1, round(4 * s)); ax = torch.arange(-r, r + 1, dtype=torch.float32, device=dev); g = torch.exp(-ax ** 2 / (2 * s * s)); return (g[:, None] * g[None])[None, None]
def lgv(z, C):
    g = z.view(C, 16, 16, -1); s = torch.zeros(C, 16, 16, device=dev); c = torch.zeros(C, 16, 16, device=dev)
    for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        sh = torch.roll(g, (dy, dx), (1, 2)); v = torch.ones(16, 16, device=dev, dtype=torch.bool)
        if dy == 1: v[0, :] = False
        if dy == -1: v[-1, :] = False
        if dx == 1: v[:, 0] = False
        if dx == -1: v[:, -1] = False
        s += ((g - sh) ** 2).sum(-1) * v; c += v
    return (s / c).reshape(-1)
def cosbar(z, C):
    d = z.shape[1]; g = z.view(C, 16, 16, d).permute(0, 3, 1, 2); k = torch.ones(1, 1, 3, 3, device=dev); k[0, 0, 1, 1] = 0
    num = F.conv2d(F.pad(g, (1,) * 4), k.expand(d, 1, 3, 3), groups=d); den = F.conv2d(F.pad(torch.ones(C, 1, 16, 16, device=dev), (1,) * 4), k)
    zb = F.normalize((num / den).permute(0, 2, 3, 1).reshape(-1, d), dim=-1)
    return 1 - (z * zb).sum(-1)
mm = lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else torch.ones_like(x)
RUNS = {"tcga_expert_vqa": f"{ROOT}/sender_relay_exp/runs/nav4_rho_latplan/nova_rho_latplan",
        "tcga_slidebench": f"{ROOT}/sender_relay_exp/runs/nav4_rho_latplan/nova_rho_latplan",
        "gtex": f"{ROOT}/sender_relay_exp/runs/nav4/nova"}
PICK = {"gtex": [35, 72, 116, 148, 152, 0, 1, 2, 3, 4]}
def cases(ds):
    seen = {}
    for d in sorted(glob.glob(f"{RUNS[ds]}/{ds}/gpu*_attempt_*/[0-9]*_{ds}__*")):
        if os.path.exists(f"{d}/result.json"): seen.setdefault(int(os.path.basename(d).split("_")[0]), d)
    return [seen[i] for i in PICK[ds] if i in seen] if ds in PICK else [seen[i] for i in sorted(seen)[:10]]
out = []
for ds in RUNS:
    for cdir in cases(ds):
        res = json.load(open(f"{cdir}/result.json")); ev = json.load(open(f"{cdir}/round_1/evidence.json"))
        sp = os.path.realpath(f"{DATA}/slides/{res['slide_id']}")
        slide = DicomSlide(Path(sp)) if os.path.isdir(sp) else openslide.OpenSlide(sp)
        imgs = [render_box(slide, types.SimpleNamespace(**p["box"]), max_side=512) for p in ev["patches"]]
        C = len(imgs)
        inp = proc.image_processor(images=imgs, return_tensors="pt"); pv, thw = inp["pixel_values"].to(dev), inp["image_grid_thw"].to(dev)
        with torch.no_grad():
            vo = vis(pv.to(torch.bfloat16), grid_thw=thw)
            Fm = vo.pooler_output; Fm = (torch.cat(list(Fm), 0) if isinstance(Fm, (list, tuple)) else Fm).float()
            N = Fm.shape[0]; z = F.normalize(Fm, dim=-1); r = Fm.norm(dim=-1)
            e2 = F.normalize(cap[2].view(N, 4, -1).mean(1), dim=-1); epe = F.normalize(cap["pe"].view(N, 4, -1).mean(1), dim=-1)
            v6 = cap["v6"].view(N, 4).mean(1)
            with contextlib.redirect_stdout(io.StringIO()):
                rho_e = RHOM.evidence_rho(pv.float(), thw, {"tau_p": 215.0, "tau_c": 0.9, "sigma": 2.0})["e"].to(dev).float()
                v1_e = SEC.evidence(pv.float(), thw, {"tau_p": 215.0, "tau_c": 0.9, "sigma": 2.0})["e"].to(dev).float()
            l2 = lgv(e2, C)
            W = {"LogDet-only": torch.ones(N, device=dev), "NOVA-ρ (대시보드)": rho_e, "NOVA v1 (대시보드 nav4)": v1_e,
                 "Norm raw": r, "Norm minmax": mm(r), "1-Hop Ctx Norm minmax": mm(conv_mean(r, C, K1)),
                 "Gauss Ctx Norm σ1 minmax": mm(conv_mean(r, C, gk(1.))), "Gauss Ctx Norm σ2 minmax": mm(conv_mean(r, C, gk(2.))),
                 "merger-LGV raw": lgv(z, C), "local-cos raw": cosbar(z, C),
                 "Early-LGV L2 raw": l2, "Early-LGV L2 /mean": l2 / l2.mean(), "Early-LGV patch-embed raw": lgv(epe, C),
                 "(보조) VN6 + G σ1": conv_mean(mm(v6), C, gk(1.))}
            B = int(round(0.25 * N)); support = torch.arange(C, device=dev).repeat_interleave(256)
            keep = {}
            for k, w in W.items():
                o, _, _ = SEC.support_logdet_greedy(z * w.clamp_min(0).sqrt()[:, None], support, B)
                kk = torch.zeros(N, dtype=torch.bool, device=dev); kk[o.to(dev)] = True; keep[k] = kk
            kk = torch.zeros(N, dtype=torch.bool, device=dev); kk[torch.topk(l2, B).indices] = True; keep["Early-LGV L2 Top-k"] = kk
        cls = np.concatenate([(lambda rh, mf, lb: np.where(rh <= 0.1, 2, lb))(*cell_labels(np.asarray(im))) for im in imgs])
        tk = {k: v.cpu().numpy() for k, v in keep.items()}
        J = lambda a, b: float((tk[a] & tk[b]).sum() / (tk[a] | tk[b]).sum())
        O = lambda a, b: float((tk[a] & tk[b]).sum() / tk[b].sum())
        rec = {"ds": ds, "id": os.path.basename(cdir).split("_", 1)[1], "C": C, "arms": {}}
        for k in tk:
            per = [int(tk[k][g * 256:(g + 1) * 256].sum()) for g in range(C)]
            rec["arms"][k] = {"min": min(per), "sd": float(np.std(per)), "J_rho": J(k, "NOVA-ρ (대시보드)"), "O_rho": O(k, "NOVA-ρ (대시보드)"), "J_v1": J(k, "NOVA v1 (대시보드 nav4)"), "O_v1": O(k, "NOVA v1 (대시보드 nav4)"), "J_logdet": J(k, "LogDet-only"),
                              "keep_glass": float(tk[k][cls == 0].mean()) if (cls == 0).any() else None,
                              "keep_fat": float(tk[k][cls == 1].mean()) if (cls == 1).any() else None,
                              "keep_tissue": float(tk[k][cls == 2].mean()) if (cls == 2).any() else None}
        out.append(rec); print(ds, rec["id"], flush=True)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "variants_sel2.json"), "w"), indent=1)
