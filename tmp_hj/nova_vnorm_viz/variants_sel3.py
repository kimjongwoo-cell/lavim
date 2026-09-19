"""후보 A: 그래프 평활 LogDet — 이웃을 e(가중치)가 아니라 선택 기하(kernel)에 넣는다.
z~_i = normalize(z_i + lam * sum_{j in N4} w_ij z_j / sum w),  w_ij = relu(cos(z_i,z_j))
선택은 기존 support_logdet_greedy 그대로(e=1), LLM에 가는 feature는 원본 z (선택만 바뀜).
평가 지표: LogDet-only 대비 겹침, 크롭별 분포, glass/fat/tissue 유지율,
coverage = 버려진 토큰이 같은 크롭 남은 토큰과 갖는 최대 cos의 평균(= facility-location 목적값).
라벨은 평가용이며 방법 입력이 아니다."""
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
def vh(mod, args, kw, out):
    hs = kw.get("hidden_states", args[0] if args else None); L = hs.shape[0]
    cap["v6"] = mod.qkv(hs).reshape(L, 3, mod.num_heads, -1)[:, 2].float().norm(dim=-1).mean(1)
vis.blocks[6].attn.register_forward_hook(vh, with_kwargs=True)
mm = lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else torch.ones_like(x)
def gk(s):
    r = max(1, round(4 * s)); ax = torch.arange(-r, r + 1, dtype=torch.float32, device=dev); g = torch.exp(-ax ** 2 / (2 * s * s)); return (g[:, None] * g[None])[None, None]
def conv_mean(x, C, k):
    g = x.view(C, 1, 16, 16); r = k.shape[-1] // 2
    return (F.conv2d(F.pad(g, (r,) * 4), k) / F.conv2d(F.pad(torch.ones_like(g), (r,) * 4), k)).reshape(-1)
def smooth(z, C, lam):
    """4-이웃 cos 가중 평균을 lam만큼 섞고 다시 정규화."""
    d = z.shape[1]; g = z.view(C, 16, 16, d)
    num = torch.zeros_like(g); den = torch.zeros(C, 16, 16, 1, device=dev)
    for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        sh = torch.roll(g, (dy, dx), (1, 2))
        v = torch.ones(16, 16, device=dev)
        if dy == 1: v[0, :] = 0
        if dy == -1: v[-1, :] = 0
        if dx == 1: v[:, 0] = 0
        if dx == -1: v[:, -1] = 0
        w = ((g * sh).sum(-1).clamp_min(0) * v)[..., None]
        num += w * sh; den += w
    nb = num / den.clamp_min(1e-6)
    return F.normalize((g + lam * nb).reshape(-1, d), dim=-1)
RUNS = {"gtex": f"{ROOT}/sender_relay_exp/runs/nav4/nova",
        "tcga_slidebench": f"{ROOT}/sender_relay_exp/runs/nav4_c1int_latplan/nova_vn6g2_latplan"}
PICK = {"gtex": [35, 72, 116, 148, 152, 0, 1, 2, 3, 4], "tcga_slidebench": [45, 49, 149, 180, 181, 196]}
def cases(ds):
    seen = {}
    for d in sorted(glob.glob(f"{RUNS[ds]}/{ds}/gpu*_attempt_*/[0-9]*_{ds}__*")):
        if os.path.exists(f"{d}/result.json"): seen.setdefault(int(os.path.basename(d).split("_")[0]), d)
    return [seen[i] for i in PICK[ds] if i in seen]
out = []
for ds in RUNS:
    for cdir in cases(ds):
        res = json.load(open(f"{cdir}/result.json")); ev = json.load(open(f"{cdir}/round_1/evidence.json"))
        sid = res["slide_id"]; sp = os.path.realpath(DATA + "/slides/" + sid)
        slide = DicomSlide(Path(sp)) if os.path.isdir(sp) else openslide.OpenSlide(sp)
        imgs = [render_box(slide, types.SimpleNamespace(**p["box"]), max_side=512) for p in ev["patches"]]
        C = len(imgs)
        inp = proc.image_processor(images=imgs, return_tensors="pt"); pv, thw = inp["pixel_values"].to(dev), inp["image_grid_thw"].to(dev)
        with torch.no_grad():
            vo = vis(pv.to(torch.bfloat16), grid_thw=thw)
            Fm = vo.pooler_output; Fm = (torch.cat(list(Fm), 0) if isinstance(Fm, (list, tuple)) else Fm).float()
            N = Fm.shape[0]; z = F.normalize(Fm, dim=-1)
            v6 = cap["v6"].view(N, 4).mean(1)
            with contextlib.redirect_stdout(io.StringIO()):
                rho_e = RHOM.evidence_rho(pv.float(), thw, {"tau_p": 215.0, "tau_c": 0.9, "sigma": 2.0})["e"].to(dev).float()
            vn = mm(v6); vng2 = 1 - conv_mean(1 - vn, C, gk(2.))
            B = int(round(0.25 * N)); support = torch.arange(C, device=dev).repeat_interleave(256)
            def sel(q):
                o, _, _ = SEC.support_logdet_greedy(q, support, B)
                kk = torch.zeros(N, dtype=torch.bool, device=dev); kk[o.to(dev)] = True; return kk
            keep = {"LogDet-only": sel(z), "NOVA-rho": sel(z * rho_e.clamp_min(0).sqrt()[:, None]),
                    "vn6g2 (세팅1)": sel(z * vng2.clamp_min(0).sqrt()[:, None])}
            for lam in (0.25, 0.5, 1.0, 2.0):
                keep[f"A lam={lam}"] = sel(smooth(z, C, lam))
            keep["A lam=0.5 + vn6g2"] = sel(smooth(z, C, 0.5) * vng2.clamp_min(0).sqrt()[:, None])
            cov = {}
            for k, kk in keep.items():
                tot = 0.0; cnt = 0
                for g in range(C):
                    zi = z[g * 256:(g + 1) * 256]; m = kk[g * 256:(g + 1) * 256]
                    if m.sum() == 0: continue
                    s = (zi[~m] @ zi[m].T).max(1).values
                    tot += float(s.sum()); cnt += int((~m).sum())
                cov[k] = tot / max(cnt, 1)
        cls = np.concatenate([(lambda rh, mf, lb: np.where(rh <= 0.1, 2, lb))(*cell_labels(np.asarray(im))) for im in imgs])
        tk = {k: v.cpu().numpy() for k, v in keep.items()}
        J = lambda a, b: float((tk[a] & tk[b]).sum() / (tk[a] | tk[b]).sum())
        rec = {"ds": ds, "id": os.path.basename(cdir).split("_", 1)[1], "C": C, "arms": {}}
        for k in tk:
            per = [int(tk[k][g * 256:(g + 1) * 256].sum()) for g in range(C)]
            rec["arms"][k] = {"min": min(per), "sd": float(np.std(per)), "cov": round(cov[k], 4),
                              "J_logdet": J(k, "LogDet-only"), "J_rho": J(k, "NOVA-rho"), "J_vn": J(k, "vn6g2 (세팅1)"),
                              "keep_glass": float(tk[k][cls == 0].mean()) if (cls == 0).any() else None,
                              "keep_fat": float(tk[k][cls == 1].mean()) if (cls == 1).any() else None,
                              "keep_tissue": float(tk[k][cls == 2].mean()) if (cls == 2).any() else None}
        out.append(rec); print(ds, rec["id"], {k: rec["arms"][k]["cov"] for k in list(tk)[:3]}, flush=True)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "variants_sel3.json"), "w"), indent=1)
