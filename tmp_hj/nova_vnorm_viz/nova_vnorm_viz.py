"""latplan + NOVA ρ 런(nav4_rho_latplan)의 실제 크롭 12장으로 NOVA 프루닝을 재현하고 e만 바꿔 비교한다.

- 크롭: 런의 round_1/evidence.json 박스 → vision_text_mas.navigation_render.render_box(max_side=512) (런과 같은 경로)
- 선택: memory.secld_select.support_logdet_greedy (NOVA 코드 그대로), q = sqrt(e) z, z = merger 출력 정규화, B = round(0.25 N)
- e 후보
    RHO     memory.nova_rho_select.evidence_rho (현재 런 설정: tau_p 215, sigma 2)
    VN6_AP1 L6 attention value norm(헤드 평균, 2x2 평균) → 문항 전체 min-max → 토큰 attention 1단계 전파(A v) → 문항 min-max
    VN6_G1  L6 value norm → 문항 min-max → in-crop 가우시안 σ1 (NOVA 필드 형태, G*v)
    VN2_AP2 L2 value norm → 문항 min-max → attention 2단계 전파(A A v) → 문항 min-max
출력: <out>/<ds>/<id>.png (행 = 방법, 열 = 크롭 12장, 어둡게 = 버림), <out>/summary.json
"""
import glob, json, math, os, sys, types
import numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageDraw
ROOT = "/home/users/whddn12316/wsi_latent_0915_decode_hj"
sys.path.insert(0, ROOT)
import openslide
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision
from vision_text_mas.navigation_render import render_box
from memory import secld_select as SEC, nova_rho_select as RHOM

RUN = os.environ.get("RUN", f"{ROOT}/sender_relay_exp/runs/nav4_rho_latplan/nova_rho_latplan")
DSETS = os.environ.get("DSETS", "tcga_expert_vqa,tcga_slidebench").split(",")
PICK = [int(x) for x in os.environ.get("PICK", "").split(",") if x]
DATA = "/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda"
OUT = sys.argv[1]; NQ = int(sys.argv[2]) if len(sys.argv) > 2 else 10
dev = "cuda"
model = Qwen3VLForConditionalGeneration.from_pretrained("/home/users/whddn12316/models/Qwen3-VL-4B-Thinking",
                                                        torch_dtype=torch.bfloat16).to(dev).eval()
proc = AutoProcessor.from_pretrained("/home/users/whddn12316/models/Qwen3-VL-4B-Thinking")
vis = model.model.visual
X = {}
def hook(li):
    def h(mod, args, kwargs, out):
        hs = kwargs.get("hidden_states", args[0] if args else None); cu = kwargs["cu_seqlens"]
        L = hs.shape[0]
        q, k, v = mod.qkv(hs).reshape(L, 3, mod.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        q, k = apply_rotary_pos_emb_vision(q, k, *kwargs["position_embeddings"])
        vn, At = [], []
        for s0, s1 in zip(cu[:-1].tolist(), cu[1:].tolist()):
            P = s1 - s0; n = P // 4
            A = torch.softmax(q[s0:s1].float().transpose(0, 1) @ k[s0:s1].float().transpose(0, 1).transpose(1, 2) * mod.scaling, -1).mean(0)
            At.append(A.view(n, 4, n, 4).sum(3).mean(1))                     # 토큰 수준, 행 합 1
            vn.append(v[s0:s1].float().norm(dim=-1).mean(1).view(n, 4).mean(1))
        X[f"vn{li}"] = torch.cat(vn); X[f"A{li}"] = At
    return h
for li in (2, 6): vis.blocks[li].attn.register_forward_hook(hook(li), with_kwargs=True)

def mm(x):
    lo, hi = x.min(), x.max(); return (x - lo) / (hi - lo) if hi > lo else torch.ones_like(x)
def blockprop(A_list, v, steps):
    out, off = [], 0
    for A in A_list:
        n = A.shape[0]; x = v[off:off + n]
        for _ in range(steps): x = A @ x
        out.append(x); off += n
    return torch.cat(out)
def gfield(v, crops, sigma=1.0):
    r = max(1, round(4 * sigma)); ax = torch.arange(-r, r + 1, dtype=torch.float32, device=v.device)
    g = torch.exp(-ax ** 2 / (2 * sigma ** 2)); k = (g[:, None] * g[None])[None, None]
    im = v.view(crops, 1, 16, 16)
    return (F.conv2d(F.pad(im, (r,) * 4), k) / F.conv2d(F.pad(torch.ones_like(im), (r,) * 4), k)).reshape(-1)

def cases(ds):
    seen = {}
    for d in sorted(glob.glob(f"{RUN}/{ds}/gpu*_attempt_*/[0-9]*_{ds}__*")):
        if not os.path.exists(f"{d}/result.json"): continue
        idx = int(os.path.basename(d).split("_")[0]); seen.setdefault(idx, d)
    if PICK: return [seen[i] for i in PICK if i in seen]
    return [seen[i] for i in sorted(seen)[:NQ]]

METHODS = (["V1_RUN"] if os.environ.get("V1") else []) + ["RHO", "VN6_AP1", "VN6_G1", "VN2_AP2"]
summary = []
os.makedirs(OUT, exist_ok=True)
for ds in DSETS:
    os.makedirs(f"{OUT}/{ds}", exist_ok=True)
    for cdir in cases(ds):
        res = json.load(open(f"{cdir}/result.json")); ev = json.load(open(f"{cdir}/round_1/evidence.json"))
        sid = res["slide_id"]
        sp = os.path.realpath(f"{DATA}/slides/{sid}")
        if os.path.isdir(sp):
            from pathlib import Path
            from vision_text_mas.dicom_slide import DicomSlide
            slide = DicomSlide(Path(sp))
        else:
            slide = openslide.OpenSlide(sp)
        imgs, mags = [], []
        for p in ev["patches"]:
            b = p["box"]; imgs.append(render_box(slide, types.SimpleNamespace(**b), max_side=512)); mags.append(p["magnification"])
        inp = proc.image_processor(images=imgs, return_tensors="pt")
        pv, thw = inp["pixel_values"].to(dev), inp["image_grid_thw"].to(dev)
        assert all(int(t[1]) == 32 and int(t[2]) == 32 for t in thw), thw
        with torch.no_grad():
            vo = vis(pv.to(torch.bfloat16), grid_thw=thw)
            Fm = vo.pooler_output; Fm = torch.cat(list(Fm), 0) if isinstance(Fm, (list, tuple)) else Fm
            z = F.normalize(Fm.float(), dim=-1); N = z.shape[0]; C = len(imgs)
            support = torch.arange(C, device=dev).repeat_interleave(256)
            cfg = {"tau_p": 215.0, "tau_c": 0.9, "sigma": 2.0}
            import io, contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                erho = RHOM.evidence_rho(pv.float(), thw, cfg)
            rho = erho["r"].to(dev).float()
            E = {"RHO": erho["e"].to(dev).float(),
                 "VN6_AP1": mm(blockprop(X["A6"], mm(X["vn6"]), 1)),
                 "VN6_G1": gfield(mm(X["vn6"]), C, 1.0),
                 "VN2_AP2": mm(blockprop(X["A2"], mm(X["vn2"]), 2))}
            if "V1_RUN" in METHODS:
                with contextlib.redirect_stdout(io.StringIO()):
                    E["V1_RUN"] = SEC.evidence(pv.float(), thw, cfg)["e"].to(dev).float()
            B = max(1, int(round(0.25 * N)))
            keep = {}
            for m, e in E.items():
                order, _, _ = SEC.support_logdet_greedy(z * e.clamp_min(0).sqrt()[:, None], support, B)
                kk = torch.zeros(N, dtype=torch.bool, device=dev); kk[order.to(dev)] = True; keep[m] = kk
        white = rho >= 0.9
        rec = {"ds": ds, "id": os.path.basename(cdir).split("_", 1)[1], "gold": res["gold_answer"],
               "pred_rho_run": (res.get("answer") or {}).get("answer") if isinstance(res.get("answer"), dict) else res.get("answer"),
               "mags": mags, "white_frac_all": float(white.float().mean())}
        for m in METHODS:
            kk = keep[m]
            rec[m] = {"per_crop": [int(kk[support == g].sum()) for g in range(C)],
                      "white_in_kept": float(white[kk].float().mean()),
                      "white_kept_of_white": float(kk[white].float().mean()) if white.any() else None,
                      "jacc_vs_rho": float((kk & keep["RHO"]).sum() / (kk | keep["RHO"]).sum()),
                      "jacc_vs_v1": float((kk & keep["V1_RUN"]).sum() / (kk | keep["V1_RUN"]).sum()) if "V1_RUN" in keep else None}
        summary.append(rec)
        # 그림: 행 = 방법, 열 = 크롭
        T = 120; W = Image.new("RGB", (90 + C * (T + 4), 36 + len(METHODS) * (T + 4)), "white"); d = ImageDraw.Draw(W)
        d.text((4, 4), f"{rec['id']}  gold={rec['gold']}  white(rho>=0.9)={rec['white_frac_all']:.2f}", fill="black")
        for j in range(C): d.text((90 + j * (T + 4), 20), f"x{mags[j]}", fill="black")
        for i, m in enumerate(METHODS):
            d.text((4, 36 + i * (T + 4) + T // 2), m, fill="black")
            kk = keep[m].view(C, 16, 16).cpu().numpy()
            for j, im in enumerate(imgs):
                a = np.asarray(im.resize((T, T))).astype(np.float32)
                mask = np.kron(kk[j], np.ones((T // 16 + 1, T // 16 + 1)))[:T, :T][..., None]
                a = a * (0.25 + 0.75 * mask)
                W.paste(Image.fromarray(a.astype(np.uint8)), (90 + j * (T + 4), 36 + i * (T + 4)))
                d.text((90 + j * (T + 4) + 2, 36 + i * (T + 4) + 2), str(int(kk[j].sum())), fill=(255, 255, 0))
        W.save(f"{OUT}/{ds}/{rec['id']}.png")
        print(rec["id"], {m: (rec[m]["per_crop"], round(rec[m]["white_in_kept"], 3), round(rec[m]["jacc_vs_rho"], 2)) for m in METHODS}, flush=True)
json.dump(summary, open(f"{OUT}/summary_{'_'.join(DSETS)}.json", "w"), indent=1)
