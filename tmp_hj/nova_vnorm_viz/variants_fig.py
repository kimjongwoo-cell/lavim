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

from PIL import Image, ImageDraw
OUTD=os.path.join(os.path.dirname(__file__),"variants_fig"); os.makedirs(OUTD,exist_ok=True)
seen={}
for d in sorted(glob.glob(f"{RUNS['gtex']}/gtex/gpu*_attempt_*/[0-9]*_gtex__*")):
    if os.path.exists(f"{d}/result.json"): seen.setdefault(os.path.basename(d).split("_",1)[1], d)
for cid in ["gtex__GTEX-1F75I-1026","gtex__GTEX-18D9A-2026"]:
    cdir=seen[cid]
    res=json.load(open(f"{cdir}/result.json")); ev=json.load(open(f"{cdir}/round_1/evidence.json"))
    sp=os.path.realpath(f"{DATA}/slides/{res['slide_id']}")
    slide=DicomSlide(Path(sp)) if os.path.isdir(sp) else openslide.OpenSlide(sp)
    imgs=[render_box(slide, types.SimpleNamespace(**p["box"]), max_side=512) for p in ev["patches"]]; mags=[p["magnification"] for p in ev["patches"]]
    C=len(imgs)
    inp=proc.image_processor(images=imgs, return_tensors="pt"); pv,thw=inp["pixel_values"].to(dev),inp["image_grid_thw"].to(dev)
    with torch.no_grad():
        vo=vis(pv.to(torch.bfloat16),grid_thw=thw)
        Fm=vo.pooler_output; Fm=(torch.cat(list(Fm),0) if isinstance(Fm,(list,tuple)) else Fm).float()
        N=Fm.shape[0]; z=F.normalize(Fm,dim=-1); r=Fm.norm(dim=-1)
        e2=F.normalize(cap[2].view(N,4,-1).mean(1),dim=-1); epe=F.normalize(cap["pe"].view(N,4,-1).mean(1),dim=-1)
        v6=cap["v6"].view(N,4).mean(1); cfg={"tau_p":215.0,"tau_c":0.9,"sigma":2.0}
        with contextlib.redirect_stdout(io.StringIO()):
            rho_e=RHOM.evidence_rho(pv.float(),thw,cfg)["e"].to(dev).float(); v1_e=SEC.evidence(pv.float(),thw,cfg)["e"].to(dev).float()
        l2=lgv(e2,C)
        W={"NOVA v1 (dashboard nav4)":v1_e,"NOVA rho (dashboard latplan)":rho_e,"LogDet-only":torch.ones(N,device=dev),
           "Norm (minmax)":mm(r),"1-Hop Ctx Norm":mm(conv_mean(r,C,K1)),"Gauss Ctx Norm s1":mm(conv_mean(r,C,gk(1.))),
           "merger-LGV":lgv(z,C),"Early-LGV L2":l2,"Early-LGV patch-embed":lgv(epe,C),"VN6 + G s1 (aux)":conv_mean(mm(v6),C,gk(1.))}
        B=int(round(0.25*N)); support=torch.arange(C,device=dev).repeat_interleave(256); keep={}
        for k,w in W.items():
            o,_,_=SEC.support_logdet_greedy(z*w.clamp_min(0).sqrt()[:,None],support,B)
            kk=torch.zeros(N,dtype=torch.bool,device=dev); kk[o.to(dev)]=True; keep[k]=kk.view(C,16,16).cpu().numpy()
    T=110; LW=190
    Wd=Image.new("RGB",(LW+C*(T+3),30+(len(keep)+1)*(T+3)),"white"); d=ImageDraw.Draw(Wd)
    d.text((4,4),f"{cid}  gold={res['gold_answer']}  keep 25% (B={B}/{N})  red = kept",fill="black")
    for j,im in enumerate(imgs):
        Wd.paste(im.resize((T,T)),(LW+j*(T+3),30)); d.text((LW+j*(T+3)+2,32),f"x{mags[j]}",fill="blue")
    for i,(k,kk) in enumerate(keep.items()):
        y=30+(i+1)*(T+3); d.text((4,y+T//2-6),k,fill="black")
        for j,im in enumerate(imgs):
            a=np.asarray(im.resize((T,T))).astype(np.float32); a=a*0.35+255*0.65
            m=np.kron(kk[j],np.ones((T//16+1,T//16+1)))[:T,:T].astype(bool)
            a[m]=a[m]*0.25+np.array([150,20,40])*0.75
            Wd.paste(Image.fromarray(a.astype(np.uint8)),(LW+j*(T+3),y)); d.text((LW+j*(T+3)+2,y+2),str(int(kk[j].sum())),fill="black")
    Wd.save(f"{OUTD}/{cid}.png"); print("saved",cid,flush=True)
