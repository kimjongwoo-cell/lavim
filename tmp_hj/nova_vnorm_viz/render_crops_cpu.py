"""CPU only: 런 크롭 박스를 render_box(max_side=512)로 다시 읽어 PNG로 저장."""
import glob, json, os, sys, types
ROOT = "/home/users/whddn12316/wsi_latent_0915_decode_hj"; sys.path.insert(0, ROOT)
import openslide
from pathlib import Path
from vision_text_mas.navigation_render import render_box
from vision_text_mas.dicom_slide import DicomSlide
DATA = "/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda"
RUNS = {"gtex": f"{ROOT}/sender_relay_exp/runs/nav4/nova", "tcga_expert_vqa": f"{ROOT}/sender_relay_exp/runs/nav4_rho_latplan/nova_rho_latplan",
        "tcga_slidebench": f"{ROOT}/sender_relay_exp/runs/nav4_rho_latplan/nova_rho_latplan"}
WANT = {"gtex": ["gtex__GTEX-1F75I-1026", "gtex__GTEX-18D9A-2026"], "tcga_expert_vqa": ["tcga_expert_vqa__8"], "tcga_slidebench": ["tcga_slidebench__2339"]}
out = sys.argv[1]; os.makedirs(out, exist_ok=True); meta = {}
for ds, ids in WANT.items():
    seen = {}
    for d in sorted(glob.glob(f"{RUNS[ds]}/{ds}/gpu*_attempt_*/[0-9]*_{ds}__*")):
        if os.path.exists(f"{d}/result.json"): seen.setdefault(os.path.basename(d).split("_", 1)[1], d)
    for cid in ids:
        cdir = seen[cid]; res = json.load(open(f"{cdir}/result.json")); ev = json.load(open(f"{cdir}/round_1/evidence.json"))
        sp = os.path.realpath(f"{DATA}/slides/{res['slide_id']}")
        slide = DicomSlide(Path(sp)) if os.path.isdir(sp) else openslide.OpenSlide(sp)
        files = []
        for j, p in enumerate(ev["patches"]):
            im = render_box(slide, types.SimpleNamespace(**p["box"]), max_side=512); fn = f"{cid}_{j:02d}.png"; im.save(f"{out}/{fn}"); files.append(fn)
        meta[cid] = {"files": files, "mags": [p["magnification"] for p in ev["patches"]], "gold": res["gold_answer"]}
        print(cid, len(files), flush=True)
json.dump(meta, open(f"{out}/meta.json", "w"), indent=1)
