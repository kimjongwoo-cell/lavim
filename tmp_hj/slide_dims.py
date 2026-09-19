import json, os, glob, sys
DATA = "/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda"
try:
    import openslide
    op = lambda p: openslide.OpenSlide(p).dimensions
except Exception:
    import tiffslide
    op = lambda p: tiffslide.TiffSlide(p).dimensions
recs0 = json.load(open(f"{DATA}/gtex.json"))
print("record keys:", list(recs0[0].keys())[:12])
for ds in ["tcga_expert_vqa", "tcga_slidebench", "gtex", "tcga", "panda"]:
    recs = json.load(open(f"{DATA}/{ds}.json"))
    small, err, seen = [], 0, {}
    for i, r in enumerate(recs):
        sid = r["Id"]
        if sid not in seen:
            path = f"{DATA}/slides/{sid}"
            if os.path.isdir(path):
                files = [f for f in glob.glob(f"{path}/*") if not f.endswith((".json", ".txt", ".png", ".jpg"))]
                path = files[0] if files else path
            try:
                seen[sid] = op(os.path.realpath(path))
            except Exception as e:
                seen[sid] = None
        d = seen[sid]
        if d is None:
            err += 1
        elif min(d) < 4096:
            small.append((i, d))
    print(f"{ds:16s} questions={len(recs)} slides={len(seen)} unreadable_q={err} short_side<4096 questions={len(small)} {small[:12]}")
