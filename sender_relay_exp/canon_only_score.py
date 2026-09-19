"""nav3 base / nav3+Canonical only / nav3+Canonical+ReRead, same dataset_index, exact scoring.

usage: canon_only_score.py [ds ...]   (default: every dataset with canon_only results)
"""
import glob
import importlib.util
import json
import os
import sys
from pathlib import Path

T = "/home/users/whddn12316/wsi_latent_0915_decode_hj/"
s = importlib.util.spec_from_file_location("rescore", T + "scripts/rescore.py")
rs = importlib.util.module_from_spec(s); sys.modules["rescore"] = rs; s.loader.exec_module(rs)
am = rs.am
R = T + "sender_relay_exp/runs/"
SIZES = {"tcga_expert_vqa": 128, "gtex": 190, "tcga_slidebench": 197, "tcga": 221, "panda": 196}


def cases(roots, ds):
    c, _ = rs.collect(rs.run_dirs([Path(r) for r in roots], None, []), {ds})
    return c.get(ds, {})


def boxes(root):
    o = {}
    for f in glob.glob(root + "/**/result.json", recursive=True):
        d = json.load(open(f)); i = d["dataset_index"]; mt = os.path.getmtime(f)
        k = tuple((p["magnification"], p["box"]["x"], p["box"]["y"]) for p in d["patches"])
        if i not in o or o[i][0] < mt:
            o[i] = (mt, k)
    return {i: k for i, (mt, k) in o.items()}


def score(ds, c, ids):
    r = rs.score_dataset(ds, {i: c[i] for i in ids}, []) if ids else None
    return (None, None, None) if not r else (r["exact"]["score"], r["exact"]["correct"], r["sec"])


ok = lambda c, i: am.normalize(c[i][1]) == am.normalize(c[i][0])
out = {}
for ds in (sys.argv[1:] or SIZES):
    only_root = R + f"canon_only_nav3/{ds}"
    if not glob.glob(only_root + "/**/result.json", recursive=True):
        continue
    o = cases([only_root], ds)
    rr = cases([R + f"canon_reread_nav3/{ds}"], ds)
    b = cases([d for d in glob.glob(R + f"nav3_prompt1_base_{ds}_*") if Path(d).is_dir()], ds)
    com = sorted(set(o) & set(b))
    com3 = sorted(set(o) & set(b) & set(rr))
    bo = boxes(only_root)
    bb = {}
    for d in glob.glob(R + f"nav3_prompt1_base_{ds}_*"):
        if Path(d).is_dir():
            bb.update(boxes(d))
    same_patch = [i for i in com if i in bo and i in bb and bo[i] == bb[i]]
    so, ko, seco = score(ds, o, com)
    sb, kb, _ = score(ds, b, com)
    sr, kr, _ = score(ds, rr, com3)
    row = {"n_only": len(o), "size": SIZES[ds], "common": len(com), "common3": len(com3),
           "only": so, "only_k": ko, "only_sec": seco, "base": sb, "base_k": kb, "reread": sr, "reread_k": kr,
           "only_vs_base_same": sum(am.normalize(o[i][1]) == am.normalize(b[i][1]) for i in com),
           "only_vs_base_gain": sum(ok(o, i) and not ok(b, i) for i in com),
           "only_vs_base_loss": sum(ok(b, i) and not ok(o, i) for i in com),
           "reread_vs_only_same": sum(am.normalize(o[i][1]) == am.normalize(rr[i][1]) for i in com3),
           "reread_vs_only_gain": sum(ok(rr, i) and not ok(o, i) for i in com3),
           "reread_vs_only_loss": sum(ok(o, i) and not ok(rr, i) for i in com3),
           "same_patch": len(same_patch)}
    if len(same_patch) < len(com):
        s2, k2, _ = score(ds, o, same_patch); b2, kb2, _ = score(ds, b, same_patch)
        row.update(samepatch_only=s2, samepatch_only_k=k2, samepatch_base=b2, samepatch_base_k=kb2)
    out[ds] = row
    print(ds, json.dumps(row))
json.dump(out, open("/tmp/canon_only_score_out.json", "w"), indent=1)
