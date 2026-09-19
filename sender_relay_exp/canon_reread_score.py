import json, glob, sys, importlib.util, re
from pathlib import Path
T = "/home/users/whddn12316/wsi_latent_0915_decode_hj/"
s = importlib.util.spec_from_file_location("rescore", T + "scripts/rescore.py"); rs = importlib.util.module_from_spec(s); sys.modules["rescore"] = rs; s.loader.exec_module(rs)
am = rs.am
R = T + "sender_relay_exp/runs/"
def cases(roots, ds):
    c, t = rs.collect(rs.run_dirs([Path(r) for r in roots], None, []), {ds})
    return c.get(ds, {}), t.get(ds)
SIZES = {"tcga_expert_vqa": 128, "gtex": 190, "tcga_slidebench": 197, "tcga": 221, "panda": 196}
out = {}
for ds in SIZES:
    m, msec = cases(glob.glob(R + f"canon_reread_nav3/{ds}"), ds)
    base_dirs = [d for d in glob.glob(R + f"nav3_prompt1_base_{ds}_*") if Path(d).is_dir()]
    b, bsec = cases(base_dirs, ds)
    com = sorted(set(m) & set(b))
    ok = lambda c, i: am.normalize(c[i][1]) == am.normalize(c[i][0])
    rm = rs.score_dataset(ds, {i: m[i] for i in com}, []) if com else None
    rb = rs.score_dataset(ds, {i: b[i] for i in com}, []) if com else None
    full_m = rs.score_dataset(ds, m, []) if m else None
    rep = sum(ok(m, i) and not ok(b, i) for i in com); brk = sum(ok(b, i) and not ok(m, i) for i in com)
    same = sum(am.normalize(m[i][1]) == am.normalize(b[i][1]) for i in com)
    row = {"n_method": len(m), "size": SIZES[ds], "n_base": len(b), "common": len(com),
           "method_full": None if not full_m else full_m["exact"]["score"],
           "method_common": None if not rm else rm["exact"]["score"], "method_correct": None if not rm else rm["exact"]["correct"],
           "base_common": None if not rb else rb["exact"]["score"], "base_correct": None if not rb else rb["exact"]["correct"],
           "repair": rep, "break": brk, "same": same,
           "sec_method": None if not full_m else full_m["sec"], "base_dirs": [Path(d).name for d in base_dirs]}
    out[ds] = row
    print(ds, json.dumps({k: v for k, v in row.items() if k != "base_dirs"}), "| base dirs:", row["base_dirs"])
json.dump(out, open("/tmp/canon_score_out.json", "w"))
# SlideBench restricted to cases whose navigator patches are identical to nav3 base
import os
def boxes(pat):
    o = {}
    for f in glob.glob(pat):
        d = json.load(open(f)); i = d["dataset_index"]; mt = os.path.getmtime(f)
        k = tuple((p["magnification"], p["box"]["x"], p["box"]["y"]) for p in d["patches"])
        if i not in o or o[i][0] < mt: o[i] = (mt, k)
    return {i: k for i, (mt, k) in o.items()}
ds = "tcga_slidebench"
a = boxes(R + f"canon_reread_nav3/{ds}/gpu*_attempt_*/*/result.json"); b = boxes(R + f"nav3_prompt1_base_{ds}_gpu*/*/result.json")
same_ids = sorted(i for i in set(a) & set(b) if a[i] == b[i])
m, _ = cases(glob.glob(R + f"canon_reread_nav3/{ds}"), ds)
bb, _ = cases([d for d in glob.glob(R + f"nav3_prompt1_base_{ds}_*") if Path(d).is_dir()], ds)
rm = rs.score_dataset(ds, {i: m[i] for i in same_ids}, []); rb = rs.score_dataset(ds, {i: bb[i] for i in same_ids}, [])
ok = lambda c, i: am.normalize(c[i][1]) == am.normalize(c[i][0])
print("SB same-patch", len(same_ids), "method", rm["exact"], "base", rb["exact"], "repair", sum(ok(m,i) and not ok(bb,i) for i in same_ids), "break", sum(ok(bb,i) and not ok(m,i) for i in same_ids))
out["tcga_slidebench"]["same_patch"] = {"n": len(same_ids), "method": rm["exact"], "base": rb["exact"]}
json.dump(out, open("/tmp/canon_score_out.json", "w"))
