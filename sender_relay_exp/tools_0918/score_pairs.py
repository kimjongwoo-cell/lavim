import importlib.util, json, os, sys
from pathlib import Path
T = Path("/home/users/whddn12316/wsi_latent_0915_decode_hj"); R = T / "sender_relay_exp/runs"
s = importlib.util.spec_from_file_location("rescore", T / "scripts/rescore.py"); rs = importlib.util.module_from_spec(s)
sys.modules["rescore"] = rs; s.loader.exec_module(rs)
nrm = rs.am.normalize
ARMS = {
    "latplan":          R / "nav4/latplan",
    "base":             R / "nav4/base",
    "nova":             R / "nav4/nova",
    "nova_rho":         R / "nav4_rho/nova_rho",
    "nova_rho_latplan": R / "nav4_rho_latplan/nova_rho_latplan",
    "glvr_rho":         R / "nav4/glvr_rho",
    "glvr_rho_nf":      R / "nav4/glvr_rho_nf",
}
ARMS["glvr_rho_fast"] = R / "nav4/glvr_rho_fast"
PAIRS = [tuple(x.split(":")) for x in os.environ.get("PAIRS", "nova_rho_latplan:latplan").split(",")]
if os.environ.get("ARM_ROOT"):                     # every run named in PAIRS lives in ARM_ROOT/<name>
    ARMS = {a: Path(os.environ["ARM_ROOT"]) / a for p in PAIRS for a in p}
DSS = [d for d in os.environ.get("DSS", "tcga_expert_vqa,tcga_slidebench").split(",") if d]
def g(root, ds):
    c, _ = rs.collect(rs.run_dirs([root / ds], None, []), {ds}); return c.get(ds, {})
for ds in DSS:
    recs = rs.load_records(ds)
    cache = {a: g(ARMS[a], ds) for a in sorted({a for p in PAIRS for a in p})}
    ok = lambda c, i: rs.am.judge(c[i][1], c[i][0], recs[i].get("Choice"), "exact")[0]
    sc = lambda c, I: rs.score_dataset(ds, {i: c[i] for i in I}, recs)["exact"]
    for a, b in PAIRS:
        A, B = cache[a], cache[b]; ids = sorted(set(A) & set(B))
        if not ids: continue
        gain = [i for i in ids if ok(A, i) and not ok(B, i)]; loss = [i for i in ids if not ok(A, i) and ok(B, i)]
        same = sum(nrm(A[i][1]) == nrm(B[i][1]) for i in ids)
        print(json.dumps({"ds": ds, "arm_name": a, "ref_name": b, "n_common": len(ids),
                          "n_arm": len(A), "n_ref": len(B), "arm": sc(A, ids), "ref": sc(B, ids),
                          "gain": len(gain), "loss": len(loss), "same": same}, ensure_ascii=False))
