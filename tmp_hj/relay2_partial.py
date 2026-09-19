import importlib.util, sys, collections, glob
from pathlib import Path
T = Path("/home/users/whddn12316/wsi_latent_0915_decode_hj"); R = T / "sender_relay_exp/runs"
s = importlib.util.spec_from_file_location("rescore", T / "scripts/rescore.py"); rs = importlib.util.module_from_spec(s)
sys.modules["rescore"] = rs; s.loader.exec_module(rs)
ds = "gtex"; recs = rs.load_records(ds); nrm = rs.am.normalize
def g(roots):
    c, _ = rs.collect(rs.run_dirs(roots, None, []), {ds}); return c.get(ds, {})
arm = g([R / "relay2_ntrs50_nav3/gtex"]); n50 = g([R / "ntrs50_nav3/gtex"]); b = g(sorted(R.glob("nav3_prompt1_base_gtex_gpu*"))); both = g(sorted(R.glob("nav3_prompt1_both_gtex_gpu*")))
ids = sorted(set(arm) & set(n50) & set(b) & set(both))
ok = lambda c, i: rs.am.judge(c[i][1], c[i][0], recs[i].get("Choice"), "exact")[0]
for name, c in (("base", b), ("NTRS50", n50), ("Both(PruneB+Relay2)", both), ("NTRS50+Relay2", arm)):
    r = rs.score_dataset(ds, {i: c[i] for i in ids}, recs)["exact"]
    print(f"{name:22s} {r['score']} ({r['correct']}/{len(ids)})")
gain = [nrm(arm[i][0]) for i in ids if ok(arm, i) and not ok(n50, i)]; loss = [nrm(arm[i][0]) for i in ids if not ok(arm, i) and ok(n50, i)]
print("vs NTRS50: same", sum(nrm(arm[i][1]) == nrm(n50[i][1]) for i in ids), "gain", collections.Counter(gain), "loss", collections.Counter(loss))
print("pred NTRS50+Relay2:", collections.Counter(nrm(arm[i][1])[:12] for i in ids).most_common(5))
