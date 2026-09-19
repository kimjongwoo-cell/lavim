import importlib.util, sys, collections
from pathlib import Path
T = Path("/home/users/whddn12316/wsi_latent_0915_decode_hj"); R = T / "sender_relay_exp/runs"
s = importlib.util.spec_from_file_location("rescore", T / "scripts/rescore.py"); rs = importlib.util.module_from_spec(s)
sys.modules["rescore"] = rs; s.loader.exec_module(rs)
ds = "gtex"; recs = rs.load_records(ds); nrm = rs.am.normalize
def g(roots):
    c, _ = rs.collect(rs.run_dirs(roots, None, []), {ds}); return c.get(ds, {})
b = g(sorted(R.glob("nav3_prompt1_base_gtex_gpu*"))); n25 = g([R / "ntrs_nav3/gtex"]); n50 = g([R / "ntrs50_nav3/gtex"])
ids = sorted(set(b) & set(n25) & set(n50))
ok = lambda c, i: rs.am.judge(c[i][1], c[i][0], recs[i].get("Choice"), "exact")[0]
for name, c in (("base", b), ("25%", n25), ("50%", n50)):
    r = rs.score_dataset(ds, {i: c[i] for i in ids}, recs)["exact"]
    rec = collections.Counter(nrm(c[i][0]) for i in ids if ok(c, i))
    print(f"{name}: {r['score']} ({r['correct']}) recall-hits {dict(sorted(rec.items()))}")
same = sum(nrm(n50[i][1]) == nrm(b[i][1]) for i in ids)
gain = [nrm(b[i][0]) for i in ids if ok(n50, i) and not ok(b, i)]; loss = [nrm(b[i][0]) for i in ids if not ok(n50, i) and ok(b, i)]
print(f"50% vs base: same {same}/{len(ids)} gain {len(gain)} {collections.Counter(gain)} loss {len(loss)} {collections.Counter(loss)}")
