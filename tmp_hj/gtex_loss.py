import importlib.util, sys, collections, json, statistics as st
from pathlib import Path
T = Path("/home/users/whddn12316/wsi_latent_0915_decode_hj"); R = T / "sender_relay_exp/runs"
s = importlib.util.spec_from_file_location("rescore", T / "scripts/rescore.py"); rs = importlib.util.module_from_spec(s)
sys.modules["rescore"] = rs; s.loader.exec_module(rs)
ds = "gtex"
def g(roots):
    c, _ = rs.collect(rs.run_dirs(roots, None, []), {ds}); return c.get(ds, {})
recs = rs.load_records(ds); nrm = rs.am.normalize
b = g(sorted(R.glob(f"nav3_prompt1_base_{ds}_gpu*"))); q = g([R / "qasc_nav3" / ds]); n = g([R / "ntrs_nav3" / ds])
ids = sorted(set(b) & set(q) & set(n))
ok = lambda c, i: rs.am.judge(c[i][1], c[i][0], recs[i].get("Choice"), "exact")[0]
lab = lambda c, i: next((ch for ch in (recs[i].get("Choice") or []) if nrm(ch) and nrm(ch) in nrm(c[i][1])), nrm(c[i][1])[:12])
print("n", len(ids))
gold = collections.Counter(nrm(b[i][0]) for i in ids); print("gold", dict(gold))
for name, c in (("base", b), ("qasc", q), ("ntrs", n)):
    pred = collections.Counter(nrm(lab(c, i)) for i in ids)
    rec = {k: sum(ok(c, i) for i in ids if nrm(c[i][0]) == k) for k in gold}
    print(f"{name:5s} pred {dict(pred)}\n      recall {rec}")
chg = [i for i in ids if nrm(n[i][1]) != nrm(b[i][1])]
tr = collections.Counter((nrm(lab(b, i)), nrm(lab(n, i)), "O->X" if ok(b, i) and not ok(n, i) else "X->O" if ok(n, i) and not ok(b, i) else "X->X") for i in chg)
print("NTRS changes vs base:", tr.most_common())
trq = collections.Counter((nrm(lab(b, i)), nrm(lab(q, i))) for i in ids if nrm(q[i][1]) != nrm(b[i][1]))
print("QASC changes vs base:", trq.most_common())
# per-question Navigator patches identical? tissue per crop? use case dirs: count crops / reasoner tokens not needed
print("--- non-choice answers")
for name, c in (("base", b), ("qasc", q), ("ntrs", n)):
    bad = [i for i in ids if not any(nrm(ch) and nrm(ch) in nrm(c[i][1]) for ch in (recs[i].get("Choice") or []))]
    print(name, len(bad), [(i, c[i][0], c[i][1][:60]) for i in bad])
