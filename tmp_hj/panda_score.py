import importlib.util, sys, collections
from pathlib import Path
T = Path("/home/users/whddn12316/wsi_latent_0915_decode_hj"); R = T / "sender_relay_exp/runs"
s = importlib.util.spec_from_file_location("rescore", T / "scripts/rescore.py"); rs = importlib.util.module_from_spec(s)
sys.modules["rescore"] = rs; s.loader.exec_module(rs)
ds = "panda"; recs = rs.load_records(ds); nrm = rs.am.normalize
def g(roots):
    c, _ = rs.collect(rs.run_dirs(roots, None, []), {ds}); return c.get(ds, {})
old = g(sorted(R.glob("nav3_prompt1_base_panda_gpu*"))); cb = g([R / "nav3clamp_base/panda"]); q = g([R / "qasc_nav3/panda"]); n = g([R / "ntrs_nav3/panda"])
FAILED = [9, 11, 46, 55, 73, 89, 109, 115, 122, 172, 184]
def sc(c, ids):
    r = rs.score_dataset(ds, {i: c[i] for i in ids}, recs); return f"{r['exact']['score']} ({r['exact']['correct']}/{r['n']})"
print("n: old base", len(old), "clamp base", len(cb), "qasc", len(q), "ntrs", len(n))
print("NTRS all 196:", sc(n, sorted(n)), "| pred dist", dict(collections.Counter(nrm(n[i][1]) for i in n)))
com = sorted(set(n) & set(old))
print("vs old base (185 common): base", sc(old, com), "ntrs", sc(n, com), "qasc", sc(q, sorted(set(com) & set(q))))
print("same answer ntrs vs old base:", sum(nrm(n[i][1]) == nrm(old[i][1]) for i in com), "/", len(com))
f = [i for i in FAILED if i in n]
print("11 former Navigator-failure slides NTRS:", [(i, n[i][0], n[i][1]) for i in f])
if cb:
    cc = sorted(set(n) & set(cb))
    print(f"vs clamp base ({len(cc)} common so far): base", sc(cb, cc), "ntrs", sc(n, cc),
          "| clamp base == old base answers on overlap:", sum(nrm(cb[i][1]) == nrm(old[i][1]) for i in set(cb) & set(old)), "/", len(set(cb) & set(old)),
          "| clamp base pred", dict(collections.Counter(nrm(cb[i][1]) for i in cb)))
