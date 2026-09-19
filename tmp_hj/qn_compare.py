import importlib.util, sys
from pathlib import Path
T = Path("/home/users/whddn12316/wsi_latent_0915_decode_hj"); R = T / "sender_relay_exp/runs"
s = importlib.util.spec_from_file_location("rescore", T / "scripts/rescore.py"); rs = importlib.util.module_from_spec(s)
sys.modules["rescore"] = rs; s.loader.exec_module(rs)
def g(roots, ds):
    c, _ = rs.collect(rs.run_dirs(roots, None, []), {ds}); return c.get(ds, {})
for ds in ["tcga_expert_vqa", "tcga_slidebench", "gtex"]:
    recs = rs.load_records(ds)
    b = g(sorted(R.glob(f"nav3_prompt1_base_{ds}_gpu*")), ds); q = g([R / "qasc_nav3" / ds], ds); n = g([R / "ntrs_nav3" / ds], ds)
    ids = sorted(set(b) & set(q) & set(n))
    ok = lambda c, i: rs.am.judge(c[i][1], c[i][0], recs[i].get("Choice"), "exact")[0]
    nrm = rs.am.normalize
    same_qn = sum(nrm(q[i][1]) == nrm(n[i][1]) for i in ids)
    q_only = [i for i in ids if ok(q, i) and not ok(n, i)]; n_only = [i for i in ids if ok(n, i) and not ok(q, i)]
    qchg = [i for i in ids if nrm(q[i][1]) != nrm(b[i][1])]; nchg = [i for i in ids if nrm(n[i][1]) != nrm(b[i][1])]
    both = set(qchg) & set(nchg)
    both_same = sum(1 for i in both if nrm(q[i][1]) == nrm(n[i][1]))
    print(f"{ds} n={len(ids)} QASC==NTRS {same_qn} | changed vs base: QASC {len(qchg)} NTRS {len(nchg)} both {len(both)} (same new answer {both_same})")
    for tag, L in (("QASC O / NTRS X", q_only), ("NTRS O / QASC X", n_only)):
        print(f"  {tag} {len(L)}: " + "; ".join(
            f"#{i} gold={q[i][0][:26]!r} base={'O' if ok(b, i) else 'X'} q={q[i][1][:20]!r} n={n[i][1][:20]!r}" for i in L))
