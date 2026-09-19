import sys, importlib.util
from pathlib import Path
T = Path("/home/users/whddn12316/wsi_latent_0915_decode_hj")
def load(n, p):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); sys.modules[n] = m; s.loader.exec_module(m); return m
rs = load("rescore", T / "scripts/rescore.py")
dr = load("dashboard_rows", T / "scripts/dashboard_rows.py")
R = T / "sender_relay_exp/runs"
def cases_of(root, ds):
    dirs = rs.run_dirs([root], None, [])
    c, t = rs.collect(dirs, {ds}); return c.get(ds, {})
def exact(ds, cases):
    r = rs.score_dataset(ds, cases, []); return r["n"], r["exact"]["correct"], r["exact"]["score"]
def old(ds, root):
    s = dr.score(ds, dr.collect_latent(root))
    return round(100 * (s["bacc"] if ds in dr.BALANCED else s["micro"]), 2)
print("== Q-PRIS 전체 5셋 (eager, budget8, mlen8192, step10, B=512)")
BASE = {"tcga_expert_vqa": "expq/strat_evqa10", "gtex": "expq/geom_audit_gtex", "tcga": "expq/tcga_base10",
        "tcga_slidebench": "expq/sb_base10", "panda": "expq/panda_base10"}
for ds, b in BASE.items():
    broot, qroot = R / b, R / f"qpris_full/qt_{ds}"
    bc, qc = cases_of(broot, ds), cases_of(qroot, ds)
    common = sorted(set(bc) & set(qc))
    be = exact(ds, {i: bc[i] for i in common}); qe = exact(ds, {i: qc[i] for i in common})
    flips_gain = sum(1 for i in common if rs.am.judge(qc[i][1], qc[i][0], None)[0] and not rs.am.judge(bc[i][1], bc[i][0], None)[0])
    flips_loss = sum(1 for i in common if not rs.am.judge(qc[i][1], qc[i][0], None)[0] and rs.am.judge(bc[i][1], bc[i][0], None)[0])
    print(f"{ds:16s} n={len(common)} (base {len(bc)}, q {len(qc)}) | old base {old(ds, broot)} q {old(ds, qroot)} "
          f"| exact base {be[2]} ({be[1]}) q {qe[2]} ({qe[1]}) d={qe[2]-be[2]:+.2f} gain/loss {flips_gain}/{flips_loss}")
print("== Q-PRIS 20케이스 3 arm (dcs20)")
for ds in BASE:
    row = []
    for arm in ("qn", "qt", "qs"):
        n, k, s = exact(ds, cases_of(R / f"qpris/{arm}_{ds}", ds)); row.append(f"{arm} {s} ({k}/{n})")
    print(f"{ds:16s} " + " | ".join(row))
for title, root, arms in (("== CSIS sweep budget8 트리 (gtex dcs20, B=512)", "csis/sweep", ("base", "pcris_v2", "csis_256", "csis_1024", "csis_4096")),
                          ("== CSIS sweep nav2v2 세팅 (gtex dcs20, B=768)", "nav2v2_mine/sweep", ("base", "csis_256", "csis_1024", "csis_4096"))):
    print(title)
    base = cases_of(R / f"{root}/base", "gtex")
    for arm in arms:
        c = cases_of(R / f"{root}/{arm}", "gtex")
        n, k, s = exact("gtex", c)
        same = sum(1 for i in c if i in base and rs.am.normalize(c[i][1]) == rs.am.normalize(base[i][1]))
        print(f"   {arm:10s} BAcc {s} ({k}/{n}) 답이 base와 같음 {same}/{n}")
