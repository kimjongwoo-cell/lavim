"""Win-set overlap among top arms (G1, 4B s10 pb8 eager) + strict scores + where gains land (class)."""
import itertools, collections
from analysis import *
from analysis3_strict import strict_pi
G = G1()
PICK = {
 "gtex": ["consol_v/c2can_gtex", "expq/gtex10_restage", "expq/gtex10_park", "expq/plipkv_gtex10", "c2_replication/gtex/4b/D_cross_step10", "expq/navsearchplip_gtex10", "expq/stoch03_gtex10", "expq/pfr_gtex10", "expq/a10_gtex10"],
 "tcga_expert_vqa": ["consol_v/c2can_tcga_expert_vqa", "expq/evqa10_restage", "expq/evqa_park", "expq/plipplan1_evqa10", "expq/x20plip_evqa10", "expq/pfr_evqa10", "expq/k75_evqa10", "expq/a10_evqa10"],
 "tcga_slidebench": ["consol_v/c2can_tcga_slidebench", "consol_v/scale512_tcga_slidebench", "expq/sb_restage10r", "expq/x20plip_sb10"],
}
def find(ds, key):
    c = [n for (n, d) in BYNAME if d == ds and n.endswith(key)]
    return c[0] if c else None
for ds, keys in PICK.items():
    base = G[ds]["base"]; bpi = get(base, ds); bs = strict_pi(base, ds); idx = set(bpi)
    print(f"\n== {SHORT[ds]} base {score_items(ds, bpi)[0]:.2f} strict {score_items(ds, bs)[0]:.2f} n={len(idx)}")
    W, Lset = {}, {}
    for k in keys:
        p = find(ds, k)
        if p is None: print("  missing", k); continue
        a = get(p, ds); s = strict_pi(p, ds)
        cov = len(set(a) & idx)
        if cov / len(idx) < 0.9: print("  low cov", k, cov); continue
        w = {i for i in idx if i in a and a[i][2] and not bpi[i][2]}
        l = {i for i in idx if i in a and not a[i][2] and bpi[i][2]}
        ws = {i for i in idx if i in s and s[i][2] and not bs[i][2]}
        ls = {i for i in idx if i in s and not s[i][2] and bs[i][2]}
        W[k], Lset[k] = w, l
        cls = collections.Counter(bpi[i][1] for i in w)
        print(f"  {k.split('/')[-1]:28s} {score_items(ds, a)[0]:6.2f} (Δ{score_items(ds, a)[0]-score_items(ds, bpi)[0]:+5.2f}) W{len(w)}/L{len(l)} | strict {score_items(ds, s)[0]:6.2f} (Δ{score_items(ds, s)[0]-score_items(ds, bs)[0]:+5.2f}) W{len(ws)}/L{len(ls)}"
              + (f" | W classes {dict(cls.most_common(5))}" if ds == 'gtex' else ""))
    ks = list(W)
    print("  pairwise |W∩W'| / |W| , |L∩L'| :")
    for a, b in itertools.combinations(ks, 2):
        print(f"    {a.split('/')[-1][:18]:18s} x {b.split('/')[-1][:18]:18s} W∩ {len(W[a]&W[b]):2d} (|W| {len(W[a])},{len(W[b])})  L∩ {len(Lset[a]&Lset[b]):2d} (|L| {len(Lset[a])},{len(Lset[b])})")
    # how often a base-wrong question is fixed by >=2 of the picked arms
    cnt = collections.Counter(i for k in ks for i in W[k])
    print("  base-wrong fixed by k picked arms:", dict(sorted(collections.Counter(cnt.values()).items())))
