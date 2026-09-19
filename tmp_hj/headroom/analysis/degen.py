import difflib, collections
from analysis import *
from analysis3_strict import strict_pi
G = G1()
def find(ds, key):
    c = [n for (n, d) in BYNAME if d == ds and n.endswith(key)]
    return c[0] if c else None
def degen_stats(path, ds):
    a = get(path, ds); s = strict_pi(path, ds)
    n = len(a); deg = 0; tie_ok = 0; exact = 0; examples = collections.Counter()
    for i, (pe, ge, c) in a.items():
        ch = [str(x) for x in DS[ds][i]["Choice"]]
        best = max(difflib.SequenceMatcher(None, pe or "", x).quick_ratio() for x in ch) if pe is not None else 0
        in_choice = pe is not None and pe.strip().lower() in [x.strip().lower() for x in ch]
        if not in_choice: deg += 1; examples[(pe or "None")[:14]] += 1
        if c and not s[i][2]: tie_ok += 1
    return n, deg, tie_ok, examples.most_common(4)
for ds, keys in {"gtex": ["base", "consol_v/c2can_gtex", "expq/gtex10_restage", "expq/gtex10_park", "expq/plipkv_gtex10", "expq/navsearchplip_gtex10"],
                 "tcga_expert_vqa": ["base", "consol_v/c2can_tcga_expert_vqa", "expq/evqa10_restage", "expq/evqa_park", "expq/plipplan1_evqa10", "expq/pfr_evqa10"],
                 "tcga_slidebench": ["base", "consol_v/c2can_tcga_slidebench", "expq/sb_restage10r"],
                 "panda": ["base", "expq/navsearchplip_panda10"], "tcga": ["base", "expq/reasm_tcga10"]}.items():
    print("==", SHORT[ds])
    for k in keys:
        p = G[ds]["base"] if k == "base" else find(ds, k)
        if p is None: print("  missing", k); continue
        n, deg, tie, ex = degen_stats(p, ds)
        print(f"  {k.split('/')[-1]:26s} n={n} not-exact-choice={deg} tie-credited={tie} top-nonchoice={ex}")
