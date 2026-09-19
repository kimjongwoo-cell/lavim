#!/usr/bin/env python3
"""Sensitivity: canonical acc_of_seq counts ties (gold ratio == another choice's max ratio) as correct,
so degenerate outputs (':', 'CHOICE', 'text', 'Grade') can score. Strict = gold is the unique argmax."""
import difflib
from analysis import *

STRICT_CACHE = {}


def strict_pi(path, ds):
    key = (path, ds)
    if key in STRICT_CACHE:
        return STRICT_CACHE[key]
    p = get(path, ds)
    if p is None:
        return None
    out = {}
    for i, (pe, ge, c) in p.items():
        if not c:
            out[i] = (pe, ge, 0); continue
        ch = DS[ds][i]["Choice"]
        g = difflib.SequenceMatcher(None, pe, ge).quick_ratio()
        ties = sum(1 for x in ch if difflib.SequenceMatcher(None, pe, str(x)).quick_ratio() == g)
        out[i] = (pe, ge, 1 if ties <= 1 else 0)
    STRICT_CACHE[key] = out
    return out


def run(gname, groups):
    print(f"\n######## {gname} (strict)")
    for ds, g in groups.items():
        bpi = strict_pi(g["base"], ds); bidx = set(bpi)
        bc = get(g["base"], ds)
        tie_base = sum(1 for i in bidx if bc[i][2] and not bpi[i][2])
        arms = []
        for a in g["core"] + g["ext"]:
            p = strict_pi(a, ds)
            if p is None or len(set(p) & bidx) / len(bidx) < 0.9:
                continue
            if CONTROL_PAT.search(os.path.basename(a)):
                continue
            arms.append(a)
        core = [a for a in arms if a in g["core"]]
        def uni(al):
            pis = [bpi] + [strict_pi(a, ds) for a in al]
            anyc = {i: int(any(p.get(i, (None, None, 0))[2] for p in pis)) for i in bidx}
            return sum(anyc.values()), score_items(ds, {i: ("x", bpi[i][1], anyc[i]) for i in bidx})[0]
        bs = score_items(ds, bpi)[0]
        best = max(((score_items(ds, strict_pi(a, ds))[0], a) for a in arms), default=(None, None))
        kc, sc = uni(core); ka, sa = uni(arms)
        print(f"== {SHORT[ds]} base strict {bs:.2f} (canonical {score_items(ds, bc)[0]:.2f}; tie-correct in base {tie_base}) "
              f"| best noctl strict {best[0]:.2f} {os.path.basename(best[1]) if best[1] else ''} "
              f"| oracle core strict {kc}/{len(bidx)} ({sc:.2f}) | oracle ext-noctl strict {ka}/{len(bidx)} ({sa:.2f}) runs={len(arms)+1} never {len(bidx)-ka}")


if __name__ == "__main__":
    run("G1", G1())
    run("G2", G2())
