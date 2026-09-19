#!/usr/bin/env python3
"""Compact per-dataset summary rows for the report."""
from analysis import *


def agree90(ds, base, arms):
    bpi = get(base, ds); bidx = set(bpi)
    pl = [get(a, ds) for a in arms]
    k = 0
    for i in bidx:
        ans = [p[i][0] for p in pl if i in p]
        if not ans or sum(1 for x in ans if x == bpi[i][0]) / len(ans) >= 0.9:
            k += 1
    return k


def summarize(gname, groups):
    print(f"\n######## {gname}")
    for ds, g in groups.items():
        rc = analyze(ds, g["base"], g["core"], "core")
        ra = analyze(ds, g["base"], g["core"] + g["ext"], "all")
        n = ra["base_n"]
        used_all = [x for x in ra["used"]]
        full = lambda s: s.replace("S:", S).replace("T:", T).replace("R:", R)
        noctl = [full(a) for a in used_all if not CONTROL_PAT.search(os.path.basename(a))]
        coreu = [full(a) for a in rc["used"]]
        rows = [a for a in ra["arms"] if a["cov"] >= 0.9 and not CONTROL_PAT.search(os.path.basename(a["arm"]))]
        best = max(rows, key=lambda a: a["score"]) if rows else None
        crow = [a for a in rc["arms"] if a["cov"] >= 0.9]
        cbest = max(crow, key=lambda a: a["score"]) if crow else None
        print(f"== {SHORT[ds]} base={ra['base_score']:.2f} ({short(g['base'])}) n={n} base_correct={ra['base_correct']}")
        if cbest:
            print(f"   core best: {os.path.basename(cbest['arm'])} {cbest['score']:.2f} Δ{cbest['delta']:+.2f} W{cbest['wins']}/L{cbest['losses']}")
        if best:
            print(f"   ext  best(noctl): {os.path.basename(best['arm'])} {best['score']:.2f} Δ{best['delta']:+.2f} W{best['wins']}/L{best['losses']}")
        print(f"   oracle core ({rc['n_used']} runs): {rc['oracle_k']}/{n} metric {rc['oracle_score']:.2f} | C2-core {rc['c2_arms']} {rc['c2_oracle_k']}/{n} {rc['c2_oracle_score']:.2f}")
        print(f"   oracle ext noctl ({ra['noctl_n']} runs): {ra['noctl_oracle_k']}/{n} metric {ra['noctl_oracle_score']:.2f} | all incl ctl ({ra['n_used']}): {ra['oracle_k']}/{n} {ra['oracle_score']:.2f} | C2-ext {[os.path.basename(x) for x in ra['c2_arms']]} {ra['c2_oracle_k']}/{n} {ra['c2_oracle_score']:.2f}")
        print(f"   never-correct: core {rc['never']} | ext noctl {ra['never_noctl']} | all {ra['never']}")
        if ds in BAL:
            print(f"   never(noctl) in never-predicted classes: {ra['never_noctl_in_never_pred_class']}; never-predicted classes {ra['classes_never_predicted_any_run']}")
            print(f"   zero-recall classes: base {len(ra['zero_recall_base'])}({sum(ra['zero_recall_base'].values())}q) → core-union {len(rc['zero_recall_all_arms'])}({sum(rc['zero_recall_all_arms'].values())}q) → ext-noctl {len(ra['zero_recall_noctl'])}({sum(ra['zero_recall_noctl'].values())}q): {ra['zero_recall_noctl']}")
            print(f"   base pred entropy {ra['base_pred_entropy_bits']:.2f}/{ra['max_entropy_bits']:.2f} bits, distinct {ra['base_distinct_pred']}, top {[(c, k, round(s*100,1)) for c,k,s in ra['base_top_pred']]}")
        print(f"   invariance all-same: core {rc['inv_all'][0]}/{n} | ext noctl {ra['inv_noctl'][0]}/{n}; ≥90% runs=base answer: core {agree90(ds, g['base'], coreu)}/{n} | ext noctl {agree90(ds, g['base'], noctl)}/{n}")
        sa = sorted(a['same_ans'] for a in rows)
        if sa:
            print(f"   per-arm same-answer-as-base median {sa[len(sa)//2]:.2f} (min {sa[0]:.2f}, max {sa[-1]:.2f})")


if __name__ == "__main__":
    summarize("G1 4B s10 pb8 eager", G1())
    summarize("G2 4B s5 pb8 eager", G2())
    summarize("G3 4B s20", G3())
    summarize("G4 8B s20", G4())
    for t, grp in OTHER().items():
        summarize("OTHER " + t, grp)
