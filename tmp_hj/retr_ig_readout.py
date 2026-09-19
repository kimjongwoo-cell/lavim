"""Readout for VLMAS_RETR=igsweep: latent-query score s_g vs causal usefulness I_g.

usage: retr_ig_readout.py <ig.jsonl> <run_output_root>   (gold from <root>/*/result.json)
  m(cond)  = log p(gold) - max_{y != gold} log p(y)
  I_g      = m(full) - m(remove g)         (>0: removing the group hurts)
  s_g      = mean over the group's tokens of S after leave-one-out position-profile
             subtraction (same absolute index, other cases with the same n_vis)
  Reports per case: argmax s_g == argmax I_g ?, Spearman(s_g, I_g) over groups;
  pooled: within-case-centred Pearson/Spearman over all (case, group) pairs,
  top-1 agreement vs chance (1/#groups), and the same with raw S (no bias removal).
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

path, root = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(path)]
gold = {}
for f in glob.glob(os.path.join(root, "*", "result.json")):
    d = json.load(open(f))
    gold[int(d.get("dataset_index", os.path.basename(os.path.dirname(f)).split("_")[0]))] = str(d.get("gold_answer", "")).strip()

def margin(lp, g):
    if g not in lp:
        # case-insensitive match
        k = next((y for y in lp if y.lower() == g.lower()), None)
        if k is None:
            return None
        g = k
    others = [v for y, v in lp.items() if y != g]
    return lp[g] - (max(others) if others else float("nan"))

def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    ra = a.argsort().argsort(); rb = b.argsort().argsort()
    return float(np.corrcoef(ra, rb)[0, 1])

# leave-one-out position profile per n_vis
by_n = defaultdict(list)
for r in rows:
    by_n[r["n_vis"]].append(np.asarray(r["S"], float))
per_case = []
pool_s, pool_i, pool_sraw = [], [], []
agree = agree_raw = n_cases = 0
for r in rows:
    g = gold.get(r["case"])
    if not g:
        continue
    mf = margin(r["logp"]["full"], g)
    if mf is None:
        continue
    S = np.asarray(r["S"], float)
    others = [x for x in by_n[r["n_vis"]] if x is not S]
    prof = np.mean(others, 0) if len(others) else np.zeros_like(S)
    Sb = S - prof
    st = r["starts"]; pages = r["pages"]
    sg, sraw, ig = [], [], []
    for root_ in r["roots"]:
        toks = []
        for p in r["groups"][str(root_)]:
            toks += list(range(st[p], st[p] + pages[p]))
        toks = np.asarray(toks)
        sg.append(float(Sb[toks].mean())); sraw.append(float(S[toks].mean()))
        mg = margin(r["logp"][f"g{root_}"], g)
        ig.append(mf - mg)
    mn = margin(r["logp"].get("novis", {}), g)
    if len(ig) >= 2:
        n_cases += 1
        agree += int(int(np.argmax(sg)) == int(np.argmax(ig)))
        agree_raw += int(int(np.argmax(sraw)) == int(np.argmax(ig)))
        pool_s += list(np.asarray(sg) - np.mean(sg)); pool_i += list(np.asarray(ig) - np.mean(ig))
        pool_sraw += list(np.asarray(sraw) - np.mean(sraw))
    per_case.append({"case": r["case"], "gold": g, "m_full": round(mf, 3), "m_novis": None if mn is None else round(mn, 3),
                     "I_g": [round(x, 3) for x in ig], "s_g": [round(x, 3) for x in sg], "s_raw": [round(x, 3) for x in sraw],
                     "rho": round(spearman(sg, ig), 2), "argmax_s": int(np.argmax(sg)), "argmax_I": int(np.argmax(ig)),
                     "groups": r["groups"]})
for c in per_case:
    print(json.dumps(c, ensure_ascii=False))
print("----")
print(f"cases {n_cases} | top-1 agree (LOO bias removed) {agree}/{n_cases} | raw S {agree_raw}/{n_cases} | "
      f"chance ≈ {np.mean([1/len(c['I_g']) for c in per_case if len(c['I_g'])>=2]):.2f}")
if pool_s:
    ps, pi, pr = np.asarray(pool_s), np.asarray(pool_i), np.asarray(pool_sraw)
    print(f"pooled within-case-centred: Pearson(s_g,I_g) {np.corrcoef(ps,pi)[0,1]:+.3f} | Spearman {spearman(ps,pi):+.3f} | "
          f"raw S Pearson {np.corrcoef(pr,pi)[0,1]:+.3f}")
    rh = [c["rho"] for c in per_case if c["rho"] == c["rho"]]
    print(f"per-case Spearman median {np.median(rh):+.2f}, >0: {sum(x>0 for x in rh)}/{len(rh)}")
    print(f"I_g > 0 (removal hurts) {int((pi + np.asarray([np.mean(c['I_g']) for c in per_case for _ in c['I_g']]) > 0).sum())}/{len(pi)} groups; "
          f"m_full median {np.median([c['m_full'] for c in per_case]):+.2f}, m_novis median "
          f"{np.median([c['m_novis'] for c in per_case if c['m_novis'] is not None]):+.2f}")
