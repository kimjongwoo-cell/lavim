#!/usr/bin/env python3
"""Extra headroom diagnostics: union growth curve, rescue multiplicity, wrong-slide noise floor,
stuck never-correct questions, duplicate answer vectors."""
import random, statistics as st
from analysis import *

random.seed(0)


def pis_for(ds, base, arms, cov_min=0.9):
    bpi = get(base, ds)
    bidx = set(bpi)
    used = []
    for a in arms:
        api = get(a, ds)
        if api is None:
            continue
        if len(set(api) & bidx) / len(bidx) >= cov_min:
            used.append(a)
    return bpi, bidx, used


def union_k(ds, bpi, bidx, arms):
    pis = [bpi] + [get(a, ds) for a in arms]
    anyc = {i: int(any(p.get(i, (None, None, 0))[2] for p in pis)) for i in bidx}
    pseudo = {i: ("x", bpi[i][1], anyc[i]) for i in bidx}
    return sum(anyc.values()), score_items(ds, pseudo)[0]


def extras(ds, base, arms, label, controls_pat=CONTROL_PAT):
    bpi, bidx, used = pis_for(ds, base, arms)
    noctl = [a for a in used if not controls_pat.search(os.path.basename(a))]
    ctl = [a for a in used if controls_pat.search(os.path.basename(a))]
    out = {"ds": ds, "label": label, "n": len(bidx), "n_noctl": len(noctl), "n_ctl": len(ctl)}
    # dedupe answer vectors
    vecs = {}
    for a in noctl:
        p = get(a, ds)
        key = tuple(p.get(i, (None,))[0] for i in sorted(bidx))
        vecs.setdefault(key, []).append(short(a))
    out["distinct_noctl"] = len(vecs)
    out["dups"] = [v for v in vecs.values() if len(v) > 1]
    # union growth curve (noctl)
    curve = {}
    for k in (1, 2, 4, 8, 16, 32, 64):
        if k > len(noctl):
            break
        vals = [union_k(ds, bpi, bidx, random.sample(noctl, k))[0] for _ in range(200)]
        curve[k] = (st.mean(vals), min(vals), max(vals))
    curve[len(noctl)] = (union_k(ds, bpi, bidx, noctl)[0],) * 3
    out["curve"] = curve
    # rescue multiplicity among base-wrong questions (noctl, distinct vectors)
    reps = [get(a[0].replace("S:", S).replace("T:", T).replace("R:", R), ds) for a in vecs.values()]
    mult = collections.Counter()
    brk = collections.Counter()
    for i in bidx:
        k = sum(1 for p in reps if i in p and p[i][2])
        if not bpi[i][2]:
            mult[k] += 1
        else:
            kb = sum(1 for p in reps if i in p and not p[i][2])
            brk[kb] += 1
    nrep = len(reps)
    def bucket(c):
        b = collections.OrderedDict((("0", 0), ("1", 0), ("2-3", 0), ("4-9", 0), (">=10", 0), (">=half", 0)))
        for k, v in c.items():
            if k == 0: b["0"] += v
            elif k == 1: b["1"] += v
            elif k <= 3: b["2-3"] += v
            elif k <= 9: b["4-9"] += v
            else: b[">=10"] += v
            if nrep and k >= nrep / 2 and k > 0: b[">=half"] += v
        return dict(b)
    out["rescue_mult_base_wrong"] = bucket(mult)
    out["break_mult_base_right"] = bucket(brk)
    out["n_base_wrong"] = sum(mult.values()); out["n_base_right"] = sum(brk.values())
    # controls
    out["ctl_rows"] = []
    for a in ctl:
        p = get(a, ds)
        w = sum(1 for i in bidx if i in p and not bpi[i][2] and p[i][2])
        l = sum(1 for i in bidx if i in p and bpi[i][2] and not p[i][2])
        out["ctl_rows"].append((short(a), w, l))
    if ctl:
        out["ctl_union"] = union_k(ds, bpi, bidx, ctl)
    # never-correct (noctl) stuck analysis: distinct wrong answers across base+noctl runs
    allp = [bpi] + [get(a, ds) for a in noctl]
    never = [i for i in bidx if not any(p.get(i, (None, None, 0))[2] for p in allp)]
    distinct = collections.Counter()
    base_share = []
    for i in never:
        answers = [pred_class(ds, i, p[i][0]) for p in allp if i in p]
        c = collections.Counter(answers)
        distinct[min(len(c), 4)] += 1
        base_share.append(c[pred_class(ds, i, bpi[i][0])] / len(answers))
    out["never_noctl"] = len(never)
    out["never_distinct_answers(1,2,3,>=4)"] = dict(sorted(distinct.items()))
    out["never_mean_share_of_base_answer"] = st.mean(base_share) if base_share else None
    return out


if __name__ == "__main__":
    RES = {}
    for gname, gfun in (("G1", G1), ("G2", G2)):
        for ds, g in gfun().items():
            RES[f"{gname}:{ds}"] = extras(ds, g["base"], g["core"] + g["ext"], gname)
    json.dump(RES, open(os.path.join(HERE, "headroom_extras.json"), "w"), ensure_ascii=False, indent=1, default=str)
    for k, v in RES.items():
        print("=====", k, "n", v["n"], "noctl", v["n_noctl"], "distinct", v["distinct_noctl"], "ctl", v["n_ctl"])
        print("  curve", {kk: (round(a, 1), b, c) for kk, (a, b, c) in v["curve"].items()})
        print("  rescue(base wrong", v["n_base_wrong"], ")", v["rescue_mult_base_wrong"])
        print("  break(base right", v["n_base_right"], ")", v["break_mult_base_right"])
        print("  ctl", v["ctl_rows"], "ctl_union", v.get("ctl_union"))
        print("  never_noctl", v["never_noctl"], "distinct answers", v["never_distinct_answers(1,2,3,>=4)"], "base-answer share", v["never_mean_share_of_base_answer"])
        print("  dups", v["dups"])
