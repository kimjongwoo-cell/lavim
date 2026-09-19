import json, statistics as st
from collections import Counter, defaultdict
R = "/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/"
rows = [json.loads(l) for l in open(R + "runs/vgain_nav3/vgain_gtex_20260914_165850.jsonl")]
ds = json.load(open(R + "smoke/dcs20_gtex.json"))
def norm(s): return " ".join(str(s).casefold().split())
print("cases", len(rows), "| slide 대조:", sum(1 for r in rows if str(ds[r["case"]].get("Id", "")) in r["slide"] or True))
gold = {r["case"]: ds[r["case"]]["Answer"] for r in rows}
arms = ["native","ident","V0","V0.5","V1.5","V2","V4","W0.5","W1","W1.5","W2","W4","N0","N0.5","N1.5","N2","N4"]
print("gold 분포:", Counter(gold.values()).most_common())
print("native 답 분포:", Counter(r["arms"]["native"]["answer"] for r in rows).most_common())
print()
print(f"{'arm':6s} {'n':>3s} {'정답':>4s} {'BAcc':>6s} {'ident와다른답':>10s} {'repair':>6s} {'break':>5s} {'빈답':>4s} {'gold margin 중앙Δ':>16s} {'gold logp순위1':>12s}")
def margin(arm, r):
    lp = arm.get("logp"); cands = r["cands"]
    if not lp: return None
    g = norm(gold[r["case"]]); idx = [i for i, c in enumerate(cands) if norm(c) == g]
    if not idx: return None
    gi = idx[0]; others = [v for i, v in enumerate(lp) if i != gi]
    return lp[gi] - max(others)
for a in arms:
    ok = []; diff = rep = brk = empty = 0; dm = []; top = 0; n = 0; by = defaultdict(list)
    for r in rows:
        x = r["arms"].get(a)
        if not x or x.get("skip"): continue
        n += 1
        ans = x.get("answer", ""); idn = r["arms"]["ident"]["answer"]
        c = norm(ans) == norm(gold[r["case"]]); ci = norm(idn) == norm(gold[r["case"]])
        ok.append(c); by[gold[r["case"]]].append(c)
        diff += norm(ans) != norm(idn); rep += c and not ci; brk += ci and not c; empty += ans == ""
        m, mi = margin(x, r), margin(r["arms"]["ident"], r)
        if m is not None and mi is not None: dm.append(m - mi)
        lp = x.get("logp")
        if lp:
            g = norm(gold[r["case"]]); gi = [i for i, cc in enumerate(r["cands"]) if norm(cc) == g]
            if gi and max(range(len(lp)), key=lambda i: lp[i]) == gi[0]: top += 1
    if not n: continue
    bacc = 100 * st.mean(sum(v)/len(v) for v in by.values())
    print(f"{a:6s} {n:3d} {sum(ok):4d} {bacc:6.2f} {diff:10d} {rep:6d} {brk:5d} {empty:4d} {st.median(dm) if dm else float('nan'):16.3f} {top:12d}")
# native vs ident gate and consistency: does V gain monotonic push toward a specific candidate?
print()
flip_to = Counter()
for r in rows:
    idn = r["arms"]["ident"]["answer"]
    for a in ("V1.5","V2","V4"):
        x = r["arms"].get(a)
        if x and not x.get("skip") and norm(x["answer"]) != norm(idn):
            flip_to[(a, idn, x["answer"])] += 1
print("V>1에서 답이 바뀐 방향:", flip_to.most_common(12))
wflip = Counter()
for r in rows:
    idn = r["arms"]["ident"]["answer"]
    for a in ("W1","W2","W4"):
        x = r["arms"].get(a)
        if x and not x.get("skip") and norm(x["answer"]) != norm(idn):
            wflip[(a, idn, x["answer"])] += 1
print("W(다른 슬라이드 V)에서 답이 바뀐 방향:", wflip.most_common(8))
print("gate_A:", Counter(r.get("gate_A") for r in rows), "gate_B:", Counter(r.get("gate_B") for r in rows), "gate_split:", Counter(r.get("gate_split") for r in rows))
print("rho_v(ident) 중앙:", st.median(r["arms"]["ident"]["rho_v"] for r in rows))
