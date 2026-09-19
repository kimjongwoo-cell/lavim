import json, re, statistics as st
from collections import defaultdict, Counter
K = "/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs/lens/"
FUNC = {"the","a","an","of","and","or","in","with","to","is","for","on","by","at","as","no","not"}
def cat(step_idx, tok):
    s = tok.strip()
    if s == "" or re.fullmatch(r"[\W_]+", s):
        return "punct/space"
    if s.lower() in FUNC:
        return "function word"
    return "first content" if step_idx == 0 else "continuation"
tot = defaultdict(list); lay = defaultdict(lambda: defaultdict(list)); examples = defaultdict(Counter)
for ds in ("gtex","tcga_expert_vqa","tcga_slidebench","tcga","panda"):
    for line in open(K + f"lens_{ds}.jsonl"):
        r = json.loads(line)
        for i, s in enumerate(r["steps"]):
            c = cat(i, s["tok"])
            key = (ds, c)
            tot[key].append(s["decision_layer"])
            examples[c][s["tok"]] += 1
            for L in (18, 24, 30, 33, 35):
                lay[key][L].append(s["layers"][L]["rank"])
cats = ("first content","continuation","function word","punct/space")
print(f"{'dataset':16s} {'category':14s} {'n':>4s} {'dec med':>7s} {'rank L18':>9s} {'L24':>6s} {'L30':>5s} {'L33':>4s} {'L35':>4s}")
for ds in ("gtex","tcga_expert_vqa","tcga_slidebench","tcga","panda"):
    for c in cats:
        v = tot.get((ds, c))
        if not v: continue
        dl = [x for x in v if x is not None]
        med = st.median(dl) if dl else None
        rk = {L: int(st.median(lay[(ds, c)][L])) for L in (18,24,30,33,35)}
        print(f"{ds:16s} {c:14s} {len(v):4d} {str(med):>7s} {rk[18]:9d} {rk[24]:6d} {rk[30]:5d} {rk[33]:4d} {rk[35]:4d}")
print("--- 전체(5셋 합)")
for c in cats:
    v = [x for (ds, cc), xs in tot.items() if cc == c for x in xs]
    if not v: continue
    dl = [x for x in v if x is not None]
    rk = {L: int(st.median([x for (ds, cc), d in lay.items() if cc == c for x in d[L]])) for L in (18,24,30,33,35)}
    print(f"{c:14s} n={len(v):3d} dec med={st.median(dl) if dl else None} rank L18 {rk[18]} L24 {rk[24]} L30 {rk[30]} L33 {rk[33]} L35 {rk[35]}  예: {examples[c].most_common(8)}")
