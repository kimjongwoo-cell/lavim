"""Offline decision-rule simulation on RVU candidate log-probs (5 sets x 20, eager, canonical addressing).
No GPU. Rules re-score the SAME stored teacher-forced log p of each candidate under full / zeroA."""
import json, statistics as st
from collections import Counter, defaultdict
R = "/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp"
BACC = {"gtex", "panda", "tcga"}
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/home/users/whddn12316/models/Qwen3-VL-4B-Thinking")
    ntok = lambda y: len(tok(f'"{y}"', add_special_tokens=False)["input_ids"])
except Exception as e:
    print("no tokenizer", e); ntok = lambda y: 1

def load(ds):
    rows, seen = [], set()
    for l in open(f"{R}/runs/rvu/rvu_{ds}.ig.jsonl"):
        r = json.loads(l)
        if r["case"] in seen: continue
        seen.add(r["case"]); rows.append(r)
    items = json.load(open(f"{R}/smoke/dcs20_{ds}.json"))
    return rows, items

def metric(ds, golds, preds):
    n = sum(g == p for g, p in zip(golds, preds))
    if ds in BACC:
        by = defaultdict(list)
        for g, p in zip(golds, preds): by[g].append(g == p)
        return n, 100 * st.mean(sum(v) / len(v) for v in by.values())
    return n, 100 * n / len(golds)

for ds in ["gtex", "tcga_expert_vqa", "tcga_slidebench", "panda", "tcga"]:
    rows, items = load(ds)
    golds = [items[r["case"]]["Answer"] for r in rows]
    fixed = len({tuple(r["cands"]) for r in rows}) == 1
    # leave-one-out class prior (only for fixed candidate sets)
    def loo_prior(i, cond):
        others = [rows[j] for j in range(len(rows)) if j != i]
        return {y: st.mean(o["logp"][cond][y] for o in others) for y in rows[i]["cands"]}
    rules = {
        "full(argmax)": lambda i, r, y: r["logp"]["full"][y],
        "full/len": lambda i, r, y: r["logp"]["full"][y] / ntok(y),
        "zeroA(no visual at Answerer)": lambda i, r, y: r["logp"]["zeroA"][y],
        "full-0.5*zeroA": lambda i, r, y: r["logp"]["full"][y] - 0.5 * r["logp"]["zeroA"][y],
        "full-1.0*zeroA": lambda i, r, y: r["logp"]["full"][y] - 1.0 * r["logp"]["zeroA"][y],
    }
    if fixed:
        rules["full-LOOprior"] = lambda i, r, y: r["logp"]["full"][y] - PRI[i][y]
        rules["(full-zeroA)-LOOprior(full-zeroA)"] = lambda i, r, y: (r["logp"]["full"][y] - r["logp"]["zeroA"][y]) - PRI2[i][y]
        PRI = [loo_prior(i, "full") for i in range(len(rows))]
        PRI2 = []
        for i in range(len(rows)):
            others = [rows[j] for j in range(len(rows)) if j != i]
            PRI2.append({y: st.mean(o["logp"]["full"][y] - o["logp"]["zeroA"][y] for o in others) for y in rows[i]["cands"]})
    print(f"\n== {ds} n={len(rows)} ncands={len(rows[0]['cands'])} fixed_set={fixed} metric={'BACC' if ds in BACC else 'ACC'} gold classes={len(set(golds))}")
    base_preds = None
    for name, f in rules.items():
        preds = [max(r["cands"], key=lambda y: f(i, r, y)) for i, r in enumerate(rows)]
        n, m = metric(ds, golds, preds)
        top = Counter(preds).most_common(1)[0]
        extra = ""
        if base_preds is None:
            base_preds = preds
        else:
            w = sum(p == g and b != g for p, b, g in zip(preds, base_preds, golds))
            lo = sum(p != g and b == g for p, b, g in zip(preds, base_preds, golds))
            extra = f" vs full: +{w}/-{lo}"
        print(f"  {name:38s} correct {n:2d}/{len(rows)}  metric {m:5.1f}  top-pred share {top[1]}/{len(rows)} ({str(top[0])[:18]}){extra}")
    # oracle over the 12 drop / zeroA / full conditions (upper bound of any single-crop mass-fixed tweak)
    conds = [c for c in rows[0]["logp"]]
    orc = sum(any(max(r["cands"], key=lambda y: r["logp"][c][y]) == g for c in conds) for r, g in zip(rows, golds))
    print(f"  oracle over stored conditions ({len(conds)}) correct {orc}/{len(rows)}; gold rank under full (median) {st.median(sorted(r['cands'], key=lambda y: -r['logp']['full'][y]).index(g) + 1 for r, g in zip(rows, golds))}")
