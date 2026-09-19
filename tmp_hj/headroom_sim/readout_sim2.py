"""Generation answer vs closed-set candidate-scoring readouts on the SAME cases (stored logps, no GPU)."""
import json, glob, sys, statistics as st
from collections import defaultdict, Counter
T = "/home/users/whddn12316/wsi_latent_0915_decode_hj"
sys.path.insert(0, T)
from eval.metrics import substring_correct
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/home/users/whddn12316/models/Qwen3-VL-4B-Thinking")
NT = {}
def ntok(y):
    if y not in NT: NT[y] = len(tok(f'"{y}"', add_special_tokens=False)["input_ids"])
    return NT[y]
S = T + "/sender_relay_exp"
SETS = [  # name, jsonl, dataset json, result dir, prior-cond, metric
    ("rvu_gtex(dcs20)", f"{S}/runs/rvu/rvu_gtex.ig.jsonl", f"{S}/smoke/dcs20_gtex.json", f"{S}/runs/rvu/rvu_gtex", "zeroA", "BACC"),
    ("rvu_evqa(dcs20)", f"{S}/runs/rvu/rvu_tcga_expert_vqa.ig.jsonl", f"{S}/smoke/dcs20_tcga_expert_vqa.json", f"{S}/runs/rvu/rvu_tcga_expert_vqa", "zeroA", "ACC"),
    ("rvu_sb(dcs20)", f"{S}/runs/rvu/rvu_tcga_slidebench.ig.jsonl", f"{S}/smoke/dcs20_tcga_slidebench.json", f"{S}/runs/rvu/rvu_tcga_slidebench", "zeroA", "ACC"),
    ("rvu_panda(dcs20)", f"{S}/runs/rvu/rvu_panda.ig.jsonl", f"{S}/smoke/dcs20_panda.json", f"{S}/runs/rvu/rvu_panda", "zeroA", "BACC"),
    ("rvu_tcga(dcs20)", f"{S}/runs/rvu/rvu_tcga.ig.jsonl", f"{S}/smoke/dcs20_tcga.json", f"{S}/runs/rvu/rvu_tcga", "zeroA", "BACC"),
    ("ig_gtex40", f"{S}/runs/retr_ig/ig_gtex40.ig.jsonl", f"{S}/smoke/gtex40.json", f"{S}/runs/retr_ig/ig_gtex40", "novis", "BACC"),
    ("ig_evqa24", f"{S}/runs/retr_ig/ig_evqa24.ig.jsonl", f"{S}/smoke/evqa24.json", f"{S}/runs/retr_ig/ig_evqa24", "novis", "ACC"),
    ("ig_sb24", f"{S}/runs/retr_ig/ig_sb24.ig.jsonl", f"{S}/smoke/sb24.json", f"{S}/runs/retr_ig/ig_sb24", "novis", "ACC"),
]
def score(metric, golds, oks):
    if metric == "BACC":
        by = defaultdict(list)
        for g, c in zip(golds, oks): by[g].append(c)
        return 100 * st.mean(sum(v) / len(v) for v in by.values())
    return 100 * sum(oks) / len(oks)
tot = defaultdict(lambda: [0, 0])
for name, jf, dsf, rdir, pc, metric in SETS:
    rows, seen = [], set()
    for l in open(jf):
        r = json.loads(l)
        if r["case"] in seen: continue
        seen.add(r["case"]); rows.append(r)
    items = json.load(open(dsf))
    gen = {}
    for f in glob.glob(f"{rdir}/**/result.json", recursive=True):
        j = json.load(open(f)); a = j.get("answer"); pa = a.get("answer") if isinstance(a, dict) else a
        gen[int(j["dataset_index"])] = str(pa)
    rows = [r for r in rows if r["case"] in gen]
    golds = [items[r["case"]]["Answer"] for r in rows]
    fixed = len({tuple(r["cands"]) for r in rows}) == 1
    out = {}
    out["generation"] = [substring_correct(str(g), gen[r["case"]]) for r, g in zip(rows, golds)]
    def pick(fn): return [max(r["cands"], key=lambda y: fn(i, r, y)) == g for i, (r, g) in enumerate(zip(rows, golds))]
    out["closed full sum"] = pick(lambda i, r, y: r["logp"]["full"][y])
    out["closed full mean/token"] = pick(lambda i, r, y: r["logp"]["full"][y] / ntok(y))
    out[f"closed {pc}(no visual) mean/token"] = pick(lambda i, r, y: r["logp"][pc][y] / ntok(y))
    out[f"closed (full-0.5*{pc})/token"] = pick(lambda i, r, y: (r["logp"]["full"][y] - 0.5 * r["logp"][pc][y]) / ntok(y))
    if fixed:
        pri = [{y: st.mean(rows[j]["logp"]["full"][y] / ntok(y) for j in range(len(rows)) if j != i) for y in rows[i]["cands"]} for i in range(len(rows))]
        out["closed full/token - LOO class prior"] = pick(lambda i, r, y: r["logp"]["full"][y] / ntok(y) - pri[i][y])
    print(f"\n== {name} n={len(rows)} ncand={len(rows[0]['cands'])} metric={metric}")
    base = out["generation"]
    for k, v in out.items():
        w = sum(a and not b for a, b in zip(v, base)); lo = sum(b and not a for a, b in zip(v, base))
        print(f"  {k:40s} correct {sum(v):2d}/{len(v)}  {metric} {score(metric, golds, v):5.1f}" + ("" if k == "generation" else f"  vs generation +{w}/-{lo}"))
        tot[k][0] += sum(v); tot[k][1] += len(v)
