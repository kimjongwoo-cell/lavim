"""Ceiling probe analysis: can ANY latent-derived query priming at the first answer-content row start the gold answer?

For each case: baseline (no priming) and every (direction k, scale s) probe give the top-50 first-content tokens.
Two readings:
  free   - is the gold answer's first token the unrestricted argmax?
  choice - among the first tokens of the question's answer options, is the gold one ranked first?
"""
import glob, json, sys
sys.path.insert(0, "/home/users/whddn12316/wsi_latent_0915_decode_hj")
sys.path.insert(0, "/home/users/whddn12316/wsi_latent_0915_decode_hj/scripts")
import rescore as rs
from transformers import AutoTokenizer
from vision_text_mas.prompts import canonical_open_answer_options

ds = sys.argv[1] if len(sys.argv) > 1 else "tcga_expert_vqa"
pat = sys.argv[2] if len(sys.argv) > 2 else "sender_relay_exp/runs/nav4/rfnm_ceil/ceil_shard*.jsonl"
tok = AutoTokenizer.from_pretrained("/home/users/whddn12316/models/Qwen3-VL-4B-Thinking")
recs = rs.load_records(ds)
first = lambda t: tok(t, add_special_tokens=False)["input_ids"][0] if t else None

def eval_one(r):
    i = r["case"]
    rec = recs[i]
    ch = list(rec.get("Choice") or canonical_open_answer_options(rec["Question"]))
    gold = rec["Answer"]
    gt = first(gold)
    cand = {c: first(c) for c in ch}
    uniq = sum(1 for c in cand if cand[c] == gt) == 1              # gold's first token not shared by another option
    def rd(ids, lp):
        d = dict(zip(ids, lp))
        free = ids[0] == gt
        best = max(cand, key=lambda c: d.get(cand[c], -99))          # restricted to option first tokens seen
        seen = any(cand[c] in d for c in cand)
        return free, (seen and best.strip().lower() == gold.strip().lower())
    base = rd(*r["base"])
    hits = [(p["k"], p["s"], *rd(p["ids"], p["lp"])) for p in r["probe"]]
    any_free = any(h[2] for h in hits)
    any_ch = any(h[3] for h in hits)
    return base, any_free, any_ch, hits, uniq

rows = [json.loads(l) for f in glob.glob(pat) for l in open(f)]
allr = [(r, eval_one(r)) for r in rows]
print(f"all cases {len(allr)}, gold first token unique among options {sum(e[4] for _, e in allr)}")
rows = [r for r, e in allr if e[4]]
n = len(rows)
bf = bc = af = ac = 0
byscale = {}
for r in rows:
    base, any_free, any_ch, hits, _ = eval_one(r)
    bf += base[0]; bc += base[1]; af += any_free or base[0]; ac += any_ch or base[1]
    for k, s, f_, c_ in hits:
        d = byscale.setdefault(s, [0, 0, 0])
        d[0] += 1; d[1] += f_; d[2] += c_
print(f"n={n}  baseline: gold is free argmax {bf}, gold wins among options {bc}")
print(f"ceiling over {len(rows[0]['probe']) if rows else 0} (direction, scale) probes: free {af}, options {ac}")
for s in sorted(byscale):
    t, f_, c_ = byscale[s]
    print(f"  scale {s}: probes {t}, gold-free-argmax {f_} ({100*f_/t:.1f}%), gold-wins-options {c_} ({100*c_/t:.1f}%)")
