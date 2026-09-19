"""Causal-pathway summary for one dataset from the W/T pair.

usage: rpath_pathways.py <tag>   (tag in gtex|evqa|sb|tcga|panda)
  true  arm = rpath_t_<tag> (pass T: normal run + patch ladder)   [gtex: rpath_true_gtex]
  wrong arm = rpath_w_<tag> (pass W: wrong-visual + KV dump)      [gtex: rpath_wrong_gtex]
Reports Level 3 (z 1-cos), Level 4 (answer flip / acc), and the pass-T ladder
(vis / R{k}:m / vis+R) flip rates + teacher-forced score shifts + KV-level dependence.
"""
import glob, json, re, sys
from pathlib import Path
import torch
tag = sys.argv[1]
Q = Path("/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs/expq")
D = Q / "rpath_out" / ("" if tag == "gtex" else tag)
true_arm = "rpath_true_gtex" if tag == "gtex" else f"rpath_t_{tag}"
wrong_arm = "rpath_wrong_gtex" if tag == "gtex" else f"rpath_w_{tag}"
CASE_RE = re.compile(r"^(\d+)_")

def cases(arm):
    out = {}
    for rj in sorted((Q / arm).glob("*/result.json")):
        mo = CASE_RE.match(rj.parent.name)
        if not mo: continue
        d = json.loads(rj.read_text()); a = d.get("answer")
        out[int(mo.group(1))] = (((a.get("answer") if isinstance(a, dict) else a) or "").strip(), d.get("gold_answer"))
    return out

def med(v): v = sorted(v); return v[len(v)//2] if v else float("nan")

true, wrong = cases(true_arm), cases(wrong_arm)
common = sorted(set(true) & set(wrong))
tz = sorted(glob.glob(str(D / "true" / "z_*.pt"))); wz = sorted(glob.glob(str(D / "wrong" / "z_*.pt")))
ot, ow = sorted(true), sorted(wrong)
rows = []
for ci in common:
    it, iw = ot.index(ci), ow.index(ci)
    if iw == 0 or it >= len(tz) or iw >= len(wz): continue     # index 0 = donor case
    a = torch.load(tz[it], map_location="cpu", weights_only=True); b = torch.load(wz[iw], map_location="cpu", weights_only=True)
    T = min(a.shape[0], b.shape[0])
    dep = [float(1 - torch.nn.functional.cosine_similarity(a[t], b[t], dim=0)) for t in range(T)]
    (pt, gold), (pw, _) = true[ci], wrong[ci]
    rows.append({"case": ci, "dep": dep, "flip": pt != pw, "tc": pt == gold, "wc": pw == gold})
n = len(rows)
print(f"[{tag}] true={len(true)} wrong={len(wrong)} paired(non-donor)={n}")
if n:
    T = min(len(r["dep"]) for r in rows)
    print("Level 3  1-cos(z_true,z_wrong): " + "  ".join(f"s{t+1}:{med([r['dep'][t] for r in rows]):.4f}" for t in range(T)))
    fl = sum(r["flip"] for r in rows)
    print(f"Level 4  flips {fl}/{n} = {100*fl/n:.0f}%   acc true {sum(r['tc'] for r in rows)}/{n} -> wrong {sum(r['wc'] for r in rows)}/{n}")
# pass-T ladder
jl = D / "kv" / "rpath_patch.jsonl"
if jl.is_file():
    recs = [json.loads(l) for l in jl.read_text().splitlines() if l.strip()]
    recs = [r for r in recs if not r.get("donor_case")]
    if recs:
        modes = [k[len("answer_"):] for k in recs[0] if k.startswith("answer_")]
        print(f"pass T ladder, n={len(recs)}:  mode  flip%  mean(score_normal - score_mode)")
        for m in modes:
            fl = sum(1 for r in recs if r.get(f"answer_{m}") != r["normal"])
            ds = [r["score_normal"] - r[f"score_{m}"] for r in recs if r.get("score_normal") is not None and r.get(f"score_{m}") is not None]
            print(f"   {m:>10s}  {100*fl/len(recs):5.1f}%   {sum(ds)/len(ds) if ds else float('nan'):+.3f}")
        ks = [r["kv_dep_k_per_step"] for r in recs if "kv_dep_k_per_step" in r]
        vs = [r["kv_dep_v_per_step"] for r in recs if "kv_dep_v_per_step" in r]
        if ks:
            T = min(len(k) for k in ks)
            print("KV-level 1-cos  K: " + " ".join(f"s{t+1}:{med([k[t] for k in ks]):.4f}" for t in range(T)))
            print("                V: " + " ".join(f"s{t+1}:{med([v[t] for v in vs]):.4f}" for t in range(T)))
        vl = [r["kv_dep_visual_lastlayer"] for r in recs if "kv_dep_visual_lastlayer" in r]
        if vl: print(f"visual cols K (last layer) true vs wrong 1-cos median: {med(vl):.4f}  (sanity: should be large)")
else:
    print("pass T ladder: not run yet")
