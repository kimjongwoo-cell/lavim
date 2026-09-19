"""Level 3 (1-cos of z) + Level 4 (answer flip / accuracy) from the true vs wrong arms."""
import glob, json, re, sys
from pathlib import Path
import torch
Q = Path("/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs/expq")
D = Q / "rpath_out"
CASE_RE = re.compile(r"^(\d+)_")

def cases(arm):
    out = {}
    for rj in sorted((Q / arm).glob("*/result.json")):
        mo = CASE_RE.match(rj.parent.name)
        if not mo: continue
        d = json.loads(rj.read_text()); a = d.get("answer")
        out[int(mo.group(1))] = ((a.get("answer") if isinstance(a, dict) else a) or "").strip(), d.get("gold_answer")
    return out

def zfiles(sub):
    return sorted(glob.glob(str(D / sub / "z_*.pt")))

true, wrong = cases("rpath_true_gtex"), cases("rpath_wrong_gtex")
common = sorted(set(true) & set(wrong))
tz, wz = zfiles("true"), zfiles("wrong")
print(f"cases true={len(true)} wrong={len(wrong)} common={len(common)}  z true={len(tz)} wrong={len(wz)}")
# processing order == ascending dataset index (shard list), so z_i <-> i-th case
order_t, order_w = sorted(true), sorted(wrong)
rows = []
for i, ci in enumerate(common):
    if ci not in order_t or ci not in order_w: continue
    it, iw = order_t.index(ci), order_w.index(ci)
    if it >= len(tz) or iw >= len(wz): continue
    a = torch.load(tz[it], map_location="cpu", weights_only=True)
    b = torch.load(wz[iw], map_location="cpu", weights_only=True)
    steps = min(a.shape[0], b.shape[0])
    dep = [float(1 - torch.nn.functional.cosine_similarity(a[t], b[t], dim=0)) for t in range(steps)]
    (pt, gold), (pw, _) = true[ci], wrong[ci]
    rows.append({"case": ci, "donor": iw == 0, "dep": dep, "flip": pt != pw,
                 "true_correct": pt == gold, "wrong_correct": pw == gold})
rows = [r for r in rows if not r["donor"]]
n = len(rows)
print(f"\nLevel 3 (z, 1-cos), n={n} (donor excluded)")
steps = min(len(r["dep"]) for r in rows)
for t in range(steps):
    v = sorted(r["dep"][t] for r in rows)
    print(f"  step {t+1:2d}: median {v[len(v)//2]:.4f}  mean {sum(v)/n:.4f}  p90 {v[int(0.9*(n-1))]:.4f}  max {v[-1]:.4f}")
flips = sum(r["flip"] for r in rows)
print(f"\nLevel 4 (answer): flips {flips}/{n} = {100*flips/n:.1f}%  "
      f"acc true {sum(r['true_correct'] for r in rows)}/{n}  acc wrong {sum(r['wrong_correct'] for r in rows)}/{n}")
# joint: do the heavy-tail Level-3 cases coincide with flips?
tail = sorted(rows, key=lambda r: -max(r["dep"]))[:max(1, n // 5)]
print(f"top-20% Level-3 tail ({len(tail)} cases): flips {sum(r['flip'] for r in tail)}/{len(tail)}; "
      f"rest: flips {sum(r['flip'] for r in rows if r not in tail)}/{n-len(tail)}")
print("per-case (case, max 1-cos, flip):", [(r["case"], round(max(r["dep"]), 3), r["flip"]) for r in sorted(rows, key=lambda r: -max(r["dep"]))[:8]])
