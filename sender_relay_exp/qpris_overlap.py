#!/usr/bin/env python3
"""Per-case prediction overlap between the three Q-PRIS arms."""
import json, glob, os
K = "/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs/qpris"
DSS = ("gtex", "tcga_expert_vqa", "tcga_slidebench", "panda", "tcga")

def preds(arm, ds):
    out = {}
    for f in sorted(glob.glob(f"{K}/{arm}_{ds}/*/result.json")):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        a = r.get("answer")
        if isinstance(a, dict):
            a = a.get("answer")
        out[r.get("dataset_index", os.path.basename(os.path.dirname(f)))] = (
            str(a).strip(), str(r.get("gold_answer", "")).strip())
    return out

print(f"{'dataset':20s} {'n':>3s} {'qn=qt':>6s} {'qt=qs':>6s} {'qn=qs':>6s} {'all3':>5s}")
print("-" * 56)
for ds in DSS:
    a, b, c = preds("qn", ds), preds("qt", ds), preds("qs", ds)
    keys = sorted(set(a) & set(b) & set(c))
    if not keys:
        print(f"{ds:20s} {'-':>3s}  (incomplete)")
        continue
    nab = sum(a[k][0] == b[k][0] for k in keys)
    nbc = sum(b[k][0] == c[k][0] for k in keys)
    nac = sum(a[k][0] == c[k][0] for k in keys)
    n3 = sum(a[k][0] == b[k][0] == c[k][0] for k in keys)
    print(f"{ds:20s} {len(keys):3d} {nab:6d} {nbc:6d} {nac:6d} {n3:5d}")

print("\n--- gtex per-case (qn | qt | qs | gold) ---")
a, b, c = preds("qn", "gtex"), preds("qt", "gtex"), preds("qs", "gtex")
for k in sorted(set(a) & set(b) & set(c)):
    mark = "" if a[k][0] == b[k][0] == c[k][0] else "  <-- differs"
    print(f"  {str(k):>3}  {a[k][0][:18]:18s} {b[k][0][:18]:18s} {c[k][0][:18]:18s} | {b[k][1][:18]}{mark}")
