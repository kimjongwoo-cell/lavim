#!/usr/bin/env python3
"""Score the Q-PRIS three-arm chain (0902) with the dashboard scorer."""
import json, subprocess, sys
K = "/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs/qpris"
PY = "/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python"
SCORER = "/home/users/whddn12316/wsi_latent_0915_decode_hj/scripts/dashboard_rows.py"
ARMS = (("qn", "none"), ("qt", "Q-PRIS"), ("qs", "shuffled"))
DSS = ("gtex", "tcga_expert_vqa", "tcga_slidebench", "panda", "tcga")

out = {}
for arm, _ in ARMS:
    for ds in DSS:
        r = subprocess.run(
            [PY, SCORER, "--arm", f"{K}/{arm}_{ds}", "--dataset", ds,
             "--param", "4b", "--method", "latent-base", "--step", "10", "--full"],
            capture_output=True, text=True, cwd="/home/users/whddn12316/wsi_latent_0915_decode_hj")
        if r.returncode != 0:
            continue
        try:
            rows = json.loads(r.stdout)
        except Exception:
            continue
        for row in rows:
            out[(arm, ds)] = row

if "--json" in sys.argv:
    print(json.dumps({f"{a}_{d}": v for (a, d), v in out.items()}, ensure_ascii=False, indent=1))
    raise SystemExit

print(f"{'dataset':20s} {'qn none':>9s} {'qt Q-PRIS':>10s} {'qs shuf':>9s} {'qt-qn':>7s} {'qt-qs':>7s}  n")
print("-" * 78)
for ds in DSS:
    def acc(a):
        r = out.get((a, ds))
        v = r.get("acc") if r else None
        return v if isinstance(v, (int, float)) else None
    qn, qt, qs = acc("qn"), acc("qt"), acc("qs")
    def s(v):
        return f"{v:9.4f}" if v is not None else f"{'-':>9s}"
    d1 = f"{qt - qn:+7.4f}" if (qt is not None and qn is not None) else f"{'-':>7s}"
    d2 = f"{qt - qs:+7.4f}" if (qt is not None and qs is not None) else f"{'-':>7s}"
    r = out.get(("qt", ds)) or out.get(("qn", ds)) or {}
    print(f"{ds:20s} {s(qn)} {s(qt)[:10]:>10s} {s(qs)} {d1} {d2}  {r.get('n', '-')}")
print()
k = ("qt", "gtex")
if k in out:
    print("row keys:", list(out[k]))
    print("sample   :", json.dumps(out[k], ensure_ascii=False))
