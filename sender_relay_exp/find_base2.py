#!/usr/bin/env python3
"""Exhaustive: which expq/consol_v dir scores exactly the dashboard base for gtex / evqa."""
import json, subprocess, glob, os
PY = "/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python"
SCORER = "/home/users/whddn12316/wsi_latent_0915_decode_hj/scripts/dashboard_rows.py"
E = "/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs"
JOBS = [("gtex", 0.2141, 190), ("tcga_expert_vqa", 0.4531, 128)]
for ds, tgt, want_n in JOBS:
    print(f"=== {ds}  target={tgt}")
    for d in sorted(glob.glob(os.path.join(E, "*", "*"))):
        if not os.path.isdir(d):
            continue
        if len(glob.glob(os.path.join(d, "*", "result.json"))) != want_n:
            continue
        r = subprocess.run([PY, SCORER, "--arm", d, "--dataset", ds, "--param", "4b",
                            "--method", "latent-base", "--step", "10", "--full"],
                           capture_output=True, text=True,
                           cwd="/home/users/whddn12316/wsi_latent_0915_decode_hj")
        if r.returncode != 0:
            continue
        try:
            rows = json.loads(r.stdout)
        except Exception:
            continue
        for row in rows:
            if abs(row["acc"] - tgt) < 0.00051:
                rel = os.path.relpath(d, E)
                print(f"   {rel:40s} acc={row['acc']}")
