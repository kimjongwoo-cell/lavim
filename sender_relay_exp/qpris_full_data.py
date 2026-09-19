#!/usr/bin/env python3
"""Score the Q-PRIS full-set arm (qt_* under runs/qpris_full)."""
import json, glob, subprocess
K = "/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs/qpris_full"
PY = "/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python"
SCORER = "/home/users/whddn12316/wsi_latent_0915_decode_hj/scripts/dashboard_rows.py"
DSS = ("tcga_expert_vqa", "gtex", "tcga", "tcga_slidebench", "panda")

out = {}
for ds in DSS:
    d = f"{K}/qt_{ds}"
    done = len(glob.glob(f"{d}/*/result.json"))
    r = subprocess.run(
        [PY, SCORER, "--arm", d, "--dataset", ds, "--param", "4b",
         "--method", "latent-base", "--step", "10", "--full"],
        capture_output=True, text=True, cwd="/home/users/whddn12316/wsi_latent_0915_decode_hj")
    row = None
    if r.returncode == 0:
        try:
            rows = json.loads(r.stdout)
            row = rows[0] if rows else None
        except Exception:
            row = None
    out[ds] = {"cases_done": done,
               "n": row["n"] if row else None,
               "acc": row["acc"] if row else None,
               "sec": row["sec"] if row else None}
print(json.dumps(out, ensure_ascii=False, indent=1))
