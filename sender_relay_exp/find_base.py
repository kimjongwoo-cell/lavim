#!/usr/bin/env python3
"""Find which expq run dir reproduces each dashboard Latent base(step10, 4b) value."""
import json, subprocess, glob, os
PY = "/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python"
SCORER = "/home/users/whddn12316/wsi_latent_0915_decode_hj/scripts/dashboard_rows.py"
E = "/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs"
TARGET = {"gtex": 0.2141, "tcga_expert_vqa": 0.4531, "tcga": 0.0733,
          "tcga_slidebench": 0.4822, "panda": 0.1826}
PAT = {"gtex": ["expq/gtex10_*", "expq/*gtex10", "consol_v/*gtex"],
       "tcga_expert_vqa": ["expq/*evqa10*", "expq/evqa*", "consol_v/*expert_vqa"],
       "tcga": ["expq/*tcga10*", "expq/tcga_*", "consol_v/*tcga"],
       "tcga_slidebench": ["expq/*sb10*", "expq/sb_*", "consol_v/*slidebench"],
       "panda": ["expq/*panda10*", "expq/panda_*", "consol_v/*panda"]}

for ds, tgt in TARGET.items():
    cands = []
    for p in PAT[ds]:
        cands += glob.glob(os.path.join(E, p))
    seen = set()
    for d in sorted(set(cands)):
        if not os.path.isdir(d) or d in seen:
            continue
        seen.add(d)
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
            if abs(row["acc"] - tgt) < 0.0006:
                print(f"{ds:18s} target={tgt} -> {os.path.basename(d):26s} acc={row['acc']} n={row['n']}")
