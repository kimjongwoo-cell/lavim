#!/usr/bin/env python3
"""Emit one JSON blob for the board's Q-PRIS section: per-arm score + arm agreement."""
import json, glob, os, subprocess
K = "/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs/qpris"
PY = "/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python"
SCORER = "/home/users/whddn12316/wsi_latent_0915_decode_hj/scripts/dashboard_rows.py"
DSS = ("tcga_expert_vqa", "gtex", "tcga", "tcga_slidebench", "panda")
ARMS = ("qn", "qt", "qs")


def score(arm, ds):
    r = subprocess.run(
        [PY, SCORER, "--arm", f"{K}/{arm}_{ds}", "--dataset", ds, "--param", "4b",
         "--method", "latent-base", "--step", "10", "--full"],
        capture_output=True, text=True, cwd="/home/users/whddn12316/wsi_latent_0915_decode_hj")
    if r.returncode != 0:
        return None
    try:
        rows = json.loads(r.stdout)
    except Exception:
        return None
    return rows[0] if rows else None


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
        out[r.get("dataset_index", os.path.basename(os.path.dirname(f)))] = str(a).strip()
    return out


blob = {}
for ds in DSS:
    row = {"dataset": ds}
    for arm in ARMS:
        s = score(arm, ds)
        row[arm] = None if s is None else {"n": s["n"], "acc": s["acc"], "sec": s["sec"]}
    p = {a: preds(a, ds) for a in ARMS}
    keys = sorted(set(p["qn"]) & set(p["qt"]) & set(p["qs"]))
    row["overlap"] = {
        "n": len(keys),
        "qn_qt": sum(p["qn"][k] == p["qt"][k] for k in keys),
        "qt_qs": sum(p["qt"][k] == p["qs"][k] for k in keys),
        "qn_qs": sum(p["qn"][k] == p["qs"][k] for k in keys),
    } if keys else None
    blob[ds] = row
print(json.dumps(blob, ensure_ascii=False, indent=1))
