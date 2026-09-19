#!/usr/bin/env bash
set -euo pipefail

repo=/home/users/whddn12316/wsi_latent_0915_decode_hj
runner="$repo/0912/run_dataset_nav2_prompt1.sh"
run_root="$repo/sender_relay_exp/runs"
dataset=tcga
variant=pruning_b_latent_kv_relay2_dual
stamp=20260914
prefix="nav3_prompt1_pruning_b_both_${dataset}_"

cd "$repo"
python - "$run_root" "$prefix" <<'PY'
import glob, json, os, sys
root, prefix = sys.argv[1:]
found = []
for path in glob.glob(os.path.join(root, prefix + "*", "run_manifest.json")):
    try:
        m = json.load(open(path))
        found.extend(m.get("requested_indices", []))
    except Exception:
        pass
if found:
    print(f"Existing Nav3 Pruning-B manifests found ({len(set(found))} indices); inspect before resuming.")
    raise SystemExit(2)
PY

count=$(python - <<'PY'
import json
data = json.load(open("/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda/tcga.json"))
print(len(data))
PY
)
ids7=$(python - "$count" <<'PY'
import sys
n = int(sys.argv[1])
print(",".join(str(i) for i in range(n) if i % 2 == 0))
PY
)
ids8=$(python - "$count" <<'PY'
import sys
n = int(sys.argv[1])
print(",".join(str(i) for i in range(n) if i % 2 == 1))
PY
)

out7="$run_root/${prefix}gpu7_${stamp}"
out8="$run_root/${prefix}gpu8_${stamp}"
echo "START dataset=$dataset method=Pruning-B+Dual-Relay nav=Nav3 prompt=Prompt1 cases=$count gpu7=$(awk -F, '{print NF}' <<< "$ids7") gpu8=$(awk -F, '{print NF}' <<< "$ids8")"

VLMAS_NAV3_INDEPENDENT=1 VLMAS_NAV2=0 bash "$runner" 7 "$dataset" "$variant" "$ids7" "$out7" >"$out7.log" 2>&1 &
pid7=$!
VLMAS_NAV3_INDEPENDENT=1 VLMAS_NAV2=0 bash "$runner" 8 "$dataset" "$variant" "$ids8" "$out8" >"$out8.log" 2>&1 &
pid8=$!
wait "$pid7"
wait "$pid8"
echo "DONE dataset=$dataset method=Pruning-B+Dual-Relay nav=Nav3 cases=$count"
