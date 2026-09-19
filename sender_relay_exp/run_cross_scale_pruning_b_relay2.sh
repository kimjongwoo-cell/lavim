#!/usr/bin/env bash
set -euo pipefail

repo=/home/users/whddn12316/wsi_latent_0915_decode_hj
data_root=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
python=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
runner="$repo/0912/run_dataset_nav2_prompt1.sh"
output_root="$repo/sender_relay_exp/runs/cross_scale_router_pruning_b_relay2_prompt1"
# Use only currently free GPUs; do not contend with jobs already on 1-3.
gpus=(0 6 7 8)
datasets=(tcga_expert_vqa tcga_slidebench tcga gtex panda)
variant=pruning_b_cross_scale_relay2

for dataset in "${datasets[@]}"; do
  phase_log="$output_root/_logs/$dataset"
  mkdir -p "$phase_log"
  count=$("$python" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$data_root/$dataset.json")
  attempt_id=$(date '+%Y%m%d_%H%M%S')
  pids=()
  for ordinal in "${!gpus[@]}"; do
    gpu="${gpus[$ordinal]}"
    selection=$("$python" - "$ordinal" "$count" "${#gpus[@]}" "$output_root" "$dataset" "$variant" <<'PY'
import json, sys
from pathlib import Path
shard, count, workers = map(int, sys.argv[1:4])
root, dataset, variant = Path(sys.argv[4]), sys.argv[5], sys.argv[6]
done = set()
for manifest in (root / dataset).rglob("run_manifest.json") if (root / dataset).exists() else ():
    try:
        if json.loads(manifest.read_text()).get("variant") != variant:
            continue
    except (OSError, json.JSONDecodeError):
        continue
    for result in manifest.parent.glob("*/result.json"):
        try:
            done.add(int(json.loads(result.read_text())["dataset_index"]))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
print(",".join(str(i) for i in range(shard, count, workers) if i not in done))
PY
)
    if [[ -z "$selection" ]]; then
      continue
    fi
    echo "[$(date '+%F %T')] START dataset=$dataset variant=$variant gpu=$gpu ids=$selection"
    bash "$runner" "$gpu" "$dataset" "$variant" "$selection" \
      "$output_root/$dataset/$variant/attempt_$attempt_id/gpu$gpu" \
      >"$phase_log/gpu$gpu.log" 2>&1 &
    pids+=("$!")
  done

  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  if (( failed )); then
    echo "[$(date '+%F %T')] FAILED dataset=$dataset; inspect $phase_log" >&2
    exit 1
  fi
  echo "[$(date '+%F %T')] DONE dataset=$dataset"
done

echo "[$(date '+%F %T')] ALL DATASETS COMPLETE"
