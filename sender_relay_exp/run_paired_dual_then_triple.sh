#!/usr/bin/env bash
set -euo pipefail

repo=/home/users/whddn12316/wsi_latent_0915_decode_hj
data_root=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
python=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
runner="$repo/0912/run_dataset_nav2_prompt1.sh"
output_root="$repo/sender_relay_exp/runs/pruning_b_reasoner_dual_triple_paired_prompt1"
gpus=(0 1 2 3 6 7 8)
datasets=(tcga_expert_vqa tcga_slidebench tcga gtex panda)

run_phase() {
  local dataset="$1" variant="$2"
  local phase_log="$output_root/_logs/$dataset/$variant"
  mkdir -p "$phase_log"
  local count
  count=$("$python" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$data_root/$dataset.json")
  local attempt_id
  attempt_id=$(date '+%Y%m%d_%H%M%S')
  local -a pids=()
  for ordinal in "${!gpus[@]}"; do
    local gpu="${gpus[$ordinal]}"
    local selection
    selection=$("$python" - "$ordinal" "$count" "${#gpus[@]}" "$output_root" "$dataset" "$variant" <<'PY'
import json, sys
from pathlib import Path
shard, count, workers = map(int, sys.argv[1:4])
root, dataset, variant = Path(sys.argv[4]), sys.argv[5], sys.argv[6]
done = set()
for manifest in (root / dataset).rglob("run_manifest.json"):
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
indices = [i for i in range(shard, count, workers) if i not in done]
print(",".join(map(str, indices)))
PY
)
    if [[ -z "$selection" ]]; then
      continue
    fi
    echo "[$(date '+%F %T')] START dataset=$dataset variant=$variant gpu=$gpu ids=$selection"
    bash "$runner" "$gpu" "$dataset" "$variant" "$selection" "$output_root/$dataset/$variant/attempt_$attempt_id/gpu$gpu" \
      >"$phase_log/gpu$gpu.log" 2>&1 &
    pids+=("$!")
  done

  local failed=0 pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if (( failed )); then
    echo "[$(date '+%F %T')] FAILED dataset=$dataset variant=$variant; inspect $phase_log" >&2
    return 1
  fi
  echo "[$(date '+%F %T')] DONE dataset=$dataset variant=$variant"
}

for dataset in "${datasets[@]}"; do
  run_phase "$dataset" pruning_b_reasoner_relay2_dual_paired
  run_phase "$dataset" pruning_b_reasoner_relay2_triple_paired
done

echo "[$(date '+%F %T')] ALL DATASETS COMPLETE"
