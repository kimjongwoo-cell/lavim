#!/usr/bin/env bash
set -euo pipefail

repo=/home/users/whddn12316/wsi_latent_0915_decode_hj
data_root=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
python=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
runner="$repo/0912/run_dataset_nav2_prompt1.sh"
gpus=(7 8)
shard_count=${#gpus[@]}
datasets=(tcga_expert_vqa tcga_slidebench tcga gtex panda)
variants=(pruning_b_reasoner_context_relay3)
run_stamp=$(date -u +%Y%m%dT%H%M%SZ)

for dataset in "${datasets[@]}"; do
  count=$("$python" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$data_root/$dataset.json")
  for variant in "${variants[@]}"; do
    output_root="$repo/sender_relay_exp/runs/${variant}_prompt1"
    pids=()
    for shard in "${!gpus[@]}"; do
      gpu=${gpus[$shard]}
      selection=$("$python" -c '
import re, sys
from pathlib import Path
shard, count, shard_count = map(int, sys.argv[1:4])
path = Path(sys.argv[4]) / sys.argv[5]
done = set()
for result in path.glob("gpu*/**/result.json"):
    match = re.match(r"(\d+)_", result.parent.name)
    if match:
        done.add(int(match.group(1)))
print(",".join(str(index) for index in range(shard, count, shard_count) if index not in done))
' "$shard" "$count" "$shard_count" "$output_root" "$dataset" "$gpu")
      if [[ -z "$selection" ]]; then
        printf 'SKIP dataset=%s variant=%s gpu=%s (shard complete)\n' "$dataset" "$variant" "$gpu"
        continue
      fi
      log="$repo/sender_relay_exp/runs/logs/${dataset}_${variant}_gpu${gpu}.log"
      mkdir -p "$(dirname "$log")"
      printf 'START dataset=%s variant=%s gpu=%s cases=%s\n' \
        "$dataset" "$variant" "$gpu" "$(awk -F, '{print NF}' <<< "$selection")"
      (
        bash "$runner" "$gpu" "$dataset" "$variant" "$selection" "$output_root/$dataset/gpu${gpu}_${run_stamp}"
      ) >"$log" 2>&1 &
      pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do
      if ! wait "$pid"; then failed=1; fi
    done
    if (( failed )); then
      printf 'INCOMPLETE dataset=%s variant=%s; rerun to resume missing indices\n' "$dataset" "$variant"
      exit 1
    fi
    printf 'DONE dataset=%s variant=%s\n' "$dataset" "$variant"
  done
done
