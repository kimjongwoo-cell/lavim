#!/usr/bin/env bash
set -euo pipefail

shard="${1:?usage: $0 SHARD_INDEX GPU}"
gpu="${2:?usage: $0 SHARD_INDEX GPU}"
continuation="${3:-resume1}"
repo=/home/users/whddn12316/wsi_latent_0915_decode_hj
data_root=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
python=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
runner="$repo/0912/run_dataset_nav2_prompt1.sh"
variant=pruning_b_latent_kv_relay2_all_agents
base_root="$repo/sender_relay_exp/runs/pruning_b_all_agents_prompt1"
output_root="$repo/sender_relay_exp/runs/pruning_b_all_agents_prompt1_$continuation"

for dataset in tcga_expert_vqa tcga_slidebench tcga gtex panda; do
  count=$("$python" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$data_root/$dataset.json")
  selection=$("$python" -c '
import re, sys
from pathlib import Path
shard, count = map(int, sys.argv[1:3])
dataset, gpu = sys.argv[3:5]
roots = (Path(sys.argv[5]), Path(sys.argv[6]))
done = set()
for root in roots:
    path = root / dataset / f"gpu{gpu}"
    for result in path.glob("*/result.json"):
        match = re.match(r"(\d+)_", result.parent.name)
        if match:
            done.add(int(match.group(1)))
remaining = (index for index in range(shard, count, 5) if index not in done)
print(",".join(map(str, remaining)))
' "$shard" "$count" "$dataset" "$gpu" "$base_root" "$output_root")
  if [[ -z "$selection" ]]; then
    printf 'SKIP dataset=%s gpu=%s shard=%s complete\n' "$dataset" "$gpu" "$shard"
    continue
  fi
  printf 'START dataset=%s gpu=%s shard=%s cases=%s\n' "$dataset" "$gpu" "$shard" "$(awk -F, '{print NF}' <<< "$selection")"
  bash "$runner" "$gpu" "$dataset" "$variant" "$selection" "$output_root/$dataset/gpu$gpu"
done
