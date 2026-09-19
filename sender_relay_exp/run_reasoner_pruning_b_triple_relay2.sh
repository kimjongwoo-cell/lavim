#!/usr/bin/env bash
set -euo pipefail

shard="${1:?usage: $0 SHARD_INDEX GPU}"
gpu="${2:?usage: $0 SHARD_INDEX GPU}"
repo=/home/users/whddn12316/wsi_latent_0915_decode_hj
data_root=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
python=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
runner="$repo/0912/run_dataset_nav2_prompt1.sh"
variant=pruning_b_reasoner_relay2_triple
output_root="$repo/sender_relay_exp/runs/pruning_b_reasoner_triple_relay2_prompt1"
shard_count=6

for dataset in tcga_expert_vqa tcga_slidebench tcga gtex panda; do
  count=$("$python" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$data_root/$dataset.json")
  selection=$("$python" -c '
import re, sys
from pathlib import Path
shard, count, shard_count = map(int, sys.argv[1:4])
path = Path(sys.argv[4]) / sys.argv[5] / f"gpu{sys.argv[6]}"
done = set()
for result in path.glob("*/result.json"):
    match = re.match(r"(\d+)_", result.parent.name)
    if match:
        done.add(int(match.group(1)))
print(",".join(str(index) for index in range(shard, count, shard_count) if index not in done))
' "$shard" "$count" "$shard_count" "$output_root" "$dataset" "$gpu")
  if [[ -z "$selection" ]]; then
    continue
  fi
  bash "$runner" "$gpu" "$dataset" "$variant" "$selection" "$output_root/$dataset/gpu$gpu"
done
