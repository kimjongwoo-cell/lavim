#!/usr/bin/env bash
set -euo pipefail

repo=/home/users/whddn12316/wsi_latent_0915_decode_hj
run_dir="$repo/0912/runs"
runner="$repo/0912/run_dataset_nav2_prompt2.sh"
gpus=(0 1 2 3 6 7 8)
datasets=(tcga tcga_slidebench panda)
sizes=(221 197 196)

run_condition() {
  local dataset="$1"
  local size="$2"
  local label="$3"
  local variant="$4"
  local worker first last gpu name

  for worker in "${!gpus[@]}"; do
    first=$((worker * size / ${#gpus[@]}))
    last=$((((worker + 1) * size / ${#gpus[@]}) - 1))
    gpu="${gpus[$worker]}"
    name="nav2_ra_prompt2_${dataset}_${label}_worker${worker}_gpu${gpu}_0913"
    bash "$runner" "$gpu" "$dataset" "$variant" "$first-$last" \
      "$run_dir/$name" >"$run_dir/$name.log" 2>&1 &
  done
  wait
}

for position in "${!datasets[@]}"; do
  dataset="${datasets[$position]}"
  size="${sizes[$position]}"
  run_condition "$dataset" "$size" base base
  run_condition "$dataset" "$size" both pruning_b_latent_kv_relay2_dual
done
