#!/usr/bin/env bash
set -euo pipefail

repo=/home/users/whddn12316/wsi_latent_0915_decode_hj
runner="$repo/0912/run_dataset_nav2_prompt2.sh"
run_root="$repo/0912/runs"

run_variant() {
  local variant="$1" label="$2"
  local pids=()
  local gpu first last ids output index
  for gpu in 0 1 2 3; do
    first=$((gpu * 32))
    last=$((first + 31))
    ids=''
    for ((index=first; index<=last; index++)); do
      ids+="${ids:+,}${index}"
    done
    output="$run_root/nav2_ra_prompt2_tcga_expert_vqa_${label}_part$(printf '%03d' "$first")_$(printf '%03d' "$last")_gpu${gpu}_0913"
    bash "$runner" "$gpu" tcga_expert_vqa "$variant" "$ids" "$output" >"$output.log" 2>&1 &
    pids+=("$!")
  done
  for index in "${pids[@]}"; do wait "$index"; done
}

cd "$repo"
run_variant base base
run_variant pruning_b_latent_kv_relay2 both
