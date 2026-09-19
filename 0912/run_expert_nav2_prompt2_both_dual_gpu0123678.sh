#!/usr/bin/env bash
set -euo pipefail

repo=/home/users/whddn12316/wsi_latent_0915_decode_hj
runner="$repo/0912/run_dataset_nav2_prompt2.sh"
run_root="$repo/0912/runs"
gpus=(0 1 2 3 6 7 8)
declare -a buckets=() pids=()

for slot in "${!gpus[@]}"; do buckets[$slot]=''; done
for ((index=0; index<128; index++)); do
  slot=$((index % ${#gpus[@]}))
  buckets[$slot]+="${buckets[$slot]:+,}${index}"
done

cd "$repo"
for slot in "${!gpus[@]}"; do
  gpu="${gpus[$slot]}"
  output="$run_root/nav2_ra_prompt2_tcga_expert_vqa_both_dual_gpu${gpu}_0913"
  bash "$runner" "$gpu" tcga_expert_vqa \
    pruning_b_latent_kv_relay2_dual "${buckets[$slot]}" "$output" \
    >"$output.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
