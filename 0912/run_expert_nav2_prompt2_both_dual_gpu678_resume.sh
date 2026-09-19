#!/usr/bin/env bash
set -euo pipefail

repo=/home/users/whddn12316/wsi_latent_0915_decode_hj
runner="$repo/0912/run_dataset_nav2_prompt2.sh"
run_root="$repo/0912/runs"
gpus=(6 7 8)
mapfile -t done_ids < <(
  find "$run_root" -maxdepth 3 -type f \
    -path '*/nav2_ra_prompt2_tcga_expert_vqa_both_dual_gpu*_0913/*/result.json' \
    | sed -E 's#.*/([0-9]+)_.*#\1#' | sed 's/^0*//' | sed 's/^$/0/' | sort -n -u
)
declare -A done=()
declare -a buckets=() pids=()
for id in "${done_ids[@]}"; do done["$id"]=1; done
for slot in "${!gpus[@]}"; do buckets[$slot]=''; done
slot=0
for ((id=0; id<128; id++)); do
  [[ -n "${done[$id]:-}" ]] && continue
  buckets[$slot]+="${buckets[$slot]:+,}${id}"
  slot=$(((slot+1)%${#gpus[@]}))
done
echo "complete=${#done_ids[@]} remaining=$((128-${#done_ids[@]}))"
cd "$repo"
for slot in "${!gpus[@]}"; do
  gpu="${gpus[$slot]}"
  output="$run_root/nav2_ra_prompt2_tcga_expert_vqa_both_dual_resume_gpu${gpu}_0913"
  bash "$runner" "$gpu" tcga_expert_vqa pruning_b_latent_kv_relay2_dual \
    "${buckets[$slot]}" "$output" >"$output.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
