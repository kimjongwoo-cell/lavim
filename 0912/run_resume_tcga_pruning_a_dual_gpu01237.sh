#!/usr/bin/env bash
set -euo pipefail

repo=/home/users/whddn12316/wsi_latent_0915_decode_hj
run_root="$repo/0912/runs"
runner="$repo/0912/run_dataset_nav2_prompt1.sh"
python=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
gpus=(0 1 2 3 6 7 8)
dataset=tcga
variant=pruning_a_latent_kv_relay2_dual
stamp="$(date +%H%M%S)"
declare -A completed=()
declare -a buckets=("" "" "" "" "" "" "") pids=()

while IFS= read -r index; do
  [[ -n "$index" ]] && completed[$index]=1
done < <(
  "$python" "$repo/0912/completed_indices.py" "$run_root" \
    nav2_prompt1_pruning_a_dual_both_tcga_ tcga
)

for ((index=0; index<221; index++)); do
  [[ -n "${completed[$index]:-}" ]] && continue
  slot=$((index % ${#gpus[@]}))
  buckets[$slot]+="${buckets[$slot]:+,}$index"
done

for slot in "${!gpus[@]}"; do
  [[ -z "${buckets[$slot]}" ]] && continue
  gpu="${gpus[$slot]}"
  output="$run_root/nav2_prompt1_pruning_a_dual_both_tcga_resume_gpu${gpu}_${stamp}"
  echo "[tcga/resume] GPU $gpu: $(awk -F, '{print NF}' <<< "${buckets[$slot]}") cases"
  bash "$runner" "$gpu" "$dataset" "$variant" "${buckets[$slot]}" "$output" \
    >"$output.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
exit "$status"
