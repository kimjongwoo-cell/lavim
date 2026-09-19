#!/usr/bin/env bash
set -euo pipefail

repo=/home/users/whddn12316/wsi_latent_0915_decode_hj
runner="$repo/0912/run_dataset_nav2_prompt1.sh"
run_root="$repo/0912/runs"
gpus=(0 1 2 3 6 7 8)

launch() {
  local gpu="$1" dataset="$2" ids="$3" output="$4"
  [[ -z "$ids" ]] && return 0
  echo "[$dataset/base] GPU $gpu: $(awk -F, '{print NF}' <<< "$ids") cases"
  bash "$runner" "$gpu" "$dataset" base "$ids" "$output" >"$output.log" 2>&1 &
  pids+=("$!")
}

completed_indices() {
  local dataset="$1"
  find "$run_root" -maxdepth 3 -type f \
    -path "$run_root/nav2_prompt1_${dataset}_base_*/*/result.json" \
    -print0 | xargs -0 -r jq -r '.dataset_index' | sort -n -u
}

declare -A tcga_done=() panda_done=()
while IFS= read -r index; do [[ -n "$index" ]] && tcga_done["$index"]=1; done \
  < <(completed_indices tcga)
while IFS= read -r index; do [[ -n "$index" ]] && panda_done["$index"]=1; done \
  < <(completed_indices panda)

declare -a tcga_buckets=("" "" "" "")
for ((index=0; index<221; index++)); do
  [[ -n "${tcga_done[$index]:-}" ]] && continue
  slot=$((index % 4))
  tcga_buckets[$slot]+="${tcga_buckets[$slot]:+,}$index"
done

declare -a panda_buckets=("" "" "" "" "" "" "")
for ((index=0; index<196; index++)); do
  [[ -n "${panda_done[$index]:-}" ]] && continue
  slot=$((index % 7))
  panda_buckets[$slot]+="${panda_buckets[$slot]:+,}$index"
done

echo "Reusing completed Prompt1 Base: GTEx 190/190, ExpertVQA 128/128, SlideBench 197/197."
echo "Resuming TCGA Base: ${#tcga_done[@]}/221; PANDA Base: ${#panda_done[@]}/196."

pids=()
for slot in 0 1 2 3; do
  gpu="${gpus[$slot]}"
  launch "$gpu" tcga "${tcga_buckets[$slot]}" \
    "$run_root/nav2_prompt1_tcga_base_tail2_gpu${gpu}_0913"
done
tcga_pids=("${pids[@]}")

for pid in "${tcga_pids[@]}"; do wait "$pid"; done

# Once the TCGA tail is clear, run all remaining PANDA shards on the requested
# seven GPUs. No pruning or relay variants are launched by this driver.
pids=()
for slot in 0 1 2 3 4 5 6; do
  gpu="${gpus[$slot]}"
  launch "$gpu" panda "${panda_buckets[$slot]}" \
    "$run_root/nav2_prompt1_panda_base_gpu${gpu}_0913"
done
panda_late_pids=("${pids[@]}")
for pid in "${panda_late_pids[@]}"; do wait "$pid"; done

echo "Prompt1 Base completed for all five datasets. No Dual Both phase is launched."
