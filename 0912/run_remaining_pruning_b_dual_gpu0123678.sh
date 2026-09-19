#!/usr/bin/env bash
set -euo pipefail

repo=/home/users/whddn12316/wsi_latent_0915_decode_hj
runner="$repo/0912/run_dataset_nav2_prompt1.sh"
indexer="$repo/0912/completed_indices.py"
python=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
run_root="$repo/0912/runs"
gpus=(0 1 2 3 6 7 8)
datasets=(tcga_slidebench tcga panda)
sizes=(197 221 196)
variant=pruning_b_latent_kv_relay2_dual
stamp="$(date +%H%M%S)"

run_dataset() {
  local dataset="$1" size="$2"
  local prefix="nav2_prompt1_pruning_b_dual_both_${dataset}_"
  local -A completed=()
  local -a buckets=("" "" "" "" "" "" "") pids=()

  while IFS= read -r index; do
    [[ -n "$index" ]] && completed[$index]=1
  done < <("$python" "$indexer" "$run_root" "$prefix" "$dataset")

  for ((index=0; index<size; index++)); do
    [[ -n "${completed[$index]:-}" ]] && continue
    slot=$((index % ${#gpus[@]}))
    buckets[$slot]+="${buckets[$slot]:+,}$index"
  done

  for slot in "${!gpus[@]}"; do
    [[ -z "${buckets[$slot]}" ]] && continue
    gpu="${gpus[$slot]}"
    output="$run_root/${prefix}shard_gpu${gpu}_${stamp}"
    echo "[$dataset/B+Dual] GPU $gpu: $(awk -F, '{print NF}' <<< "${buckets[$slot]}") cases"
    bash "$runner" "$gpu" "$dataset" "$variant" "${buckets[$slot]}" "$output" \
      >"$output.log" 2>&1 &
    pids+=("$!")
  done

  status=0
  for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
  if (( status != 0 )); then
    echo "[$dataset/B+Dual] a shard failed; stopping sequence" >&2
    return "$status"
  fi
  echo "[$dataset/B+Dual] complete"
}

for slot in "${!datasets[@]}"; do
  run_dataset "${datasets[$slot]}" "${sizes[$slot]}"
done
