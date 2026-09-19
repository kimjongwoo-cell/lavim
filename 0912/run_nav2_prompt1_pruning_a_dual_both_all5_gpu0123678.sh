#!/usr/bin/env bash
set -euo pipefail

repo=/home/users/whddn12316/wsi_latent_0915_decode_hj
runner="$repo/0912/run_dataset_nav2_prompt1.sh"
completed_indexer="$repo/0912/completed_indices.py"
run_root="$repo/0912/runs"
datasets=(tcga_expert_vqa gtex tcga_slidebench tcga panda)
sizes=(128 190 197 221 196)
gpus=(0 1 2 3 6 7 8)
variant=pruning_a_latent_kv_relay2_dual
stamp="$(date +%H%M%S)"
run_dataset_root() {
  local dataset="$1"
  local size="$2"
  local -a buckets=()
  local -A completed=()
  local -a pids=()

  for slot in "${!gpus[@]}"; do buckets[$slot]=''; done
  while IFS= read -r index; do
    [[ -n "$index" ]] && completed[$index]=1
  done < <(
    /home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python \
      "$completed_indexer" "$run_root" \
      "nav2_prompt1_pruning_a_dual_both_${dataset}_" "$dataset"
  )

  for ((index=0; index<size; index++)); do
    [[ -n "${completed[$index]:-}" ]] && continue
    slot=$((index % ${#gpus[@]}))
    buckets[$slot]+="${buckets[$slot]:+,}$index"
  done

  for slot in "${!gpus[@]}"; do
    [[ -z "${buckets[$slot]}" ]] && continue
    gpu="${gpus[$slot]}"
    output="$run_root/nav2_prompt1_pruning_a_dual_both_${dataset}_shard_gpu${gpu}_${stamp}"
    echo "[$dataset] shard GPU $gpu, $(awk -F, '{print NF}' <<< "${buckets[$slot]}") cases"
    bash "$runner" "$gpu" "$dataset" "$variant" "${buckets[$slot]}" "$output" \
      >"$output.log" 2>&1 &
    pids+=("$!")
  done

  status=0
  for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
  if (( status != 0 )); then
    echo "[$dataset] one or more shards failed; stopping before the next dataset" >&2
    return "$status"
  fi
  echo "[$dataset] all shards finished; continuing to the next dataset"
}

for slot in "${!datasets[@]}"; do
  run_dataset_root "${datasets[$slot]}" "${sizes[$slot]}"
done
