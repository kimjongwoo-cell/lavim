#!/usr/bin/env bash
set -euo pipefail

repo=/home/users/whddn12316/wsi_latent_0915_decode_hj
runner="$repo/0912/run_dataset_nav2_prompt1.sh"
run_root="$repo/0912/runs"
gpus=(6 7 8)

while pgrep -f 'python -m.*nav2_ra_prompt2_tcga_expert_vqa_both_dual_resume_gpu' >/dev/null 2>&1; do
  sleep 10
done

run_phase() {
  local dataset="$1" variant="$2" size="$3" label="$4"
  mapfile -t done_ids < <(
    find "$run_root" -maxdepth 3 -type f \
      -path "*/nav2_prompt1_${dataset}_${label}_*/*/result.json" \
      | sed -E 's#.*/([0-9]+)_.*#\1#' | sed 's/^0*//' | sed 's/^$/0/' | sort -n -u
  )
  declare -A done=()
  local -a buckets=() pids=()
  local id slot gpu output
  for id in "${done_ids[@]}"; do done["$id"]=1; done
  for slot in "${!gpus[@]}"; do buckets[$slot]=''; done
  slot=0
  for ((id=0; id<size; id++)); do
    [[ -n "${done[$id]:-}" ]] && continue
    buckets[$slot]+="${buckets[$slot]:+,}${id}"
    slot=$(((slot+1)%${#gpus[@]}))
  done
  echo "[$dataset/$label] complete=${#done_ids[@]} remaining=$((size-${#done_ids[@]}))"
  ((${#done_ids[@]} == size)) && return 0
  for slot in "${!gpus[@]}"; do
    [[ -z "${buckets[$slot]}" ]] && continue
    gpu="${gpus[$slot]}"
    output="$run_root/nav2_prompt1_${dataset}_${label}_resume_gpu${gpu}_0913"
    bash "$runner" "$gpu" "$dataset" "$variant" "${buckets[$slot]}" "$output" >"$output.log" 2>&1 &
    pids+=("$!")
  done
  for id in "${pids[@]}"; do wait "$id"; done
}

cd "$repo"
run_phase tcga base 221 base
run_phase tcga pruning_b_latent_kv_relay2_dual 221 dual_both
run_phase tcga_slidebench base 197 base
run_phase tcga_slidebench pruning_b_latent_kv_relay2_dual 197 dual_both
run_phase panda base 196 base
run_phase panda pruning_b_latent_kv_relay2_dual 196 dual_both
