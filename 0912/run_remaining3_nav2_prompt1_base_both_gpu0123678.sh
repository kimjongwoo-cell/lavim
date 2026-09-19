#!/usr/bin/env bash
set -euo pipefail

repo=/home/users/whddn12316/wsi_latent_0915_decode_hj
runner="$repo/0912/run_dataset_nav2_prompt1.sh"
run_root="$repo/0912/runs"
gpus=(0 1 2 3 6 7 8)

# Do not compete with the currently finishing ExpertVQA Prompt2 run.
while pgrep -f 'python -m .*nav2_ra_prompt2_tcga_expert_vqa_both_resume' >/dev/null 2>&1; do sleep 10; done

run_phase() {
  local dataset="$1" variant="$2" size="$3" label="$4"
  local -a buckets=() pids=()
  local slot id gpu output
  for slot in "${!gpus[@]}"; do buckets[$slot]=''; done
  slot=0
  for ((id=0; id<size; id++)); do
    buckets[$slot]+="${buckets[$slot]:+,}${id}"
    slot=$(((slot+1)%${#gpus[@]}))
  done
  echo "[$dataset/$label] total=$size"
  for slot in "${!gpus[@]}"; do
    gpu="${gpus[$slot]}"
    output="$run_root/nav2_prompt1_${dataset}_${label}_gpu${gpu}_0913"
    bash "$runner" "$gpu" "$dataset" "$variant" "${buckets[$slot]}" "$output" >"$output.log" 2>&1 &
    pids+=("$!")
  done
  for id in "${pids[@]}"; do wait "$id"; done
}

cd "$repo"
run_phase tcga base 221 base
run_phase tcga pruning_b_latent_kv_relay2_dual 221 both
run_phase tcga_slidebench base 197 base
run_phase tcga_slidebench pruning_b_latent_kv_relay2_dual 197 both
run_phase panda base 196 base
run_phase panda pruning_b_latent_kv_relay2_dual 196 both
