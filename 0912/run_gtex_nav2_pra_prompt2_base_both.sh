#!/usr/bin/env bash
set -euo pipefail

repo=/home/users/whddn12316/wsi_latent_0915_decode_hj
run_dir="$repo/0912/runs"
runner="$repo/0912/run_gtex_nav2_sparse_resume.sh"
gpus=(0 1 2 3 6 7 8)
ranges=(0-27 28-54 55-81 82-108 109-135 136-162 163-189)

run_condition() {
  local label="$1"
  local variant="$2"

  for worker in "${!gpus[@]}"; do
    gpu="${gpus[$worker]}"
    selection="${ranges[$worker]}"
    name="nav2_navguard_pra_prompt2_gtex_${label}_worker${worker}_gpu${gpu}_0913"
    VLMAS_PROMPT_SET=prompt2 bash "$runner" \
      "$gpu" "$variant" "$selection" "$run_dir/$name" \
      >"$run_dir/$name.log" 2>&1 &
  done
  wait
}

run_condition base base
run_condition both pruning_b_latent_kv_relay2_dual
