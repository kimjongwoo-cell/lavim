#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?usage: $0 GPU BASE_RANGES RELAY_RANGES}"
base_ranges="${2:?usage: $0 GPU BASE_RANGES RELAY_RANGES}"
relay_ranges="${3:?usage: $0 GPU BASE_RANGES RELAY_RANGES}"
repo=/home/users/whddn12316/wsi_latent_0915_decode_hj
run_dir="$repo/0912/runs"

cd "$repo"
bash 0912/run_gtex_nav2_sparse_resume.sh \
  "$gpu" base "$base_ranges" \
  "$run_dir/nav2v2_gtex_balanced_base_gpu${gpu}_0913" \
  >"$run_dir/nav2v2_gtex_balanced_base_gpu${gpu}_0913.log" 2>&1

bash 0912/run_gtex_nav2_sparse_resume.sh \
  "$gpu" pruning_b_latent_kv_relay2_dual "$relay_ranges" \
  "$run_dir/nav2v2_gtex_balanced_relay_gpu${gpu}_0913" \
  >"$run_dir/nav2v2_gtex_balanced_relay_gpu${gpu}_0913.log" 2>&1
