#!/usr/bin/env bash
set -euo pipefail

repo=/home/users/whddn12316/wsi_latent_0915_decode_hj
run_dir="$repo/0912/runs"
runner="$repo/0912/run_gtex_nav2_sparse_resume.sh"

launch() {
  local gpu="$1"
  local variant="$2"
  local indices="$3"
  local name="$4"

  VLMAS_PROMPT_SET=prompt2 bash "$runner" \
    "$gpu" "$variant" "$indices" "$run_dir/$name" \
    >"$run_dir/$name.log" 2>&1
}

# Finish the complete Base condition first, using all requested GPUs.
launch 0 base 0-27 nav2_prompt2_gtex_base7_part000_027_gpu0_0913 &
launch 1 base 28-54 nav2_prompt2_gtex_base7_part028_054_gpu1_0913 &
launch 2 base 55-81 nav2_prompt2_gtex_base7_part055_081_gpu2_0913 &
launch 3 base 82-108 nav2_prompt2_gtex_base7_part082_108_gpu3_0913 &
launch 6 base 109-135 nav2_prompt2_gtex_base7_part109_135_gpu6_0913 &
launch 7 base 136-162 nav2_prompt2_gtex_base7_part136_162_gpu7_0913 &
launch 8 base 163-189 nav2_prompt2_gtex_base7_part163_189_gpu8_0913 &
wait

# Only after Base is complete, run the complete Both condition on the same GPUs.
launch 0 pruning_b_latent_kv_relay2_dual 0-27 \
  nav2_prompt2_gtex_both7_part000_027_gpu0_0913 &
launch 1 pruning_b_latent_kv_relay2_dual 28-54 \
  nav2_prompt2_gtex_both7_part028_054_gpu1_0913 &
launch 2 pruning_b_latent_kv_relay2_dual 55-81 \
  nav2_prompt2_gtex_both7_part055_081_gpu2_0913 &
launch 3 pruning_b_latent_kv_relay2_dual 82-108 \
  nav2_prompt2_gtex_both7_part082_108_gpu3_0913 &
launch 6 pruning_b_latent_kv_relay2_dual 109-135 \
  nav2_prompt2_gtex_both7_part109_135_gpu6_0913 &
launch 7 pruning_b_latent_kv_relay2_dual 136-162 \
  nav2_prompt2_gtex_both7_part136_162_gpu7_0913 &
launch 8 pruning_b_latent_kv_relay2_dual 163-189 \
  nav2_prompt2_gtex_both7_part163_189_gpu8_0913 &
wait
