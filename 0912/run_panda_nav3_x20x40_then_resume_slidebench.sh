#!/usr/bin/env bash
set -euo pipefail

repo=/home/users/whddn12316/wsi_latent_0915_decode_hj
runner="$repo/0912/run_dataset_nav2_prompt1.sh"
runs="$repo/sender_relay_exp/runs"
ids7=0,2,4,6,10,12,14,17,24,37
ids8=1,3,5,8,11,13,15,20,36,39

VLMAS_NAV3_INDEPENDENT=1 VLMAS_NAV3_X20_X40=1 VLMAS_NAV2=0 \
  bash "$runner" 7 panda base "$ids7" \
  "$runs/nav3_x20x40_prompt1_base_panda_gpu7_20260914" \
  >"$runs/nav3_x20x40_prompt1_base_panda_gpu7_20260914.log" 2>&1 &
pid7=$!
VLMAS_NAV3_INDEPENDENT=1 VLMAS_NAV3_X20_X40=1 VLMAS_NAV2=0 \
  bash "$runner" 8 panda base "$ids8" \
  "$runs/nav3_x20x40_prompt1_base_panda_gpu8_20260914" \
  >"$runs/nav3_x20x40_prompt1_base_panda_gpu8_20260914.log" 2>&1 &
pid8=$!

failed=0
wait "$pid7" || failed=1
wait "$pid8" || failed=1
(( failed == 0 )) || exit 1

exec bash "$repo/0912/run_nav3_prompt1_slidebench_base_both_gpu78.sh"
