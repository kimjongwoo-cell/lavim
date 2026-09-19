#!/usr/bin/env bash
set -euo pipefail

repo=/home/users/whddn12316/wsi_latent_0915_decode_hj
python=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
data=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
model=/home/users/whddn12316/models/Qwen3-VL-4B-Thinking
runs="$repo/sender_relay_exp/runs"

run_shard() {
  local gpu="$1" ids="$2" output="$3"
  local args=()
  IFS=',' read -r -a selected <<<"$ids"
  for index in "${selected[@]}"; do args+=(--dataset-index "$index"); done
  env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$repo" PYTHONUNBUFFERED=1 \
    VLMAS_PROMPT_SET=prompt1 VLMAS_NAV3_INDEPENDENT=1 VLMAS_NAV2=0 \
    VLMAS_ATTN_IMPLEMENTATION=sdpa MATCHED_VLMAS_PATCH_BUDGET=12 \
    "$python" -m wsi_latentmas.pipeline.text_mas \
    --dataset "$data/panda.json" --slide-root "$data/slides" \
    --model "$model" --output-root "$output" --device cuda:0 \
    --max-model-len 12288 --temperature 0.0 --top-p 1.0 --seed 42 \
    --answerer-max-new-tokens 512 --answerer-protocol structured_json \
    --canonical-open-options --fast-json --io-pipeline "${args[@]}"
}

run_shard 7 0,2,4,6,10,12,14,17,24,37 "$runs/nav3_prompt1_vlmas_panda20_gpu7_20260914" \
  >"$runs/nav3_prompt1_vlmas_panda20_gpu7_20260914.log" 2>&1 &
pid7=$!
run_shard 8 1,3,5,8,11,13,15,20,36,39 "$runs/nav3_prompt1_vlmas_panda20_gpu8_20260914" \
  >"$runs/nav3_prompt1_vlmas_panda20_gpu8_20260914.log" 2>&1 &
pid8=$!
wait "$pid7"
wait "$pid8"
