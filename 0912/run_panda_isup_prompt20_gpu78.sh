#!/usr/bin/env bash
set -euo pipefail

repo=/home/users/whddn12316/wsi_latent_0915_decode_hj
python=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
model=/home/users/whddn12316/models/Qwen3-VL-4B-Thinking
slides=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda/slides
ids=(0 1 2 3 4 5 6 8 10 11 12 13 14 15 17 20 24 36 37 39)
index_args=()
for id in "${ids[@]}"; do index_args+=(--dataset-index "$id"); done

run() {
  local gpu="$1" dataset="$2" output="$3"
  env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$repo" PYTHONUNBUFFERED=1 \
    VLMAS_PROMPT_SET=prompt1 VLMAS_ATTN_IMPLEMENTATION=sdpa \
    VLMAS_NAV3_INDEPENDENT=1 \
    "$python" -m wsi_latentmas.pipeline.latent_mas \
    --variant base --backbone qwen3-vl --dataset "$dataset" \
    --slide-root "$slides" --model "$model" --output-root "$output" \
    --device cuda:0 --latent-steps 10 --patch-budget 12 --max-model-len 12288 \
    --navigator-control-tokens 512 --temperature 0.0 --top-p 1.0 --seed 42 \
    --deterministic --answerer-greedy --no-answerer-thinking \
    --answerer-max-new-tokens 512 --answerer-rationale \
    --answerer-protocol structured_json --canonical-open-options \
    --navigator-kv --io-pipeline --no-save-navigation-pngs --case-retries 0 \
    "${index_args[@]}"
}

cd "$repo"
run 7 "$repo/0912/panda_prompt_experiment/panda_isup_prompt.json" \
  "$repo/sender_relay_exp/runs/nav3_panda_isup_prompt20_gpu7_20260914" \
  >"$repo/sender_relay_exp/runs/nav3_panda_isup_prompt20_gpu7_20260914.log" 2>&1 &
pid7=$!
run 8 "$repo/0912/panda_prompt_experiment/panda_isup_prompt_shuffled.json" \
  "$repo/sender_relay_exp/runs/nav3_panda_isup_prompt20_shuffled_gpu8_20260914" \
  >"$repo/sender_relay_exp/runs/nav3_panda_isup_prompt20_shuffled_gpu8_20260914.log" 2>&1 &
pid8=$!
wait "$pid7"
wait "$pid8"
