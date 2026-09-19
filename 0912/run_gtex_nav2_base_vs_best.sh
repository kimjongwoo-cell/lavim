#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?usage: $0 GPU FIRST LAST}"
first="${2:?usage: $0 GPU FIRST LAST}"
last="${3:?usage: $0 GPU FIRST LAST}"
repo=/home/users/whddn12316/wsi_latent_0915_decode_hj
data_root=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
python=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
model=/home/users/whddn12316/models/Qwen3-VL-4B-Thinking
variants=(base pruning_b_latent_kv_relay2_dual)
labels=(base pruning_b_dual_relay)
ids=()

for ((index=first; index<=last; index++)); do
  ids+=(--dataset-index "$index")
done

cd "$repo"
for pos in "${!variants[@]}"; do
  output="$repo/0912/runs/nav2v2_gtex_${labels[$pos]}_part$(printf '%03d' "$first")_$(printf '%03d' "$last")_gpu${gpu}_0913"
  env \
    CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH="$repo" \
    PYTHONUNBUFFERED=1 \
    VLMAS_ATTN_IMPLEMENTATION=sdpa \
    VLMAS_NAV2=1 \
    "$python" -m wsi_latentmas.pipeline.latent_mas \
    --variant "${variants[$pos]}" \
    --backbone qwen3-vl \
    --dataset "$data_root/gtex.json" \
    --slide-root "$data_root/slides" \
    --model "$model" \
    --output-root "$output" \
    --device cuda:0 \
    --latent-steps 10 \
    --patch-budget 12 \
    --max-model-len 12288 \
    --navigator-control-tokens 512 \
    --temperature 0.0 \
    --top-p 1.0 \
    --seed 42 \
    --deterministic \
    --answerer-greedy \
    --no-answerer-thinking \
    --answerer-max-new-tokens 512 \
    --answerer-rationale \
    --answerer-protocol structured_json \
    --canonical-open-options \
    --navigator-kv \
    --io-pipeline \
    --no-save-navigation-pngs \
    --case-retries 0 \
    "${ids[@]}" >"$output.log" 2>&1
done
