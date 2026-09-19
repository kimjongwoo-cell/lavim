#!/usr/bin/env bash
set -euo pipefail

project_root=/home/users/whddn12316/wsi_latent_0915_decode_hj
venv_python=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
dataset=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda/tcga_expert_vqa.json
slide_root=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda/slides
model=/home/users/whddn12316/models/Qwen3-VL-4B-Thinking
base_manifest="$project_root/runs/expertvqa_latent_base_x5_4_x20_8_latent10_gpu8_mlen12288/run_manifest.json"
output_root="$project_root/runs/expertvqa_reallocation_a_direct_relay_x5_4_x20_8_latent10_gpu8_mlen12288_full"

cd "$project_root"
export CUDA_VISIBLE_DEVICES=8
exec "$venv_python" -m wsi_latentmas.pipeline.latent_mas \
  --variant reallocation_a \
  --backbone qwen3-vl \
  --dataset "$dataset" \
  --slide-root "$slide_root" \
  --model "$model" \
  --output-root "$output_root" \
  --resume \
  --expected-base-manifest "$base_manifest" \
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
  --case-retries 0
