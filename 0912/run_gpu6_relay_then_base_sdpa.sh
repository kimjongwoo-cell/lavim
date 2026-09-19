#!/usr/bin/env bash
set -euo pipefail

project_root=/home/users/whddn12316/wsi_latent_0915_decode_hj
python_bin=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
dataset=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda/tcga_expert_vqa.json
slide_root=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda/slides
model=/home/users/whddn12316/models/Qwen3-VL-4B-Thinking
common=(--backbone qwen3-vl --dataset "$dataset" --slide-root "$slide_root" --model "$model" --device cuda:0 --latent-steps 10 --patch-budget 12 --max-model-len 12288 --navigator-control-tokens 512 --temperature 0.0 --top-p 1.0 --seed 42 --deterministic --answerer-greedy --no-answerer-thinking --answerer-max-new-tokens 512 --answerer-rationale --answerer-protocol structured_json --canonical-open-options --navigator-kv --io-pipeline --no-save-navigation-pngs --case-retries 0)
relay_indices=(72 74 76 78 80 82 84 86 88 90 92 94 96 98 100 102 104 106 108 110 112 114 116 118 120 122 124 126)
relay_args=()
for index in "${relay_indices[@]}"; do relay_args+=(--dataset-index "$index"); done

cd "$project_root"
export CUDA_VISIBLE_DEVICES=6
export VLMAS_ATTN_IMPLEMENTATION=sdpa
export PYTHONPATH="$project_root"
export PYTHONUNBUFFERED=1

"$python_bin" -m wsi_latentmas.pipeline.latent_mas --variant latent_kv_relay --output-root "$project_root/0912/runs/relay_remaining_gpu6" "${common[@]}" "${relay_args[@]}"
"$python_bin" -m wsi_latentmas.pipeline.latent_mas --variant base --output-root "$project_root/0912/runs/base_sdpa_gpu6" "${common[@]}"
