#!/usr/bin/env bash
set -euo pipefail

project_root=/home/users/whddn12316/wsi_latent_0915_decode_hj
python_bin=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
mode=${1:?spatial or shuffled}
gpu=${2:?GPU index}
output_name=${3:-spatial_${mode}_pilot_gpu${gpu}}
shift 2
if [ "$#" -gt 0 ]; then shift; fi
indices=()
if [ "$#" -gt 0 ]; then
  for index in "$@"; do indices+=(--dataset-index "$index"); done
else
  for index in {0..120..8}; do indices+=(--dataset-index "$index"); done
fi

cd "$project_root"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="$project_root/0912:$project_root"
export PYTHONUNBUFFERED=1
export VLMAS_ATTN_IMPLEMENTATION=sdpa
export SPATIAL_MODE="$mode"
uv run --python "$python_bin" --no-project python 0912/run_spatial_latent.py \
  --variant base --backbone qwen3-vl \
  --dataset /home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda/tcga_expert_vqa.json \
  --slide-root /home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda/slides \
  --model /home/users/whddn12316/models/Qwen3-VL-4B-Thinking \
  --output-root "$project_root/0912/runs/$output_name" \
  --device cuda:0 --latent-steps 10 --patch-budget 12 --max-model-len 12288 \
  --navigator-control-tokens 512 --temperature 0.0 --top-p 1.0 --seed 42 \
  --deterministic --answerer-greedy --no-answerer-thinking \
  --answerer-max-new-tokens 512 --answerer-rationale \
  --answerer-protocol structured_json --canonical-open-options --navigator-kv \
  --io-pipeline --no-save-navigation-pngs --case-retries 0 "${indices[@]}"
