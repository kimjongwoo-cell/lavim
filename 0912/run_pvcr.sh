#!/usr/bin/env bash
set -euo pipefail

variant=${1:?spatial shuffled visual original off or native}
gpu=${2:?GPU index}
if [[ "$gpu" != 7 && "$gpu" != 8 ]]; then echo 'This experiment is restricted to GPU 7 or 8' >&2; exit 2; fi
name=${3:?new output name}
if [[ ! "$name" =~ ^[a-zA-Z0-9_-]+$ ]]; then echo 'Output name must be a simple directory name' >&2; exit 2; fi
shift 3
project_root=/home/users/whddn12316/wsi_latent_0915_decode_hj
python_bin=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
if [ "$#" -eq 0 ]; then echo 'Explicit dataset IDs are required' >&2; exit 2; fi
indices=()
for index in "$@"; do indices+=(--dataset-index "$index"); done
cd "$project_root"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="$project_root/0912:$project_root"
export PYTHONUNBUFFERED=1
export VLMAS_ATTN_IMPLEMENTATION=sdpa
export PVCR_VARIANT="$variant"
uv run --python "$python_bin" --no-project python 0912/run_pvcr.py \
  --variant base --backbone qwen3-vl --transport-mode cumulative \
  --dataset /home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda/tcga_expert_vqa.json \
  --slide-root /home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda/slides \
  --model /home/users/whddn12316/models/Qwen3-VL-4B-Thinking \
  --output-root "$project_root/0912/runs/$name" \
  --device cuda:0 --latent-steps 10 --patch-budget 12 --max-model-len 12288 \
  --navigator-control-tokens 512 --temperature 0.0 --top-p 1.0 --seed 42 \
  --deterministic --answerer-greedy --no-answerer-thinking \
  --answerer-max-new-tokens 512 --answerer-rationale \
  --answerer-protocol structured_json --canonical-open-options --navigator-kv \
  --io-pipeline --no-save-navigation-pngs --case-retries 0 "${indices[@]}"
