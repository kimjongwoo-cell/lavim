#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?usage: $0 GPU DATASET VARIANT IDS OUTPUT_ROOT}"
dataset="${2:?usage: $0 GPU DATASET VARIANT IDS OUTPUT_ROOT}"
variant="${3:?usage: $0 GPU DATASET VARIANT IDS OUTPUT_ROOT}"
selection="${4:?usage: $0 GPU DATASET VARIANT IDS OUTPUT_ROOT}"
output="${5:?usage: $0 GPU DATASET VARIANT IDS OUTPUT_ROOT}"
repo=/home/users/whddn12316/wsi_latent_0915_decode_hj
data_root=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
python=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
model=/home/users/whddn12316/models/Qwen3-VL-4B-Thinking
ids=()
cross_scale_router=0
if [[ "$variant" == "pruning_b_cross_scale_relay2" ]]; then
  cross_scale_router=1
fi
IFS=',' read -r -a raw_ids <<< "$selection"
for index in "${raw_ids[@]}"; do ids+=(--dataset-index "$index"); done

cd "$repo"
env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$repo" PYTHONUNBUFFERED=1 \
  VLMAS_PROMPT_SET=prompt1 VLMAS_ATTN_IMPLEMENTATION="${VLMAS_ATTN_IMPLEMENTATION:-sdpa}" VLMAS_NAV2=1 \
  VLMAS_CROSS_SCALE_ROUTER="$cross_scale_router" \
  "$python" -m wsi_latentmas.pipeline.latent_mas \
  --variant "$variant" --backbone qwen3-vl \
  --dataset "$data_root/$dataset.json" --slide-root "$data_root/slides" \
  --model "$model" --output-root "$output" --device cuda:0 \
  --latent-steps 10 --patch-budget 12 --max-model-len 12288 \
  --navigator-control-tokens 512 --temperature 0.0 --top-p 1.0 --seed 42 \
  --deterministic --answerer-greedy --no-answerer-thinking \
  --answerer-max-new-tokens 512 --answerer-rationale \
  --answerer-protocol structured_json --canonical-open-options \
  --navigator-kv --io-pipeline --no-save-navigation-pngs --case-retries 0 \
  "${ids[@]}"
