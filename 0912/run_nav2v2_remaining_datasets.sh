#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?usage: $0 GPU SHARD}"
shard="${2:?usage: $0 GPU SHARD}"
if [[ "$shard" != "left" && "$shard" != "right" ]]; then
  echo "SHARD must be left or right" >&2
  exit 2
fi

repo=/home/users/whddn12316/wsi_latent_0915_decode_hj
data_root=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
model=/home/users/whddn12316/models/Qwen3-VL-4B-Thinking
python=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
run_root="$repo/0912/runs"

if [[ "$gpu" == "6" ]]; then
  prior_session=nav2-both-rem20-gpu6
elif [[ "$gpu" == "8" ]]; then
  prior_session=nav2-both-rem21-gpu8
else
  echo "Only GPU 6 or 8 is allowed" >&2
  exit 2
fi

while tmux has-session -t "$prior_session" 2>/dev/null; do
  sleep 10
done

datasets=(gtex tcga tcga_slidebench panda)
sizes=(190 221 197 196)
variants=(base pruning_b latent_kv_relay2 pruning_b_latent_kv_relay2)
labels=(base pruning_b relay2 both)

cd "$repo"
for dataset_pos in "${!datasets[@]}"; do
  dataset="${datasets[$dataset_pos]}"
  size="${sizes[$dataset_pos]}"
  midpoint=$((size / 2))
  if [[ "$shard" == "left" ]]; then
    first=0
    last=$((midpoint - 1))
  else
    first=$midpoint
    last=$((size - 1))
  fi

  ids=()
  for ((index=first; index<=last; index++)); do
    ids+=(--dataset-index "$index")
  done

  for variant_pos in "${!variants[@]}"; do
    variant="${variants[$variant_pos]}"
    label="${labels[$variant_pos]}"
    output="$run_root/nav2v2_${label}_${dataset}_part$(printf '%03d' "$first")_$(printf '%03d' "$last")_gpu${gpu}_0913"
    env \
      CUDA_VISIBLE_DEVICES="$gpu" \
      PYTHONPATH="$repo" \
      PYTHONUNBUFFERED=1 \
      VLMAS_ATTN_IMPLEMENTATION=sdpa \
      VLMAS_NAV2=1 \
      "$python" -m wsi_latentmas.pipeline.latent_mas \
      --variant "$variant" \
      --backbone qwen3-vl \
      --dataset "$data_root/$dataset.json" \
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
done
