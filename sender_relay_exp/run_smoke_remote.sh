#!/bin/bash
# Sender-relay smoke on the remote tree (GPUs 6/7/8): fixed indices 0-9, seed 42.
set -u
TREE=/home/users/whddn12316/wsi_latent_0915_decode_hj
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
OUT=$TREE/sender_relay_exp/runs/smoke
IDX="--dataset-index 0 --dataset-index 1 --dataset-index 2 --dataset-index 3 --dataset-index 4 --dataset-index 5 --dataset-index 6 --dataset-index 7 --dataset-index 8 --dataset-index 9"
COMMON="--backbone qwen3-vl --dataset /home/users/whddn12316/datasets/WSI-VQA/WsiVQA_test.json \
 --slide-root /home/users/whddn12316/datasets/WSI-VQA/DATA_SVS \
 --model /home/users/whddn12316/models/Qwen3-VL-4B-Thinking --device cuda:0 \
 --latent-steps 5 --patch-budget 8 --max-model-len 8192 --navigator-control-tokens 512 \
 --temperature 0.0 --top-p 1.0 --seed 42 --answerer-max-new-tokens 512 \
 --terminal-no-repeat-ngram-size 0 --terminal-repetition-penalty 1.0 \
 --terminal-antiloop-method none --answerer-protocol structured_json \
 --transport-mode cumulative --realign-method wa --case-retries 0 \
 --deterministic --answerer-greedy --no-answerer-thinking --canonical-open-options \
 --navigator-kv --io-pipeline --answerer-rationale --no-save-navigation-pngs"

run_arm () {
  local gpu=$1 name=$2; shift 2
  local t0=$SECONDS
  PYTHONPATH=$TREE VLMAS_RELAY_DEBUG=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$gpu \
    $PY -m wsi_latentmas.pipeline.latent_mas "$@" $COMMON $IDX \
    --output-root $OUT/$name > $OUT/$name.log 2>&1
  echo "ARM $name gpu=$gpu rc=$? elapsed=$((SECONDS-t0))s"
}

mkdir -p $OUT
(
  run_arm 6 pruning_v3         --variant pruning_v3
  run_arm 6 pr_sender_priority --variant pruning_v3_reallocation --relay-source sender_priority
) &
(
  run_arm 7 pr_key_norm        --variant pruning_v3_reallocation --relay-source key_norm
  run_arm 7 pr_shuffled        --variant pruning_v3_reallocation --relay-source shuffled_sender_priority
) &
(
  run_arm 8 pr_uniform         --variant pruning_v3_reallocation --relay-source uniform
) &
wait
echo "ALL DONE"
