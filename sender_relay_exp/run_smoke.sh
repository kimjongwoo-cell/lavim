#!/bin/bash
# Sender-relay smoke (§11): fixed indices 0-9, seed 42, deterministic.
# latent-base reuses the existing canonical 735-run (identical protocol).
set -u
TREE=/home/super/hj/0902_2155
PY=/home/super/hj/venvs/wsi-latentmas-py312/bin/python
OUT=$TREE/sender_relay_exp/runs/smoke
IDX="--dataset-index 0 --dataset-index 1 --dataset-index 2 --dataset-index 3 --dataset-index 4 --dataset-index 5 --dataset-index 6 --dataset-index 7 --dataset-index 8 --dataset-index 9"
COMMON="--backbone qwen3-vl --dataset /home/super/hj/datasets/WSI-VQA/WsiVQA_test.json \
 --slide-root /home/super/hj/datasets/WSI-VQA/DATA_SVS \
 --model /home/super/hj/models/Qwen3-VL-4B-Thinking --device cuda:0 \
 --latent-steps 5 --patch-budget 8 --max-model-len 8192 --navigator-control-tokens 512 \
 --temperature 0.0 --top-p 1.0 --seed 42 --answerer-max-new-tokens 512 \
 --terminal-no-repeat-ngram-size 0 --terminal-repetition-penalty 1.0 \
 --terminal-antiloop-method none --answerer-protocol structured_json \
 --transport-mode cumulative --realign-method wa --case-retries 0 \
 --deterministic --answerer-greedy --no-answerer-thinking --canonical-open-options \
 --navigator-kv --io-pipeline --answerer-rationale --no-save-navigation-pngs"

run_arm () {
  local name=$1; shift
  echo "===== ARM $name ====="
  local t0=$SECONDS
  PYTHONPATH=$TREE VLMAS_RELAY_DEBUG=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
    $PY -m wsi_latentmas.pipeline.latent_mas "$@" $COMMON $IDX \
    --output-root $OUT/$name > $OUT/$name.log 2>&1
  echo "ARM $name rc=$? elapsed=$((SECONDS-t0))s"
}

mkdir -p $OUT
run_arm pruning_v3                 --variant pruning_v3
run_arm pr_key_norm                --variant pruning_v3_reallocation --relay-source key_norm
run_arm pr_sender_priority         --variant pruning_v3_reallocation --relay-source sender_priority
run_arm pr_shuffled                --variant pruning_v3_reallocation --relay-source shuffled_sender_priority
run_arm pr_uniform                 --variant pruning_v3_reallocation --relay-source uniform
echo "ALL DONE"
