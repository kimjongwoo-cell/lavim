#!/bin/bash
# Sender-relay main experiment on MultiPathQA (ExpertVQA 128 + GTEx 190),
# qwen-4b, latent step 5, seed 42 — 6 arms on GPUs 6/7/8.
# Results: $TREE/sender_relay_exp/runs/multipath_qwen4b_step5_20260903/
set -u
TREE=/home/users/whddn12316/wsi_latent_0915_decode_hj
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
MP=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
OUT=$TREE/sender_relay_exp/runs/multipath_qwen4b_step5_20260903
COMMON="--backbone qwen3-vl --model /home/users/whddn12316/models/Qwen3-VL-4B-Thinking \
 --device cuda:0 --latent-steps 5 --patch-budget 8 --max-model-len 8192 \
 --navigator-control-tokens 512 --temperature 0.0 --top-p 1.0 --seed 42 \
 --answerer-max-new-tokens 512 --terminal-no-repeat-ngram-size 0 \
 --terminal-repetition-penalty 1.0 --terminal-antiloop-method none \
 --answerer-protocol structured_json --transport-mode cumulative --realign-method wa \
 --case-retries 0 --deterministic --answerer-greedy --no-answerer-thinking \
 --canonical-open-options --navigator-kv --io-pipeline --answerer-rationale \
 --no-save-navigation-pngs"

run_job () {
  local gpu=$1 dataset=$2 name=$3; shift 3
  local t0=$SECONDS
  PYTHONPATH=$TREE PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$gpu \
    $PY -m wsi_latentmas.pipeline.latent_mas "$@" $COMMON \
    --dataset $MP/$dataset.json --slide-root $MP/slides \
    --output-root $OUT/$dataset/$name > $OUT/$dataset.$name.log 2>&1
  echo "JOB $dataset/$name gpu=$gpu rc=$? elapsed=$((SECONDS-t0))s"
}

mkdir -p $OUT
(
  for ds in tcga_expert_vqa gtex; do
    run_job 6 $ds base            --variant base
    run_job 6 $ds pr_sender      --variant pruning_v3_reallocation --relay-source sender_priority
  done
) &
(
  for ds in tcga_expert_vqa gtex; do
    run_job 7 $ds pruning_v3     --variant pruning_v3
    run_job 7 $ds pr_key_norm    --variant pruning_v3_reallocation --relay-source key_norm
  done
) &
(
  for ds in tcga_expert_vqa gtex; do
    run_job 8 $ds pr_uniform     --variant pruning_v3_reallocation --relay-source uniform
    run_job 8 $ds pr_shuffled    --variant pruning_v3_reallocation --relay-source shuffled_sender_priority
  done
) &
wait
echo "MULTIPATH ALL DONE"
