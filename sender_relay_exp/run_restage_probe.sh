#!/bin/bash
# KV re-staging probe (refeed-minus-pixels) — MultiPathQA ONLY (user directive).
#   Phase 1  identity check (gtex idx 0-4): move columns, keep positions —
#            answers MUST equal multipath base (plumbing gate).
#   Phase 2  restage arms: full gtex(190) + tcga_expert_vqa(128).
# Judgment (BACC for gtex / ACC for evqa) vs runs/multipath_qwen4b_step5_20260903
# base; the refeed arm on the SAME datasets is the position+freshness yardstick.
# Usage: run_restage_probe.sh <GPU>
set -u
GPU=${1:?usage: run_restage_probe.sh GPU}
TREE=/home/users/whddn12316/wsi_latent_0915_decode_hj
if [ "$GPU" = "8" ]; then
  exec 9>>$TREE/sender_relay_exp/gpu8.lock
  flock -n 9 || { echo "GPU8 LOCK HELD — refusing to start"; exit 1; }
fi
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
OUT=$TREE/sender_relay_exp/runs/restage_probe_mp
MP=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
COMMON="--variant base --backbone qwen3-vl \
 --model /home/users/whddn12316/models/Qwen3-VL-4B-Thinking --device cuda:0 \
 --latent-steps 5 --patch-budget 8 --max-model-len 8192 --navigator-control-tokens 512 \
 --temperature 0.0 --top-p 1.0 --seed 42 --answerer-max-new-tokens 512 \
 --terminal-no-repeat-ngram-size 0 --terminal-repetition-penalty 1.0 \
 --terminal-antiloop-method none --answerer-protocol structured_json \
 --transport-mode cumulative --realign-method wa --case-retries 0 \
 --deterministic --answerer-greedy --no-answerer-thinking --canonical-open-options \
 --navigator-kv --io-pipeline --answerer-rationale --no-save-navigation-pngs"

mkdir -p $OUT
IDX5=""; for i in $(seq 0 4); do IDX5="$IDX5 --dataset-index $i"; done
if [ ! -d $OUT/identity ]; then
  t0=$SECONDS
  VLMAS_KV_RESTAGE=1 VLMAS_KV_RESTAGE_IDENTITY=1 \
  PYTHONPATH=$TREE PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU \
    $PY -m wsi_latentmas.pipeline.latent_mas $COMMON \
    --dataset $MP/gtex.json --slide-root $MP/slides $IDX5 \
    --output-root $OUT/identity > $OUT/identity.log 2>&1
  echo "RESTAGE-IDENTITY rc=$? elapsed=$((SECONDS-t0))s"
fi

for ds in gtex tcga_expert_vqa; do
  dest=$OUT/$ds/restage
  if [ -d "$dest" ]; then echo "SKIP $ds (exists)"; continue; fi
  t0=$SECONDS
  VLMAS_KV_RESTAGE=1 \
  PYTHONPATH=$TREE PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU \
    $PY -m wsi_latentmas.pipeline.latent_mas $COMMON \
    --dataset $MP/$ds.json --slide-root $MP/slides \
    --output-root $dest > $OUT/$ds.restage.log 2>&1
  echo "RESTAGE-ARM $ds rc=$? elapsed=$((SECONDS-t0))s"
done
echo "RESTAGE PROBE MP DONE gpu=$GPU"
