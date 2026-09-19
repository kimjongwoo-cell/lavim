#!/bin/bash
# General experiment queue worker. Jobs file: runs/expq/jobs.txt, one job per
# line:  NAME|ENV1=V1 ENV2=V2|DATASET|MODEL|STEPS
# Results: runs/expq/<NAME>/ (skip-if-exists). flock queue so any GPU can join.
# Usage: run_expq.sh <GPU>
set -u
GPU=${1:?usage: run_expq.sh GPU}
TREE=/home/users/whddn12316/wsi_latent_0915_decode_hj
if [ "$GPU" = "8" ]; then
  exec 9>>$TREE/sender_relay_exp/gpu8.lock
  flock -n 9 || { echo "GPU8 LOCK HELD — refusing to start"; exit 1; }
fi
export VLMAS_EFF_DUMP=1
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
Q=$TREE/sender_relay_exp/runs/expq
MP=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
mkdir -p $Q
while true; do
  job=$(flock $Q/jobs.lock bash -c "head -1 $Q/jobs.txt 2>/dev/null; sed -i '1d' $Q/jobs.txt 2>/dev/null")
  [ -z "$job" ] && break
  NAME=$(echo "$job" | cut -d"|" -f1)
  ENVS=$(echo "$job" | cut -d"|" -f2)
  DS=$(echo "$job" | cut -d"|" -f3)
  MODEL=$(echo "$job" | cut -d"|" -f4)
  STEPS=$(echo "$job" | cut -d"|" -f5)
  SHARD=$(echo "$job" | cut -d"|" -f6)
  dest=$Q/$NAME
  if [ -d "$dest" ]; then echo "SKIP $NAME (exists)"; continue; fi
  IDX=""
  if [ -n "$SHARD" ]; then
    K=${SHARD%%/*}; N=${SHARD##*/}
    IDX=$($PY -c "import json;n=len(json.load(open('$MP/$DS.json')));print(' '.join('--dataset-index %d'%i for i in range($K,n,$N)))")
    if [ -z "$IDX" ]; then echo "SKIP $NAME (empty shard)"; continue; fi
  fi
  t0=$SECONDS
  env $ENVS PYTHONPATH=$TREE PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU \
    $PY -m wsi_latentmas.pipeline.latent_mas \
    --variant base --backbone qwen3-vl \
    --dataset $MP/$DS.json --slide-root $MP/slides \
    --model /home/users/whddn12316/models/Qwen3-VL-${MODEL}-Thinking --device cuda:0 \
    --latent-steps $STEPS --patch-budget 8 --max-model-len 8192 \
    --navigator-control-tokens 512 --temperature 0.0 --top-p 1.0 --seed 42 \
    --answerer-max-new-tokens 512 --terminal-no-repeat-ngram-size 0 \
    --terminal-repetition-penalty 1.0 --terminal-antiloop-method none \
    --answerer-protocol structured_json --transport-mode cumulative \
    --realign-method wa --case-retries 0 --deterministic --answerer-greedy \
    --no-answerer-thinking --canonical-open-options --navigator-kv \
    --io-pipeline --answerer-rationale --no-save-navigation-pngs \
    $IDX \
    --output-root $dest > $Q/$NAME.log 2>&1
  echo "EXPQ $NAME gpu=$GPU rc=$? elapsed=$((SECONDS-t0))s"
done
echo "EXPQ WORKER gpu=$GPU drained"
