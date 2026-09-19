#!/bin/bash
# {single, vlmas, latent-base} trio for one MultiPathQA dataset/model on one GPU.
# usage: run_mp_trio.sh <dataset> <qwen-2b|qwen-4b|qwen-8b> <gpu>
set -u
TREE=/home/users/whddn12316/wsi_latent_0915_decode_hj
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
DS=${1:?dataset}
MODEL=${2:?model}
GPU=${3:?gpu}
TAG=${MODEL#qwen-}
OUT=$TREE/sender_relay_exp/runs/mp_${DS}_20260905/$TAG

mkdir -p "$OUT"
for method in single vlmas latent-base; do
  dest=$OUT/$method
  if [ -d "$dest" ]; then echo "SKIP $DS/$TAG/$method (exists)"; continue; fi
  t0=$SECONDS
  PYTHONPATH=$TREE PYTHONUNBUFFERED=1 \
    $PY -m wsi_latentmas.execution.cli \
    --dataset "$DS" --model "$MODEL" --method $method --gpus "$GPU" \
    --output-root "$dest" > "$OUT/$method.log" 2>&1
  echo "$(date --iso-8601=seconds) JOB $DS/$TAG/$method gpu=$GPU rc=$? elapsed=$((SECONDS-t0))s"
done
echo "$DS $TAG ALL DONE"
