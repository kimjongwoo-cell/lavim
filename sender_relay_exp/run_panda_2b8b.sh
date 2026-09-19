#!/bin/bash
# PANDA 2B/8B x {single, vlmas, latent-base} via the tree's execution CLI.
# One model size per GPU, methods sequential (exclusive occupancy, no co-tenancy OOM).
# usage: run_panda_2b8b.sh <qwen-2b|qwen-8b> <gpu>
set -u
TREE=/home/users/whddn12316/wsi_latent_0915_decode_hj
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
MODEL=${1:?model}
GPU=${2:?gpu}
TAG=${MODEL#qwen-}
OUT=$TREE/sender_relay_exp/runs/panda_full_20260904/$TAG

mkdir -p "$OUT"
for method in single vlmas latent-base; do
  dest=$OUT/$method
  if [ -d "$dest" ]; then echo "SKIP $TAG/$method (exists)"; continue; fi
  t0=$SECONDS
  PYTHONPATH=$TREE PYTHONUNBUFFERED=1 \
    $PY -m wsi_latentmas.execution.cli \
    --dataset panda --model "$MODEL" --method $method --gpus "$GPU" \
    --output-root "$dest" > "$OUT/$method.log" 2>&1
  echo "$(date --iso-8601=seconds) JOB panda/$TAG/$method gpu=$GPU rc=$? elapsed=$((SECONDS-t0))s"
done
echo "PANDA $TAG ALL DONE"
