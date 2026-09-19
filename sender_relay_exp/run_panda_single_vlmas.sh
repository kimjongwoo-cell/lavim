#!/bin/bash
# PANDA Single + VLMAS baselines (qwen-4b), queued behind the running latent-base
# job on GPU 8. Uses the tree's execution CLI (canonical presets, panda already
# registered in config/experiment.py).
set -u
TREE=/home/users/whddn12316/wsi_latent_0915_decode_hj
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
OUT=$TREE/sender_relay_exp/runs/panda_full_20260904
BASE_PID=${1:?usage: run_panda_single_vlmas.sh <base-run-pid>}

while kill -0 "$BASE_PID" 2>/dev/null; do sleep 60; done
echo "$(date --iso-8601=seconds) base pid $BASE_PID gone; starting single"

for method in single vlmas; do
  t0=$SECONDS
  PYTHONPATH=$TREE PYTHONUNBUFFERED=1 \
    $PY -m wsi_latentmas.execution.cli run \
    --dataset panda --model qwen-4b --method "$method" --gpus 8 \
    --output-root $OUT/panda/$method \
    > $OUT/panda.$method.log 2>&1
  echo "$(date --iso-8601=seconds) JOB panda/$method rc=$? elapsed=$((SECONDS-t0))s"
done
echo "PANDA single+vlmas DONE"
