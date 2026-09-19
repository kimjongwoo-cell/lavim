#!/bin/bash
# Restage budget-gain curve: §3.3-v2 selection (s_c hook + kappa=cos) at
# keep ∈ {0.25, 0.5, 0.75} composed with KV re-staging. Anchor points:
# keep=1.0 = runs/restage_probe_mp (full restage), no-restage baselines exist.
# Usage: run_budget_curve.sh <GPU> <dataset>
set -u
GPU=${1:?gpu}; DS=${2:?dataset}
TREE=/home/users/whddn12316/wsi_latent_0915_decode_hj
if [ "$GPU" = "8" ]; then
  exec 9>>$TREE/sender_relay_exp/gpu8.lock
  flock -n 9 || { echo "GPU8 LOCK HELD — refusing to start"; exit 1; }
fi
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
OUT=$TREE/sender_relay_exp/runs/budget_curve
MP=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
mkdir -p $OUT
for KEEP in 0.5 0.75 0.25; do
  dest=$OUT/$DS/keep${KEEP}
  if [ -d "$dest" ]; then echo "SKIP $DS keep$KEEP (exists)"; continue; fi
  t0=$SECONDS
  VLMAS_WSI_CONSOL=1 VLMAS_WSI_CONSOL_KEEP=$KEEP \
  VLMAS_WSI_CONSOL_SC=1 VLMAS_WSI_CONSOL_KAPPA=cos \
  VLMAS_KV_RESTAGE=1 \
  PYTHONPATH=$TREE PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU \
    $PY -m wsi_latentmas.pipeline.latent_mas \
    --variant base --backbone qwen3-vl \
    --dataset $MP/$DS.json --slide-root $MP/slides \
    --model /home/users/whddn12316/models/Qwen3-VL-4B-Thinking --device cuda:0 \
    --latent-steps 5 --patch-budget 8 --max-model-len 8192 \
    --navigator-control-tokens 512 --temperature 0.0 --top-p 1.0 --seed 42 \
    --answerer-max-new-tokens 512 --terminal-no-repeat-ngram-size 0 \
    --terminal-repetition-penalty 1.0 --terminal-antiloop-method none \
    --answerer-protocol structured_json --transport-mode cumulative \
    --realign-method wa --case-retries 0 --deterministic --answerer-greedy \
    --no-answerer-thinking --canonical-open-options --navigator-kv \
    --io-pipeline --answerer-rationale --no-save-navigation-pngs \
    --output-root $dest > $OUT/$DS.keep$KEEP.log 2>&1
  echo "CURVE $DS keep$KEEP gpu=$GPU rc=$? elapsed=$((SECONDS-t0))s"
done
echo "BUDGET CURVE $DS DONE gpu=$GPU"
