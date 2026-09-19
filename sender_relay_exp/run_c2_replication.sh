#!/bin/bash
# C2 cross-scale replication worker (C2_REPLICATION_PREREG.md).
# Usage: run_c2_replication.sh <GPU>   — flock queue so 6/7 can join later.
set -u
GPU=${1:?usage: run_c2_replication.sh GPU}
TREE=/home/users/whddn12316/wsi_latent_0915_decode_hj
if [ "$GPU" = "8" ]; then
  exec 9>>$TREE/sender_relay_exp/gpu8.lock
  flock -n 9 || { echo "GPU8 LOCK HELD — refusing to start"; exit 1; }
fi
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
OUT=$TREE/sender_relay_exp/runs/c2_replication
MP=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
QUEUE=$OUT/jobs.txt
LOCK=$OUT/jobs.lock
model_path () { case $1 in 4b) echo /home/users/whddn12316/models/Qwen3-VL-4B-Thinking;; 8b) echo /home/users/whddn12316/models/Qwen3-VL-8B-Thinking;; esac; }

mkdir -p $OUT
if [ ! -f $QUEUE ]; then
  cat > $QUEUE <<'JOBS'
4b 10 gtex
4b 10 tcga_expert_vqa
4b 20 gtex
4b 20 tcga_expert_vqa
8b 20 gtex
8b 20 tcga_expert_vqa
JOBS
fi

while true; do
  job=$(flock $LOCK bash -c "head -1 $QUEUE; sed -i '1d' $QUEUE")
  [ -z "$job" ] && break
  set -- $job; model=$1 step=$2 ds=$3
  dest=$OUT/$ds/$model/D_cross_step$step
  if [ -d "$dest" ]; then echo "SKIP $job (exists)"; continue; fi
  t0=$SECONDS
  VLMAS_CROSS_SCALE=1 \
  PYTHONPATH=$TREE PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU \
    $PY -m wsi_latentmas.pipeline.latent_mas \
    --variant pruning_v3 --backbone qwen3-vl \
    --dataset $MP/$ds.json --slide-root $MP/slides \
    --model $(model_path $model) --device cuda:0 \
    --latent-steps $step --patch-budget 8 --max-model-len 8192 \
    --navigator-control-tokens 512 --temperature 0.0 --top-p 1.0 --seed 42 \
    --answerer-max-new-tokens 512 --terminal-no-repeat-ngram-size 0 \
    --terminal-repetition-penalty 1.0 --terminal-antiloop-method none \
    --answerer-protocol structured_json --transport-mode cumulative \
    --realign-method wa --case-retries 0 --deterministic --answerer-greedy \
    --no-answerer-thinking --canonical-open-options --navigator-kv \
    --io-pipeline --answerer-rationale --no-save-navigation-pngs \
    --output-root $dest > $OUT/$ds.$model.step$step.log 2>&1
  echo "C2-JOB $ds/$model/step$step gpu=$GPU rc=$? elapsed=$((SECONDS-t0))s"
done
echo "C2 WORKER gpu=$GPU drained"
