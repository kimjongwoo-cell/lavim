#!/bin/bash
# Visual-relay Experiment 1 (GPU 8): arms B(global), C(roi8), D(roi8+cross-slide V)
# on WSI-VQA idx 0-29, 4B step5 canonical config. Arm A(latent-base) is reused
# from runs/canonical. Afterwards this script REJOINS the allsteps matrix as
# worker 8 (same flock queue), so matrix throughput returns to 3 GPUs.
set -u
TREE=/home/users/whddn12316/wsi_latent_0915_decode_hj
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
OUT=$TREE/sender_relay_exp/runs/visual_relay_exp1
IDX=""; for i in $(seq 0 29); do IDX="$IDX --dataset-index $i"; done
COMMON="--variant base --backbone qwen3-vl \
 --dataset /home/users/whddn12316/datasets/WSI-VQA/WsiVQA_test.json \
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
  local name=$1 relay=$2 cross=$3
  local t0=$SECONDS
  VLMAS_VISUAL_RELAY=$relay VLMAS_VISUAL_RELAY_CROSS=$cross \
  PYTHONPATH=$TREE PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=8 \
    $PY -m wsi_latentmas.pipeline.latent_mas $COMMON $IDX \
    --output-root $OUT/$name > $OUT/$name.log 2>&1
  echo "RELAY-ARM $name rc=$? elapsed=$((SECONDS-t0))s"
}

mkdir -p $OUT
run_arm B_global_relay  global 0
run_arm C_roi_relay     roi    0
run_arm D_roi_cross     roi    1
echo "RELAY SMOKE DONE"

# ── rejoin the allsteps matrix as worker 8 ─────────────────────────────────
MOUT=$TREE/sender_relay_exp/runs/allsteps_matrix_20260903
QUEUE=$MOUT/jobs.txt
LOCK=$MOUT/jobs.lock
MP=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
model_path () { case $1 in 2b) echo /home/users/whddn12316/models/Qwen3-VL-2B-Thinking;; 4b) echo /home/users/whddn12316/models/Qwen3-VL-4B-Thinking;; 8b) echo /home/users/whddn12316/models/Qwen3-VL-8B-Thinking;; esac; }
while true; do
  job=$(flock $LOCK bash -c "head -1 $QUEUE; sed -i '1d' $QUEUE")
  [ -z "$job" ] && break
  set -- $job; model=$1 step=$2 arm=$3 ds=$4
  dest=$MOUT/$ds/$model/${arm}_step${step}
  if [ -d "$dest" ]; then echo "SKIP $job (exists)"; continue; fi
  t0=$SECONDS
  PYTHONPATH=$TREE PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=8 \
    $PY -m wsi_latentmas.pipeline.latent_mas \
    --variant $arm --backbone qwen3-vl \
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
    --output-root $dest > $MOUT/$ds.$model.${arm}_step${step}.log 2>&1
  echo "JOB $ds/$model/${arm}_step${step} gpu=8 rc=$? elapsed=$((SECONDS-t0))s"
done
echo "WORKER gpu=8 drained (rejoined)"
