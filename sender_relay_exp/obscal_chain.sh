#!/bin/bash
# C2 feasibility: WSI Observation-Level Receiver Calibration (Notion Experiment Log).
# Smoke gate (2 cases) then 5 sets x 20. usage: obscal_chain.sh <wait_pid|-> <gpu>
set -u
WAITPID=$1; GPU=$2
TREE=/home/users/whddn12316/wsi_latent_0915_decode_hj
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
MP=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
K=$TREE/sender_relay_exp/runs/obscal
SM=$TREE/sender_relay_exp/smoke
mkdir -p $K

run_one () {  # $1 name  $2 dataset.json  $3... extra env
  NAME=$1; DSJ=$2; shift 2
  rm -rf $K/$NAME $K/obscal_$NAME.jsonl
  cd $TREE && env VLMAS_OBSCAL=$K/obscal_$NAME.jsonl "$@" PYTHONPATH=$TREE PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout 28800 \
    $PY -m wsi_latentmas.pipeline.latent_mas --variant base --backbone qwen3-vl --dataset $SM/$DSJ \
    --slide-root $MP/slides --model /home/users/whddn12316/models/Qwen3-VL-4B-Thinking --device cuda:0 \
    --latent-steps 10 --patch-budget 8 --max-model-len 8192 --navigator-control-tokens 512 \
    --temperature 0.0 --top-p 1.0 --seed 42 --answerer-max-new-tokens 512 --terminal-no-repeat-ngram-size 0 \
    --terminal-repetition-penalty 1.0 --terminal-antiloop-method none --answerer-protocol structured_json \
    --transport-mode cumulative --realign-method wa --case-retries 0 --deterministic --answerer-greedy \
    --no-answerer-thinking --canonical-open-options --navigator-kv --io-pipeline --answerer-rationale \
    --no-save-navigation-pngs --output-root $K/$NAME > $K/$NAME.log 2>&1 < /dev/null
  echo "DONE $NAME rc=$? rows=$(wc -l < $K/obscal_$NAME.jsonl 2>/dev/null || echo 0) marks=$(grep -c "^\[ObsCal\] case [0-9]* l\*" $K/$NAME.log) skips=$(grep -c "^\[ObsCal\] case .*SKIP" $K/$NAME.log) tb=$(grep -ci traceback $K/$NAME.log) results=$(find $K/$NAME -name result.json 2>/dev/null | wc -l)" >> $K/chain_gpu$GPU.log
}

if [ "$WAITPID" != "-" ]; then
  while kill -0 $WAITPID 2>/dev/null; do sleep 30; done
fi
echo "START $(date +%F_%T) gpu=$GPU" >> $K/chain_gpu$GPU.log
run_one smoke gtex2.json CUDA_LAUNCH_BLOCKING=1
n=$(grep -c "^\[ObsCal\] case [0-9]* l\*" $K/smoke.log); tb=$(grep -ci traceback $K/smoke.log)
sk=$(grep -c "^\[ObsCal\] case .*SKIP" $K/smoke.log); res=$(find $K/smoke -name result.json 2>/dev/null | wc -l)
echo "$(date +%H:%M) smoke: marks=$n skips=$sk traceback=$tb result=$res" >> $K/chain_gpu$GPU.log
if [ "${n:-0}" -ge 2 ] && [ "${tb:-1}" -eq 0 ] && [ "${res:-0}" -ge 2 ]; then
  for ds in gtex tcga_expert_vqa tcga_slidebench panda tcga; do run_one $ds dcs20_$ds.json; done
  echo "ALL DONE $(date +%F_%T) gpu=$GPU" >> $K/chain_gpu$GPU.log
else
  echo "SMOKE GATE FAILED — full run not launched" >> $K/chain_gpu$GPU.log
fi
