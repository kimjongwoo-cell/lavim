#!/bin/bash
# C2 structural diagnosis: Non-local Observation Integration (Notion Experiment Log), Stage A.
# Exact single/pair observation value-cut forwards at the Answerer, true vs shuffled grouping.
# Smoke gate (gtex 2 cases) -> native 5 sets x 20 -> canonical 5 sets x 20 (secondary control).
# usage: nloi_chain.sh <wait_pid|-> <gpu>
set -u
WAITPID=$1; GPU=$2
TREE=/home/users/whddn12316/wsi_latent_0915_decode_hj
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
MP=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
K=$TREE/sender_relay_exp/runs/nloi
SM=$TREE/sender_relay_exp/smoke
mkdir -p $K

busy_hj () {  # hj-tree runs (sender_relay_exp chains) already on this GPU, excluding ourselves
  local n=0
  for p in $(pgrep -u "$(id -u)" -f "wsi_latentmas.pipeline.latent_mas"); do
    [ "$p" = "$$" ] && continue
    tr "\0" "\n" < /proc/$p/cmdline 2>/dev/null | grep -q "sender_relay_exp" || continue
    tr "\0" "\n" < /proc/$p/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$GPU" && n=$((n+1))
  done
  echo $n
}

run_one () {   # $1 name  $2 dataset.json  $3... extra env
  NAME=$1; DSJ=$2; shift 2
  rm -rf $K/$NAME $K/nloi_$NAME.jsonl $K/dump_$NAME
  cd $TREE && env VLMAS_NLOI=$K/nloi_$NAME.jsonl VLMAS_NLOI_DUMP=$K/dump_$NAME "$@" \
    PYTHONPATH=$TREE PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout 43200 \
    $PY -m wsi_latentmas.pipeline.latent_mas --variant base --backbone qwen3-vl \
    --dataset $SM/$DSJ --slide-root $MP/slides \
    --model /home/users/whddn12316/models/Qwen3-VL-4B-Thinking --device cuda:0 \
    --latent-steps 10 --patch-budget 8 --max-model-len 8192 --navigator-control-tokens 512 \
    --temperature 0.0 --top-p 1.0 --seed 42 --answerer-max-new-tokens 512 \
    --terminal-no-repeat-ngram-size 0 --terminal-repetition-penalty 1.0 --terminal-antiloop-method none \
    --answerer-protocol structured_json --transport-mode cumulative --realign-method wa --case-retries 0 \
    --deterministic --answerer-greedy --no-answerer-thinking --canonical-open-options --navigator-kv \
    --io-pipeline --answerer-rationale --no-save-navigation-pngs \
    --output-root $K/$NAME > $K/$NAME.log 2>&1 < /dev/null
  echo "DONE $NAME rc=$? $(date +%H:%M) rows=$(wc -l < $K/nloi_$NAME.jsonl 2>/dev/null || echo 0) marks=$(grep -c "^\[NLOI\] case [0-9]* obs" $K/$NAME.log) skips=$(grep -c "^\[NLOI\] case .*SKIP" $K/$NAME.log) geom0=$(grep -c "^\[NLOI\] case [0-9]* .*geom=0" $K/$NAME.log) tb=$(grep -ci traceback $K/$NAME.log) oom=$(grep -c "out of memory" $K/$NAME.log) results=$(find $K/$NAME -name result.json 2>/dev/null | wc -l)" >> $K/chain_gpu$GPU.log
}

if [ "$WAITPID" != "-" ]; then
  while kill -0 $WAITPID 2>/dev/null; do sleep 30; done
fi
while [ "$(busy_hj)" -gt 0 ]; do sleep 60; done
echo "START $(date +%F_%T) gpu=$GPU pid=$$" >> $K/chain_gpu$GPU.log
run_one smoke gtex2.json CUDA_LAUNCH_BLOCKING=1
n=$(grep -c "^\[NLOI\] case [0-9]* obs" $K/smoke.log); tb=$(grep -ci traceback $K/smoke.log)
sk=$(grep -c "^\[NLOI\] case .*SKIP" $K/smoke.log); g0=$(grep -c "^\[NLOI\] case [0-9]* .*geom=0" $K/smoke.log)
res=$(find $K/smoke -name result.json 2>/dev/null | wc -l)
echo "$(date +%H:%M) smoke: marks=$n skips=$sk geom0=$g0 traceback=$tb result=$res" >> $K/chain_gpu$GPU.log
if [ "${n:-0}" -ge 2 ] && [ "${tb:-1}" -eq 0 ] && [ "${res:-0}" -ge 2 ] && [ "${sk:-1}" -eq 0 ] && [ "${g0:-1}" -eq 0 ]; then
  for ds in gtex tcga_expert_vqa tcga_slidebench panda tcga; do run_one nat_$ds dcs20_$ds.json; done
  echo "NATIVE DONE $(date +%F_%T) gpu=$GPU" >> $K/chain_gpu$GPU.log
  for ds in gtex tcga_expert_vqa tcga_slidebench panda tcga; do
    run_one can_$ds dcs20_$ds.json VLMAS_KV_ROUTE=1 VLMAS_KV_ROUTE_MODE=canonical
  done
  echo "ALL DONE $(date +%F_%T) gpu=$GPU" >> $K/chain_gpu$GPU.log
else
  echo "SMOKE GATE FAILED - full run not launched" >> $K/chain_gpu$GPU.log
fi
