#!/bin/bash
# NLOI Stage A on the nav2v2 setting. Run flags are sender_relay_exp/nav2v2_mine_run.sh
# verbatim (sdpa, VLMAS_NAV2=1, patch-budget 12, max-model-len 12288, arm base); only
# --dataset points at the fixed dcs20_* 20-case sets and --output-root at runs/nloi_nav2v2.
# Smoke gate (gtex 2 cases) -> native 5 sets x 20.
# usage: nloi_nav2v2_chain.sh <wait_pid|-> <gpu>
set -u
WAITPID=$1; GPU=$2
REPO=/home/users/whddn12316/wsi_latent_0915_decode_hj
DATA=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
MODEL=/home/users/whddn12316/models/Qwen3-VL-4B-Thinking
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
K=$REPO/sender_relay_exp/runs/nloi_nav2v2
SM=$REPO/sender_relay_exp/smoke
mkdir -p $K

run_one () {   # $1 name  $2 dataset.json  $3... extra env
  NAME=$1; DSJ=$2; shift 2
  rm -rf $K/$NAME $K/nloi_$NAME.jsonl $K/dump_$NAME
  cd $REPO && env \
    CUDA_VISIBLE_DEVICES="$GPU" \
    PYTHONPATH="$REPO" \
    PYTHONUNBUFFERED=1 \
    VLMAS_ATTN_IMPLEMENTATION=sdpa \
    VLMAS_NAV2=1 \
    VLMAS_NLOI=$K/nloi_$NAME.jsonl \
    VLMAS_NLOI_DUMP=$K/dump_$NAME \
    VLMAS_NLOI_ARM=native_nav2v2 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$@" \
    timeout 86400 "$PY" -m wsi_latentmas.pipeline.latent_mas \
    --variant base \
    --backbone qwen3-vl \
    --dataset "$SM/$DSJ" \
    --slide-root "$DATA/slides" \
    --model "$MODEL" \
    --output-root "$K/$NAME" \
    --device cuda:0 \
    --latent-steps 10 \
    --patch-budget 12 \
    --max-model-len 12288 \
    --navigator-control-tokens 512 \
    --temperature 0.0 \
    --top-p 1.0 \
    --seed 42 \
    --deterministic \
    --answerer-greedy \
    --no-answerer-thinking \
    --answerer-max-new-tokens 512 \
    --answerer-rationale \
    --answerer-protocol structured_json \
    --canonical-open-options \
    --navigator-kv \
    --io-pipeline \
    --no-save-navigation-pngs \
    --case-retries 0 > "$K/$NAME.log" 2>&1 < /dev/null
  echo "DONE $NAME rc=$? $(date +%H:%M) rows=$(wc -l < $K/nloi_$NAME.jsonl 2>/dev/null || echo 0) marks=$(grep -c "^\[NLOI\] case [0-9]* obs" $K/$NAME.log) skips=$(grep -c "^\[NLOI\] case .*SKIP" $K/$NAME.log) geom0=$(grep -c "^\[NLOI\] case [0-9]* .*geom=0" $K/$NAME.log) tb=$(grep -ci traceback $K/$NAME.log) oom=$(grep -c "out of memory" $K/$NAME.log) results=$(find $K/$NAME -name result.json 2>/dev/null | wc -l)" >> $K/chain_gpu$GPU.log
}

if [ "$WAITPID" != "-" ]; then
  while kill -0 $WAITPID 2>/dev/null; do sleep 30; done
fi
echo "START $(date +%F_%T) gpu=$GPU pid=$$" >> $K/chain_gpu$GPU.log
run_one smoke gtex2.json CUDA_LAUNCH_BLOCKING=1
n=$(grep -c "^\[NLOI\] case [0-9]* obs" $K/smoke.log); tb=$(grep -ci traceback $K/smoke.log)
sk=$(grep -c "^\[NLOI\] case .*SKIP" $K/smoke.log); g0=$(grep -c "^\[NLOI\] case [0-9]* .*geom=0" $K/smoke.log)
res=$(find $K/smoke -name result.json 2>/dev/null | wc -l)
echo "$(date +%H:%M) smoke: marks=$n skips=$sk geom0=$g0 traceback=$tb result=$res" >> $K/chain_gpu$GPU.log
if [ "${n:-0}" -ge 2 ] && [ "${tb:-1}" -eq 0 ] && [ "${res:-0}" -ge 2 ] && [ "${sk:-1}" -eq 0 ] && [ "${g0:-1}" -eq 0 ]; then
  for ds in gtex tcga_expert_vqa tcga_slidebench panda tcga; do run_one nat_$ds dcs20_$ds.json; done
  echo "ALL DONE $(date +%F_%T) gpu=$GPU" >> $K/chain_gpu$GPU.log
else
  echo "SMOKE GATE FAILED - full run not launched" >> $K/chain_gpu$GPU.log
fi
