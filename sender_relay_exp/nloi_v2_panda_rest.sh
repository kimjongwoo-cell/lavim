#!/bin/bash
# PANDA cases 7-9 were lost to an external SIGTERM at 22:54 (rc=143; the other session's alsi_v3
# run died at the same second). Re-run only those dataset indices after the GPU8 worker ends.
# Output nat_panda_rest / nloi_nat_panda_rest.jsonl (case = absolute dataset index 7..9).
set -u
GPU=8; WAIT=$1
REPO=/home/users/whddn12316/wsi_latent_0915_decode_hj
DATA=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
MODEL=/home/users/whddn12316/models/Qwen3-VL-4B-Thinking
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
K=$REPO/sender_relay_exp/runs/nloi_nav2v2
LOG=$K/chain_v2.log
while kill -0 $WAIT 2>/dev/null; do sleep 30; done
NAME=nat_panda_rest
rm -rf $K/$NAME $K/nloi_$NAME.jsonl $K/dump_$NAME
echo "PANDA_REST start gpu=$GPU $(date +%F_%T)" >> $LOG
cd $REPO && env CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$REPO" PYTHONUNBUFFERED=1 \
  VLMAS_ATTN_IMPLEMENTATION=sdpa VLMAS_NAV2=1 \
  VLMAS_NLOI=$K/nloi_$NAME.jsonl VLMAS_NLOI_DUMP=$K/dump_$NAME VLMAS_NLOI_ARM=native_nav2v2 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  timeout 86400 "$PY" -m wsi_latentmas.pipeline.latent_mas --variant base --backbone qwen3-vl \
  --dataset "$K/sets/dcs10_panda.json" --slide-root "$DATA/slides" --model "$MODEL" \
  --output-root "$K/$NAME" --device cuda:0 --latent-steps 10 --patch-budget 12 --max-model-len 12288 \
  --navigator-control-tokens 512 --temperature 0.0 --top-p 1.0 --seed 42 --deterministic --answerer-greedy \
  --no-answerer-thinking --answerer-max-new-tokens 512 --answerer-rationale --answerer-protocol structured_json \
  --canonical-open-options --navigator-kv --io-pipeline --no-save-navigation-pngs --case-retries 0 \
  --dataset-index 7 --dataset-index 8 --dataset-index 9 > "$K/$NAME.log" 2>&1 < /dev/null
echo "DONE $NAME rc=$? $(date +%H:%M) rows=$(wc -l < $K/nloi_$NAME.jsonl 2>/dev/null || echo 0) marks=$(grep -c "^\[NLOI\] case [0-9]* obs" $K/$NAME.log) cases=$(grep -o "^\[NLOI\] case [0-9]*" $K/$NAME.log | awk '{print $3}' | tr '\n' ,) tb=$(grep -ci traceback $K/$NAME.log) oom=$(grep -c "out of memory" $K/$NAME.log) results=$(find $K/$NAME -name result.json | wc -l)" >> $LOG
