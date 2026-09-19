#!/bin/bash
# C2 diagnosis: Receiver-Margin Visual Reach (Notion Experiment Log, 5 sets x 20).
# usage: rmvr_chain.sh <gpu> <name:dataset.json> ...      (canonical pipeline, eager)
set -u
GPU=$1; shift
TREE=/home/users/whddn12316/wsi_latent_0915_decode_hj
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
MP=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
K=$TREE/sender_relay_exp/runs/rmvr
SM=$TREE/sender_relay_exp/smoke
mkdir -p $K
echo "START $(date +%F_%T) gpu=$GPU jobs=$*" >> $K/chain_gpu$GPU.log
for job in "$@"; do
  NAME=${job%%:*}; DSJ=${job#*:}
  rm -rf $K/$NAME $K/rmvr_$NAME.jsonl
  cd $TREE && env VLMAS_RMVR=$K/rmvr_$NAME.jsonl PYTHONPATH=$TREE PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout 28800 \
    $PY -m wsi_latentmas.pipeline.latent_mas --variant base --backbone qwen3-vl --dataset $SM/$DSJ \
    --slide-root $MP/slides --model /home/users/whddn12316/models/Qwen3-VL-4B-Thinking --device cuda:0 \
    --latent-steps 10 --patch-budget 8 --max-model-len 8192 --navigator-control-tokens 512 \
    --temperature 0.0 --top-p 1.0 --seed 42 --answerer-max-new-tokens 512 --terminal-no-repeat-ngram-size 0 \
    --terminal-repetition-penalty 1.0 --terminal-antiloop-method none --answerer-protocol structured_json \
    --transport-mode cumulative --realign-method wa --case-retries 0 --deterministic --answerer-greedy \
    --no-answerer-thinking --canonical-open-options --navigator-kv --io-pipeline --answerer-rationale \
    --no-save-navigation-pngs --output-root $K/$NAME > $K/$NAME.log 2>&1 < /dev/null
  rc=$?
  echo "DONE $NAME rc=$rc rows=$(wc -l < $K/rmvr_$NAME.jsonl 2>/dev/null || echo 0) markers=$(grep -c '^\[RMVR\] case [0-9]* gen_tok' $K/$NAME.log) skips=$(grep -c '^\[RMVR\] case .*SKIP' $K/$NAME.log) gate_false=$(grep -c 'gate_ok=False' $K/$NAME.log) tb=$(grep -ci traceback $K/$NAME.log) results=$(find $K/$NAME -name result.json 2>/dev/null | wc -l)" >> $K/chain_gpu$GPU.log
done
echo "ALL DONE $(date +%F_%T) gpu=$GPU" >> $K/chain_gpu$GPU.log
