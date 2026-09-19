#!/bin/bash
# C2 Visual-to-Candidate Contrast Alignment diagnosis runs (VLMAS_VCCA).
# Same pipeline settings and eager attention as the existing C2 causal diagnoses
# (the Notion plan §3 says "기존 C2 causal diagnosis와 동일한 ... 압축 없음").
# usage: vcca_chain.sh <gpu> <name:dataset.json> ...
set -u
GPU=$1; shift
TREE=/home/users/whddn12316/wsi_latent_0915_decode_hj
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
MP=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
K=$TREE/sender_relay_exp/runs/vcca
SM=$TREE/sender_relay_exp/smoke
mkdir -p $K
echo "START $(date +%F_%T) gpu=$GPU jobs=$*" >> $K/chain_gpu$GPU.log
for job in "$@"; do
  NAME=${job%%:*}; DSJ=${job#*:}
  rm -rf $K/$NAME $K/vcca_$NAME.jsonl
  cd $TREE && env VLMAS_VCCA=$K/vcca_$NAME.jsonl \
    PYTHONPATH=$TREE PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout 28800 \
    $PY -m wsi_latentmas.pipeline.latent_mas --variant base --backbone qwen3-vl --dataset $SM/$DSJ --slide-root $MP/slides \
    --model /home/users/whddn12316/models/Qwen3-VL-4B-Thinking --device cuda:0 \
    --latent-steps 10 --patch-budget 8 --max-model-len 8192 --navigator-control-tokens 512 --temperature 0.0 --top-p 1.0 --seed 42 \
    --answerer-max-new-tokens 512 --terminal-no-repeat-ngram-size 0 --terminal-repetition-penalty 1.0 --terminal-antiloop-method none \
    --answerer-protocol structured_json --transport-mode cumulative --realign-method wa --case-retries 0 --deterministic --answerer-greedy \
    --no-answerer-thinking --canonical-open-options --navigator-kv --io-pipeline --answerer-rationale --no-save-navigation-pngs \
    --output-root $K/$NAME > $K/$NAME.log 2>&1 < /dev/null
  echo "DONE $NAME rc=$? vcca_rows=$(wc -l < $K/vcca_$NAME.jsonl 2>/dev/null || echo 0) markers=$(grep -c '\[VCCA\] case' $K/$NAME.log) skips=$(grep -c '\[VCCA\].*SKIP' $K/$NAME.log) tb=$(grep -c Traceback $K/$NAME.log) results=$(find $K/$NAME -name result.json 2>/dev/null | wc -l)" >> $K/chain_gpu$GPU.log
done
echo "ALL DONE $(date +%F_%T) gpu=$GPU" >> $K/chain_gpu$GPU.log
