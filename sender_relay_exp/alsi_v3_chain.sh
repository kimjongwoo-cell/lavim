#!/bin/bash
# ALSI v3: float32 suffix + relative eps sweep 0.1/0.05/0.02 (middle = estimate).
# usage: alsi_v3_chain.sh <gpu> smoke|full
set -u
GPU=$1; MODE=$2
TREE=/home/users/whddn12316/wsi_latent_0915_decode_hj
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
MP=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
K=$TREE/sender_relay_exp/runs/alsi_v3
SM=$TREE/sender_relay_exp/smoke
mkdir -p $K

run_one () {
  NAME=$1; DSJ=$2; shift 2
  rm -rf $K/$NAME $K/alsi_$NAME.jsonl
  cd $TREE && env VLMAS_ALSI=$K/alsi_$NAME.jsonl "$@" \
    PYTHONPATH=$TREE PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout 28800 \
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
  echo "DONE $NAME rc=$? rows=$(wc -l < $K/alsi_$NAME.jsonl 2>/dev/null || echo 0) marks=$(grep -c '^\[ALSI\] case [0-9]* arm' $K/$NAME.log) tb=$(grep -ci traceback $K/$NAME.log) oom=$(grep -ci 'out of memory' $K/$NAME.log) results=$(find $K/$NAME -name result.json 2>/dev/null | wc -l)" >> $K/chain_gpu$GPU.log
}

echo "START $MODE $(date +%F_%T) gpu=$GPU" >> $K/chain_gpu$GPU.log
if [ "$MODE" = "smoke" ]; then
  run_one smoke gtex2.json CUDA_LAUNCH_BLOCKING=1
else
  for ds in gtex tcga_expert_vqa tcga_slidebench panda tcga; do run_one nat_$ds dcs20_$ds.json; done
  for ds in gtex tcga_expert_vqa tcga_slidebench panda tcga; do
    run_one can_$ds dcs20_$ds.json VLMAS_KV_ROUTE=1 VLMAS_KV_ROUTE_MODE=canonical
  done
fi
echo "END $MODE $(date +%F_%T)" >> $K/chain_gpu$GPU.log
