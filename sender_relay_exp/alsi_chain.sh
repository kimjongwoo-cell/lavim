#!/bin/bash
# C2 structural diagnosis: Answerer Latent Source Interference (Notion sections 3-8).
# Arm 1 native addressing, arm 2 canonical addressing (section 6). Arm 3
# (canonical + native-total-mass) is NOT wired yet — it needs the mass-restoring hook and
# is only worth building if arms 1 and 2 differ.
# usage: alsi_chain.sh <wait_pid|-> <gpu>
set -u
WAITPID=$1; GPU=$2
TREE=/home/users/whddn12316/wsi_latent_0915_decode_hj
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
MP=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
K=$TREE/sender_relay_exp/runs/alsi
SM=$TREE/sender_relay_exp/smoke
mkdir -p $K

run_one () {   # $1 name  $2 dataset.json  $3... extra env
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
  echo "DONE $NAME rc=$? rows=$(wc -l < $K/alsi_$NAME.jsonl 2>/dev/null || echo 0) marks=$(grep -c "^\[ALSI\] case [0-9]* arm" $K/$NAME.log) skips=$(grep -c "^\[ALSI\] case .*SKIP" $K/$NAME.log) epsdec=$(grep -o "eps_dec_max=[0-9.e+-]*" $K/$NAME.log | sort -u | head -3 | tr "\n" ";") tb=$(grep -ci traceback $K/$NAME.log) results=$(find $K/$NAME -name result.json 2>/dev/null | wc -l)" >> $K/chain_gpu$GPU.log
}

if [ "$WAITPID" != "-" ]; then
  while kill -0 $WAITPID 2>/dev/null; do sleep 30; done
fi
echo "START $(date +%F_%T) gpu=$GPU" >> $K/chain_gpu$GPU.log
run_one smoke gtex2.json CUDA_LAUNCH_BLOCKING=1
n=$(grep -c "^\[ALSI\] case [0-9]* arm" $K/smoke.log); tb=$(grep -ci traceback $K/smoke.log)
res=$(find $K/smoke -name result.json 2>/dev/null | wc -l)
bad=$(grep -o "eps_dec_max=[0-9.e+-]*" $K/smoke.log | awk -F= "\$2>1e-3{c++}END{print c+0}")
echo "$(date +%H:%M) smoke: marks=$n traceback=$tb result=$res eps_dec_over_1e-3=$bad" >> $K/chain_gpu$GPU.log
if [ "${n:-0}" -ge 2 ] && [ "${tb:-1}" -eq 0 ] && [ "${res:-0}" -ge 2 ] && [ "${bad:-1}" -eq 0 ]; then
  for ds in gtex tcga_expert_vqa tcga_slidebench panda tcga; do run_one nat_$ds dcs20_$ds.json; done
  for ds in gtex tcga_expert_vqa tcga_slidebench panda tcga; do
    run_one can_$ds dcs20_$ds.json VLMAS_KV_ROUTE=1 VLMAS_KV_ROUTE_MODE=canonical
  done
  echo "ALL DONE $(date +%F_%T) gpu=$GPU" >> $K/chain_gpu$GPU.log
else
  echo "SMOKE GATE FAILED - full run not launched" >> $K/chain_gpu$GPU.log
fi
