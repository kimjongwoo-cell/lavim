#!/bin/bash
# LatentMAS Appendix-E mirroring, matched arms on the same 100 cases.
#   off_<ds>  current prompt ("...do not restate it.")
#   on_<ds>   VLMAS_LATENT_RESTATE=1 ("First restate, in text, the original contents...")
# Both arms carry the paper-style readout probe (ask the model to restate what it received)
# AND its no-latent control, so we can see whether the restate framing makes the latent
# handoff actually recoverable, and what it costs in accuracy.
# usage: restate_chain.sh <wait_pid|-> <gpu>
set -u
WAITPID=$1; GPU=$2
TREE=/home/users/whddn12316/wsi_latent_0915_decode_hj
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
MP=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
K=$TREE/sender_relay_exp/runs/restate
SM=$TREE/sender_relay_exp/smoke
mkdir -p $K

run_one () {
  NAME=$1; DSJ=$2; shift 2
  rm -rf $K/$NAME $K/latdec_$NAME.jsonl
  cd $TREE && env VLMAS_LATDEC=$K/latdec_$NAME.jsonl VLMAS_LATDEC_STAGES=reasoner \
    VLMAS_LATDEC_ASK=384 VLMAS_LATDEC_CTRL=1 "$@" \
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
  echo "DONE $NAME rc=$? rows=$(wc -l < $K/latdec_$NAME.jsonl 2>/dev/null || echo 0) marks=$(grep -c '^\[LatDec\] case' $K/$NAME.log) err=$(grep -c ask_error $K/$NAME.log) tb=$(grep -ci traceback $K/$NAME.log) results=$(find $K/$NAME -name result.json 2>/dev/null | wc -l)" >> $K/chain_gpu$GPU.log
}

if [ "$WAITPID" != "-" ]; then
  while kill -0 $WAITPID 2>/dev/null; do sleep 30; done
fi
echo "START $(date +%F_%T) gpu=$GPU" >> $K/chain_gpu$GPU.log
run_one smoke_off gtex2.json CUDA_LAUNCH_BLOCKING=1
run_one smoke_on gtex2.json CUDA_LAUNCH_BLOCKING=1 VLMAS_LATENT_RESTATE=1
n1=$(grep -c '^\[LatDec\] case' $K/smoke_off.log); n2=$(grep -c '^\[LatDec\] case' $K/smoke_on.log)
tb=$(( $(grep -ci traceback $K/smoke_off.log) + $(grep -ci traceback $K/smoke_on.log) ))
r1=$(find $K/smoke_off -name result.json 2>/dev/null | wc -l)
r2=$(find $K/smoke_on -name result.json 2>/dev/null | wc -l)
# the ON arm must actually change the prompt that reaches the model
chg=$(grep -c "First restate" $K/smoke_on/*/result.json 2>/dev/null | awk -F: "{s+=\$2} END {print s+0}")
echo "$(date +%H:%M) smoke: off marks=$n1 res=$r1 | on marks=$n2 res=$r2 | prompt_changed=$chg tb=$tb" >> $K/chain_gpu$GPU.log
if [ "${n1:-0}" -ge 2 ] && [ "${n2:-0}" -ge 2 ] && [ "${tb:-1}" -eq 0 ] \
   && [ "${r1:-0}" -ge 2 ] && [ "${r2:-0}" -ge 2 ] && [ "${chg:-0}" -ge 1 ]; then
  for ds in gtex tcga_expert_vqa tcga_slidebench panda tcga; do
    run_one off_$ds dcs20_$ds.json
    run_one on_$ds dcs20_$ds.json VLMAS_LATENT_RESTATE=1
  done
  echo "ALL DONE $(date +%F_%T) gpu=$GPU" >> $K/chain_gpu$GPU.log
else
  echo "SMOKE GATE FAILED - full run not launched" >> $K/chain_gpu$GPU.log
fi
