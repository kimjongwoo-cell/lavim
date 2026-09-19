#!/bin/bash
# Q-PRIS three-arm chain on the FULL MultiPathQA sets (not the dcs20 subsets).
# Flags match the dashboard Latent base(step 10, budget 8, mlen 8192) arm; attn defaults to
# eager because backbone/qwen3vl.py:69 defaults to eager and the expq *_base10 runs behind the
# dashboard set no VLMAS_ATTN_IMPLEMENTATION. ATTN=sdpa overrides.
#   usage: qpris_full_chain.sh <wait_pid|-> <gpu> <ds>:<arm> [<ds>:<arm> ...]
#   arm in {none,true,shuffled} -> output prefix qn_/qt_/qs_
set -u
WAITPID=$1; GPU=$2; shift 2
TREE=/home/users/whddn12316/wsi_latent_0915_decode_hj
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
MP=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
K=$TREE/sender_relay_exp/runs/qpris_full
mkdir -p $K

run_one () {
  DS=$1; QARM=$2
  case $QARM in none) P=qn ;; true) P=qt ;; shuffled) P=qs ;; *) echo "bad arm $QARM"; return 1 ;; esac
  NAME=${P}_${DS}
  rm -rf $K/$NAME
  cd $TREE && env VLMAS_EFF_DUMP=1 VLMAS_ATTN_IMPLEMENTATION=${ATTN:-eager} VLMAS_WSI_CONSOL=1 VLMAS_WSI_CONSOL_MODE=qpris \
    VLMAS_WSI_CONSOL_BUDGET=512 VLMAS_QPRIS_Q=$QARM \
    PYTHONPATH=$TREE PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout 14400 \
    $PY -m wsi_latentmas.pipeline.latent_mas --variant base --backbone qwen3-vl \
    --dataset $MP/$DS.json --slide-root $MP/slides \
    --model /home/users/whddn12316/models/Qwen3-VL-4B-Thinking --device cuda:0 \
    --latent-steps 10 --patch-budget 8 --max-model-len 8192 --navigator-control-tokens 512 \
    --temperature 0.0 --top-p 1.0 --seed 42 --answerer-max-new-tokens 512 \
    --terminal-no-repeat-ngram-size 0 --terminal-repetition-penalty 1.0 --terminal-antiloop-method none \
    --answerer-protocol structured_json --transport-mode cumulative --realign-method wa --case-retries 0 \
    --deterministic --answerer-greedy --no-answerer-thinking --canonical-open-options --navigator-kv \
    --io-pipeline --answerer-rationale --no-save-navigation-pngs \
    --output-root $K/$NAME > $K/$NAME.log 2>&1 < /dev/null
  rc=$?
  echo "DONE $NAME rc=$rc $(date +%H:%M) marks=$(grep -c 'WSIConsol. qpris' $K/$NAME.log) qsrc=$(grep -o 'q_src=[a-z_]*' $K/$NAME.log | sort | uniq -c | tr -s ' ' | tr '\n' ';') tb=$(grep -c Traceback $K/$NAME.log) oom=$(grep -c 'out of memory' $K/$NAME.log) results=$(find $K/$NAME -name result.json 2>/dev/null | wc -l)" >> $K/full_gpu$GPU.log
}

if [ "$WAITPID" != "-" ]; then
  while kill -0 $WAITPID 2>/dev/null; do sleep 30; done
fi
echo "FULL START $(date +%F_%T) gpu=$GPU jobs=$*" >> $K/full_gpu$GPU.log
for job in "$@"; do
  run_one "${job%%:*}" "${job##*:}"
done
echo "FULL DONE $(date +%F_%T) gpu=$GPU" >> $K/full_gpu$GPU.log
