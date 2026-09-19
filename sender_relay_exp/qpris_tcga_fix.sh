#!/bin/bash
# Repair: the Q-PRIS chain died mid-run at qn_tcga (12/15 arms done). Re-run the three
# TCGA arms with exactly the settings qpris_chain.sh used, then hand GPU7 to ALSI.
# usage: qpris_tcga_fix.sh <wait_pid|-> <gpu>
set -u
WAITPID=$1; GPU=$2
TREE=/home/users/whddn12316/wsi_latent_0915_decode_hj
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
MP=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
K=$TREE/sender_relay_exp/runs/qpris
SM=$TREE/sender_relay_exp/smoke
mkdir -p $K

run_one () {
  NAME=$1; DSJ=$2; shift 2
  rm -rf $K/$NAME
  cd $TREE && env VLMAS_ATTN_IMPLEMENTATION=sdpa VLMAS_WSI_CONSOL=1 VLMAS_WSI_CONSOL_MODE=qpris \
    VLMAS_WSI_CONSOL_BUDGET=512 "$@" \
    PYTHONPATH=$TREE PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout 18000 \
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
  echo "DONE $NAME rc=$? marks=$(grep -c '\[WSIConsol\] qpris' $K/$NAME.log) qsrc=$(grep -o 'q_src=[a-z_]*' $K/$NAME.log | sort | uniq -c | tr -s ' ' | tr '\n' ';') tb=$(grep -c Traceback $K/$NAME.log) results=$(find $K/$NAME -name result.json 2>/dev/null | wc -l)" >> $K/chain_gpu$GPU.log
}

if [ "$WAITPID" != "-" ]; then
  while kill -0 $WAITPID 2>/dev/null; do sleep 30; done
fi
echo "REPAIR START $(date +%F_%T) gpu=$GPU (tcga arms only)" >> $K/chain_gpu$GPU.log
run_one qn_tcga dcs20_tcga.json VLMAS_QPRIS_Q=none
run_one qt_tcga dcs20_tcga.json VLMAS_QPRIS_Q=true
run_one qs_tcga dcs20_tcga.json VLMAS_QPRIS_Q=shuffled
echo "REPAIR DONE $(date +%F_%T) gpu=$GPU" >> $K/chain_gpu$GPU.log
