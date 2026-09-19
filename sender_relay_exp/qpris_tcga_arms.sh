#!/bin/bash
# Re-run the remaining Q-PRIS TCGA arms with qpris_chain.sh settings verbatim.
# The 04:41 repair died with CUDA OOM because it shared GPU7 with an ALSI run.
# usage: qpris_tcga_arms.sh <gpu> <arm> [arm ...]      arm in {qn,qt,qs}
set -u
GPU=$1; shift
TREE=/home/users/whddn12316/wsi_latent_0915_decode_hj
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
MP=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
K=$TREE/sender_relay_exp/runs/qpris
SM=$TREE/sender_relay_exp/smoke
mkdir -p $K

run_one () {
  NAME=$1; QARM=$2
  rm -rf $K/$NAME
  cd $TREE && env VLMAS_ATTN_IMPLEMENTATION=sdpa VLMAS_WSI_CONSOL=1 VLMAS_WSI_CONSOL_MODE=qpris \
    VLMAS_WSI_CONSOL_BUDGET=512 VLMAS_QPRIS_Q=$QARM \
    PYTHONPATH=$TREE PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout 18000 \
    $PY -m wsi_latentmas.pipeline.latent_mas --variant base --backbone qwen3-vl \
    --dataset $SM/dcs20_tcga.json --slide-root $MP/slides \
    --model /home/users/whddn12316/models/Qwen3-VL-4B-Thinking --device cuda:0 \
    --latent-steps 10 --patch-budget 8 --max-model-len 8192 --navigator-control-tokens 512 \
    --temperature 0.0 --top-p 1.0 --seed 42 --answerer-max-new-tokens 512 \
    --terminal-no-repeat-ngram-size 0 --terminal-repetition-penalty 1.0 --terminal-antiloop-method none \
    --answerer-protocol structured_json --transport-mode cumulative --realign-method wa --case-retries 0 \
    --deterministic --answerer-greedy --no-answerer-thinking --canonical-open-options --navigator-kv \
    --io-pipeline --answerer-rationale --no-save-navigation-pngs \
    --output-root $K/$NAME > $K/$NAME.log 2>&1 < /dev/null
  rc=$?
  marks=$(grep -c 'WSIConsol. qpris' $K/$NAME.log)
  qsrc=$(grep -o 'q_src=[a-z_]*' $K/$NAME.log | sort | uniq -c | tr -s ' ' | tr '\n' ';')
  tb=$(grep -c Traceback $K/$NAME.log)
  oom=$(grep -c 'out of memory' $K/$NAME.log)
  res=$(find $K/$NAME -name result.json 2>/dev/null | wc -l)
  echo "DONE $NAME rc=$rc marks=$marks qsrc=$qsrc tb=$tb oom=$oom results=$res" >> $K/chain_gpu$GPU.log
}

echo "ARMS START $(date +%F_%T) gpu=$GPU arms=$*" >> $K/chain_gpu$GPU.log
for a in "$@"; do
  case $a in
    qn) run_one qn_tcga none ;;
    qt) run_one qt_tcga true ;;
    qs) run_one qs_tcga shuffled ;;
  esac
done
echo "ARMS DONE $(date +%F_%T) gpu=$GPU" >> $K/chain_gpu$GPU.log
