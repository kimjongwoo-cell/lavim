#!/bin/bash
# CSIS sigma sweep on gtex 20 cases, with the two references needed to read it:
#   base       = 압축 없음            (Δ의 기준)
#   pcris_v2   = 공간 마스크 없음      (sigma -> inf 끝점)
#   csis_<s>   = sigma 256 / 1024 / 4096
# 전부 eager · step 10 · budget 8 · mlen 8192 · B=512 로 고정. 오직 selection 만 다름.
# usage: csis_sweep.sh <gpu>
set -u
GPU=$1
TREE=/home/users/whddn12316/wsi_latent_0915_decode_hj
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
MP=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
K=$TREE/sender_relay_exp/runs/csis/sweep
mkdir -p $K

run_one () {   # $1 name  $2... extra env
  NAME=$1; shift
  rm -rf $K/$NAME
  cd $TREE && env VLMAS_EFF_DUMP=1 VLMAS_ATTN_IMPLEMENTATION=eager "$@" \
    PYTHONPATH=$TREE PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout 7200 \
    $PY -m wsi_latentmas.pipeline.latent_mas --variant base --backbone qwen3-vl \
    --dataset $TREE/sender_relay_exp/smoke/dcs20_gtex.json --slide-root $MP/slides \
    --model /home/users/whddn12316/models/Qwen3-VL-4B-Thinking --device cuda:0 \
    --latent-steps 10 --patch-budget 8 --max-model-len 8192 --navigator-control-tokens 512 \
    --temperature 0.0 --top-p 1.0 --seed 42 --answerer-max-new-tokens 512 \
    --terminal-no-repeat-ngram-size 0 --terminal-repetition-penalty 1.0 --terminal-antiloop-method none \
    --answerer-protocol structured_json --transport-mode cumulative --realign-method wa --case-retries 0 \
    --deterministic --answerer-greedy --no-answerer-thinking --canonical-open-options --navigator-kv \
    --io-pipeline --answerer-rationale --no-save-navigation-pngs \
    --output-root $K/$NAME > $K/$NAME.log 2>&1 < /dev/null
  echo "DONE $NAME rc=$? $(date +%H:%M) marks=$(grep -c 'WSIConsol.' $K/$NAME.log) x5=$(grep -o 'kept_by_mag={5: [0-9]*' $K/$NAME.log | grep -o '[0-9]*$' | awk '{s+=$1;n++} END{if(n)printf "%.0f",s/n}') tb=$(grep -c Traceback $K/$NAME.log) oom=$(grep -c 'out of memory' $K/$NAME.log) results=$(find $K/$NAME -name result.json 2>/dev/null | wc -l)" >> $K/sweep_gpu$GPU.log
}

echo "SWEEP START $(date +%F_%T) gpu=$GPU" >> $K/sweep_gpu$GPU.log
run_one base
run_one pcris_v2 VLMAS_WSI_CONSOL=1 VLMAS_WSI_CONSOL_MODE=qpris VLMAS_WSI_CONSOL_BUDGET=512 VLMAS_QPRIS_Q=none
for s in 256 1024 4096; do
  run_one csis_$s VLMAS_WSI_CONSOL=1 VLMAS_WSI_CONSOL_MODE=csis VLMAS_WSI_CONSOL_BUDGET=512 VLMAS_CSIS_SIGMA=$s
done
echo "SWEEP DONE $(date +%F_%T) gpu=$GPU" >> $K/sweep_gpu$GPU.log
