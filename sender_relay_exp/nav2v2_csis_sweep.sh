#!/bin/bash
# CSIS on the nav2v2 setting. Flags are nav2v2_mine_run.sh verbatim (sdpa, VLMAS_NAV2=1,
# patch-budget 12, max-model-len 12288); only --dataset points at the same dcs20_gtex 20
# cases the 2048-token sweep uses, so the two lines are read on the same items.
# 12 crops x 256 = 3072 tokens, so B=768 keeps the 25% ratio of the B=512 / 2048 line.
# usage: nav2v2_csis_sweep.sh <gpu>
set -u
GPU=$1
REPO=/home/users/whddn12316/wsi_latent_0915_decode_hj
DATA=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
MODEL=/home/users/whddn12316/models/Qwen3-VL-4B-Thinking
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
K=$REPO/sender_relay_exp/runs/nav2v2_mine/sweep
mkdir -p $K

run_one () {   # $1 name  $2... extra env
  NAME=$1; shift
  rm -rf $K/$NAME
  cd $REPO && env \
    CUDA_VISIBLE_DEVICES="$GPU" \
    PYTHONPATH="$REPO" \
    PYTHONUNBUFFERED=1 \
    VLMAS_ATTN_IMPLEMENTATION=sdpa \
    VLMAS_NAV2=1 \
    VLMAS_EFF_DUMP=1 \
    "$@" \
    timeout 7200 "$PY" -m wsi_latentmas.pipeline.latent_mas \
    --variant base \
    --backbone qwen3-vl \
    --dataset "$REPO/sender_relay_exp/smoke/dcs20_gtex.json" \
    --slide-root "$DATA/slides" \
    --model "$MODEL" \
    --output-root "$K/$NAME" \
    --device cuda:0 \
    --latent-steps 10 \
    --patch-budget 12 \
    --max-model-len 12288 \
    --navigator-control-tokens 512 \
    --temperature 0.0 \
    --top-p 1.0 \
    --seed 42 \
    --deterministic \
    --answerer-greedy \
    --no-answerer-thinking \
    --answerer-max-new-tokens 512 \
    --answerer-rationale \
    --answerer-protocol structured_json \
    --canonical-open-options \
    --navigator-kv \
    --io-pipeline \
    --no-save-navigation-pngs \
    --case-retries 0 > "$K/$NAME.log" 2>&1
  rc=$?
  echo "DONE $NAME rc=$rc $(date +%H:%M) marks=$(grep -c 'WSIConsol. csis' $K/$NAME.log) selfpar=$(grep -o 'self_parent_dropped=[0-9]*' $K/$NAME.log | grep -o '[0-9]*$' | awk '{s+=$1} END{print s+0}') tb=$(grep -c Traceback $K/$NAME.log) oom=$(grep -c 'out of memory' $K/$NAME.log) results=$(find $K/$NAME -name result.json 2>/dev/null | wc -l)" >> $K/sweep_gpu$GPU.log
}

echo "NAV2V2 CSIS SWEEP START $(date +%F_%T) gpu=$GPU B=768" >> $K/sweep_gpu$GPU.log
run_one base
for s in 256 1024 4096; do
  run_one csis_$s VLMAS_WSI_CONSOL=1 VLMAS_WSI_CONSOL_MODE=csis \
          VLMAS_WSI_CONSOL_BUDGET=768 VLMAS_CSIS_SIGMA=$s
done
echo "NAV2V2 CSIS SWEEP DONE $(date +%F_%T) gpu=$GPU" >> $K/sweep_gpu$GPU.log
