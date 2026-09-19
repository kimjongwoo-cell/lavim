#!/bin/bash
# Visual Binding Step — protocol steps [4]+[5] (GPU 8, WSI-VQA idx 0-29, 4B step5,
# latent-base: pruning/realloc OFF, architecture frozen).
#
# Usage: bash run_visual_bind_exp.sh "<LAYERS>"   e.g. "18" or "14,18,22"
#   LAYERS should come from the step-[3] coarse layer probe (visual-carrying band).
#
# Arms (last Reasoner latent step only; everything else untouched):
#   base          — reuse runs/kv_swap_probe/base_probe (identical flags/cases)
#   matched       — bind z_5 to this slide's visual KV at LAYERS
#   wrong_slide   — bind to a FROZEN donor slide's visual K/V (case 0 = donor,
#                   captured no-swap; exclude idx 0, treat idx 1-3 as same-slide)
#   shuffled_v    — keep K, permute V across visual columns (content-binding control)
#
# Pre-registered gate (VISUAL_BIND_PREREG.md): matched > base AND
# matched > wrong_slide AND matched > shuffled_v (paired flips + accuracy).
set -u
TREE_LOCK=/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/gpu8.lock
exec 9>>$TREE_LOCK
flock -n 9 || { echo "GPU8 LOCK HELD by another process — refusing to start"; exit 1; }
TREE=/home/users/whddn12316/wsi_latent_0915_decode_hj
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
OUT=$TREE/sender_relay_exp/runs/visual_bind_exp
LAYERS=${1:?usage: run_visual_bind_exp.sh LAYERS (e.g. 18 or 14,18,22)}
IDX=""; for i in $(seq 0 29); do IDX="$IDX --dataset-index $i"; done
COMMON="--variant base --backbone qwen3-vl \
 --dataset /home/users/whddn12316/datasets/WSI-VQA/WsiVQA_test.json \
 --slide-root /home/users/whddn12316/datasets/WSI-VQA/DATA_SVS \
 --model /home/users/whddn12316/models/Qwen3-VL-4B-Thinking --device cuda:0 \
 --latent-steps 5 --patch-budget 8 --max-model-len 8192 --navigator-control-tokens 512 \
 --temperature 0.0 --top-p 1.0 --seed 42 --answerer-max-new-tokens 512 \
 --terminal-no-repeat-ngram-size 0 --terminal-repetition-penalty 1.0 \
 --terminal-antiloop-method none --answerer-protocol structured_json \
 --transport-mode cumulative --realign-method wa --case-retries 0 \
 --deterministic --answerer-greedy --no-answerer-thinking --canonical-open-options \
 --navigator-kv --io-pipeline --answerer-rationale --no-save-navigation-pngs"

run_arm () {
  local name=$1 mode=$2
  local t0=$SECONDS
  VLMAS_VISUAL_BIND=1 VLMAS_VISUAL_BIND_LAYERS=$LAYERS VLMAS_VISUAL_BIND_MODE=$mode \
  PYTHONPATH=$TREE PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=8 \
    $PY -m wsi_latentmas.pipeline.latent_mas $COMMON $IDX \
    --output-root $OUT/$name > $OUT/$name.log 2>&1
  echo "BIND-ARM $name layers=$LAYERS rc=$? elapsed=$((SECONDS-t0))s"
}

mkdir -p $OUT
echo "layers=$LAYERS" > $OUT/LAYERS.txt
run_arm matched_L${LAYERS//,/_}   matched
run_arm wrong_L${LAYERS//,/_}     wrong_slide
run_arm shuffled_L${LAYERS//,/_}  shuffled_v
echo "VISUAL BIND DONE"
