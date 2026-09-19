#!/bin/bash
# LaViM Method (Sec. 3.3 + 3.4) smoke, GPU 8, WSI-VQA idx 0-29, 4B step5.
# Base = runs/kv_swap_probe2/base_probe (same cases/flags).
# Arms:
#   consol_only — §3.3 consolidation (task-weighted coverage selection +
#                 physical K/V compaction at the Reasoner boundary), keep 0.25
#   bind_all    — §3.4 evidence-bound terminal latent step over the FULL
#                 visual cache (all layers, R*-only softmax)
#   lavim_full  — §3.3 + §3.4 composed (bind reads the consolidated R*)
set -u
TREE_LOCK=/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/gpu8.lock
exec 9>>$TREE_LOCK
flock -n 9 || { echo "GPU8 LOCK HELD by another process — refusing to start"; exit 1; }
TREE=/home/users/whddn12316/wsi_latent_0915_decode_hj
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
OUT=$TREE/sender_relay_exp/runs/lavim_method_smoke
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
  local name=$1; shift
  local dest=$OUT/$name
  if [ -d "$dest" ]; then echo "SKIP $name (exists)"; return; fi
  local t0=$SECONDS
  env "$@" \
  PYTHONPATH=$TREE PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=8 \
    $PY -m wsi_latentmas.pipeline.latent_mas $COMMON $IDX \
    --output-root $dest > $OUT/$name.log 2>&1
  echo "LAVIM-ARM $name rc=$? elapsed=$((SECONDS-t0))s"
}

mkdir -p $OUT
run_arm consol_only VLMAS_WSI_CONSOL=1 VLMAS_WSI_CONSOL_KEEP=0.25
run_arm bind_all    VLMAS_VISUAL_BIND=1 VLMAS_VISUAL_BIND_LAYERS=all VLMAS_VISUAL_BIND_NO_SELF=1
run_arm lavim_full  VLMAS_WSI_CONSOL=1 VLMAS_WSI_CONSOL_KEEP=0.25 \
                    VLMAS_VISUAL_BIND=1 VLMAS_VISUAL_BIND_LAYERS=all VLMAS_VISUAL_BIND_NO_SELF=1
echo "LAVIM METHOD SMOKE DONE"
