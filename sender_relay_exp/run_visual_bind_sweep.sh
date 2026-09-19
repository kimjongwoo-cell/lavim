#!/bin/bash
# Visual Binding sweep (user-ordered despite [2] null): matched at L10/L18/L26,
# controls (wrong_slide, shuffled_v) at L18. Base = runs/kv_swap_probe2/base_probe
# (same 30 cases, same flags). Afterwards relaunches the consolidation ablation.
set -u
TREE_LOCK=/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/gpu8.lock
exec 9>>$TREE_LOCK
flock -n 9 || { echo "GPU8 LOCK HELD by another process — refusing to start"; exit 1; }
TREE=/home/users/whddn12316/wsi_latent_0915_decode_hj
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
OUT=$TREE/sender_relay_exp/runs/visual_bind_exp
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
  local name=$1 layers=$2 mode=$3
  local dest=$OUT/$name
  if [ -d "$dest" ]; then echo "SKIP $name (exists)"; return; fi
  local t0=$SECONDS
  VLMAS_VISUAL_BIND=1 VLMAS_VISUAL_BIND_LAYERS=$layers VLMAS_VISUAL_BIND_MODE=$mode \
  PYTHONPATH=$TREE PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=8 \
    $PY -m wsi_latentmas.pipeline.latent_mas $COMMON $IDX \
    --output-root $dest > $OUT/$name.log 2>&1
  echo "BIND-ARM $name rc=$? elapsed=$((SECONDS-t0))s"
}

mkdir -p $OUT
run_arm matched_L10  10 matched
run_arm matched_L18  18 matched
run_arm matched_L26  26 matched
run_arm wrong_L18    18 wrong_slide
run_arm shuffled_L18 18 shuffled_v
echo "VISUAL BIND SWEEP DONE"

# release the GPU8 lock BEFORE relaunching consolidation (the child would
# otherwise inherit fd 9 and its own flock -n would refuse to start)
exec 9>&-
nohup bash $TREE/sender_relay_exp/run_consolidation_ablation.sh \
  >> $TREE/sender_relay_exp/consolidation_driver.log 2>&1 &
echo "consolidation relaunched"
