#!/usr/bin/env bash
set -euo pipefail

#######################################
# Run all 3 local VQA baselines on WSI-VQA (WsiVQA_test.json) sequentially.
#
# Unlike BCNB (flat JPGs + per-task report), WSI-VQA reads pyramidal .svs slides
# from SLIDE_DIR and each per-model script runs the open-ended general evaluator
# (eval/metrics.py) internally — so there is no separate per-task report step.
#
# Results layout (shared RUN_ID timestamp across models):
#   <RESULTS_ROOT>/qwen3-vl-4b/<RUN_ID>/
#   <RESULTS_ROOT>/huatuogpt-vision-7b/<RUN_ID>/
#   <RESULTS_ROOT>/patho-r1-7b/<RUN_ID>/
#
# Each model shows only a tqdm bar on the terminal; detailed output goes to
# <model>/<RUN_ID>/logs/*.log. Runs are resumable: re-invoke with the same
# RUN_ID to continue a crashed run.
#
# NOTE: MCQ_STYLE is intentionally NOT exported here — it differs per model
# (qwen3-vl=text, huatuo/patho=letter). Each per-model script keeps its default.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash baselines/run_wsivqa_all.sh
#   RUN_ID=20260701_120000 bash baselines/run_wsivqa_all.sh   # resume
#######################################

HJ_ROOT="${HJ_ROOT:-/home/super/hj/HJ}"
LATENTMAS_ROOT="${LATENTMAS_ROOT:-/home/super/hj/wsi_latentmas_0701}"

export TEST_JSON="${TEST_JSON:-$HJ_ROOT/WSI-VQA/dataset/WSI_captions/WsiVQA_test.json}"
export INPUT_JSON="${INPUT_JSON:-$TEST_JSON}"
export SLIDE_DIR="${SLIDE_DIR:-/media/super/4TB/hj/DATA_SVS}"
export THUMBNAIL_DIR="${THUMBNAIL_DIR:-$HJ_ROOT/baseline_results/wsi-vqa/shared_thumbnails_dx1}"
# WSI-VQA reads slides from SLIDE_DIR (not a flat image dir) — keep IMAGE_ROOT empty.
export IMAGE_ROOT="${IMAGE_ROOT:-}"
export DATASET_TAG="${DATASET_TAG:-WsiVQA_test}"
export PROMPT_STYLE="${PROMPT_STYLE:-wsi}"
export RESULTS_ROOT="${RESULTS_ROOT:-$LATENTMAS_ROOT/results/wsi-vqa}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Shared timestamp so the 3 model dirs line up under one run.
export RUN_ID="${RUN_ID:-$(date '+%Y%m%d_%H%M%S')}"

MODELS=(qwen3-vl-4b huatuogpt-vision-7b patho-r1-7b)

cd "$LATENTMAS_ROOT"
echo "RUN_ID=$RUN_ID"
echo "RESULTS_ROOT=$RESULTS_ROOT  PROMPT_STYLE=$PROMPT_STYLE  GPU=$CUDA_VISIBLE_DEVICES"
echo "INPUT_JSON=$INPUT_JSON"
echo "SLIDE_DIR=$SLIDE_DIR  THUMBNAIL_DIR=$THUMBNAIL_DIR"

# ── inference: stop the chain if any model fails ──────────────────────────────
echo "[1/3] qwen3-vl-4b"        && bash baselines/qwen3-vl-4b.sh \
&& echo "[2/3] huatuogpt-vision-7b" && bash baselines/huatuogpt-vision-7b.sh \
&& echo "[3/3] patho-r1-7b"        && bash baselines/patho-r1-7b.sh

# ── summary: each per-model script already ran eval/metrics.py internally ──────
echo "=== general eval summary (per model) ==="
for m in "${MODELS[@]}"; do
  txt="$RESULTS_ROOT/$m/$RUN_ID/eval/general_eval.txt"
  echo "----- $m -----"
  if [[ -f "$txt" ]]; then
    cat "$txt"
  else
    echo "  (no general_eval.txt at $txt)"
  fi
done

echo "Done. Per-model eval dirs: $RESULTS_ROOT/<model>/$RUN_ID/eval/"
