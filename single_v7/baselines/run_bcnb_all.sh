#!/usr/bin/env bash
set -euo pipefail

#######################################
# Run all 3 local VQA baselines on SlideBench-VQA-BCNB sequentially, then write
# per-task accuracy reports.
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
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash baselines/run_bcnb_all.sh
#   RUN_ID=20260630_021500 bash baselines/run_bcnb_all.sh   # resume
#######################################

HJ_ROOT="${HJ_ROOT:-/home/super/hj/HJ}"
LATENTMAS_ROOT="${LATENTMAS_ROOT:-/home/super/hj/wsi_latentmas_0701}"
BCNB_DIR="${BCNB_DIR:-/media/super/4TB/hj/SlideBench_BCNB}"

export INPUT_JSON="${INPUT_JSON:-$BCNB_DIR/SlideBench_BCNB_test.json}"
export THUMBNAIL_DIR="${THUMBNAIL_DIR:-$BCNB_DIR/thumbnails}"
export IMAGE_ROOT="${IMAGE_ROOT:-$BCNB_DIR/WSIs/WSIs}"
export DATASET_TAG="${DATASET_TAG:-SlideBench_BCNB}"
export PROMPT_STYLE="${PROMPT_STYLE:-bcnb}"
export RESULTS_ROOT="${RESULTS_ROOT:-$LATENTMAS_ROOT/results/bcnb}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Shared timestamp so the 3 model dirs line up under one run.
export RUN_ID="${RUN_ID:-$(date '+%Y%m%d_%H%M%S')}"

MODELS=(qwen3-vl-4b huatuogpt-vision-7b patho-r1-7b)

cd "$LATENTMAS_ROOT"
echo "RUN_ID=$RUN_ID"
echo "RESULTS_ROOT=$RESULTS_ROOT  PROMPT_STYLE=$PROMPT_STYLE  GPU=$CUDA_VISIBLE_DEVICES"
echo "INPUT_JSON=$INPUT_JSON"

# ── inference: stop the chain if any model fails ──────────────────────────────
echo "[1/3] qwen3-vl-4b"        && bash baselines/qwen3-vl-4b.sh \
&& echo "[2/3] huatuogpt-vision-7b" && bash baselines/huatuogpt-vision-7b.sh \
&& echo "[3/3] patho-r1-7b"        && bash baselines/patho-r1-7b.sh

# ── per-task reports: one per model (no cross-model merge) ────────────────────
echo "=== per-task accuracy (per model) ==="
for m in "${MODELS[@]}"; do
  python eval/bcnb.py "$RESULTS_ROOT/$m/$RUN_ID" --questions_file "$INPUT_JSON"
done

echo "Done. Per-model per-task reports: $RESULTS_ROOT/<model>/$RUN_ID/eval/bcnb_task_report.txt"
