#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 <dataset.json> <slide-root> <model-dir> <output-root>" >&2
  exit 2
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"
DATASET="$1"
SLIDE_ROOT="$2"
MODEL="$3"
OUTPUT_ROOT="$4"

# Match the established fast Nav3 runner: independent-scale navigation, SDPA,
# bounded patch/model budgets, queued slide I/O, and no navigation image dumps.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export VLMAS_PROMPT_SET=prompt1
export VLMAS_ATTN_IMPLEMENTATION=sdpa
export VLMAS_NAV2=1
export VLMAS_NAV3_INDEPENDENT=1
export VLMAS_CROSS_SCALE_ROUTER=0

mkdir -p "$OUTPUT_ROOT"
for variant in \
  pruning_eadaprune_pathology_latent_kv_relay2_dual \
  pruning_atp_sap_pathology_latent_kv_relay2_dual; do
  run_root="$OUTPUT_ROOT/$variant"
  if [[ -e "$run_root" ]]; then
    echo "output already exists: $run_root" >&2
    exit 1
  fi
  "$PYTHON" -m vision_text_mas.latent_hf_ablation_cli \
    --variant "$variant" \
    --dataset "$DATASET" \
    --slide-root "$SLIDE_ROOT" \
    --model "$MODEL" \
    --output-root "$run_root" \
    --device "$DEVICE" \
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
    --case-retries 0
done
