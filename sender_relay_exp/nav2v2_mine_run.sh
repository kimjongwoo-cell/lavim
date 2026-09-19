#!/bin/bash
# nav2v2 setting, copied verbatim from the other session's launchers so results are
# comparable, but writing ONLY into our own tree. Nothing under 0912/ is read or written
# at run time; the flags below were transcribed from
#   0912/run_expert_factorial_new.sh          (arms relay_nav / relay_dual /
#                                              pruning_relay_nav / pruning_relay_dual)
#   0912/run_nav2v2_remaining_datasets.sh     (arms base / pruning_b / relay2 / both)
# and differ from ours in four places: sdpa, VLMAS_NAV2=1, patch-budget 12,
# max-model-len 12288.
#
# usage: run_nav2v2.sh <gpu> <dataset> <arm> [first last] [EXTRA_ENV=V ...]
#   dataset : gtex | tcga_expert_vqa | tcga_slidebench | panda | tcga
#   arm     : base | pruning_b | relay2 | both
#             | relay_nav | relay_dual | pruning_relay_nav | pruning_relay_dual
#   first/last : dataset-index range, inclusive. omit = whole set.
set -u
GPU=${1:?usage: run_nav2v2.sh GPU DATASET ARM [FIRST LAST] [ENV=V ...]}
DS=${2:?}
ARM=${3:?}
shift 3
FIRST=""; LAST=""
case "${1:-}" in [0-9]*) FIRST=$1; LAST=$2; shift 2 ;; esac

REPO=/home/users/whddn12316/wsi_latent_0915_decode_hj
DATA=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
MODEL=/home/users/whddn12316/models/Qwen3-VL-4B-Thinking
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
OUT=$REPO/sender_relay_exp/runs/nav2v2_mine
mkdir -p $OUT

case "$ARM" in
  base)               VARIANT=base ;;
  pruning_b)          VARIANT=pruning_b ;;
  relay2)             VARIANT=latent_kv_relay2 ;;
  both)               VARIANT=pruning_b_latent_kv_relay2 ;;
  relay_nav)          VARIANT=latent_kv_relay2_navigator ;;
  relay_dual)         VARIANT=latent_kv_relay2_dual ;;
  pruning_relay_nav)  VARIANT=pruning_b_latent_kv_relay2_navigator ;;
  pruning_relay_dual) VARIANT=pruning_b_latent_kv_relay2_dual ;;
  *) echo "unknown arm: $ARM" >&2; exit 2 ;;
esac

IDS=()
NAME="${ARM}_${DS}"
if [ -n "$FIRST" ]; then
  for ((i=FIRST; i<=LAST; i++)); do IDS+=(--dataset-index "$i"); done
  NAME="${NAME}_part$(printf '%03d' "$FIRST")_$(printf '%03d' "$LAST")"
fi
DEST=$OUT/$NAME
rm -rf "$DEST"

cd $REPO && env \
  CUDA_VISIBLE_DEVICES="$GPU" \
  PYTHONPATH="$REPO" \
  PYTHONUNBUFFERED=1 \
  VLMAS_ATTN_IMPLEMENTATION=sdpa \
  VLMAS_NAV2=1 \
  "$@" \
  "$PY" -m wsi_latentmas.pipeline.latent_mas \
  --variant "$VARIANT" \
  --backbone qwen3-vl \
  --dataset "$DATA/$DS.json" \
  --slide-root "$DATA/slides" \
  --model "$MODEL" \
  --output-root "$DEST" \
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
  --case-retries 0 \
  "${IDS[@]}" > "$DEST.log" 2>&1
rc=$?
echo "DONE $NAME variant=$VARIANT rc=$rc results=$(find $DEST -name result.json 2>/dev/null | wc -l) tb=$(grep -ci traceback $DEST.log)" >> $OUT/chain_gpu$GPU.log
