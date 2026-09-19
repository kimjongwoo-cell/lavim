#!/bin/bash
# Order 88 §12.5 kill test 1 (5 diagnostic cases): does the context-conditioned query actually differ
# from the native one, and does it change the current-slide visual read?
#   identity : q_hat := q (correction must be exactly 0, answers must equal the base)
#   ctx      : main candidate, lambda 1 (measurement + answers)
# Base = latplan + C1 NOVA rho. Visual columns come from the Reasoner generation capture (no replay).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=/home/users/whddn12316/wsi_latent_0915_decode_hj
DATA=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
OUT=$HERE/runs/rcvr_kill1
SMOKE=$REPO/sender_relay_exp/smoke/glvr5.json
MODES=${MODES:-"identity ctx"}
NOVA="VLMAS_C1_QASC=1 VLMAS_C1_SELECT=nova VLMAS_NOVA_KEEP=0.25 VLMAS_C1_META_BRIDGE=1 VLMAS_NOVA_RHO=1"
BASE_ENV="VLMAS_PLAN_LATENT_ONLY=1 VLMAS_PROMPT_SET=prompt1 VLMAS_ATTN_IMPLEMENTATION=sdpa VLMAS_NAV4=1 VLMAS_NAV2=1 VLMAS_CROSS_SCALE_ROUTER=0 VLMAS_EFF_DUMP=1 VLMAS_ANSWERER_GRADE_RULE=1"
RTASK=$(printf '%b' "$(sed -n "s/^RTASK=\$'\(.*\)'$/\1/p" $REPO/sender_relay_exp/rtask_nav4_nonav2_arm.sh | head -1)")
ARGS=(--variant base --backbone qwen3-vl --model /home/users/whddn12316/models/Qwen3-VL-4B-Thinking --slide-root ${SLIDE_ROOT:-/tmp/hj_slide_root} --device cuda:0
      --latent-steps 10 --patch-budget 12 --max-model-len 12288 --navigator-control-tokens 512
      --temperature 0.0 --top-p 1.0 --seed 42 --deterministic --answerer-greedy --no-answerer-thinking
      --answerer-max-new-tokens 512 --answerer-rationale --answerer-protocol structured_json
      --canonical-open-options --navigator-kv --io-pipeline --no-save-navigation-pngs --case-retries 0
      --dataset-index 0 --dataset-index 1 --dataset-index 2 --dataset-index 3 --dataset-index 4)
g=6
for mode in $MODES; do
  o=$OUT/$mode/gpu${g}_attempt_$(date +%Y%m%d_%H%M%S); mkdir -p $OUT/$mode
  extra="$NOVA VLMAS_RNLCR=diag VLMAS_GLVR=$OUT/$mode/glvr.jsonl VLMAS_GLVR_GENCAP=1 VLMAS_GLVR_CHAIN=nova_rho_latplan_hooks VLMAS_RCVR=$OUT/$mode/rcvr.jsonl VLMAS_RCVR_LAMBDA=${LAMBDA:-1} VLMAS_RCVR_MODE=$mode"
  ( cd $REPO && env CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$REPO/tmp_hj/rcvr:$REPO PYTHONUNBUFFERED=1 $BASE_ENV $extra VLMAS_REASONER_TASK="$RTASK" \
      $PY -m wsi_latentmas.pipeline.latent_mas --dataset $SMOKE --output-root $o "${ARGS[@]}" > $o.log 2>&1 < /dev/null ) &
  echo "$mode gpu$g pid $! -> $o"; g=$((g + 1)); sleep 2
done | tee -a $OUT/launch.log
wait; echo "ALL DONE $(date +%H:%M)" | tee -a $OUT/launch.log
