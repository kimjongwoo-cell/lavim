#!/bin/bash
# Kill test for the GLVR relay signal on SlideBench (09-18).
#   K0  latplan + NOVA rho, rerun as is                      -> run-to-run noise floor
#   K2  same as K1 but u = 0: o' = o - 0.75*m_B (only the H_C share is removed)
#   K3  same as K1 but random directions with each u_k's own norm
#   (KARMS="K2 K3" bash kill_test_0918.sh)
#   K1  + GLVR relay only (formation off, lambda 0.75) but the relayed visual reads U_V come from the
#       PREVIOUS case (VLMAS_GLVR_USWAP=1): same size, wrong slide.
# Questions: the 7 where relay-only changed correctness vs NOVA rho (gain 49 73 127 138 151 180, loss 69)
# plus 20 where both gave the same answer. Each group starts with a concordant question (0/1/2) that
# only primes the swap and is excluded from the verdict.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=/home/users/whddn12316/wsi_latent_0915_decode_hj
DATA=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
OUT=$HERE/runs/kill_0918
SLIDE_ROOT=${SLIDE_ROOT:-/tmp/hj_slide_root}
GROUPS_=("0 14 40 69 86 114 138 151 173" "1 29 49 74 102 119 143 164 180" "2 61 73 80 109 127 135 137 144")
NOVA="VLMAS_C1_QASC=1 VLMAS_C1_SELECT=nova VLMAS_NOVA_KEEP=0.25 VLMAS_C1_META_BRIDGE=1 VLMAS_NOVA_RHO=1"
BASE_ENV="VLMAS_PLAN_LATENT_ONLY=1 VLMAS_PROMPT_SET=prompt1 VLMAS_ATTN_IMPLEMENTATION=sdpa VLMAS_NAV4=1 VLMAS_NAV2=1 VLMAS_CROSS_SCALE_ROUTER=0 VLMAS_EFF_DUMP=1 VLMAS_ANSWERER_GRADE_RULE=1"
RTASK=$(sed -n "s/^RTASK=\$'\(.*\)'$/\1/p" $REPO/sender_relay_exp/rtask_nav4_nonav2_arm.sh | head -1)
RTASK=$(printf '%b' "$RTASK")
ARGS=(--variant base --backbone qwen3-vl --model /home/users/whddn12316/models/Qwen3-VL-4B-Thinking --slide-root "$SLIDE_ROOT" --device cuda:0
      --latent-steps 10 --patch-budget 12 --max-model-len 12288 --navigator-control-tokens 512
      --temperature 0.0 --top-p 1.0 --seed 42 --deterministic --answerer-greedy --no-answerer-thinking
      --answerer-max-new-tokens 512 --answerer-rationale --answerer-protocol structured_json
      --canonical-open-options --navigator-kv --io-pipeline --no-save-navigation-pngs --case-retries 0)
launch() {   # launch <K0|K1> <gpu> <ids>
  local arm=$1 g=$2 ids=$3 hooks extra idx="" i
  for i in $ids; do idx="$idx --dataset-index $i"; done
  if [ "$arm" = K0 ]; then hooks=$REPO/tmp_hj/nova_rho_latplan_hooks; extra="$NOVA"
  elif [ "${arm:0:1}" = R ]; then     # §12 re-read put in the GLVR H_C slot: R1 native query, R2 context query, R4 R2 on the previous slide's visual KV
       hooks=$REPO/tmp_hj/rcvr
       case $arm in R1) rm=identity ;; R2) rm=ctx ;; R4) rm=wrong ;; esac
       extra="$NOVA VLMAS_RNLCR=diag VLMAS_GLVR=$OUT/$arm/glvr_gpu$g.jsonl VLMAS_GLVR_GENCAP=1 VLMAS_GLVR_CHAIN=nova_rho_latplan_hooks VLMAS_RCVR=$OUT/$arm/rcvr_gpu$g.jsonl VLMAS_RCVR_LAMBDA=0.75 VLMAS_RCVR_MODE=$rm VLMAS_RCVR_INJECT=hc"
  else hooks=$REPO/tmp_hj/glvr_kill
       case $arm in K1) ctl="VLMAS_GLVR_USWAP=1" ;; K2) ctl="VLMAS_GLVR_UMODE=zero" ;; K3) ctl="VLMAS_GLVR_UMODE=random" ;; esac
       extra="$NOVA VLMAS_RNLCR=ground VLMAS_RNLCR_S1_STEPS=0 VLMAS_GLVR=$OUT/$arm/glvr_gpu$g.jsonl VLMAS_GLVR_APPLY=0.75 VLMAS_GLVR_CHAIN=nova_rho_latplan_hooks $ctl"; fi
  local o=$OUT/$arm/tcga_slidebench/gpu${g}_attempt_$(date +%Y%m%d_%H%M%S)
  mkdir -p "$OUT/$arm/tcga_slidebench"
  ( cd $REPO && env CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$hooks:$REPO PYTHONUNBUFFERED=1 $BASE_ENV $extra VLMAS_REASONER_TASK="$RTASK" \
      $PY -m wsi_latentmas.pipeline.latent_mas --dataset $DATA/tcga_slidebench.json --output-root $o "${ARGS[@]}" $idx > $o.log 2>&1 < /dev/null ) &
  echo "$arm gpu$g pid $! ids [$ids] -> $o"
}
mkdir -p $OUT
KARMS=${KARMS:-"K0 K1"}   # K2: relay with u = 0 (H_C suppression only), K3: random directions of the same norm
for j in 0 1 2; do for a in $KARMS; do launch $a $((6 + j)) "${GROUPS_[$j]}"; sleep 2; done; done | tee -a $OUT/launch.log
wait
echo "ALL DONE $(date +%H:%M)" | tee -a $OUT/launch.log
