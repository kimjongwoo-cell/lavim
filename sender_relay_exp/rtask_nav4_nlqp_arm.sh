#!/bin/bash
# 0919 copy of rtask_nav4_nonav2_arm.sh with the C2 NLQP / VCDLP arms (vcdlp_rho, vcdlp_rho_check, vcdlp_rho_uncentered) (tmp_hj/nlqp_hooks, on NOVA rho + latplan):
#   nlqp_rho = canonical NLQP, nlqp_rho_uniform = uniform latent pre-read, nlqp_rho_direct = direct latent residual.
# 0917 nav4 family on the decode-fix base (copy tree wsi_latent_0915_decode_hj).
#   ARM=base   : nav4 + decode fix, no C1            <- the new baseline for the nav4 board
#   ARM=eovc   : + --variant pruning_b + VLMAS_EOVC=1 (C2 #14 unexplained-variation coverage)
#   ARM=pruneb : + --variant pruning_b                (novelty + 0.25*texture + quota; §12 baseline)
#   ARM=secld  : + --variant pruning_b + VLMAS_SECLD=1 (C1 #19 spatial evidence-coverage log-det)
#   ARM=nova   : --variant base + VLMAS_C1_QASC=1 VLMAS_C1_SELECT=nova KEEP=0.25 (C1 #21 NOVA, post-vision, + C1META bridge)
#   ARM=fulr      : nav4 base + VLMAS_RNLCR=fulr (C2 #18 FULR: question-grounded latent formation + sink-norm-conserved read)
#   ARM=nova_fulr : nova arm + VLMAS_RNLCR=fulr
#   ARM=nonav2   : nav4 base WITHOUT VLMAS_NAV2 (Planner decodes no x5/x20 text; the nav4 prompt then carries
#                  fixed_patch_plan's placeholder targets). Compare against runs/nav4/base on the same questions.
# (copy of rtask_nav4_nova_arm.sh with the two FULR arms; memory/rnlcr.py mode fulr, engine unchanged.)
# (copy of rtask_nav4_secld_arm.sh with the nova arm and MAXUSED; same ROOT runs/nav4 and claim dir.)
# (copy of rtask_nav4_arm.sh with the secld arm; same ROOT runs/nav4 and claim dir.)
#
# nav4 is pinned: vision_text_mas/onepass_navigation_nav4.py md5 4f6e14a4 and
# vision_text_mas/onepass_navigator.py md5 198bf899, copied from wsi_latent_0902_2155_hj on 09-17.
# The navigator checks VLMAS_NAV4 BEFORE the nav3/nav2 branches, so nav4 wins; VLMAS_NAV2=1 is kept
# only because the Planner uses it to decode plan targets, and CROSS_SCALE_ROUTER stays 0.
# The PANDA x5 clamp patches onepass_navigation_roots, which nav4 also uses, so it still applies
# (the flag is named VLMAS_NAV3_X5_CLAMP for historical reasons).
# Decode fix is this tree's default; PANDA also gets VLMAS_ANSWERER_GRADE_RULE=1.
# usage: rtask_nav4_arm.sh <arm> <shard|smoke> <nshard> "<gpus>" [ds:n ...]
#   env OUTROOT / CLAIMDIR override the output root and the GPU claim dir.
set -u
ARM=${1:?arm}; SHARD=${2:?shard}; NSHARD=${3:?nshard}; GPUS=${4:?gpus}; shift 4
SPECS=${*:-tcga_expert_vqa:128 tcga_slidebench:197 gtex:190 tcga:221 panda:196}
MAXUSED=${MAXUSED:-2000}   # a GPU counts as free below this used-MiB (co-locate with one other run at 20000)
REPO=/home/users/whddn12316/wsi_latent_0915_decode_hj
DATA=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
MODEL=/home/users/whddn12316/models/Qwen3-VL-4B-Thinking
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
HOOKS=${HOOKS:-$REPO/tmp_hj/lpvh_ntrs_hooks}
[ "$ARM" = latplan ] && [ -z "${HOOKS_OVERRIDE:-}" ] && HOOKS=$REPO/tmp_hj/latplan_hooks
[ -n "${HOOKS_OVERRIDE:-}" ] && HOOKS=$HOOKS_OVERRIDE
ROOT=${OUTROOT:-$REPO/sender_relay_exp/runs/nav4}
VARIANT=base
case $ARM in
  base)   K=$ROOT/base;   ARMENV="" ;;
  eovc)   K=$ROOT/eovc;   VARIANT=pruning_b; ARMENV="VLMAS_EOVC=1" ;;
  pruneb) K=$ROOT/pruneb; VARIANT=pruning_b; ARMENV="" ;;
  secld)  K=$ROOT/secld;  VARIANT=pruning_b; ARMENV="VLMAS_SECLD=1" ;;
  nova)   K=$ROOT/nova;   ARMENV="VLMAS_C1_QASC=1 VLMAS_C1_SELECT=nova VLMAS_NOVA_KEEP=0.25 VLMAS_C1_META_BRIDGE=1" ;;
  fulr)      K=$ROOT/fulr;      ARMENV="VLMAS_RNLCR=fulr VLMAS_RNLCR_LOG=$ROOT/fulr/rnlcr_shard$SHARD.jsonl" ;;
  nonav2)    K=$ROOT/nonav2;    ARMENV="" ;;   # nav4 base with VLMAS_NAV2 unset (see NAV2 line below)
  latplan)   K=$ROOT/latplan;   ARMENV="VLMAS_PLAN_LATENT_ONLY=1" ;;   # nav4 base, Planner text off AND the two target lines removed (tmp_hj/latplan_hooks)
  glvr)      K=$ROOT/glvr;       ARMENV="VLMAS_PLAN_LATENT_ONLY=1 VLMAS_RNLCR=ground VLMAS_ALSI=$ROOT/glvr/alsi_shard$SHARD.jsonl VLMAS_GLVR=$ROOT/glvr/glvr_shard$SHARD.jsonl VLMAS_GLVR_SKIP_ALSI=1 ${GLVR_ENV:-}" ;;   # GLVR canonical-A diagnostic (tmp_hj/glvr)
  glvr_rho)  K=$ROOT/glvr_rho;   ARMENV="VLMAS_C1_QASC=1 VLMAS_C1_SELECT=nova VLMAS_NOVA_KEEP=0.25 VLMAS_C1_META_BRIDGE=1 VLMAS_NOVA_RHO=1 VLMAS_PLAN_LATENT_ONLY=1 VLMAS_RNLCR=ground VLMAS_GLVR=$ROOT/glvr_rho/glvr_shard$SHARD.jsonl VLMAS_GLVR_APPLY=${GLVR_LAMBDA:-0.75} VLMAS_GLVR_CHAIN=nova_rho_latplan_hooks ${GLVR_ENV:-}" ;;   # GLVR canonical-A applied on NOVA rho + latplan (tmp_hj/glvr)
  glvr_rho_nf) K=$ROOT/glvr_rho_nf;   ARMENV="VLMAS_C1_QASC=1 VLMAS_C1_SELECT=nova VLMAS_NOVA_KEEP=0.25 VLMAS_C1_META_BRIDGE=1 VLMAS_NOVA_RHO=1 VLMAS_PLAN_LATENT_ONLY=1 VLMAS_RNLCR=ground VLMAS_GLVR=$ROOT/glvr_rho_nf/glvr_shard$SHARD.jsonl VLMAS_RNLCR_S1_STEPS=0 VLMAS_GLVR_APPLY=${GLVR_LAMBDA:-0.75} VLMAS_GLVR_CHAIN=nova_rho_latplan_hooks ${GLVR_ENV:-}" ;;   # same, grounding formation off (0 Adam steps): replay + relay only
  glvr_rho_fast) K=$ROOT/glvr_rho_fast; ARMENV="VLMAS_C1_QASC=1 VLMAS_C1_SELECT=nova VLMAS_NOVA_KEEP=0.25 VLMAS_C1_META_BRIDGE=1 VLMAS_NOVA_RHO=1 VLMAS_PLAN_LATENT_ONLY=1 VLMAS_RNLCR=diag VLMAS_GLVR=$ROOT/glvr_rho_fast/glvr_shard$SHARD.jsonl VLMAS_GLVR_GENCAP=1 VLMAS_GLVR_APPLY=${GLVR_LAMBDA:-0.75} VLMAS_GLVR_ANSWER_ONLY=1 VLMAS_GLVR_CHAIN=nova_rho_latplan_hooks ${GLVR_ENV:-}" ;;   # glvr_rho_nf, efficient: trajectory captured during generation (no replay), relay on the answer span only
  nova_fulr) K=$ROOT/nova_fulr; ARMENV="VLMAS_C1_QASC=1 VLMAS_C1_SELECT=nova VLMAS_NOVA_KEEP=0.25 VLMAS_C1_META_BRIDGE=1 VLMAS_RNLCR=fulr VLMAS_RNLCR_LOG=$ROOT/nova_fulr/rnlcr_shard$SHARD.jsonl" ;;
  nlqp_rho) K=$ROOT/nlqp_rho; HOOKS=$REPO/tmp_hj/nlqp_hooks; ARMENV="VLMAS_C1_QASC=1 VLMAS_C1_SELECT=nova VLMAS_NOVA_KEEP=0.25 VLMAS_C1_META_BRIDGE=1 VLMAS_NOVA_RHO=1 VLMAS_PLAN_LATENT_ONLY=1 VLMAS_NLQP=native VLMAS_NLQP_LOG=$ROOT/nlqp_rho/nlqp_shard$SHARD.jsonl" ;;
  nlqp_rho_uniform) K=$ROOT/nlqp_rho_uniform; HOOKS=$REPO/tmp_hj/nlqp_hooks; ARMENV="VLMAS_C1_QASC=1 VLMAS_C1_SELECT=nova VLMAS_NOVA_KEEP=0.25 VLMAS_C1_META_BRIDGE=1 VLMAS_NOVA_RHO=1 VLMAS_PLAN_LATENT_ONLY=1 VLMAS_NLQP=uniform VLMAS_NLQP_LOG=$ROOT/nlqp_rho_uniform/nlqp_shard$SHARD.jsonl" ;;
  nlqp_rho_identity) K=$ROOT/nlqp_rho_identity; HOOKS=$REPO/tmp_hj/nlqp_hooks; ARMENV="VLMAS_C1_QASC=1 VLMAS_C1_SELECT=nova VLMAS_NOVA_KEEP=0.25 VLMAS_C1_META_BRIDGE=1 VLMAS_NOVA_RHO=1 VLMAS_PLAN_LATENT_ONLY=1 VLMAS_NLQP=identity VLMAS_NLQP_LOG=$ROOT/nlqp_rho_identity/nlqp_shard$SHARD.jsonl" ;;
  nlqp_rho_direct) K=$ROOT/nlqp_rho_direct; HOOKS=$REPO/tmp_hj/nlqp_hooks; ARMENV="VLMAS_C1_QASC=1 VLMAS_C1_SELECT=nova VLMAS_NOVA_KEEP=0.25 VLMAS_C1_META_BRIDGE=1 VLMAS_NOVA_RHO=1 VLMAS_PLAN_LATENT_ONLY=1 VLMAS_NLQP=direct VLMAS_NLQP_LOG=$ROOT/nlqp_rho_direct/nlqp_shard$SHARD.jsonl" ;;
  vcdlp_rho) K=$ROOT/vcdlp_rho; HOOKS=$REPO/tmp_hj/nlqp_hooks; ARMENV="VLMAS_C1_QASC=1 VLMAS_C1_SELECT=nova VLMAS_NOVA_KEEP=0.25 VLMAS_C1_META_BRIDGE=1 VLMAS_NOVA_RHO=1 VLMAS_PLAN_LATENT_ONLY=1 VLMAS_NLQP=vcdlp VLMAS_NLQP_LOG=$ROOT/vcdlp_rho/nlqp_shard$SHARD.jsonl" ;;
  vcdlp_rho_check) K=$ROOT/vcdlp_rho_check; HOOKS=$REPO/tmp_hj/nlqp_hooks; ARMENV="VLMAS_C1_QASC=1 VLMAS_C1_SELECT=nova VLMAS_NOVA_KEEP=0.25 VLMAS_C1_META_BRIDGE=1 VLMAS_NOVA_RHO=1 VLMAS_PLAN_LATENT_ONLY=1 VLMAS_NLQP=vcdlp_check VLMAS_NLQP_LOG=$ROOT/vcdlp_rho_check/nlqp_shard$SHARD.jsonl" ;;
  vcdlp_rho_uncentered) K=$ROOT/vcdlp_rho_uncentered; HOOKS=$REPO/tmp_hj/nlqp_hooks; ARMENV="VLMAS_C1_QASC=1 VLMAS_C1_SELECT=nova VLMAS_NOVA_KEEP=0.25 VLMAS_C1_META_BRIDGE=1 VLMAS_NOVA_RHO=1 VLMAS_PLAN_LATENT_ONLY=1 VLMAS_NLQP=vcdlp_uncentered VLMAS_NLQP_LOG=$ROOT/vcdlp_rho_uncentered/nlqp_shard$SHARD.jsonl" ;;
  vcdlp_rho_fast) K=$ROOT/vcdlp_rho_fast; HOOKS=$REPO/tmp_hj/nlqp_hooks; ARMENV="VLMAS_C1_QASC=1 VLMAS_C1_SELECT=nova VLMAS_NOVA_KEEP=0.25 VLMAS_C1_META_BRIDGE=1 VLMAS_NOVA_RHO=1 VLMAS_PLAN_LATENT_ONLY=1 VLMAS_NLQP=vcdlp VLMAS_VCDLP_FAST=1 VLMAS_NLQP_ATTNLOG=0 VLMAS_NLQP_LOG=$ROOT/vcdlp_rho_fast/nlqp_shard$SHARD.jsonl" ;;
  vcdlp_rho_fast_check) K=$ROOT/vcdlp_rho_fast_check; HOOKS=$REPO/tmp_hj/nlqp_hooks; ARMENV="VLMAS_C1_QASC=1 VLMAS_C1_SELECT=nova VLMAS_NOVA_KEEP=0.25 VLMAS_C1_META_BRIDGE=1 VLMAS_NOVA_RHO=1 VLMAS_PLAN_LATENT_ONLY=1 VLMAS_NLQP=vcdlp_check VLMAS_VCDLP_FAST=1 VLMAS_NLQP_ATTNLOG=0 VLMAS_NLQP_LOG=$ROOT/vcdlp_rho_fast_check/nlqp_shard$SHARD.jsonl" ;;
  rfnm_rho) K=$ROOT/rfnm_rho; HOOKS=$REPO/tmp_hj/nlqp_hooks; ARMENV="VLMAS_C1_QASC=1 VLMAS_C1_SELECT=nova VLMAS_NOVA_KEEP=0.25 VLMAS_C1_META_BRIDGE=1 VLMAS_NOVA_RHO=1 VLMAS_PLAN_LATENT_ONLY=1 VLMAS_NLQP=rfnm VLMAS_NLQP_LOG=$ROOT/rfnm_rho/nlqp_shard$SHARD.jsonl" ;;
  rfnm_rho_check) K=$ROOT/rfnm_rho_check; HOOKS=$REPO/tmp_hj/nlqp_hooks; ARMENV="VLMAS_C1_QASC=1 VLMAS_C1_SELECT=nova VLMAS_NOVA_KEEP=0.25 VLMAS_C1_META_BRIDGE=1 VLMAS_NOVA_RHO=1 VLMAS_PLAN_LATENT_ONLY=1 VLMAS_NLQP=rfnm_check VLMAS_NLQP_LOG=$ROOT/rfnm_rho_check/nlqp_shard$SHARD.jsonl" ;;
  rfnm_rho_raw) K=$ROOT/rfnm_rho_raw; HOOKS=$REPO/tmp_hj/nlqp_hooks; ARMENV="VLMAS_C1_QASC=1 VLMAS_C1_SELECT=nova VLMAS_NOVA_KEEP=0.25 VLMAS_C1_META_BRIDGE=1 VLMAS_NOVA_RHO=1 VLMAS_PLAN_LATENT_ONLY=1 VLMAS_NLQP=rfnm_raw VLMAS_NLQP_LOG=$ROOT/rfnm_rho_raw/nlqp_shard$SHARD.jsonl" ;;
  *) echo "bad arm $ARM"; exit 2 ;;
esac
NAV2ENV="VLMAS_NAV2=1"; [ "$ARM" = nonav2 ] && NAV2ENV=""
CLAIM=${CLAIMDIR:-$ROOT/claims}
LOG=$K/chain_${ARM}_shard$SHARD.log
RTASK=$'For each patch, examine its image quality, architecture, cellularity, cytology, stroma, necrosis, tissue boundary, artifacts, and relevance to the question. Focus only on visually observable findings and distinguish true tissue features from artifacts. State your best morphological read for every feature even when uncertain; do not report a feature as unresolvable.\n\nFormat your response as follows:\n- One bullet point per patch, starting with its patch ID, using terse 2-4 word phrases per feature.\n- Consistent findings: findings shared across patches and the patch IDs that support them.\n- Contradictions: disagreements between patches.\n- Unresolved evidence: what remains uncertain.\n- Diagnostically usable patches: patch IDs.\n\nDo not diagnose, choose an answer, or navigate.\nNow, output your response below:'
mkdir -p $K $CLAIM
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }
mem() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' -v g=$1 '$1==g{print $2}'; }

if [ "$SHARD" != smoke ]; then
  log "WAIT smoke gate"
  while [ ! -f $K/SMOKE_GATE_$ARM ]; do sleep 30; done
  grep -q ^PASS $K/SMOKE_GATE_$ARM || { log "SMOKE FAILED -> stop"; exit 1; }
fi

G=""
log "WAIT one of gpus [$GPUS] shard $SHARD/$NSHARD maxused=$MAXUSED"
while [ -z "$G" ]; do
  for g in $GPUS; do
    u=$(mem $g)
    if [ -n "$u" ] && [ "$u" -lt "$MAXUSED" ] && mkdir $CLAIM/claim_gpu$g 2>/dev/null; then G=$g; break; fi
  done
  [ -z "$G" ] && sleep 60
done
trap 'rmdir $CLAIM/claim_gpu$G 2>/dev/null' EXIT
log "CLAIM gpu$G shard=$SHARD arm=$ARM variant=$VARIANT nav4_md5=$(md5sum $REPO/vision_text_mas/onepass_navigation_nav4.py | cut -c1-8) navigator_md5=$(md5sum $REPO/vision_text_mas/onepass_navigator.py | cut -c1-8) backbone_md5=$(md5sum $REPO/backbone/qwen3vl.py | cut -c1-8) eovc_md5=$(md5sum $REPO/memory/eovc.py | cut -c1-8) secld_md5=$(md5sum $REPO/memory/secld_select.py | cut -c1-8) nova_md5=$(md5sum $REPO/memory/nova_select.py | cut -c1-8) qasc_md5=$(md5sum $REPO/memory/qasc_select.py | cut -c1-8) rnlcr_md5=$(md5sum $REPO/memory/rnlcr.py | cut -c1-8) lu_md5=$(md5sum $REPO/memory/lu.py | cut -c1-8) engine_md5=$(md5sum $REPO/vision_text_mas/latent_qwen_engine.py | cut -c1-8) env=[$ARMENV] used=$(mem $G)"

run() {
  local tag=$1 dsjson=$2 dest=$3; shift 3
  local out=$dest/gpu${G}_attempt_$(date +%Y%m%d_%H%M%S)
  mkdir -p $dest
  log "START $tag pending=$(( $# / 2 )) -> ${out#$K/}"
  cd $REPO && env \
    CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$HOOKS:$REPO PYTHONUNBUFFERED=1 $ARMENV \
    VLMAS_PROMPT_SET=prompt1 VLMAS_ATTN_IMPLEMENTATION=sdpa \
    VLMAS_NAV4=1 $NAV2ENV VLMAS_CROSS_SCALE_ROUTER=0 \
    VLMAS_NAV3_X5_CLAMP=1 VLMAS_EFF_DUMP=1 VLMAS_ANSWERER_GRADE_RULE=1 \
    VLMAS_REASONER_TASK="$RTASK" \
    $PY -m wsi_latentmas.pipeline.latent_mas \
    --variant $VARIANT --backbone qwen3-vl --dataset "$dsjson" --slide-root "${SLIDE_ROOT:-$DATA/slides}" \
    --model "$MODEL" --output-root "$out" --device cuda:0 \
    --latent-steps 10 --patch-budget 12 --max-model-len 12288 --navigator-control-tokens 512 \
    --temperature 0.0 --top-p 1.0 --seed 42 --deterministic --answerer-greedy --no-answerer-thinking \
    --answerer-max-new-tokens 512 --answerer-rationale --answerer-protocol structured_json \
    --canonical-open-options --navigator-kv --io-pipeline --no-save-navigation-pngs --case-retries 0 \
    "$@" > "$out.log" 2>&1 < /dev/null
  local rc=$?
  log "DONE $ARM $tag rc=$rc results=$(find $out -name result.json | wc -l) nav4=$(grep -l 'independent of the x5 selection' $(find $out -name navigator_calls.json) 2>/dev/null | wc -l) clamp=$(grep -c 'NAV3-X5CLAMP' $out.log) eovc=$(grep -c '^\[EOVC\]' $out.log) secld=$(grep -c '^\[SECLD\]' $out.log) nova=$(grep -c '^\[NOVA\] kept' $out.log) c1meta_ok=$(grep -c '^\[C1META\] keep .* used=True meta=ok' $out.log) empty_crops=$(grep -o 'empty_crops=[0-9]*' $out.log | grep -vc 'empty_crops=0') prunev4=$(grep -c 'PruneVision/v4' $out.log) fulr=$(grep -c '^\[RNLCR\] case .* mode=fulr applied=True' $out.log) fulr_skip=$(grep -c '^\[RNLCR.*SKIP' $out.log) empty=$(grep -c 'EMPTY terminal answer' $out.log) tb=$(grep -c Traceback $out.log) oom=$(grep -ci 'out of memory' $out.log) eff=$(find $out -name efficiency.json | wc -l)"
}

if [ "$SHARD" = smoke ]; then
  SMOKE_N=${SMOKE_N:-2}; run smoke ${SMOKE_JSON:-$REPO/sender_relay_exp/smoke/gtex2.json} $K/smoke_$ARM $(for i in $(seq 0 $((SMOKE_N-1))); do printf -- "--dataset-index %s " $i; done)
  o=$(ls -d $K/smoke_$ARM/gpu*_attempt_* | grep -v log$ | tail -1)
  r=$(find $o -name result.json | wc -l)
  n4=$(grep -l 'independent of the x5 selection' $(find $o -name navigator_calls.json) 2>/dev/null | wc -l); t=$(grep -c Traceback $o.log)
  em=$(grep -c 'EMPTY terminal answer' $o.log)
  ev=$(grep -c '^\[EOVC\]' $o.log); v4=$(grep -c 'PruneVision/v4' $o.log); sl=$(grep -c '^\[SECLD\]' $o.log); nv=$(grep -c '^\[NOVA\] kept' $o.log)
  need_nv=0; { [ "$ARM" = nova ] || [ "$ARM" = nova_fulr ] || [ "$ARM" = glvr_rho ] || [ "$ARM" = glvr_rho_nf ] || [ "$ARM" = glvr_rho_fast ] || { [ "${ARM#nlqp_rho}" != "$ARM" ] || [ "${ARM#vcdlp_rho}" != "$ARM" ] || [ "${ARM#rfnm_rho}" != "$ARM" ]; }; } && need_nv=$SMOKE_N
  fu=$(grep -c '^\[RNLCR\] case .* mode=fulr applied=True' $o.log); fs=$(grep -c '^\[RNLCR.*SKIP' $o.log); nq=$(grep -c '^\[NLQP\] case .* applied=36/36' $o.log); nqs=$(grep -c '^\[NLQP\] case .* SKIP' $o.log); need_nq=0; { [ "${ARM#nlqp_rho}" != "$ARM" ] || [ "${ARM#vcdlp_rho}" != "$ARM" ] || [ "${ARM#rfnm_rho}" != "$ARM" ]; } && need_nq=$SMOKE_N
  pt=$(grep -c '^\[PlanTargets\]' $o.log); need_pt_zero=0; { [ "$ARM" = nonav2 ] || [ "$ARM" = latplan ]; } && need_pt_zero=1
  need_fu=0; { [ "$ARM" = fulr ] || [ "$ARM" = nova_fulr ]; } && need_fu=2
  need_ev=0; [ "$ARM" = eovc ] && need_ev=$v4
  need_sl=0; [ "$ARM" = secld ] && need_sl=$v4
  need_v4=0; { [ "$ARM" = eovc ] || [ "$ARM" = pruneb ] || [ "$ARM" = secld ]; } && need_v4=1
  okv4=1; [ "$need_v4" -eq 1 ] && { [ "$v4" -gt 0 ] || okv4=0; }
  if [ "$r" -eq "$SMOKE_N" ] && [ "$n4" -gt 0 ] && [ "$t" -eq 0 ] && [ "$em" -eq 0 ] \
     && [ "$ev" -eq "$need_ev" ] && [ "$sl" -eq "$need_sl" ] && [ "$nv" -eq "$need_nv" ] && [ "$okv4" -eq 1 ] \
     && [ "$fu" -eq "$need_fu" ] && [ "$fs" -eq 0 ] && [ "$nq" -ge "$need_nq" ] && [ "$nqs" -eq 0 ] \
     && { [ "$need_pt_zero" -eq 0 ] || [ "$pt" -eq 0 ]; }; then
    echo "PASS nlqp=$nq results=$r nav4=$n4 prunev4=$v4 eovc=$ev secld=$sl nova=$nv fulr=$fu fulr_skip=$fs plantargets=$pt empty=$em tb=$t" > $K/SMOKE_GATE_$ARM
  else
    echo "FAIL nlqp=$nq need_nlqp=$need_nq nlqp_skip=$nqs plantargets=$pt results=$r nav4=$n4 prunev4=$v4 eovc=$ev secld=$sl nova=$nv fulr=$fu fulr_skip=$fs need_nova=$need_nv need_fulr=$need_fu need_eovc=$need_ev need_secld=$need_sl need_prunev4=$need_v4 empty=$em tb=$t" > $K/SMOKE_GATE_$ARM
  fi
  log "SMOKE $(cat $K/SMOKE_GATE_$ARM)"; exit 0
fi

for spec in $SPECS; do
  ds=${spec%%:*}; n=${spec##*:}
  done_ids=$(find $K/$ds -name result.json 2>/dev/null -exec $PY -c \
    'import json,sys; [print(json.load(open(f))["dataset_index"]) for f in sys.argv[1:]]' {} + | sort -n -u)
  ids=()
  for ((i=0; i<n; i++)); do
    [ $((i % NSHARD)) -eq "$SHARD" ] || continue
    grep -qx "$i" <<< "$done_ids" || ids+=(--dataset-index "$i")
  done
  if [ ${#ids[@]} -eq 0 ]; then log "SKIP $ds shard $SHARD complete"; continue; fi
  run $ds $DATA/$ds.json $K/$ds "${ids[@]}"
done
log "ALL DONE shard $SHARD"
touch $K/SHARD${SHARD}_${ARM}_DONE
