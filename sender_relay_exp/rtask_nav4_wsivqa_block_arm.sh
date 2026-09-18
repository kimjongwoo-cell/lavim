#!/bin/bash
# 0917 WSI-VQA copy of rtask_nav4_nonav2_arm.sh (arms base / latplan used); only DATA, SPECS, ROOT and the smoke json differ.
# 0918 BLOCK copy: shards take contiguous slide-aligned blocks from runs/nav4_wsivqa/latplan/wsivqa_block_plan.json
#   (stride split opened every slide once per shard, ~360 s CPU each). Everything else = rtask_nav4_wsivqa_arm.sh.
# 0917 22:0x VLMAS_CHOICE_FORMAT=list: choices go in as the json list (Choice: [...]) instead of CHOICE n: lines; smoke gate requires it.
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
SPECS=${*:-wsivqa:735}
MAXUSED=${MAXUSED:-2000}   # a GPU counts as free below this used-MiB (co-locate with one other run at 20000)
REPO=/home/users/whddn12316/wsi_latent_0915_decode_hj
DATA=/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/data_wsivqa   # WSI-VQA test 735: wsivqa.json + slides -> datasets/WSI-VQA/DATA_SVS
MODEL=/home/users/whddn12316/models/Qwen3-VL-4B-Thinking
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
HOOKS=$REPO/tmp_hj/lpvh_ntrs_hooks
[ "$ARM" = latplan ] && HOOKS=$REPO/tmp_hj/latplan_hooks
ROOT=${OUTROOT:-$REPO/sender_relay_exp/runs/nav4_wsivqa}
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
  nova_fulr) K=$ROOT/nova_fulr; ARMENV="VLMAS_C1_QASC=1 VLMAS_C1_SELECT=nova VLMAS_NOVA_KEEP=0.25 VLMAS_C1_META_BRIDGE=1 VLMAS_RNLCR=fulr VLMAS_RNLCR_LOG=$ROOT/nova_fulr/rnlcr_shard$SHARD.jsonl" ;;
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
    VLMAS_CHOICE_FORMAT=list VLMAS_PROMPT_SET=prompt1 VLMAS_ATTN_IMPLEMENTATION=sdpa \
    VLMAS_NAV4=1 $NAV2ENV VLMAS_CROSS_SCALE_ROUTER=0 \
    VLMAS_NAV3_X5_CLAMP=1 VLMAS_EFF_DUMP=1 VLMAS_ANSWERER_GRADE_RULE=1 \
    VLMAS_REASONER_TASK="$RTASK" \
    $PY -m wsi_latentmas.pipeline.latent_mas \
    --variant $VARIANT --backbone qwen3-vl --dataset "$dsjson" --slide-root "$DATA/slides" \
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
  run smoke $REPO/sender_relay_exp/smoke/wsivqa2.json $K/smoke_$ARM --dataset-index 0 --dataset-index 1
  o=$(ls -d $K/smoke_$ARM/gpu*_attempt_* | grep -v log$ | tail -1)
  r=$(find $o -name result.json | wc -l)
  n4=$(grep -l 'independent of the x5 selection' $(find $o -name navigator_calls.json) 2>/dev/null | wc -l); t=$(grep -c Traceback $o.log)
  em=$(grep -c 'EMPTY terminal answer' $o.log)
  ev=$(grep -c '^\[EOVC\]' $o.log); v4=$(grep -c 'PruneVision/v4' $o.log); sl=$(grep -c '^\[SECLD\]' $o.log); nv=$(grep -c '^\[NOVA\] kept' $o.log)
  need_nv=0; { [ "$ARM" = nova ] || [ "$ARM" = nova_fulr ]; } && need_nv=2
  fu=$(grep -c '^\[RNLCR\] case .* mode=fulr applied=True' $o.log); fs=$(grep -c '^\[RNLCR.*SKIP' $o.log)
  pt=$(grep -c '^\[PlanTargets\]' $o.log); need_pt_zero=0; { [ "$ARM" = nonav2 ] || [ "$ARM" = latplan ]; } && need_pt_zero=1
  need_fu=0; { [ "$ARM" = fulr ] || [ "$ARM" = nova_fulr ]; } && need_fu=2
  need_ev=0; [ "$ARM" = eovc ] && need_ev=$v4
  need_sl=0; [ "$ARM" = secld ] && need_sl=$v4
  need_v4=0; { [ "$ARM" = eovc ] || [ "$ARM" = pruneb ] || [ "$ARM" = secld ]; } && need_v4=1
  okv4=1; [ "$need_v4" -eq 1 ] && { [ "$v4" -gt 0 ] || okv4=0; }
  cl=$(grep -l "Choice: \[" $(find $o -name answerer_call.json) 2>/dev/null | wc -l); cn=$(grep -l "CHOICE 1:" $(find $o -name answerer_call.json) 2>/dev/null | wc -l)
  if [ "$cl" -eq 2 ] && [ "$cn" -eq 0 ] && [ "$r" -eq 2 ] && [ "$n4" -gt 0 ] && [ "$t" -eq 0 ] && [ "$em" -eq 0 ] \
     && [ "$ev" -eq "$need_ev" ] && [ "$sl" -eq "$need_sl" ] && [ "$nv" -eq "$need_nv" ] && [ "$okv4" -eq 1 ] \
     && [ "$fu" -eq "$need_fu" ] && [ "$fs" -eq 0 ] \
     && { [ "$need_pt_zero" -eq 0 ] || [ "$pt" -eq 0 ]; }; then
    echo "PASS choicelist=$cl CHOICEn=$cn results=$r nav4=$n4 prunev4=$v4 eovc=$ev secld=$sl nova=$nv fulr=$fu fulr_skip=$fs plantargets=$pt empty=$em tb=$t" > $K/SMOKE_GATE_$ARM
  else
    echo "FAIL choicelist=$cl CHOICEn=$cn plantargets=$pt results=$r nav4=$n4 prunev4=$v4 eovc=$ev secld=$sl nova=$nv fulr=$fu fulr_skip=$fs need_nova=$need_nv need_fulr=$need_fu need_eovc=$need_ev need_secld=$need_sl need_prunev4=$need_v4 empty=$em tb=$t" > $K/SMOKE_GATE_$ARM
  fi
  log "SMOKE $(cat $K/SMOKE_GATE_$ARM)"; exit 0
fi

for spec in $SPECS; do
  ds=${spec%%:*}; n=${spec##*:}
  # 0918 block split: indices come from the frozen slide-aligned plan (tmp_hj/wsivqa_block_plan.py), minus finished ones.
  PLAN=$K/wsivqa_block_plan.json
  [ -f $PLAN ] || { log "NO PLAN $PLAN -> stop"; exit 1; }
  ids=()
  for i in $($PY - "$PLAN" "$SHARD" "$K/$ds" <<'PYEOF'
import glob, json, sys
plan, shard, root = sys.argv[1], sys.argv[2], sys.argv[3]
done = set()
for p in glob.glob(root + "/gpu*_attempt_*/*/result.json"):
    try:
        done.add(int(json.load(open(p))["dataset_index"]))
    except Exception:
        pass
print(" ".join(str(i) for i in json.load(open(plan))[shard] if i not in done))
PYEOF
); do ids+=(--dataset-index "$i"); done
  if [ ${#ids[@]} -eq 0 ]; then log "SKIP $ds shard $SHARD complete"; continue; fi
  run $ds $DATA/$ds.json $K/$ds "${ids[@]}"
done
log "ALL DONE shard $SHARD"
touch $K/SHARD${SHARD}_${ARM}_DONE
