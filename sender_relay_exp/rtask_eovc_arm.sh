#!/bin/bash
# 0917 C2 #14 EOVC vs Pruning-B on the decode-fix base (copy tree wsi_latent_0915_decode_hj).
#   ARM=eovc   : --variant pruning_b + VLMAS_EOVC=1  (early-observed unexplained-variation coverage)
#   ARM=pruneb : --variant pruning_b                 (novelty + 0.25*texture, quota; page §12 baseline)
# Same early-observation blocks, same keep ratio, same budget: only the keep-mask rule differs.
# Decode fix is this tree's default (bullet Reasoner task via VLMAS_REASONER_TASK, no digit repair).
# PANDA uses VLMAS_ANSWERER_GRADE_RULE=1 (fires only when the choices are 0..N).
# usage: rtask_eovc_arm.sh <arm> <shard|smoke> <nshard> "<gpus>" [ds:n ...]
#   env OUTROOT / CLAIMDIR override the output root and the GPU claim dir.
set -u
ARM=${1:?arm}; SHARD=${2:?shard}; NSHARD=${3:?nshard}; GPUS=${4:?gpus}; shift 4
SPECS=${*:-tcga_expert_vqa:128 tcga_slidebench:197 gtex:190 tcga:221 panda:196}
REPO=/home/users/whddn12316/wsi_latent_0915_decode_hj
DATA=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
MODEL=/home/users/whddn12316/models/Qwen3-VL-4B-Thinking
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
ROOT=${OUTROOT:-$REPO/sender_relay_exp/runs/rtask_eovc}
case $ARM in
  eovc)   K=$ROOT/eovc;   ARMENV="VLMAS_EOVC=1" ;;
  pruneb) K=$ROOT/pruneb; ARMENV="" ;;
  *) echo "bad arm $ARM"; exit 2 ;;
esac
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
log "WAIT one of gpus [$GPUS] shard $SHARD/$NSHARD"
while [ -z "$G" ]; do
  for g in $GPUS; do
    u=$(mem $g)
    if [ -n "$u" ] && [ "$u" -lt 2000 ] && mkdir $CLAIM/claim_gpu$g 2>/dev/null; then G=$g; break; fi
  done
  [ -z "$G" ] && sleep 60
done
trap 'rmdir $CLAIM/claim_gpu$G 2>/dev/null' EXIT
log "CLAIM gpu$G shard=$SHARD backbone_md5=$(md5sum $REPO/backbone/qwen3vl.py | cut -c1-8) eovc_md5=$(md5sum $REPO/memory/eovc.py | cut -c1-8)"

run() {
  local tag=$1 dsjson=$2 dest=$3; shift 3
  local out=$dest/gpu${G}_attempt_$(date +%Y%m%d_%H%M%S)
  mkdir -p $dest
  log "START $tag pending=$(( $# / 2 )) -> ${out#$K/}"
  cd $REPO && env \
    CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$REPO PYTHONUNBUFFERED=1 $ARMENV \
    VLMAS_PROMPT_SET=prompt1 VLMAS_ATTN_IMPLEMENTATION=sdpa VLMAS_NAV2=1 VLMAS_NAV3_INDEPENDENT=1 \
    VLMAS_CROSS_SCALE_ROUTER=0 VLMAS_EFF_DUMP=1 VLMAS_ANSWERER_GRADE_RULE=1 \
    VLMAS_REASONER_TASK="$RTASK" \
    $PY -m wsi_latentmas.pipeline.latent_mas \
    --variant pruning_b --backbone qwen3-vl --dataset "$dsjson" --slide-root "$DATA/slides" \
    --model "$MODEL" --output-root "$out" --device cuda:0 \
    --latent-steps 10 --patch-budget 12 --max-model-len 12288 --navigator-control-tokens 512 \
    --temperature 0.0 --top-p 1.0 --seed 42 --deterministic --answerer-greedy --no-answerer-thinking \
    --answerer-max-new-tokens 512 --answerer-rationale --answerer-protocol structured_json \
    --canonical-open-options --navigator-kv --io-pipeline --no-save-navigation-pngs --case-retries 0 \
    "$@" > "$out.log" 2>&1 < /dev/null
  local rc=$?
  log "DONE $ARM $tag rc=$rc results=$(find $out -name result.json | wc -l) eovc=$(grep -c '^\[EOVC\]' $out.log) prunev4=$(grep -c 'PruneVision/v4' $out.log) empty=$(grep -c 'EMPTY terminal answer' $out.log) tb=$(grep -c Traceback $out.log) oom=$(grep -ci 'out of memory' $out.log) eff=$(find $out -name efficiency.json | wc -l)"
}

if [ "$SHARD" = smoke ]; then
  run smoke $REPO/sender_relay_exp/smoke/gtex2.json $K/smoke_$ARM --dataset-index 0 --dataset-index 1
  o=$(ls -d $K/smoke_$ARM/gpu*_attempt_* | grep -v log$ | tail -1)
  r=$(find $o -name result.json | wc -l)
  v4=$(grep -c 'PruneVision/v4' $o.log); ev=$(grep -c '^\[EOVC\]' $o.log)
  t=$(grep -c Traceback $o.log); em=$(grep -c 'EMPTY terminal answer' $o.log)
  need_ev=0; [ "$ARM" = eovc ] && need_ev=$v4
  sd=$(grep -o 'sd=[0-9.]*' $o.log | tr '\n' ' ')
  ex=$(grep -o 'explained=[0-9.]*' $o.log | tr '\n' ' ')
  if [ "$r" -eq 2 ] && [ "$v4" -gt 0 ] && [ "$ev" -eq "$need_ev" ] && [ "$t" -eq 0 ] && [ "$em" -eq 0 ]; then
    echo "PASS results=$r prunev4=$v4 eovc=$ev empty=$em tb=$t sd=[$sd] explained=[$ex]" > $K/SMOKE_GATE_$ARM
  else
    echo "FAIL results=$r prunev4=$v4 eovc=$ev need_eovc=$need_ev empty=$em tb=$t" > $K/SMOKE_GATE_$ARM
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
