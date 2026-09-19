#!/bin/bash
#   VELH (memory/velh.py) and Text-cut (memory/textcut.py) arms on top of NTRS 50% + decode fix
#   (same setting as rtask_ntrs50_arm.sh; comparison base = runs/rtask_ntrs50_nav3, LU base = runs/rtask_lu_s12_nav3).
#   arms:  velh_diag | velh_bind | velh_handoff | velh_full | velh_direct | velh_shuffle
#          rn_diag | rn_cap | rn_ground | rn_full   (RN-LCR, memory/rnlcr.py; MAXUSED=<MiB> env lets a shard co-locate on a busy GPU)
#          tc_diag | tc_drop | tc_filler | tc_space | tc_drop_lu | tc_filler_lu | tc_space_lu   (lu = VLMAS_LU=s12)
#   usage: rtask_c2diag_arm.sh <arm> {diag5|smoke|<shard>} <nshard> "<gpus>" [ds:n ...]
#          diag5 = dataset indices 0..4 of the FIRST spec's dataset (default gtex) -> runs/rtask_<arm>_nav3/diag5_<ds>
set -u
ARM=${1:?arm}; SHARD=${2:?shard}; NSHARD=${3:?nshard}; GPUS=${4:?gpus}; shift 4; SPECS=${*:-gtex:190}; MAXUSED=${MAXUSED:-2000}
REPO=/home/users/whddn12316/wsi_latent_0915_decode_hj
DATA=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
MODEL=/home/users/whddn12316/models/Qwen3-VL-4B-Thinking
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
K=$REPO/sender_relay_exp/runs/rtask_${ARM}_nav3
EXTRA=""; TAG=""
case $ARM in
  velh_*)   EXTRA="VLMAS_VELH=${ARM#velh_} VLMAS_VELH_LOG=$K/velh_shard${SHARD}.jsonl"; TAG="VELH" ;;
  rn_*)     EXTRA="VLMAS_RNLCR=${ARM#rn_} VLMAS_RNLCR_LOG=$K/rnlcr_shard${SHARD}.jsonl"; TAG="RNLCR" ;;
  tc_*)     base=${ARM#tc_}; lu=""
            case $base in *_lu) base=${base%_lu}; lu="VLMAS_LU=s12 VLMAS_LU_LOG=$K/lu_shard${SHARD}.jsonl" ;; esac
            case $base in
              diag)   EXTRA="VLMAS_TEXTCUT=diag" ;;
              drop)   EXTRA="VLMAS_TEXTCUT=drop" ;;
              filler) EXTRA="VLMAS_TEXTCUT=filler VLMAS_TEXTCUT_FILLER=shuffle" ;;
              space)  EXTRA="VLMAS_TEXTCUT=filler VLMAS_TEXTCUT_FILLER=space" ;;
              *) echo bad arm; exit 2 ;;
            esac
            EXTRA="$EXTRA VLMAS_TEXTCUT_LOG=$K/textcut_shard${SHARD}.jsonl $lu"; TAG="TextCut" ;;
  *) echo bad arm; exit 2 ;;
esac
ARMENV="VLMAS_C1_QASC=1 VLMAS_C1_SELECT=ntrs VLMAS_QASC_KEEP=0.5 VLMAS_C1_META_BRIDGE=1 $EXTRA"
HOOKS=$REPO/tmp_hj/lpvh_ntrs_hooks
CLAIM=/home/users/whddn12316/wsi_latent_0902_2155_hj/sender_relay_exp/runs/vgain
LOG=$K/chain_shard$SHARD.log
RTASK=$'For each patch, examine its image quality, architecture, cellularity, cytology, stroma, necrosis, tissue boundary, artifacts, and relevance to the question. Focus only on visually observable findings and distinguish true tissue features from artifacts. State your best morphological read for every feature even when uncertain; do not report a feature as unresolvable.\n\nFormat your response as follows:\n- One bullet point per patch, starting with its patch ID, using terse 2-4 word phrases per feature.\n- Consistent findings: findings shared across patches and the patch IDs that support them.\n- Contradictions: disagreements between patches.\n- Unresolved evidence: what remains uncertain.\n- Diagnostically usable patches: patch IDs.\n\nDo not diagnose, choose an answer, or navigate.\nNow, output your response below:'
mkdir -p $K $CLAIM
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }
mem() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' -v g=$1 '$1==g{print $2}'; }
G=""
if [ "$SHARD" != smoke ] && [ "$SHARD" != diag5 ]; then
  log "WAIT smoke gate"
  while [ ! -f $K/SMOKE_GATE ]; do sleep 30; done
  grep -q ^PASS $K/SMOKE_GATE || { log "SMOKE FAILED -> stop"; exit 1; }
fi
log "WAIT one of gpus [$GPUS] shard $SHARD/$NSHARD"
while [ -z "$G" ]; do
  for g in $GPUS; do
    u=$(mem $g)
    if [ -n "$u" ] && [ "$u" -lt "${MAXUSED:-2000}" ] && mkdir $CLAIM/claim_gpu$g 2>/dev/null; then G=$g; break; fi
  done
  [ -z "$G" ] && sleep 60
done
trap 'rmdir $CLAIM/claim_gpu$G 2>/dev/null' EXIT
log "CLAIM gpu$G shard=$SHARD arm=$ARM env=[$EXTRA] velh_md5=$(md5sum $REPO/memory/velh.py | cut -c1-8) textcut_md5=$(md5sum $REPO/memory/textcut.py | cut -c1-8) lu_md5=$(md5sum $REPO/memory/lu.py | cut -c1-8) engine_md5=$(md5sum $REPO/vision_text_mas/latent_qwen_engine.py | cut -c1-8)"

run() {
  local tag=$1 dsjson=$2 dest=$3; shift 3
  local out=$dest/gpu${G}_attempt_$(date +%Y%m%d_%H%M%S)
  mkdir -p $dest
  log "START $tag pending=$(( $# / 2 )) -> ${out#$K/}"
  cd $REPO && env \
    CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$HOOKS:$REPO PYTHONUNBUFFERED=1 $ARMENV \
    VLMAS_PROMPT_SET=prompt1 VLMAS_ATTN_IMPLEMENTATION=sdpa VLMAS_NAV2=1 VLMAS_NAV3_INDEPENDENT=1 \
    VLMAS_CROSS_SCALE_ROUTER=0 VLMAS_EFF_DUMP=1 VLMAS_REASONER_TASK="$RTASK" \
    $PY -m wsi_latentmas.pipeline.latent_mas \
    --variant base --backbone qwen3-vl --dataset "$dsjson" --slide-root "$DATA/slides" \
    --model "$MODEL" --output-root "$out" --device cuda:0 \
    --latent-steps 10 --patch-budget 12 --max-model-len 12288 --navigator-control-tokens 512 \
    --temperature 0.0 --top-p 1.0 --seed 42 --deterministic --answerer-greedy --no-answerer-thinking \
    --answerer-max-new-tokens 512 --answerer-rationale --answerer-protocol structured_json \
    --canonical-open-options --navigator-kv --io-pipeline --no-save-navigation-pngs --case-retries 0 \
    "$@" > "$out.log" 2>&1 < /dev/null
  local rc=$?
  log "DONE $tag rc=$rc results=$(find $out -name result.json | wc -l) ntrs1536=$(grep -c '^\[NTRS\] kept 1536' $out.log) c1meta_ok=$(grep -c '^\[C1META\] keep .* used=True meta=ok' $out.log) ${TAG}=$(grep -c "^\[$TAG\] case .* mode=" $out.log) ${TAG}_skip=$(grep -c "^\[$TAG\] case .*SKIP" $out.log) lu=$(grep -c '^\[LU\] case .* mode=' $out.log) lu_skip=$(grep -c '^\[LU\] case .*SKIP' $out.log) empty=$(grep -c 'EMPTY terminal answer' $out.log) tb=$(grep -c Traceback $out.log) oom=$(grep -ci 'out of memory' $out.log) eff=$(find $out -name efficiency.json | wc -l)"
}

gate() {  # $1 = expected results, $2 = attempt dir
  local n=$1 o=$2
  local r=$(find $o -name result.json | wc -l) t=$(grep -c Traceback $o.log)
  local c=$(grep -c "^\[$TAG\] case .* mode=" $o.log) s=$(grep -c "^\[$TAG\] case .*SKIP" $o.log) e=$(grep -c "EMPTY terminal answer" $o.log)
  if [ "$r" -eq $n ] && [ "$t" -eq 0 ] && [ "$c" -eq $n ] && [ "$s" -eq 0 ] && [ "$e" -eq 0 ]; then echo "PASS results=$r tb=$t $TAG=$c skip=$s empty=$e"; else echo "FAIL results=$r tb=$t $TAG=$c skip=$s empty=$e"; fi
}

if [ "$SHARD" = smoke ]; then
  run smoke $REPO/sender_relay_exp/smoke/gtex2.json $K/smoke --dataset-index 0 --dataset-index 1
  o=$(ls -td $K/smoke/gpu*_attempt_*/ | head -1); o=${o%/}
  gate 2 $o > $K/SMOKE_GATE; log "SMOKE $(cat $K/SMOKE_GATE)"; exit 0
fi
if [ "$SHARD" = diag5 ]; then
  ds=${SPECS%% *}; ds=${ds%%:*}
  run diag5_$ds $DATA/$ds.json $K/diag5_$ds --dataset-index 0 --dataset-index 1 --dataset-index 2 --dataset-index 3 --dataset-index 4
  o=$(ls -td $K/diag5_$ds/gpu*_attempt_*/ | head -1); o=${o%/}
  gate 5 $o > $K/DIAG5_${ds}_GATE; log "DIAG5 $ds $(cat $K/DIAG5_${ds}_GATE)"; exit 0
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
touch $K/SHARD${SHARD}_DONE
