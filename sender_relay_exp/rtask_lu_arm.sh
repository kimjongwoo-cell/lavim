#!/bin/bash
#   Latent Unsilencing (memory/lu.py) on top of NTRS 50% + decode fix (same setting as rtask_ntrs50_arm.sh).
#   ARM=lu_replay : VLMAS_LU=replay (Z untouched, crop + replay once; numerical identity check) -> runs/rtask_lu_replay_nav3
#   ARM=lu_s1     : VLMAS_LU=s1     (Stage I only)                                              -> runs/rtask_lu_s1_nav3
#   ARM=lu_s12    : VLMAS_LU=s12    (Stage I + Stage II NES)                                    -> runs/rtask_lu_s12_nav3
#   usage: rtask_lu_arm.sh <arm> {smoke|<shard>} <nshard> "<gpus>" [ds:n ...]     (default specs = gtex:190)
#   Comparison base = runs/rtask_ntrs50_nav3 (same questions, same C1, LU off).
set -u
ARM=${1:?arm}; SHARD=${2:?shard}; NSHARD=${3:?nshard}; GPUS=${4:?gpus}; shift 4; SPECS=${*:-gtex:190}; HGPU=; HPID=
REPO=/home/users/whddn12316/wsi_latent_0915_decode_hj
DATA=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
MODEL=/home/users/whddn12316/models/Qwen3-VL-4B-Thinking
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
case $ARM in
  lu_replay) LU=replay ;;
  lu_s1)     LU=s1 ;;
  lu_s12)    LU=s12 ;;
  *) echo bad arm; exit 2 ;;
esac
K=$REPO/sender_relay_exp/runs/rtask_${ARM}_nav3
ARMENV="VLMAS_C1_QASC=1 VLMAS_C1_SELECT=ntrs VLMAS_QASC_KEEP=0.5 VLMAS_C1_META_BRIDGE=1 VLMAS_LU=$LU VLMAS_LU_LOG=$K/lu_shard${SHARD}.jsonl"
HOOKS=$REPO/tmp_hj/lpvh_ntrs_hooks
CLAIM=/home/users/whddn12316/wsi_latent_0902_2155_hj/sender_relay_exp/runs/vgain
LOG=$K/chain_shard$SHARD.log
RTASK=$'For each patch, examine its image quality, architecture, cellularity, cytology, stroma, necrosis, tissue boundary, artifacts, and relevance to the question. Focus only on visually observable findings and distinguish true tissue features from artifacts. State your best morphological read for every feature even when uncertain; do not report a feature as unresolvable.\n\nFormat your response as follows:\n- One bullet point per patch, starting with its patch ID, using terse 2-4 word phrases per feature.\n- Consistent findings: findings shared across patches and the patch IDs that support them.\n- Contradictions: disagreements between patches.\n- Unresolved evidence: what remains uncertain.\n- Diagnostically usable patches: patch IDs.\n\nDo not diagnose, choose an answer, or navigate.\nNow, output your response below:'
mkdir -p $K $CLAIM
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }
mem() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' -v g=$1 '$1==g{print $2}'; }
G=""
if [ "$SHARD" != smoke ]; then
  log "WAIT smoke gate"
  while [ ! -f $K/SMOKE_GATE ]; do sleep 30; done
  grep -q ^PASS $K/SMOKE_GATE || { log "SMOKE FAILED -> stop"; exit 1; }
fi
log "WAIT one of gpus [$GPUS] shard $SHARD/$NSHARD"
while [ -z "$G" ]; do
  for g in $GPUS; do
    u=$(mem $g)
    if [ -n "$u" ] && [ "$u" -lt 2000 ] && mkdir $CLAIM/claim_gpu$g 2>/dev/null; then G=$g; break; fi
  done
  [ -z "$G" ] && sleep 60
done
trap 'rmdir $CLAIM/claim_gpu$G 2>/dev/null' EXIT
log "CLAIM gpu$G shard=$SHARD lu=$LU lu_md5=$(md5sum $REPO/memory/lu.py | cut -c1-8) qasc_md5=$(md5sum $REPO/memory/qasc_select.py | cut -c1-8) backbone_md5=$(md5sum $REPO/backbone/qwen3vl.py | cut -c1-8) engine_md5=$(md5sum $REPO/vision_text_mas/latent_qwen_engine.py | cut -c1-8)"

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
  log "DONE $tag rc=$rc results=$(find $out -name result.json | wc -l) ntrs1536=$(grep -c '^\[NTRS\] kept 1536' $out.log) c1meta_ok=$(grep -c '^\[C1META\] keep .* used=True meta=ok' $out.log) lu=$(grep -c '^\[LU\] case .* mode=' $out.log) lu_skip=$(grep -c '^\[LU\] case .*SKIP' $out.log) empty=$(grep -c 'EMPTY terminal answer' $out.log) tb=$(grep -c Traceback $out.log) oom=$(grep -ci 'out of memory' $out.log) eff=$(find $out -name efficiency.json | wc -l)"
}

if [ "$SHARD" = smoke ]; then
  run smoke $REPO/sender_relay_exp/smoke/gtex2.json $K/smoke --dataset-index 0 --dataset-index 1
  o=$(ls -td $K/smoke/gpu*_attempt_*/ | head -1); o=${o%/}   # newest attempt by mtime (name order would pick gpu8 over gpu6/7)
  r=$(find $o -name result.json | wc -l); k=$(grep -c '^\[NTRS\] kept 1536/3072' $o.log); t=$(grep -c Traceback $o.log)
  l=$(grep -c '^\[LU\] case .* mode=' $o.log); s=$(grep -c '^\[LU\] case .*SKIP' $o.log)
  if [ "$r" -eq 2 ] && [ "$k" -eq 2 ] && [ "$t" -eq 0 ] && [ "$l" -eq 2 ] && [ "$s" -eq 0 ]; then echo "PASS results=$r kept1536=$k tb=$t lu=$l skip=$s" > $K/SMOKE_GATE; else echo "FAIL results=$r kept1536=$k tb=$t lu=$l skip=$s" > $K/SMOKE_GATE; fi
  log "SMOKE $(cat $K/SMOKE_GATE)"; exit 0
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
