#!/bin/bash
#   ARM=mcladr : 디코딩 수정 + NTRS 50% + C2 12.8 MC-LADR (VLMAS_LADR=mc, visual 총 mass 보존) -> runs/rtask_mcladr_gtex
#   ARM=realloc : 디코딩 수정 + NTRS 50% + 재분배(--variant reallocation, alpha 0.15, L8-20, handoff 경계) -> runs/rtask_realloc_gtex
#   ARM=realloc_v2 : 같은 위치에서 receiver 재분배(reallocation_v2, alpha 0.05, answerer prefill only)
#   ARM=ladr : 디코딩 수정 + NTRS 50% + C2 12.8 LADR (VLMAS_LADR=1, memory/ladr.py) -> runs/rtask_ladr_gtex
#   ARM=ladr_identity : 같은 경로, p_role=p_acq (native 재현 확인)
#   ARM=psfp : 디코딩 수정 + NTRS 50% + C2 12.7 PSFP (VLMAS_PSFP=1, memory/psfp.py) -> runs/rtask_psfp_gtex
#   ARM=psfp_diag : 같은 경로, logits 보정 없이 conflict만 계측
#   ARM=lrpsr : 디코딩 수정 + NTRS 50% + C2 12.6 LR-PSR (VLMAS_LRPSR=1, memory/lrpsr.py) -> runs/rtask_lrpsr_gtex
#   ARM=lrpsr_identity : 같은 경로, native 출력(경로 확인용)
#   Everything else = greedy_base_rtask_full.sh (Latent base + 디코딩 수정).
set -u
ARM=${1:?arm}; SHARD=${2:?shard}; NSHARD=${3:?nshard}; GPUS=${4:?gpus}; shift 4; SPECS=${*:-gtex:190}; HGPU=; HPID=
REPO=/home/users/whddn12316/wsi_latent_0915_decode_hj
DATA=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
MODEL=/home/users/whddn12316/models/Qwen3-VL-4B-Thinking
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
VARIANT=base
case $ARM in
  realloc) K=$REPO/sender_relay_exp/runs/rtask_realloc_gtex; VARIANT=reallocation; ARMENV="VLMAS_C1_QASC=1 VLMAS_C1_SELECT=ntrs VLMAS_QASC_KEEP=0.5 VLMAS_C1_META_BRIDGE=1" ;;
  realloc_v2) K=$REPO/sender_relay_exp/runs/rtask_realloc_gtex; VARIANT=reallocation_v2; ARMENV="VLMAS_C1_QASC=1 VLMAS_C1_SELECT=ntrs VLMAS_QASC_KEEP=0.5 VLMAS_C1_META_BRIDGE=1" ;;
  mcladr) K=$REPO/sender_relay_exp/runs/rtask_mcladr_gtex; ARMENV="VLMAS_C1_QASC=1 VLMAS_C1_SELECT=ntrs VLMAS_QASC_KEEP=0.5 VLMAS_C1_META_BRIDGE=1 VLMAS_LADR=mc" ;;
  mcladr_identity) K=$REPO/sender_relay_exp/runs/rtask_mcladr_gtex; ARMENV="VLMAS_C1_QASC=1 VLMAS_C1_SELECT=ntrs VLMAS_QASC_KEEP=0.5 VLMAS_C1_META_BRIDGE=1 VLMAS_LADR=identity" ;;
  ladr) K=$REPO/sender_relay_exp/runs/rtask_ladr_gtex; ARMENV="VLMAS_C1_QASC=1 VLMAS_C1_SELECT=ntrs VLMAS_QASC_KEEP=0.5 VLMAS_C1_META_BRIDGE=1 VLMAS_LADR=1" ;;
  ladr_identity) K=$REPO/sender_relay_exp/runs/rtask_ladr_gtex; ARMENV="VLMAS_C1_QASC=1 VLMAS_C1_SELECT=ntrs VLMAS_QASC_KEEP=0.5 VLMAS_C1_META_BRIDGE=1 VLMAS_LADR=identity" ;;
  psfp) K=$REPO/sender_relay_exp/runs/rtask_psfp_gtex; ARMENV="VLMAS_C1_QASC=1 VLMAS_C1_SELECT=ntrs VLMAS_QASC_KEEP=0.5 VLMAS_C1_META_BRIDGE=1 VLMAS_PSFP=1" ;;
  psfp_diag) K=$REPO/sender_relay_exp/runs/rtask_psfp_gtex; ARMENV="VLMAS_C1_QASC=1 VLMAS_C1_SELECT=ntrs VLMAS_QASC_KEEP=0.5 VLMAS_C1_META_BRIDGE=1 VLMAS_PSFP=diag" ;;
  lrpsr) K=$REPO/sender_relay_exp/runs/rtask_lrpsr_gtex; ARMENV="VLMAS_C1_QASC=1 VLMAS_C1_SELECT=ntrs VLMAS_QASC_KEEP=0.5 VLMAS_C1_META_BRIDGE=1 VLMAS_LRPSR=1" ;;
  lrpsr_identity) K=$REPO/sender_relay_exp/runs/rtask_lrpsr_gtex; ARMENV="VLMAS_C1_QASC=1 VLMAS_C1_SELECT=ntrs VLMAS_QASC_KEEP=0.5 VLMAS_C1_META_BRIDGE=1 VLMAS_LRPSR=identity" ;;
  *) echo bad arm; exit 2 ;;
esac
HOOKS=$REPO/tmp_hj/lpvh_ntrs_hooks
CLAIM=/home/users/whddn12316/wsi_latent_0902_2155_hj/sender_relay_exp/runs/vgain
LOG=$K/chain_${ARM}_shard$SHARD.log
RTASK=$'For each patch, examine its image quality, architecture, cellularity, cytology, stroma, necrosis, tissue boundary, artifacts, and relevance to the question. Focus only on visually observable findings and distinguish true tissue features from artifacts. State your best morphological read for every feature even when uncertain; do not report a feature as unresolvable.\n\nFormat your response as follows:\n- One bullet point per patch, starting with its patch ID, using terse 2-4 word phrases per feature.\n- Consistent findings: findings shared across patches and the patch IDs that support them.\n- Contradictions: disagreements between patches.\n- Unresolved evidence: what remains uncertain.\n- Diagnostically usable patches: patch IDs.\n\nDo not diagnose, choose an answer, or navigate.\nNow, output your response below:'
mkdir -p $K $CLAIM
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }
mem() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' -v g=$1 '$1==g{print $2}'; }
G=""
if [ -n "$HGPU" ] && [ -n "$HPID" ]; then
  log "HANDOVER wait for pid $HPID on gpu$HGPU (previous launcher, same shard) then take its claim"
  while kill -0 "$HPID" 2>/dev/null; do sleep 5; done
  rmdir $CLAIM/claim_gpu$HGPU 2>/dev/null
  if mkdir $CLAIM/claim_gpu$HGPU 2>/dev/null; then G=$HGPU; fi
fi
if [ "$SHARD" != smoke ]; then
  log "WAIT smoke gate"
  while [ ! -f $K/SMOKE_GATE_$ARM ]; do sleep 30; done
  grep -q ^PASS $K/SMOKE_GATE_$ARM || { log "SMOKE FAILED -> stop"; exit 1; }
fi
if [ -n "${WAIT_LADR:-}" ]; then
  W=$REPO/sender_relay_exp/runs/rtask_ladr_gtex
  log "WAIT ${WAIT_DIR:-rtask_ladr_gtex} ${WAIT_DS:-panda} ${WAIT_N:-196}"
  W=$REPO/sender_relay_exp/runs/${WAIT_DIR:-rtask_ladr_gtex}
  while [ "$(find $W/${WAIT_DS:-panda} -name result.json 2>/dev/null | wc -l)" -lt ${WAIT_N:-196} ]; do sleep 60; done
fi
if [ -n "${WAIT_GTEX:-}" ]; then
  log "WAIT LR-PSR GTEx 190 results"
  while [ "$(find $K/gtex -name result.json 2>/dev/null | wc -l)" -lt 190 ]; do sleep 30; done
fi
log "WAIT one of gpus [$GPUS] shard $SHARD/$NSHARD (handover=${G:-none})"
while [ -z "$G" ]; do
  for g in $GPUS; do
    u=$(mem $g)
    if [ -n "$u" ] && [ "$u" -lt 2000 ] && mkdir $CLAIM/claim_gpu$g 2>/dev/null; then G=$g; break; fi
  done
  [ -z "$G" ] && sleep 60
done
trap 'rmdir $CLAIM/claim_gpu$G 2>/dev/null' EXIT
log "CLAIM gpu$G shard=$SHARD nav3_md5=$(md5sum $REPO/vision_text_mas/onepass_navigator.py | cut -c1-8) qasc_md5=$(md5sum $REPO/memory/qasc_select.py | cut -c1-8) ssda_md5=$(md5sum $REPO/memory/ssda_select.py | cut -c1-8) backbone_md5=$(md5sum $REPO/backbone/qwen3vl.py | cut -c1-8) engine_md5=$(md5sum $REPO/vision_text_mas/latent_qwen_engine.py | cut -c1-8)"

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
    --variant $VARIANT --backbone qwen3-vl --dataset "$dsjson" --slide-root "$DATA/slides" \
    --model "$MODEL" --output-root "$out" --device cuda:0 \
    --latent-steps 10 --patch-budget 12 --max-model-len 12288 --navigator-control-tokens 512 \
    --temperature 0.0 --top-p 1.0 --seed 42 --deterministic --answerer-greedy --no-answerer-thinking \
    --answerer-max-new-tokens 512 --answerer-rationale --answerer-protocol structured_json \
    --canonical-open-options --navigator-kv --io-pipeline --no-save-navigation-pngs --case-retries 0 \
    "$@" > "$out.log" 2>&1 < /dev/null
  local rc=$?
  log "DONE $ARM $tag rc=$rc results=$(find $out -name result.json | wc -l) psfp_case=$(grep -c '^\[PSFP\] case' $out.log) ladr_case=$(grep -c '^\[LADR\] case' $out.log) realloc=$(grep -c '^\[reallocate\]' $out.log) empty=$(grep -c 'EMPTY terminal answer' $out.log) ntrs1536=$(grep -c '^\[NTRS\] kept 1536' $out.log) c1meta_ok=$(grep -c '^\[C1META\] keep .* used=True meta=ok' $out.log) empty=$(grep -c 'EMPTY terminal answer' $out.log) tb=$(grep -c Traceback $out.log) oom=$(grep -ci 'out of memory' $out.log) eff=$(find $out -name efficiency.json | wc -l)"
}

if [ "$SHARD" = smoke ]; then
  run smoke $REPO/sender_relay_exp/smoke/gtex2.json $K/smoke_$ARM --dataset-index 0 --dataset-index 1
  o=$(ls -d $K/smoke_$ARM/gpu*_attempt_* | grep -v log$ | tail -1)
  r=$(find $o -name result.json | wc -l); k=$(grep -c '^\[NTRS\] kept 1536/3072' $o.log)
  t=$(grep -c Traceback $o.log); c=$(grep -cE '^\[(LRPSR|PSFP|LADR)\] case' $o.log); rl=$(grep -c '^\[reallocate\]' $o.log); sk=$(grep -cE '^\[(LRPSR|PSFP|LADR)\] SKIP' $o.log); em=$(grep -c 'EMPTY terminal answer' $o.log)
  md=$(grep -oE '(maxdiff_identity=[0-9.e+-]*|activated=[0-9]+ \([0-9.]+%\) argmax_flips=[0-9]+)' $o.log | tail -2 | tr '\n' ' ')
  GATE_CASE=2; case $ARM in realloc|realloc_v2) GATE_CASE=$c ;; esac
  if [ "$r" -eq 2 ] && [ "$k" -eq 2 ] && [ "$t" -eq 0 ] && [ "$c" -eq "$GATE_CASE" ] && [ "$sk" -eq 0 ] && [ "$em" -eq 0 ]; then
    echo "PASS results=$r kept1536=$k case=$c realloc_lines=$rl skip=$sk empty=$em tb=$t $md" > $K/SMOKE_GATE_$ARM
  else
    echo "FAIL results=$r kept1536=$k case=$c realloc_lines=$rl skip=$sk empty=$em tb=$t $md" > $K/SMOKE_GATE_$ARM
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
