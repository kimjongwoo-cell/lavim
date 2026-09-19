#!/bin/bash
# PSAR 2-case GPU smoke (gtex2), board baseline setting (sdpa · VLMAS_NAV2=1 · patch 12 · mlen 12288 · step 10).
#   arm c1psar : Question-Agnostic CSIS (branch, B=768) + VLMAS_PSAR=1   -> provenance after C1 pruning
#   arm ident  : no C1 + VLMAS_PSAR=identity                              -> provenance without C1, path check
# Gate per arm: traceback 0, results 2, "[PSAR] mode=" 2 and "[PSAR] SKIP" 0, "[PSAR] case" 2.
# GPUs 8/7 only (6 is held by the QA-CSIS chain); claim dirs shared with the VGain driver. Smoke only.
set -u
REPO=/home/users/whddn12316/wsi_latent_0915_decode_hj
DATA=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
MODEL=/home/users/whddn12316/models/Qwen3-VL-4B-Thinking
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
K=$REPO/sender_relay_exp/runs/psar
CLAIM=$REPO/sender_relay_exp/runs/vgain
mkdir -p $K $CLAIM
LOG=$K/smoke_chain.log
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }
mem() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' -v g=$1 '$1==g{print $2}'; }

log "WAIT gpu (8/7, shared claims)"
G=""
while [ -z "$G" ]; do
  for g in 8 7; do
    [ -d $CLAIM/claim_gpu$g ] && continue
    u=$(mem $g); [ -n "$u" ] && [ "$u" -lt 2000 ] || continue
    sleep 60
    u2=$(mem $g); [ -d $CLAIM/claim_gpu$g ] && continue
    if [ -n "$u2" ] && [ "$u2" -lt 2000 ] && mkdir $CLAIM/claim_gpu$g 2>/dev/null; then G=$g; break; fi
  done
  [ -z "$G" ] && sleep 120
done
trap 'rmdir $CLAIM/claim_gpu$G 2>/dev/null' EXIT
log "CLAIM gpu$G"

arm () {   # $1 name, rest env
  local NAME=$1; shift
  local OUT=$K/smoke_${NAME}_$(date +%Y%m%d_%H%M%S)
  cd $REPO && env CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$REPO PYTHONUNBUFFERED=1 \
    VLMAS_ATTN_IMPLEMENTATION=sdpa VLMAS_NAV2=1 "$@" \
    timeout 3600 "$PY" -m wsi_latentmas.pipeline.latent_mas \
    --variant base --backbone qwen3-vl --dataset "$REPO/sender_relay_exp/smoke/gtex2.json" \
    --slide-root "$DATA/slides" --model "$MODEL" --output-root "$OUT" --device cuda:0 \
    --latent-steps 10 --patch-budget 12 --max-model-len 12288 --navigator-control-tokens 512 \
    --temperature 0.0 --top-p 1.0 --seed 42 --deterministic --answerer-greedy --no-answerer-thinking \
    --answerer-max-new-tokens 512 --answerer-rationale --answerer-protocol structured_json \
    --canonical-open-options --navigator-kv --io-pipeline --no-save-navigation-pngs --case-retries 0 \
    > "$OUT.log" 2>&1 < /dev/null
  local rc=$?
  local on=$(grep -c '^\[PSAR\] mode=' $OUT.log) skip=$(grep -c '^\[PSAR\] SKIP' $OUT.log)
  local cs=$(grep -c '^\[PSAR\] case' $OUT.log) tb=$(grep -c Traceback $OUT.log)
  local res=$(find $OUT -name result.json 2>/dev/null | wc -l)
  local gate=PASS
  { [ "$tb" -eq 0 ] && [ "$res" -eq 2 ] && [ "$on" -ge 2 ] && [ "$skip" -eq 0 ] && [ "$cs" -ge 2 ]; } || gate=FAIL
  log "DONE $NAME rc=$rc results=$res psar_on=$on skip=$skip case_lines=$cs tb=$tb GATE $gate"
  grep '^\[PSAR\]\|^\[WSIConsol\] csis' $OUT.log | cut -c1-400 >> $LOG
}

arm c1psar VLMAS_WSI_CONSOL=1 VLMAS_WSI_CONSOL_MODE=csis VLMAS_WSI_CONSOL_BUDGET=768 VLMAS_CSIS_SUPPORT=branch VLMAS_PSAR=1
arm ident VLMAS_PSAR=identity VLMAS_PSAR_ROWS=all
log "ALL DONE"
