#!/bin/bash
# Question-Agnostic Cross-Scale Innovation-Support Selection (CSIS support=branch) — 2-case GPU smoke.
# Setting = nav2v2_csis_sweep.sh verbatim (sdpa, VLMAS_NAV2=1, patch 12, max-model-len 12288, B=768),
# only VLMAS_CSIS_SUPPORT=branch added, on smoke/gtex2.json, with a selection dump for offline checks.
# Waits for a GPU among 8/7/6 that is <2GB twice 60 s apart and not claimed by the VGain driver
# (shares its claim dirs in runs/vgain so the two never stack). Smoke only — no full run.
set -u
REPO=/home/users/whddn12316/wsi_latent_0915_decode_hj
DATA=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
MODEL=/home/users/whddn12316/models/Qwen3-VL-4B-Thinking
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
K=$REPO/sender_relay_exp/runs/csis_branch
CLAIM=$REPO/sender_relay_exp/runs/vgain
mkdir -p $K $CLAIM
LOG=$K/smoke_chain.log
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }
mem() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' -v g=$1 '$1==g{print $2}'; }

log "WAIT gpu (8/7/6, shared claims with vgain)"
G=""
while [ -z "$G" ]; do
  for g in 8 7 6; do
    [ -d $CLAIM/claim_gpu$g ] && continue
    u=$(mem $g); [ -n "$u" ] && [ "$u" -lt 2000 ] || continue
    sleep 60
    u2=$(mem $g); [ -d $CLAIM/claim_gpu$g ] && continue
    if [ -n "$u2" ] && [ "$u2" -lt 2000 ] && mkdir $CLAIM/claim_gpu$g 2>/dev/null; then G=$g; break; fi
  done
  [ -z "$G" ] && sleep 120
done
trap 'rmdir $CLAIM/claim_gpu$G 2>/dev/null' EXIT
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=$K/smoke_$STAMP
log "CLAIM gpu$G -> $(basename $OUT)"
cd $REPO && env \
  CUDA_VISIBLE_DEVICES="$G" PYTHONPATH="$REPO" PYTHONUNBUFFERED=1 \
  VLMAS_ATTN_IMPLEMENTATION=sdpa VLMAS_NAV2=1 VLMAS_EFF_DUMP=1 \
  VLMAS_WSI_CONSOL=1 VLMAS_WSI_CONSOL_MODE=csis VLMAS_WSI_CONSOL_BUDGET=768 \
  VLMAS_CSIS_SUPPORT=branch VLMAS_CSIS_DUMP=$OUT.dump \
  timeout 3600 "$PY" -m wsi_latentmas.pipeline.latent_mas \
  --variant base --backbone qwen3-vl --dataset "$REPO/sender_relay_exp/smoke/gtex2.json" \
  --slide-root "$DATA/slides" --model "$MODEL" --output-root "$OUT" --device cuda:0 \
  --latent-steps 10 --patch-budget 12 --max-model-len 12288 --navigator-control-tokens 512 \
  --temperature 0.0 --top-p 1.0 --seed 42 --deterministic --answerer-greedy --no-answerer-thinking \
  --answerer-max-new-tokens 512 --answerer-rationale --answerer-protocol structured_json \
  --canonical-open-options --navigator-kv --io-pipeline --no-save-navigation-pngs --case-retries 0 \
  > "$OUT.log" 2>&1 < /dev/null
rc=$?
marks=$(grep -c '^\[WSIConsol\] csis .*support=branch' $OUT.log)
tb=$(grep -c Traceback $OUT.log); oom=$(grep -ci 'out of memory' $OUT.log)
res=$(find $OUT -name result.json 2>/dev/null | wc -l); dumps=$(ls $OUT.dump/csis_*.pt 2>/dev/null | wc -l)
gate=PASS
{ [ "$tb" -eq 0 ] && [ "$res" -eq 2 ] && [ "$marks" -ge 2 ]; } || gate=FAIL
log "DONE gpu$G rc=$rc marks=$marks results=$res dumps=$dumps tb=$tb oom=$oom GATE $gate"
grep '^\[WSIConsol\] csis' $OUT.log | cut -c1-600 >> $LOG
