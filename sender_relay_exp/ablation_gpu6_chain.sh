#!/bin/bash
# GPU 6 ablation chain (09-14 user plan), board setting 4B · patch 12 · step 10 Latent base
# (sb_sdpa_base.sh flags: sdpa · VLMAS_NAV2=1 · max-model-len 12288), full datasets, exact rescore.
#
#   QA     = Question-Agnostic Cross-Scale Innovation-Support Selection (csis, support=branch, B=768)
#   PSAR   = Physical-Support Additive Readout alone (defaults: L24-30, eta 1, target visual, rows gen, support branch)
#   QAPSAR = QA + PSAR
# Order: QA evqa, QA sb, PSAR evqa, PSAR sb, QAPSAR evqa, QAPSAR sb, then QA gtex, QA panda, QA tcga.
#
# 1) wait for the SlideBench sdpa base rerun (launcher PID) to exit, GPU 6 < 2GB
# 2) 2-case smokes on gtex2: QA, PSAR, QAPSAR. QA gate: tb 0 · results 2 · ">=2 csis support=branch B=768".
#    PSAR gates additionally: ">=2 [PSAR] mode=", "0 [PSAR] SKIP", ">=2 [PSAR] case".
#    A failed QA smoke stops everything; a failed PSAR/QAPSAR smoke drops only that arm.
# 3) full runs in the order above, resumable, exact rescore of each into chain.log.
# GPU 6 stays claimed (runs/vgain/claim_gpu6) for the whole chain.
set -u
REPO=/home/users/whddn12316/wsi_latent_0915_decode_hj
DATA=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
MODEL=/home/users/whddn12316/models/Qwen3-VL-4B-Thinking
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
RUNS=$REPO/sender_relay_exp/runs
K=$RUNS/ablation_gpu6
CLAIM=$RUNS/vgain
SB_LAUNCHER_PID=${SB_LAUNCHER_PID:-2770536}
G=6
mkdir -p $K $CLAIM
LOG=$K/chain.log
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }
mem() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' -v g=$1 '$1==g{print $2}'; }

QA_ENV="VLMAS_WSI_CONSOL=1 VLMAS_WSI_CONSOL_MODE=csis VLMAS_WSI_CONSOL_BUDGET=768 VLMAS_CSIS_SUPPORT=branch"
PSAR_ENV="VLMAS_PSAR=1"
declare -A ARM_ENV=( [qa]="$QA_ENV" [psar]="$PSAR_ENV" [qapsar]="$QA_ENV $PSAR_ENV" )
declare -A ARM_DIR=( [qa]=$RUNS/qacsis_b768 [psar]=$RUNS/psar_p12 [qapsar]=$RUNS/qacsis_psar_b768 )
declare -A ARM_OK=( [qa]=1 [psar]=1 [qapsar]=1 )

if ! mkdir $CLAIM/claim_gpu$G 2>/dev/null; then log "ABORT: claim_gpu$G already exists"; exit 1; fi
trap 'rmdir $CLAIM/claim_gpu$G 2>/dev/null' EXIT
log "CLAIM gpu$G; waiting for SB launcher pid $SB_LAUNCHER_PID"
while kill -0 $SB_LAUNCHER_PID 2>/dev/null; do sleep 30; done
log "SB launcher exited: $(tail -1 $RUNS/sb_sdpa_base/chain.log)"
while true; do
  u=$(mem $G)
  [ -n "$u" ] && [ "$u" -lt 2000 ] && break
  log "gpu$G busy (${u}MiB) — another run is on it, waiting"
  sleep 60
done

run_set () {   # $1 arm  $2 name  $3 dataset.json  $4 dest ; extra env from ARM_ENV (+ $5 extra)
  local ARM=$1 NAME=$2 DSJ=$3 DEST=$4 EXTRA=${5:-}
  mkdir -p "$DEST"
  local STAMP=$(date +%Y%m%d_%H%M%S)
  local ATT=$DEST/attempt_$STAMP
  local N=$("$PY" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$DSJ")
  local DONE_IDS=$(find "$DEST" -name result.json 2>/dev/null -exec "$PY" -c \
    'import json,sys; [print(json.load(open(f))["dataset_index"]) for f in sys.argv[1:]]' {} + | sort -n -u)
  local IDS=()
  for ((i=0; i<N; i++)); do grep -qx "$i" <<< "$DONE_IDS" || IDS+=(--dataset-index "$i"); done
  if [ ${#IDS[@]} -eq 0 ]; then log "$ARM/$NAME already complete ($N)"; LAST_LOG=""; return 0; fi
  env VLMAS_ATTN_IMPLEMENTATION=sdpa VLMAS_NAV2=1 ${ARM_ENV[$ARM]} $EXTRA "$PY" - "$ATT.meta.json" "$DSJ" "$N" "$ARM" <<'PYMETA'
import datetime, json, os, sys
json.dump({"started_at": datetime.datetime.now().isoformat(timespec="seconds"),
           "script": "sender_relay_exp/ablation_gpu6_chain.sh", "gpu": "6", "arm": sys.argv[4],
           "dataset": sys.argv[2], "n": int(sys.argv[3]),
           "env": {k: os.environ[k] for k in sorted(os.environ) if k.startswith("VLMAS_")}},
          open(sys.argv[1], "w"), indent=1)
PYMETA
  log "START $ARM/$NAME pending=$(( ${#IDS[@]} / 2 ))/$N -> ${DEST#$RUNS/}/$(basename $ATT)"
  cd $REPO && env CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$REPO PYTHONUNBUFFERED=1 \
    VLMAS_ATTN_IMPLEMENTATION=sdpa VLMAS_NAV2=1 ${ARM_ENV[$ARM]} $EXTRA \
    "$PY" -m wsi_latentmas.pipeline.latent_mas \
    --variant base --backbone qwen3-vl --dataset "$DSJ" --slide-root "$DATA/slides" --model "$MODEL" \
    --output-root "$ATT" --device cuda:0 --latent-steps 10 --patch-budget 12 --max-model-len 12288 \
    --navigator-control-tokens 512 --temperature 0.0 --top-p 1.0 --seed 42 --deterministic \
    --answerer-greedy --no-answerer-thinking --answerer-max-new-tokens 512 --answerer-rationale \
    --answerer-protocol structured_json --canonical-open-options --navigator-kv --io-pipeline \
    --no-save-navigation-pngs --case-retries 0 "${IDS[@]}" > "$ATT.log" 2>&1 < /dev/null
  local rc=$?
  LAST_LOG=$ATT.log
  log "DONE $ARM/$NAME rc=$rc results=$(find $DEST -name result.json | wc -l)/$N csis=$(grep -c '^\[WSIConsol\] csis .*support=branch' $ATT.log) psar_on=$(grep -c '^\[PSAR\] mode=' $ATT.log) psar_skip=$(grep -c '^\[PSAR\] SKIP' $ATT.log) tb=$(grep -c Traceback $ATT.log) oom=$(grep -ci 'out of memory' $ATT.log)"
  return $rc
}

rescore () {   # $1 dest  $2 dataset  $3 label
  (cd $REPO && "$PY" scripts/rescore.py --root "$1" --dataset $2 2>&1 | "$PY" -c 'import json,sys
lab=sys.argv[1]
try:
    for r in json.load(sys.stdin):
        print("RESCORE", lab, r["dataset"], r["score_name"], "n", r["n"], "exact", r["exact"]["score"], "correct", r["exact"]["correct"], "sec", r["sec"])
except Exception as e:
    print("RESCORE", lab, "parse failed", e)' "$3") >> $LOG
}

# ---- smokes
for arm in qa psar qapsar; do
  SM=${ARM_DIR[$arm]}/smoke
  extra=""; [ "$arm" != "psar" ] && extra="VLMAS_CSIS_DUMP=${ARM_DIR[$arm]}/smoke_dump"
  run_set $arm smoke $REPO/sender_relay_exp/smoke/gtex2.json $SM "$extra"
  SL=$(ls -t $SM/attempt_*.log 2>/dev/null | head -1)
  if [ -z "$SL" ]; then
    res=$(find $SM -name result.json | wc -l); log "SMOKE $arm: no new attempt (results=$res already)"; [ "$res" -eq 2 ] || ARM_OK[$arm]=0; continue
  fi
  tb=$(grep -c Traceback $SL); res=$(find $SM -name result.json | wc -l)
  cs=$(grep -c '^\[WSIConsol\] csis .*support=branch.* B=768' $SL)
  pon=$(grep -c '^\[PSAR\] mode=' $SL); psk=$(grep -c '^\[PSAR\] SKIP' $SL); pcs=$(grep -c '^\[PSAR\] case' $SL)
  ok=1
  { [ "$tb" -eq 0 ] && [ "$res" -eq 2 ]; } || ok=0
  [ "$arm" != "psar" ] && [ "$cs" -lt 2 ] && ok=0
  [ "$arm" != "qa" ] && { [ "$pon" -lt 2 ] || [ "$psk" -ne 0 ] || [ "$pcs" -lt 2 ]; } && ok=0
  ARM_OK[$arm]=$ok
  log "SMOKE $arm GATE $([ $ok -eq 1 ] && echo PASS || echo FAIL) tb=$tb results=$res csis=$cs psar_on=$pon psar_skip=$psk psar_case=$pcs"
  grep '^\[PSAR\]\|^\[WSIConsol\] csis' $SL | cut -c1-500 >> $LOG
  if [ $ok -eq 0 ]; then tail -25 $SL >> $LOG; fi
done
if [ "${ARM_OK[qa]}" -ne 1 ]; then log "QA smoke failed — chain stops"; exit 1; fi

# ---- full runs
for step in qa:tcga_expert_vqa qa:tcga_slidebench psar:tcga_expert_vqa psar:tcga_slidebench \
            qapsar:tcga_expert_vqa qapsar:tcga_slidebench qa:gtex qa:panda qa:tcga; do
  arm=${step%%:*}; ds=${step#*:}
  if [ "${ARM_OK[$arm]}" -ne 1 ]; then log "SKIP $arm/$ds (smoke gate failed)"; continue; fi
  run_set $arm $ds $DATA/$ds.json ${ARM_DIR[$arm]}/$ds
  rescore ${ARM_DIR[$arm]}/$ds $ds $arm
done
log "ALL DONE"
