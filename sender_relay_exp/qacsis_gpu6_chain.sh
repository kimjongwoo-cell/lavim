#!/bin/bash
# Question-Agnostic CSIS (support=branch, B=768) on the board's 4B · patch 12 · step 10 Latent base
# setting, full 5 datasets, GPU 6 only — starts as soon as the SlideBench sdpa base rerun on GPU 6 ends.
#
# Flags = sender_relay_exp/sb_sdpa_base.sh (the board baseline setting: sdpa · VLMAS_NAV2=1 · patch 12 ·
# max-model-len 12288 · step 10) + VLMAS_WSI_CONSOL=1 MODE=csis SUPPORT=branch BUDGET=768.
# 1) wait for the SB launcher PID to exit   2) GPU 6 must be < 2GB (never stack on another run)
# 3) 2-case smoke on gtex2 (gate: traceback 0, results 2, >=2 "[WSIConsol] csis ... support=branch ... B=768")
# 4) full datasets one after another, resumable (indices with a result.json are skipped)
# 5) exact rescore per dataset into chain.log (scripts/rescore.py)
# GPU 6 is claimed in runs/vgain/claim_gpu6 for the whole chain so the VGain driver never takes it.
set -u
REPO=/home/users/whddn12316/wsi_latent_0915_decode_hj
DATA=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
MODEL=/home/users/whddn12316/models/Qwen3-VL-4B-Thinking
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
K=$REPO/sender_relay_exp/runs/qacsis_b768
CLAIM=$REPO/sender_relay_exp/runs/vgain
SB_LAUNCHER_PID=${SB_LAUNCHER_PID:-2770536}
G=6
mkdir -p $K $CLAIM
LOG=$K/chain.log
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }
mem() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' -v g=$1 '$1==g{print $2}'; }

if ! mkdir $CLAIM/claim_gpu$G 2>/dev/null; then log "ABORT: claim_gpu$G already exists"; exit 1; fi
trap 'rmdir $CLAIM/claim_gpu$G 2>/dev/null' EXIT
log "CLAIM gpu$G; waiting for SB launcher pid $SB_LAUNCHER_PID"
while kill -0 $SB_LAUNCHER_PID 2>/dev/null; do sleep 30; done
log "SB launcher exited: $(tail -1 $REPO/sender_relay_exp/runs/sb_sdpa_base/chain.log)"
while true; do
  u=$(mem $G)
  [ -n "$u" ] && [ "$u" -lt 2000 ] && break
  log "gpu$G busy (${u}MiB) — another run is on it, waiting"
  sleep 60
done

run_set () {   # $1 name  $2 dataset.json  $3 dest dir ; resumable
  local NAME=$1 DSJ=$2 DEST=$3
  mkdir -p "$DEST"
  local STAMP=$(date +%Y%m%d_%H%M%S)
  local ATT=$DEST/attempt_$STAMP
  local N=$("$PY" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$DSJ")
  local DONE_IDS=$(find "$DEST" -name result.json 2>/dev/null -exec "$PY" -c \
    'import json,sys; [print(json.load(open(f))["dataset_index"]) for f in sys.argv[1:]]' {} + | sort -n -u)
  local IDS=()
  for ((i=0; i<N; i++)); do grep -qx "$i" <<< "$DONE_IDS" || IDS+=(--dataset-index "$i"); done
  if [ ${#IDS[@]} -eq 0 ]; then log "$NAME already complete ($N)"; return 0; fi
  "$PY" - "$ATT.meta.json" "$DSJ" "$N" "$(( ${#IDS[@]} / 2 ))" <<'PYMETA'
import datetime, json, os, sys
json.dump({"started_at": datetime.datetime.now().isoformat(timespec="seconds"),
           "script": "sender_relay_exp/qacsis_gpu6_chain.sh", "gpu": "6", "dataset": sys.argv[2],
           "n": int(sys.argv[3]), "pending": int(sys.argv[4]),
           "env": {k: os.environ.get(k) for k in sorted(os.environ) if k.startswith("VLMAS_")}},
          open(sys.argv[1], "w"), indent=1)
PYMETA
  log "START $NAME pending=$(( ${#IDS[@]} / 2 ))/$N -> $(basename $DEST)/$(basename $ATT)"
  cd $REPO && env CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$REPO PYTHONUNBUFFERED=1 \
    "$PY" -m wsi_latentmas.pipeline.latent_mas \
    --variant base --backbone qwen3-vl --dataset "$DSJ" --slide-root "$DATA/slides" --model "$MODEL" \
    --output-root "$ATT" --device cuda:0 --latent-steps 10 --patch-budget 12 --max-model-len 12288 \
    --navigator-control-tokens 512 --temperature 0.0 --top-p 1.0 --seed 42 --deterministic \
    --answerer-greedy --no-answerer-thinking --answerer-max-new-tokens 512 --answerer-rationale \
    --answerer-protocol structured_json --canonical-open-options --navigator-kv --io-pipeline \
    --no-save-navigation-pngs --case-retries 0 "${IDS[@]}" > "$ATT.log" 2>&1 < /dev/null
  local rc=$?
  log "DONE $NAME rc=$rc results=$(find $DEST -name result.json | wc -l)/$N marks=$(grep -c '^\[WSIConsol\] csis .*support=branch' $ATT.log) tb=$(grep -c Traceback $ATT.log) oom=$(grep -ci 'out of memory' $ATT.log)"
  return $rc
}

export VLMAS_ATTN_IMPLEMENTATION=sdpa VLMAS_NAV2=1
export VLMAS_WSI_CONSOL=1 VLMAS_WSI_CONSOL_MODE=csis VLMAS_WSI_CONSOL_BUDGET=768 VLMAS_CSIS_SUPPORT=branch

# ---- smoke (with selection dump)
SM=$K/smoke
export VLMAS_CSIS_DUMP=$K/smoke_dump
run_set smoke $REPO/sender_relay_exp/smoke/gtex2.json $SM
unset VLMAS_CSIS_DUMP
SL=$(ls -t $SM/attempt_*.log | head -1)
marks=$(grep -c '^\[WSIConsol\] csis .*support=branch.* B=768' $SL)
tb=$(grep -c Traceback $SL); res=$(find $SM -name result.json | wc -l)
if [ "$tb" -ne 0 ] || [ "$res" -ne 2 ] || [ "$marks" -lt 2 ]; then
  log "SMOKE GATE FAIL tb=$tb results=$res marks=$marks — full run NOT launched"; tail -30 $SL >> $LOG; exit 1
fi
log "SMOKE GATE PASS tb=0 results=2 marks=$marks"
grep '^\[WSIConsol\] csis' $SL | cut -c1-500 >> $LOG

# ---- full datasets (slowest last)
for ds in gtex tcga_expert_vqa panda tcga_slidebench tcga; do
  run_set $ds $DATA/$ds.json $K/$ds
  (cd $REPO && "$PY" scripts/rescore.py --root "$K/$ds" --dataset $ds 2>&1 \
     | "$PY" -c 'import json,sys
try:
    rows=json.load(sys.stdin)
    for r in rows: print("RESCORE", r["dataset"], r["score_name"], "n", r["n"], "exact", r["exact"]["score"], "correct", r["exact"]["correct"], "sec", r["sec"])
except Exception as e: print("RESCORE parse failed", e)') >> $LOG
done
log "ALL DONE"
