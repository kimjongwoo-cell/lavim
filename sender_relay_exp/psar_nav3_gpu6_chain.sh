#!/bin/bash
# PSAR (Physical-Support Additive Readout) on the nav3 setting, full 5 datasets, GPU 6 only.
# Env = the nav3 base runs of 09-14 (sender_relay_exp/runs/nav3_prompt1_base_*):
#   VLMAS_NAV3_INDEPENDENT=1 VLMAS_NAV2=1 VLMAS_PROMPT_SET=prompt1 VLMAS_ATTN_IMPLEMENTATION=sdpa VLMAS_CROSS_SCALE_ROUTER=0
#   + same CLI flags (4B · patch 12 · step 10 · mlen 12288) + VLMAS_PSAR=1 (defaults: L24-30, eta 1, target visual,
#   rows gen, support branch -> on nav3 every crop is its own support since no x20 lies inside an x5).
# 1) GPU 6 claimed (runs/vgain/claim_gpu6) and < 2GB   2) gtex2 smoke gate: tb 0, results 2, "[PSAR] mode=" >= 2,
#    "[PSAR] SKIP" 0, "[PSAR] case" >= 2   3) evqa, slidebench, gtex, panda, tcga (resumable) + exact rescore each.
# The md5 of vision_text_mas/onepass_navigation_nav3.py is written into every attempt's meta (the file is being edited).
set -u
REPO=/home/users/whddn12316/wsi_latent_0915_decode_hj
DATA=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
MODEL=/home/users/whddn12316/models/Qwen3-VL-4B-Thinking
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
RUNS=$REPO/sender_relay_exp/runs
K=$RUNS/psar_nav3
CLAIM=$RUNS/vgain
G=6
mkdir -p $K $CLAIM
LOG=$K/chain.log
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }
mem() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' -v g=$1 '$1==g{print $2}'; }

if ! mkdir $CLAIM/claim_gpu$G 2>/dev/null; then log "ABORT: claim_gpu$G already exists"; exit 1; fi
trap 'rmdir $CLAIM/claim_gpu$G 2>/dev/null' EXIT
while true; do
  u=$(mem $G); [ -n "$u" ] && [ "$u" -lt 2000 ] && break
  log "gpu$G busy (${u}MiB) — waiting"; sleep 60
done
log "CLAIM gpu$G"

ENVS="VLMAS_NAV3_INDEPENDENT=1 VLMAS_NAV2=1 VLMAS_PROMPT_SET=prompt1 VLMAS_ATTN_IMPLEMENTATION=sdpa VLMAS_CROSS_SCALE_ROUTER=0 VLMAS_PSAR=1"

run_set () {   # $1 name  $2 dataset.json  $3 dest
  local NAME=$1 DSJ=$2 DEST=$3
  mkdir -p "$DEST"
  local ATT=$DEST/attempt_$(date +%Y%m%d_%H%M%S)
  local N=$("$PY" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$DSJ")
  local DONE_IDS=$(find "$DEST" -name result.json 2>/dev/null -exec "$PY" -c \
    'import json,sys; [print(json.load(open(f))["dataset_index"]) for f in sys.argv[1:]]' {} + | sort -n -u)
  local IDS=()
  for ((i=0; i<N; i++)); do grep -qx "$i" <<< "$DONE_IDS" || IDS+=(--dataset-index "$i"); done
  if [ ${#IDS[@]} -eq 0 ]; then log "$NAME already complete ($N)"; LAST_LOG=""; return 0; fi
  env $ENVS "$PY" - "$ATT.meta.json" "$DSJ" "$N" "$(md5sum $REPO/vision_text_mas/onepass_navigation_nav3.py | cut -d' ' -f1)" <<'PYMETA'
import datetime, json, os, sys
json.dump({"started_at": datetime.datetime.now().isoformat(timespec="seconds"),
           "script": "sender_relay_exp/psar_nav3_gpu6_chain.sh", "gpu": "6", "dataset": sys.argv[2],
           "n": int(sys.argv[3]), "nav3_md5": sys.argv[4],
           "env": {k: os.environ[k] for k in sorted(os.environ) if k.startswith("VLMAS_")}},
          open(sys.argv[1], "w"), indent=1)
PYMETA
  log "START $NAME pending=$(( ${#IDS[@]} / 2 ))/$N -> $(basename $DEST)/$(basename $ATT) nav3_md5=$(md5sum $REPO/vision_text_mas/onepass_navigation_nav3.py | cut -c1-8)"
  cd $REPO && env CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$REPO PYTHONUNBUFFERED=1 $ENVS \
    "$PY" -m wsi_latentmas.pipeline.latent_mas \
    --variant base --backbone qwen3-vl --dataset "$DSJ" --slide-root "$DATA/slides" --model "$MODEL" \
    --output-root "$ATT" --device cuda:0 --latent-steps 10 --patch-budget 12 --max-model-len 12288 \
    --navigator-control-tokens 512 --temperature 0.0 --top-p 1.0 --seed 42 --deterministic \
    --answerer-greedy --no-answerer-thinking --answerer-max-new-tokens 512 --answerer-rationale \
    --answerer-protocol structured_json --canonical-open-options --navigator-kv --io-pipeline \
    --no-save-navigation-pngs --case-retries 0 "${IDS[@]}" > "$ATT.log" 2>&1 < /dev/null
  local rc=$?
  LAST_LOG=$ATT.log
  log "DONE $NAME rc=$rc results=$(find $DEST -name result.json | wc -l)/$N psar_on=$(grep -c '^\[PSAR\] mode=' $ATT.log) psar_skip=$(grep -c '^\[PSAR\] SKIP' $ATT.log) tb=$(grep -c Traceback $ATT.log) oom=$(grep -ci 'out of memory' $ATT.log)"
  return $rc
}

# ---- smoke
run_set smoke $REPO/sender_relay_exp/smoke/gtex2.json $K/smoke
SL=$(ls -t $K/smoke/attempt_*.log 2>/dev/null | head -1)
tb=$(grep -c Traceback $SL); res=$(find $K/smoke -name result.json | wc -l)
pon=$(grep -c '^\[PSAR\] mode=' $SL); psk=$(grep -c '^\[PSAR\] SKIP' $SL); pcs=$(grep -c '^\[PSAR\] case' $SL)
if [ "$tb" -ne 0 ] || [ "$res" -ne 2 ] || [ "$pon" -lt 2 ] || [ "$psk" -ne 0 ] || [ "$pcs" -lt 2 ]; then
  log "SMOKE GATE FAIL tb=$tb results=$res psar_on=$pon psar_skip=$psk psar_case=$pcs"; tail -30 $SL >> $LOG; exit 1
fi
log "SMOKE GATE PASS tb=0 results=2 psar_on=$pon psar_case=$pcs"
grep '^\[PSAR\]' $SL | cut -c1-500 >> $LOG

# ---- full
for ds in tcga_expert_vqa tcga_slidebench gtex panda tcga; do
  run_set $ds $DATA/$ds.json $K/$ds
  (cd $REPO && "$PY" scripts/rescore.py --root "$K/$ds" --dataset $ds 2>&1 | "$PY" -c 'import json,sys
try:
    for r in json.load(sys.stdin):
        print("RESCORE psar_nav3", r["dataset"], r["score_name"], "n", r["n"], "exact", r["exact"]["score"], "correct", r["exact"]["correct"], "sec", r["sec"])
except Exception as e:
    print("RESCORE parse failed", e)') >> $LOG
done
log "ALL DONE"
