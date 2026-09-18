#!/bin/bash
# nav4 stack, in board order: base -> latplan -> + C1 NOVA rho -> + C2 GLVR -> GLVR relay only.
# Each arm: 2-case smoke (must PASS) -> N shards in the background -> wait -> exact score vs the
# arm above it -> next arm. See README_nav4_latplan_glvr_0918.md for what each arm changes.
#
# usage (from sender_relay_exp/):
#   DRY_RUN=1 bash run_stack_0918.sh                 # print every command, run nothing
#   bash run_stack_0918.sh                            # run the whole stack into a NEW root
#   ARMS="glvr_rho glvr_rho_nf" bash run_stack_0918.sh   # only some arms (the ref arm must exist in OUTROOT)
#
# env:
#   OUTROOT   results root, default runs/stack_<date>; every arm goes to $OUTROOT/<arm>
#   ARMS      default "base latplan nova_rho_latplan glvr_rho glvr_rho_nf"
#   GPUS      default "6 7 8"      NSHARD default 3      MAXUSED default 20000
#   SPECS     default "tcga_expert_vqa:128 tcga_slidebench:197"
#   SLIDE_ROOT  optional; default is the dataset's own slides/ (the original /home path)
set -u
cd "$(dirname "$0")"
SR=$PWD
REPO=$(cd .. && pwd)
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
OUTROOT=${OUTROOT:-$SR/runs/stack_$(date +%m%d_%H%M)}
ARMS=${ARMS:-"base latplan nova_rho_latplan glvr_rho glvr_rho_nf"}
GPUS=${GPUS:-"6 7 8"}
NSHARD=${NSHARD:-3}
MAXUSED=${MAXUSED:-20000}
SPECS=${SPECS:-"tcga_expert_vqa:128 tcga_slidebench:197"}
DRY=${DRY_RUN:-0}
CLAIM=$OUTROOT/claims
SMOKE=$SR/smoke/glvr_rho2.json          # EVQA __4 + SlideBench __2339
DSS=$(for s in $SPECS; do printf "%s," "${s%%:*}"; done | sed 's/,$//')

# arm -> launcher, hooks override, the arm it is compared with (the row above it on the board)
launcher() { [ "$1" = nova_rho_latplan ] && echo rtask_nav4_nova_rho_latplan_arm.sh || echo rtask_nav4_nonav2_arm.sh; }
hooks() {
  case $1 in
    glvr_rho|glvr_rho_nf) echo "$REPO/tmp_hj/glvr_fast2" ;;
    glvr_rho_fast)        echo "$REPO/tmp_hj/glvr_fast3" ;;
    *)                    echo "" ;;       # launcher default (base: lpvh_ntrs_hooks, latplan: latplan_hooks)
  esac
}
ref_of() {
  case $1 in
    latplan) echo base ;; nova_rho_latplan) echo latplan ;;
    glvr_rho|glvr_rho_nf|glvr_rho_fast) echo nova_rho_latplan ;; *) echo "" ;;
  esac
}
log() { echo "[stack $(date +%H:%M:%S)] $*" | tee -a "$OUTROOT/stack.log"; }
run() { if [ "$DRY" = 1 ]; then echo "  + $*"; else eval "$@"; fi; }

mkdir -p "$OUTROOT" "$CLAIM"
log "OUTROOT=$OUTROOT ARMS=[$ARMS] GPUS=[$GPUS] NSHARD=$NSHARD SPECS=[$SPECS] SLIDE_ROOT=${SLIDE_ROOT:-default} DRY=$DRY"
log "md5 nav4=$(md5sum $REPO/vision_text_mas/onepass_navigation_nav4.py | cut -c1-8) answerer=$(md5sum $REPO/vision_text_mas/latent_answerer.py | cut -c1-8) backbone=$(md5sum $REPO/backbone/qwen3vl.py | cut -c1-8) rnlcr=$(md5sum $REPO/memory/rnlcr.py | cut -c1-8) nova_rho=$(md5sum $REPO/memory/nova_rho_select.py | cut -c1-8)"

for ARM in $ARMS; do
  L=$(launcher "$ARM"); H=$(hooks "$ARM")
  ENV="OUTROOT=$OUTROOT CLAIMDIR=$CLAIM MAXUSED=$MAXUSED${H:+ HOOKS_OVERRIDE=$H}${SLIDE_ROOT:+ SLIDE_ROOT=$SLIDE_ROOT}"
  log "== $ARM ($L${H:+, hooks $H})"

  # 1) smoke; the nova_rho launcher has its own fixed 2-case smoke
  if [ "$L" = rtask_nav4_nonav2_arm.sh ]; then
    run "env $ENV SMOKE_JSON=$SMOKE SMOKE_N=2 bash ./$L $ARM smoke $NSHARD \"$GPUS\""
  else
    run "env $ENV bash ./$L $ARM smoke $NSHARD \"$GPUS\""
  fi
  GATE=$OUTROOT/$ARM/SMOKE_GATE_$ARM
  if [ "$DRY" != 1 ] && ! grep -q ^PASS "$GATE" 2>/dev/null; then
    log "SMOKE FAILED for $ARM ($(head -c 200 "$GATE" 2>/dev/null)) -> stop"; exit 1
  fi
  log "smoke PASS"

  # 2) shards in the background, then wait for all of them
  PIDS=""
  for s in $(seq 0 $((NSHARD - 1))); do
    if [ "$DRY" = 1 ]; then
      echo "  + env $ENV nohup bash ./$L $ARM $s $NSHARD \"$GPUS\" $SPECS &"
    else
      env $ENV nohup bash ./$L $ARM $s $NSHARD "$GPUS" $SPECS > /dev/null 2>&1 < /dev/null &
      PIDS="$PIDS $!"; sleep 3
    fi
  done
  [ "$DRY" = 1 ] || { log "shards pids:$PIDS"; for p in $PIDS; do wait $p; done; }
  log "$ARM shards done: $(find $OUTROOT/$ARM -name result.json 2>/dev/null | wc -l) results"

  # 3) exact score against the arm above it
  R=$(ref_of "$ARM")
  if [ -n "$R" ]; then
    run "cd $REPO && ARM_ROOT=$OUTROOT PAIRS=$ARM:$R DSS=$DSS $PY sender_relay_exp/tools_0918/score_pairs.py | tee -a $OUTROOT/scores.jsonl; cd $SR"
  fi
done
log "ALL DONE -> $OUTROOT (scores: $OUTROOT/scores.jsonl)"
