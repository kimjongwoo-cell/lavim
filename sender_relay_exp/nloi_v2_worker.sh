#!/bin/bash
# NLOI v2 (native generated token ids) on the nav2v2 setting, 10 cases per dataset.
# One worker per GPU; datasets are claimed atomically (mkdir), so GPU8 and GPU7 never run
# the same set. The GPU8 worker runs the smoke gate first; other workers wait for its verdict.
# usage: nloi_v2_worker.sh <gpu> [wait_pid ...]
set -u
GPU=$1; shift
REPO=/home/users/whddn12316/wsi_latent_0915_decode_hj
DATA=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
MODEL=/home/users/whddn12316/models/Qwen3-VL-4B-Thinking
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
K=$REPO/sender_relay_exp/runs/nloi_nav2v2
SETS=$K/sets
SM=$SETS
LOG=$K/chain_v2.log
source <(sed -n '/^run_one () {/,/^}/p' $REPO/sender_relay_exp/nloi_nav2v2_chain.sh)

busy_hj () {
  local n=0
  for p in $(pgrep -u "$(id -u)" -f "wsi_latentmas.pipeline.latent_mas"); do
    tr "\0" "\n" < /proc/$p/cmdline 2>/dev/null | grep -q "sender_relay_exp" || continue
    tr "\0" "\n" < /proc/$p/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$GPU" && n=$((n+1))
  done
  echo $n
}

for w in "$@"; do while kill -0 $w 2>/dev/null; do sleep 30; done; done
while [ "$(busy_hj)" -gt 0 ]; do sleep 60; done
echo "WORKER START gpu=$GPU pid=$$ $(date +%F_%T)" >> $LOG

if mkdir $K/claim_smoke 2>/dev/null; then
  run_one smoke gtex2.json CUDA_LAUNCH_BLOCKING=1 > /dev/null
  n=$(grep -c "^\[NLOI\] case [0-9]* obs.*nloi=v2" $K/smoke.log); tb=$(grep -ci traceback $K/smoke.log)
  sk=$(grep -c "^\[NLOI\] case .*SKIP" $K/smoke.log); g0=$(grep -c "^\[NLOI\] case [0-9]* .*geom=0" $K/smoke.log)
  tm=$(grep -c "^\[NLOI\] case [0-9]* .*text_match=0" $K/smoke.log); res=$(find $K/smoke -name result.json | wc -l)
  echo "$(date +%H:%M) smoke gpu=$GPU: marks_v2=$n skips=$sk geom0=$g0 text_match0=$tm traceback=$tb result=$res" >> $LOG
  if [ "${n:-0}" -ge 2 ] && [ "${tb:-1}" -eq 0 ] && [ "${res:-0}" -ge 2 ] && [ "${sk:-1}" -eq 0 ] && [ "${g0:-1}" -eq 0 ] && [ "${tm:-1}" -eq 0 ]; then
    touch $K/SMOKE_OK
  else
    touch $K/SMOKE_FAIL; echo "SMOKE GATE FAILED gpu=$GPU" >> $LOG; exit 1
  fi
fi
while [ ! -e $K/SMOKE_OK ] && [ ! -e $K/SMOKE_FAIL ]; do sleep 30; done
[ -e $K/SMOKE_FAIL ] && { echo "WORKER EXIT (smoke failed) gpu=$GPU" >> $LOG; exit 1; }

for ds in gtex tcga_expert_vqa tcga_slidebench panda tcga; do
  mkdir $K/claim_$ds 2>/dev/null || continue
  echo "CLAIM $ds gpu=$GPU $(date +%H:%M)" >> $LOG
  run_one nat_$ds dcs10_$ds.json
  tail -1 $K/chain_gpu$GPU.log >> $LOG
done
echo "WORKER DONE gpu=$GPU $(date +%F_%T)" >> $LOG
