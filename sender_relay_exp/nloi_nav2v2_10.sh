#!/bin/bash
# User request 2026-09-13 20:2x: 10 cases per dataset (dcs20 indices 0-9), 50 total.
# 1) let the running chain finish GTEx case 9 (10 result.json), then stop that chain (PID-checked)
# 2) run EVQA / SB / PANDA / TCGA on the first 10 cases, same nav2v2 flags as nloi_nav2v2_chain.sh
# usage: nloi_nav2v2_10.sh <old_chain_pid> <gpu>
set -u
OLD=$1; GPU=$2
REPO=/home/users/whddn12316/wsi_latent_0915_decode_hj
K=$REPO/sender_relay_exp/runs/nloi_nav2v2
SETS=$K/sets
mkdir -p $SETS
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
DATA=/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda
MODEL=/home/users/whddn12316/models/Qwen3-VL-4B-Thinking
for ds in gtex tcga_expert_vqa tcga_slidebench panda tcga; do
  $PY -c "import json; d=json.load(open('$REPO/sender_relay_exp/smoke/dcs20_$ds.json')); json.dump(d[:10], open('$SETS/dcs10_$ds.json','w'))"
done
echo "SUBSET10 $(date +%F_%T) sets=$(ls $SETS | tr '\n' ' ')" >> $K/chain_gpu$GPU.log

# 1) wait for GTEx case 9 to finish in the running chain
while [ "$(find $K/nat_gtex -name result.json 2>/dev/null | wc -l)" -lt 10 ]; do
  kill -0 $OLD 2>/dev/null || break
  sleep 20
done
if kill -0 $OLD 2>/dev/null && tr "\0" " " < /proc/$OLD/cmdline | grep -q "nloi_nav2v2_chain.sh"; then
  kids=$(pgrep -P $OLD); grand=""; for c in $kids; do grand="$grand $(pgrep -P $c)"; done
  kill $OLD; sleep 1; kill $kids $grand 2>/dev/null; sleep 5
  echo "STOPPED old chain $OLD (+$kids $grand) $(date +%F_%T) gtex_results=$(find $K/nat_gtex -name result.json | wc -l) gtex_rows=$(wc -l < $K/nloi_nat_gtex.jsonl)" >> $K/chain_gpu$GPU.log
fi

# 2) remaining four sets, 10 cases each
source <(sed -n '/^run_one () {/,/^}/p' $REPO/sender_relay_exp/nloi_nav2v2_chain.sh)
SM=$SETS
for ds in tcga_expert_vqa tcga_slidebench panda tcga; do run_one nat_$ds dcs10_$ds.json; done
echo "ALL DONE (10/set) $(date +%F_%T) gpu=$GPU" >> $K/chain_gpu$GPU.log
