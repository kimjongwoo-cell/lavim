#!/bin/bash
# Keep the latplan arm (SlideBench 197) running: relaunch the 3 shards whenever none is alive, until 197 unique results.
REPO=/home/users/whddn12316/wsi_latent_0915_decode_hj
L=$REPO/sender_relay_exp/rtask_nav4_nonav2_arm.sh
K=$REPO/sender_relay_exp/runs/nav4/latplan
CLAIM=$REPO/sender_relay_exp/runs/nav4/claims_latplan_sb
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
LOG=$K/keepalive_sb.log
DS=tcga_slidebench
N=197
cd $REPO
echo "$(date '+%m-%d %H:%M:%S') KEEPALIVE_SB START pid=$$" >> $LOG
while true; do
  done_n=$($PY -c "
import glob, json
print(len({json.load(open(f))['dataset_index'] for f in glob.glob('$K/$DS/**/result.json', recursive=True)}))" 2>/dev/null || echo 0)
  if [ "${done_n:-0}" -ge "$N" ]; then echo "$(date '+%m-%d %H:%M:%S') COMPLETE $done_n/$N" >> $LOG; touch $K/LATPLAN_SB_COMPLETE; break; fi
  alive=$(ps -eo args | grep -c "[r]task_nav4_nonav2_arm.sh latplan [012] .*$DS")
  if [ "$alive" -eq 0 ]; then
    rm -rf $CLAIM; mkdir -p $CLAIM
    for s in 0 1 2; do
      (setsid nohup env MAXUSED=20000 CLAIMDIR=$CLAIM bash $L latplan $s 3 "6 7 8" $DS:$N > /dev/null 2>&1 < /dev/null &)
    done
    echo "$(date '+%m-%d %H:%M:%S') RELAUNCH shards 0-2 (done ${done_n:-0}/$N)" >> $LOG
  fi
  sleep 120
done
