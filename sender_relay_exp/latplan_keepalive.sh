#!/bin/bash
# Keep the latplan arm (ExpertVQA 128) running: relaunch the 3 shards whenever none is alive, until 128 unique results.
REPO=/home/users/whddn12316/wsi_latent_0915_decode_hj
L=$REPO/sender_relay_exp/rtask_nav4_nonav2_arm.sh
K=$REPO/sender_relay_exp/runs/nav4/latplan
CLAIM=$K/../claims_latplan
PY=/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python
LOG=$K/keepalive.log
N=128
cd $REPO
echo "$(date '+%m-%d %H:%M:%S') KEEPALIVE START pid=$$" >> $LOG
while true; do
  done_n=$(find $K/tcga_expert_vqa -name result.json 2>/dev/null -exec $PY -c 'import json,sys; [print(json.load(open(f))["dataset_index"]) for f in sys.argv[1:]]' {} + 2>/dev/null | sort -u | wc -l)
  if [ "$done_n" -ge "$N" ]; then echo "$(date '+%m-%d %H:%M:%S') COMPLETE $done_n/$N" >> $LOG; touch $K/LATPLAN_EVQA_COMPLETE; break; fi
  alive=$(ps -eo args | grep -c "[r]task_nav4_nonav2_arm.sh latplan [012] ")
  if [ "$alive" -eq 0 ]; then
    rm -rf $CLAIM; mkdir -p $CLAIM
    for s in 0 1 2; do
      (setsid nohup env MAXUSED=20000 CLAIMDIR=$CLAIM bash $L latplan $s 3 "6 7 8" tcga_expert_vqa:$N > /dev/null 2>&1 < /dev/null &)
    done
    echo "$(date '+%m-%d %H:%M:%S') RELAUNCH shards 0-2 (done $done_n/$N)" >> $LOG
  fi
  sleep 120
done
