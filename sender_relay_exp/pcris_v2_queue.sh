#!/bin/bash
# PCRIS v2 (= null-calibrated residual, MODE=qpris with VLMAS_QPRIS_Q=none) on the FULL
# sets, same flags as the Q-PRIS full run. Output lands as runs/qpris_full/qn_<ds>.
#
# GPU 6/7/8 are shared with other sessions right now, and the 04:41 OOM came from sharing
# a card. So this claims a GPU only when it is actually idle (<2GB, no compute apps) and
# runs one job on it at a time. It never touches anyone else's process.
set -u
TREE=/home/users/whddn12316/wsi_latent_0915_decode_hj
CHAIN=$TREE/sender_relay_exp/qpris_full_chain.sh
K=$TREE/sender_relay_exp/runs/qpris_full
Q=$K/queue_pcris_v2.log
mkdir -p $K
JOBS="tcga gtex tcga_slidebench panda tcga_expert_vqa"

gpu_used () {   # $1 = gpu index -> MiB in use
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -v g="$1" -F', *' '$1==g {print $2}'
}
mine_on () {    # $1 = gpu index -> 1 if this queue already put a job there
  [ -f "$K/.claim_$1" ] && kill -0 "$(cat $K/.claim_$1)" 2>/dev/null
}

echo "QUEUE START $(date +%F_%T) jobs=$JOBS" >> $Q
for ds in $JOBS; do
  gpu=""
  while [ -z "$gpu" ]; do
    for g in 6 7 8; do
      mine_on $g && continue
      u=$(gpu_used $g)
      if [ -n "$u" ] && [ "$u" -lt 2000 ]; then gpu=$g; break; fi
    done
    [ -z "$gpu" ] && sleep 120
  done
  echo "$(date +%H:%M) claim gpu$gpu for $ds" >> $Q
  setsid nohup bash "$CHAIN" - "$gpu" "$ds:none" > "$K/launch_pcrisv2_gpu$gpu.out" 2>&1 < /dev/null &
  echo $! > "$K/.claim_$gpu"
  sleep 90     # 모델 로딩으로 메모리가 잡힐 때까지 대기 후 다음 GPU 탐색
done
echo "QUEUE DISPATCHED $(date +%F_%T)" >> $Q
