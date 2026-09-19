#!/bin/bash
# Join GPU7 only after it has had NO hj-tree (sender_relay_exp) run for 5 consecutive minutes,
# so a follow-up job another session queues right after its chain is not raced (21:40 collision).
set -u
GPU=${1:-7}
REPO=/home/users/whddn12316/wsi_latent_0915_decode_hj
LOG=$REPO/sender_relay_exp/runs/nloi_nav2v2/chain_v2.log
busy () {
  local n=0
  for p in $(pgrep -u "$(id -u)" -f "wsi_latentmas.pipeline.latent_mas"); do
    tr "\0" "\n" < /proc/$p/cmdline 2>/dev/null | grep -q "sender_relay_exp" || continue
    tr "\0" "\n" < /proc/$p/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$GPU" && n=$((n+1))
  done
  echo $n
}
echo "WAIT7 start $(date +%F_%T) pid=$$" >> $LOG
streak=0
while [ $streak -lt 5 ]; do
  if [ "$(busy)" -eq 0 ]; then streak=$((streak+1)); else streak=0; fi
  sleep 60
done
# all remaining sets already claimed -> nothing to do
left=0; for ds in gtex tcga_expert_vqa tcga_slidebench panda tcga; do [ -d $REPO/sender_relay_exp/runs/nloi_nav2v2/claim_$ds ] || left=$((left+1)); done
[ $left -eq 0 ] && { echo "WAIT7 exit: nothing left to claim $(date +%H:%M)" >> $LOG; exit 0; }
echo "WAIT7 GPU$GPU free 5 min -> worker $(date +%F_%T)" >> $LOG
exec bash $REPO/sender_relay_exp/nloi_v2_worker.sh $GPU
