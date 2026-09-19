#!/bin/bash
# GPU 6/7/8 중 하나가 실제로 비면(<2GB) sb_sdpa_base.sh 를 그 카드에 한 번만 띄운다.
# 남의 런 위에 얹지 않는다. 결과는 runs/sb_sdpa_base/chain.log.
set -u
REPO=/home/users/whddn12316/wsi_latent_0915_decode_hj
LOG=$REPO/sender_relay_exp/runs/sb_sdpa_base/chain.log
mkdir -p $(dirname $LOG)
echo "$(date '+%m-%d %H:%M') WAIT start (GPU 6/7/8 중 빈 카드 대기)" >> $LOG
while true; do
  for g in 6 7 8; do
    u=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' -v g=$g '$1==g{print $2}')
    if [ -n "$u" ] && [ "$u" -lt 2000 ]; then
      sleep 60   # 다른 세션이 연달아 잡는지 한 번 더 확인
      u2=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' -v g=$g '$1==g{print $2}')
      if [ "$u2" -lt 2000 ]; then
        echo "$(date '+%m-%d %H:%M') CLAIM gpu$g" >> $LOG
        exec bash $REPO/sender_relay_exp/sb_sdpa_base.sh $g
      fi
    fi
  done
  sleep 120
done
