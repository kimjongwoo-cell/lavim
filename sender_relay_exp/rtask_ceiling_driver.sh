#!/bin/bash
# 상한 측정 드라이버: (1) 앞선 큐(MC-LADR 5셋 → 재분배 GTEx)가 끝나길 기다린 뒤
# (2) 디코딩 수정 기준선을 seed 1..N으로 샘플링해 GTEx 190문항을 반복 실행한다.
# 채점은 tmp_hj/score_ceiling.py: greedy 대비 'seed 중 한 번이라도 맞힌' 비율 = 파이프라인 상한.
set -u
R=/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp
SEEDS=${SEEDS:-'1 2 3'}
WAITDIR=${WAITDIR:-runs/rtask_realloc_gtex/gtex}; WAITN=${WAITN:-190}
L=$R/runs/rtask_ceiling/driver.log
mkdir -p $R/runs/rtask_ceiling
log(){ echo "$(date '+%m-%d %H:%M:%S') $*" >> $L; }
log "WAIT $WAITDIR $WAITN"
while [ "$(find $R/$WAITDIR -name result.json 2>/dev/null | wc -l)" -lt $WAITN ]; do sleep 60; done
for s in $SEEDS; do
  log "START seed $s"
  for i in 0 1 2; do (setsid nohup bash $R/rtask_ceiling.sh $s $i 3 '6 7 8' gtex:190 >/dev/null 2>&1 </dev/null &); done
  while [ "$(find $R/runs/rtask_ceiling/seed$s/gtex -name result.json 2>/dev/null | wc -l)" -lt 190 ]; do sleep 60; done
  log "DONE seed $s ($(find $R/runs/rtask_ceiling/seed$s/gtex -name result.json | wc -l)/190)"
done
log "ALL SEEDS DONE"
