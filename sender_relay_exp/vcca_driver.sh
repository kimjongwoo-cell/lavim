#!/bin/bash
# Wait for the budget-sweep chains to finish, run the mandatory 2-case VCCA smoke,
# and only launch the 5x20 run if the smoke passes the 3-point gate
# (log marker / result.json / no traceback).
set -u
TREE=/home/users/whddn12316/wsi_latent_0915_decode_hj
R=$TREE/sender_relay_exp
K=$R/runs/vcca

echo "== waiting for the budget sweep to finish =="
for i in $(seq 1 180); do
  n=$(pgrep -f 'acc_chain.sh' | wc -l)
  echo "$(date +%T) acc_chain procs=$n"
  [ "$n" -eq 0 ] && break
  sleep 60
done
sleep 30

echo
echo "== VCCA 2-case smoke on gpu6 =="
cd "$R" && bash vcca_chain.sh 6 smoke:gtex2.json
rows=$(wc -l < "$K/vcca_smoke.jsonl" 2>/dev/null || echo 0)
mark=$(grep -c '\[VCCA\] case' "$K/smoke.log" 2>/dev/null || echo 0)
skip=$(grep -c '\[VCCA\].*SKIP' "$K/smoke.log" 2>/dev/null || echo 0)
tb=$(grep -c Traceback "$K/smoke.log" 2>/dev/null || echo 0)
res=$(find "$K/smoke" -name result.json 2>/dev/null | wc -l)
echo "smoke gate: rows=$rows markers=$mark skips=$skip traceback=$tb results=$res"
grep '\[VCCA\]' "$K/smoke.log" 2>/dev/null | head -5
if [ "$tb" -ne 0 ]; then echo "GATE FAIL: traceback present — not launching"; tail -30 "$K/smoke.log"; exit 1; fi
if [ "$res" -ne 2 ]; then echo "GATE FAIL: results=$res != 2 — not launching"; tail -30 "$K/smoke.log"; exit 1; fi
if [ "$mark" -lt 2 ]; then echo "GATE FAIL: only $mark VCCA markers — not launching"; tail -30 "$K/smoke.log"; exit 1; fi
if [ "$rows" -lt 1 ]; then echo "GATE FAIL: no JSONL row — not launching"; tail -30 "$K/smoke.log"; exit 1; fi
echo "GATE PASS"

echo
echo "== launching 5x20 on gpu 6/7/8 =="
cd "$K" || exit 1
setsid nohup bash "$R/vcca_chain.sh" 6 gtex:dcs20_gtex.json panda:dcs20_panda.json \
  > "$K/chain6.out" 2>&1 < /dev/null &
sleep 5
setsid nohup bash "$R/vcca_chain.sh" 7 tcga_expert_vqa:dcs20_tcga_expert_vqa.json tcga:dcs20_tcga.json \
  > "$K/chain7.out" 2>&1 < /dev/null &
sleep 5
setsid nohup bash "$R/vcca_chain.sh" 8 tcga_slidebench:dcs20_tcga_slidebench.json \
  > "$K/chain8.out" 2>&1 < /dev/null &
sleep 10
echo "launched at $(date +%T)"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed -n '7,9p'

echo
echo "== waiting for the 5 datasets =="
for i in $(seq 1 300); do
  t=0
  for d in gtex tcga_expert_vqa tcga_slidebench tcga panda; do
    n=$(wc -l < "$K/vcca_$d.jsonl" 2>/dev/null || echo 0); t=$((t+n))
  done
  echo "$(date +%T) vcca rows=$t/100"
  live=$(pgrep -f 'vcca_chain.sh' | wc -l)
  [ "$live" -eq 0 ] && { echo "chains finished"; break; }
  sleep 60
done

echo
echo "== chain log =="
cat "$K"/chain_gpu*.log
