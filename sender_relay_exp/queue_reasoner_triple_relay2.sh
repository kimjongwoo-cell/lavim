#!/usr/bin/env bash
set -euo pipefail

repo=/home/users/whddn12316/wsi_latent_0915_decode_hj
for session in reasoner-relay2-gpu0 reasoner-relay2-gpu1 reasoner-relay2-gpu2 reasoner-relay2-gpu3 reasoner-relay2-gpu6 reasoner-relay2-gpu8; do
  while tmux has-session -t "$session" 2>/dev/null; do
    sleep 30
  done
done

for shard_gpu in "0 0" "1 1" "2 2" "3 3" "4 6" "5 8"; do
  read -r shard gpu <<< "$shard_gpu"
  tmux new-session -d -s "reasoner-triple-gpu${gpu}" \
    "bash $repo/sender_relay_exp/run_reasoner_pruning_b_triple_relay2.sh $shard $gpu"
done
