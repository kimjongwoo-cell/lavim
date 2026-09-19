#!/bin/bash
# Sequential: gpfx (what fraction of the answer margin is the JSON prefill) then lens
# (which layer locks the answer). Both wired and unit-tested, neither ever completed.
set -u
GPU=${1:-7}
T=/home/users/whddn12316/wsi_latent_0915_decode_hj
S=$T/sender_relay_exp
LOG=$S/runs/queue2.log
echo "QUEUE2 START $(date +%F_%T) gpu=$GPU pid=$$" >> $LOG
$S/gpfx_chain.sh - $GPU
echo "QUEUE2 gpfx done $(date +%F_%T)" >> $LOG
$S/lens_chain.sh - $GPU
echo "QUEUE2 lens done $(date +%F_%T)" >> $LOG
