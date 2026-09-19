#!/bin/bash
# Sequential GPU7 queue: gpfx -> lens -> ALSI rerun (relative-eps fix).
set -u
GPU=${1:-7}
T=/home/users/whddn12316/wsi_latent_0915_decode_hj
S=$T/sender_relay_exp
LOG=$S/runs/queue3.log
echo "QUEUE3 START $(date +%F_%T) gpu=$GPU pid=$$" >> $LOG
$S/gpfx_chain.sh - $GPU
echo "QUEUE3 gpfx done $(date +%F_%T)" >> $LOG
$S/lens_chain.sh - $GPU
echo "QUEUE3 lens done $(date +%F_%T)" >> $LOG
mv $S/runs/alsi $S/runs/alsi_eps1 2>/dev/null
mkdir -p $S/runs/alsi
$S/alsi_chain.sh - $GPU
echo "QUEUE3 alsi(v2) done $(date +%F_%T)" >> $LOG
