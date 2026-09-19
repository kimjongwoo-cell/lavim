#!/bin/bash
# Single sequential driver for GPU7 (no PID cross-chaining — that is what let several of
# my runs land on the same GPU at once). Each step blocks until it finishes.
#   1. gpfx        how much of the answer margin g is the JSON prefill worth
#   2. lens        where does the answer lock in, layer by layer (off-by-one fixed)
#   3. qpris fix   the three TCGA arms the crashed Q-PRIS chain never ran
#   4. alsi        Answerer latent source interference, native + canonical
# usage: run_queue_0913.sh <gpu>
set -u
GPU=${1:-7}
T=/home/users/whddn12316/wsi_latent_0915_decode_hj
S=$T/sender_relay_exp
LOG=$S/runs/queue_0913.log

echo "QUEUE START $(date +%F_%T) gpu=$GPU pid=$$" >> $LOG
$S/gpfx_chain.sh - $GPU
echo "QUEUE step1 gpfx done $(date +%F_%T)" >> $LOG
$S/lens_chain.sh - $GPU
echo "QUEUE step2 lens done $(date +%F_%T)" >> $LOG
$S/qpris_tcga_fix.sh - $GPU
echo "QUEUE step3 qpris-tcga done $(date +%F_%T)" >> $LOG
$S/alsi_chain.sh - $GPU
echo "QUEUE step4 alsi done $(date +%F_%T)" >> $LOG
echo "QUEUE ALL DONE $(date +%F_%T)" >> $LOG
