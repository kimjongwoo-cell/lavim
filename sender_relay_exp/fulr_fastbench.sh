#!/bin/bash
# FULR reference hook vs VLMAS_RNLCR_FAST=1 on the same cases, alternating on one GPU slot (claims serialize the steps).
# cases: GTEx idx 0-1 (smoke), ExpertVQA idx 0-7, TCGA idx 0-7. nav4 base env from rtask_nav4_fulr_arm.sh.
REPO=/home/users/whddn12316/wsi_latent_0915_decode_hj
L=$REPO/sender_relay_exp/rtask_nav4_fulr_arm.sh
B=$REPO/sender_relay_exp/runs/fulr_fastbench
mkdir -p $B
export CLAIMDIR=$B/claims MAXUSED=20000
cd $REPO
echo "$(date '+%m-%d %H:%M:%S') CHAIN START" >> $B/chain.log
VLMAS_RNLCR_FAST=1 OUTROOT=$B/fast bash $L fulr smoke 1 "6 7 8"
OUTROOT=$B/ref bash $L fulr smoke 1 "6 7 8"
for spec in tcga_expert_vqa:8 tcga:8; do
  VLMAS_RNLCR_FAST=1 OUTROOT=$B/fast bash $L fulr 0 1 "6 7 8" $spec
  OUTROOT=$B/ref bash $L fulr 0 1 "6 7 8" $spec
done
echo "$(date '+%m-%d %H:%M:%S') CHAIN DONE" >> $B/chain.log
touch $B/CHAIN_DONE
