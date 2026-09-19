#!/usr/bin/env bash
set -euo pipefail

mode=${1:?spatial or shuffled}
gpu=${2:?GPU index}
name=${3:?output name}
shift 3

export RELATION_LATENT=1
export RELATION_CONTRAST=1
export RELATION_ALPHA=0.1
export CONTRAST_STRENGTH=0.1
export SPATIAL_GAIN=4
bash /home/users/whddn12316/wsi_latent_0915_decode_hj/0912/run_spatial_pilot.sh \
  "$mode" "$gpu" "$name" "$@"
