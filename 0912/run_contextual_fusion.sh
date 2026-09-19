#!/usr/bin/env bash
set -euo pipefail

mode=${1:?spatial or shuffled}
gpu=${2:?GPU index}
name=${3:?output name}
shift 3

export CONTEXTUAL_FUSION=1
export SPATIAL_GAIN=4
export FUSION_STRENGTH=0.4
bash /home/users/whddn12316/wsi_latent_0915_decode_hj/0912/run_spatial_pilot.sh \
  "$mode" "$gpu" "$name" "$@"
