#!/usr/bin/env bash
set -euo pipefail

mode=${1:?spatial or shuffled}
gpu=${2:?GPU index}
alpha=${3:-0.1}
name=${4:-relation_${mode}_alpha${alpha}_gpu${gpu}}
shift 2
if [ "$#" -gt 0 ]; then shift; fi
if [ "$#" -gt 0 ]; then shift; fi

export RELATION_LATENT=1
export RELATION_ALPHA="$alpha"
export SPATIAL_GAIN=4
bash /home/users/whddn12316/wsi_latent_0915_decode_hj/0912/run_spatial_pilot.sh \
  "$mode" "$gpu" "$name" "$@"
