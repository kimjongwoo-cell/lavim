#!/bin/bash
# GPU chain: canonical model-level diagnosis, then the canonical-only control run on the same GPU/shard.
# usage: canon_diag_then_only.sh <gpu> <shard> <smoke|wait>
D=/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp
bash $D/canon_diag_nav3.sh "$1" "$2" "$3"
bash $D/canon_only_nav3.sh "$1" "$2" "$3"
