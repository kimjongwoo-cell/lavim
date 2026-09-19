"""Wire EOVC (C2 #14) into the Pruning-B early-observation selection. Off = byte-identical.

Replaces ONLY the keep-mask decision inside _hierarchy_pruned_vision_features: instead of
hierarchy_prefill_keep_mask over `novelty + 0.25*texture`, the mask comes from
memory.eovc.select on the same early-observation group states, with the same global budget.
The observation blocks, the merge-group granularity and everything downstream are untouched.
"""
from pathlib import Path

P = Path("/home/users/whddn12316/wsi_latent_0915_decode_hj/backbone/qwen3vl.py")
s = P.read_text()
bak = P.with_suffix(P.suffix + ".bak_0917_eovc")
if not bak.exists():
    bak.write_text(s)

old = """        keep_groups = hierarchy_prefill_keep_mask(
            scores,
            token_counts=token_counts,
            parent_indices=parent_indices,
            keep_ratio=adaptive_keep_ratio,
            min_tokens_per_image=min_tokens_per_image,
            significance_sigma=significance_sigma,
        )"""
new = """        # C2 #14 EOVC (memory/eovc.py, env VLMAS_EOVC=1): the same early-observation group
        # states, but the keep mask minimises the support variation left unexplained by the
        # retained span instead of ranking novelty + 0.25*texture under a per-image quota.
        # Off = unchanged.
        from memory import eovc as _eovc
        if _eovc.enabled():
            _eovc_budget = int(round(float(adaptive_keep_ratio) * int(pooled.shape[0])))
            _eovc_info = _eovc.select(grouped, token_counts, _eovc_budget)
            keep_groups = _eovc_info["keep"].to(device=pooled.device)
            print(f"[EOVC] {_eovc.summary(_eovc_info)} keep_ratio={float(adaptive_keep_ratio):.3f}",
                  flush=True)
        else:
            keep_groups = hierarchy_prefill_keep_mask(
                scores,
                token_counts=token_counts,
                parent_indices=parent_indices,
                keep_ratio=adaptive_keep_ratio,
                min_tokens_per_image=min_tokens_per_image,
                significance_sigma=significance_sigma,
            )"""
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new, 1)
P.write_text(s)

import hashlib
print("patched", P)
print("md5", hashlib.md5(P.read_bytes()).hexdigest()[:8])
