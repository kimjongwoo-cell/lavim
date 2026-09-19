import sys
p = sys.argv[1]; s = open(p).read()
# (1) absolute budget override, only when env set
a1 = '''        ratio = float(os.environ.get("VLMAS_WSI_CONSOL_KEEP", "0.25"))
        budget = max(1, _math.ceil(nV * ratio))
'''
n1 = a1 + '''        if os.environ.get("VLMAS_WSI_CONSOL_BUDGET", "").strip():
            # WSI-aware spec (0910 v2): case-level absolute budget B (e.g. 64/96/128)
            budget = max(1, int(os.environ["VLMAS_WSI_CONSOL_BUDGET"]))
            ratio = round(budget / max(nV, 1), 4)
'''
assert s.count(a1) == 1 and "VLMAS_WSI_CONSOL_BUDGET" not in s
s = s.replace(a1, n1)
# (2) mode branch
a2 = '''        elif mode in ("hier", "hier_shuf"):'''
n2 = '''        elif mode in ("wsi", "wsi_flat", "wsi_shuf"):
            # WSI-Aware Visual KV Compression (0910 spec v2): seeds = center
            # token per observation U bridge token (parent token nearest the
            # child's footprint center) per real zoom edge; remaining slots by
            # greedy covering-radius (largest R_p observation, farthest token).
            # Requires B >= B0 (seed set never cut). wsi_flat = no bridge
            # constraint; wsi_shuf = deranged parents (relative pos transplanted).
            from memory.stratified import wsi_aware_keep_mask
            keep, wsi_info = wsi_aware_keep_mask(
                token_counts, grid_hws, boxes,
                tuple(getattr(self, "_prune_image_parent_indices", ()) or ()),
                budget,
                constraint={"wsi": "hier", "wsi_flat": "flat",
                            "wsi_shuf": "shuffled"}[mode],
                return_info=True)
            keep = keep.to(self.device)
            print(
                f"[WSIConsol] wsi constraint={wsi_info['constraint']} "
                f"B={budget} B0={wsi_info['B0']} edges={wsi_info['edges']} "
                f"infeasible={wsi_info['infeasible']} "
                f"parents={wsi_info['parents']} per_crop={wsi_info['per_crop']}",
                flush=True)
'''
assert s.count(a2) == 1 and 'mode in ("wsi"' not in s
s = s.replace(a2, n2 + a2)
open(p, "w").write(s); print("patched", p)
