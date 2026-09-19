import sys
p = sys.argv[1]; s = open(p).read()
a = '''        elif mode in ("wsi", "wsi_flat", "wsi_shuf"):'''
new = '''        elif mode in ("scale", "scale_shuf"):
            # Scale-Structured Visual KV Compression (0910 spec v3): B_C =
            # floor(alpha*B) coarse (bridge tokens + childless one-centers,
            # then GLOBAL farthest-point coverage in the common WSI frame) and
            # B_F = B - B_C fine (b_c >= 1 ∝ N_c; one-center + FPS per child).
            # scale_shuf = deranged parents (relative center transplanted).
            from memory.stratified import scale_structured_keep_mask
            keep, sc_info = scale_structured_keep_mask(
                token_counts, grid_hws, boxes,
                tuple(getattr(self, "_prune_image_parent_indices", ()) or ()),
                magnifications, budget,
                alpha=float(os.environ.get("VLMAS_WSI_CONSOL_ALPHA", "0.5")),
                shuffle_parents=(mode == "scale_shuf"), return_info=True)
            keep = keep.to(self.device)
            print(
                f"[WSIConsol] scale B={budget} B_C={sc_info['B_C']} "
                f"B_F={sc_info['B_F']} coarse={sc_info['coarse']} "
                f"fine={sc_info['fine']} bridges={sc_info['bridges']} "
                f"mandatory_C={sc_info['mandatory_C']} "
                f"infeasible={sc_info['infeasible']} "
                f"wsi_coords={sc_info['wsi_coords']} parents={sc_info['parents']} "
                f"per_crop={sc_info['per_crop']}", flush=True)
'''
assert s.count(a) == 1 and 'mode in ("scale"' not in s
s = s.replace(a, new + a); open(p, "w").write(s); print("patched", p)
