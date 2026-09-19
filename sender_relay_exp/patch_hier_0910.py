import sys
p = sys.argv[1]
s = open(p).read()
anchor = '''        elif mode == "strat":
            # C1 = Observation-Stratified compression (method draft 0907):'''
new = '''        elif mode in ("hier", "hier_shuf"):
            # Hierarchy-Constrained Spatial Pruning (0910 spec): mandatory
            # {>=m tokens per crop  U  parent token pi(c) per real 5x->20x
            # edge}, remainder by the strat rule (b_p ∝ |V_p|, equal-area
            # thinning). No scores. hier_shuf = Shuffled-Hierarchy control
            # (children re-attached to a different anchor; crops, tokens and
            # budget unchanged).
            from memory.stratified import hierarchy_keep_mask
            keep, hier_info = hierarchy_keep_mask(
                token_counts, grid_hws, boxes,
                tuple(getattr(self, "_prune_image_parent_indices", ()) or ()),
                budget,
                min_per_crop=int(os.environ.get("VLMAS_WSI_CONSOL_HIER_MIN", "1")),
                shuffle_parents=(mode == "hier_shuf"), return_info=True)
            keep = keep.to(self.device)
            print(
                f"[WSIConsol] hier mandatory={hier_info['mandatory']} "
                f"edges={hier_info['edges']} parents={hier_info['parents']} "
                f"per_crop={hier_info['per_crop']}", flush=True)
'''
assert s.count(anchor) == 1, s.count(anchor)
assert 'mode in ("hier", "hier_shuf")' not in s
s = s.replace(anchor, new + anchor)
open(p, "w").write(s)
print("patched", p)
