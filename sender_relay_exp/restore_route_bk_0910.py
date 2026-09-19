import sys
p=sys.argv[1]; s=open(p).read()
a='''        self._aiva_visual_cols = (past_len + vis_idx).detach()

        # Level 3/4 probe: substitute a frozen donor slide's visual K/V so the'''
assert s.count(a)==1, s.count(a)
assert '_route_vis_cols = ' not in s
n='''        self._aiva_visual_cols = (past_len + vis_idx).detach()

        # Single-Pass Dual-Address Visual Routing (env VLMAS_KV_ROUTE=1):
        # remember the retained visual columns, their native 3-D MRoPE
        # coordinates and the per-observation page sizes at the Reasoner
        # boundary; the terminal Answerer resolves each page's address from
        # its own boundary query (memory/route_address.py). Base layout only.
        # (Restored 0910 22:40 -- the block had been dropped by a concurrent
        # edit; without it ROUTE_MODE=canonical logs "SKIP: no visual
        # bookkeeping" and silently runs the base layout.)
        if (os.environ.get("VLMAS_KV_ROUTE", "") == "1"
                and getattr(self, "_prune_morphology_enabled", False)):
            self._route_vis_cols = (past_len + vis_idx).detach().clone()
            self._route_vis_pos = pos[:, 0, vis_idx].detach().clone()
            self._route_pages = list(
                getattr(self, "_reassembly_meta", (None,))[0]
                or self._aavm_token_counts(inputs.get("image_grid_thw")))

        # Level 3/4 probe: substitute a frozen donor slide's visual K/V so the'''
s=s.replace(a,n); open(p,'w').write(s); print("restored",p)
