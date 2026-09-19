import sys
q, e = sys.argv[1], sys.argv[2]
s = open(q).read()
a = "            last_hidden = o.hidden_states[-1][:, -1:, :]\n"
assert s.count(a) >= 1 and "_retr_hidden" not in s
n = a + '''            if os.environ.get("VLMAS_RETR", ""):
                # retrieval kill test: per-layer residual input at the latest
                # latent step (overwritten each step -> the last step remains)
                self._retr_hidden = [h[0, -1].detach().float().clone() for h in o.hidden_states[:-1]]
'''
s = s.replace(a, n); open(q, "w").write(s); print("patched", q, s.count("_retr_hidden"))
s = open(e).read()
a2 = '''        _vcut_ctx = contextlib.nullcontext()
        if is_terminal and _vcut.enabled("A") and getattr(self, "_rpath_visual_cols", None) is not None:
            _vcut_ctx = _vcut.install_visual_cut(
                self._backbone, self._rpath_visual_cols.to(self._backbone.device), stage="A")
'''
n2 = '''        _vcut_ctx = contextlib.nullcontext()
        # Retrieval kill test (VLMAS_RETR=context|token|random): keep one
        # reasoning-selected cross-scale group of visual columns, mask the rest
        # (mask mode of the visual cut, Answerer stage; VER scored the same way).
        self._vcut_a_cols = getattr(self, "_rpath_visual_cols", None)
        if is_terminal and self._vcut_a_cols is not None and os.environ.get("VLMAS_RETR", "").strip():
            from memory import retrieval_group as _rg
            _bb = self._backbone
            _cols_all = self._vcut_a_cols.to(_bb.device)
            _rc = getattr(_bb, "_route_vis_cols", None)
            _pos = getattr(_bb, "_route_vis_pos", None)
            _pages = getattr(_bb, "_route_pages", None)
            if _rc is not None and int(_rc.numel()) == int(_cols_all.numel()) and _pos is not None and _pages:
                _keep_local, _ = _rg.select(
                    _bb, cache, _rc.to(_bb.device), _pos.to(_bb.device), _pages,
                    tuple(getattr(_bb, "_prune_image_parent_indices", ()) or ()),
                    int(getattr(self, "_rpath_case_index", -1)))
                _mask = torch.ones(int(_cols_all.numel()), dtype=torch.bool, device=_bb.device)
                _mask[_keep_local.to(_bb.device)] = False
                self._vcut_a_cols = _cols_all[_mask].cpu()
            else:
                print("[Retr] SKIP: visual bookkeeping missing/mismatch", flush=True)
        if is_terminal and _vcut.enabled("A") and self._vcut_a_cols is not None:
            _vcut_ctx = _vcut.install_visual_cut(
                self._backbone, self._vcut_a_cols.to(self._backbone.device), stage="A")
'''
assert s.count(a2) == 1
s = s.replace(a2, n2)
a3 = '''            if _vcut.enabled("A") and getattr(self, "_rpath_visual_cols", None) is not None:
                _vcut_ver = _vcut.install_visual_cut(
                    self._backbone, self._rpath_visual_cols.to(self._backbone.device), stage="A")
'''
n3 = '''            if _vcut.enabled("A") and getattr(self, "_vcut_a_cols", None) is not None:
                _vcut_ver = _vcut.install_visual_cut(
                    self._backbone, self._vcut_a_cols.to(self._backbone.device), stage="A")
'''
assert s.count(a3) == 1
s = s.replace(a3, n3); open(e, "w").write(s); print("patched", e)
