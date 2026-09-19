import sys, re
q = sys.argv[1]; e = sys.argv[2]
s = open(q).read()
a = '''                    with visual_bind_context:
                        o = self.lm(
'''
n = '''                    with visual_bind_context, self._visual_cut_ctx(cur_vis):
                        o = self.lm(
'''
assert s.count(a) == 2, s.count(a)
assert "_visual_cut_ctx" not in s
s = s.replace(a, n)
a2 = '''    def _visual_bind_step(self, past_kv, cur_vis, cur_len):'''
n2 = '''    def _visual_cut_ctx(self, cur_vis):
        """Visual-contribution cut at Reasoner latent steps (env VLMAS_VCUT,
        VLMAS_VCUT_STAGE contains R; off = nullcontext)."""
        import contextlib as _cl
        from memory import visual_cut as _vc
        if not _vc.enabled("R") or not getattr(self, "_prune_morphology_enabled", False):
            return _cl.nullcontext(None)
        if not isinstance(cur_vis, torch.Tensor) or cur_vis.numel() == 0:
            return _cl.nullcontext(None)
        return _vc.install_visual_cut(self, cur_vis.to(self.device), stage="R")

''' + a2
assert s.count(a2) == 1
s = s.replace(a2, n2)
open(q, "w").write(s); print("patched", q)

s = open(e).read()
a3 = '''        with context as realloc_probe, _acc_terminal, _pfr_ctx:
            output = generate_terminal_json('''
n3 = '''        # Visual-contribution cut at the Answerer (VLMAS_VCUT=zero|mask|identity,
        # VLMAS_VCUT_STAGE contains A): hooks on every layer for the prompt
        # prefill + generation (and again for VER score-only below). Off = none.
        from memory import visual_cut as _vcut
        _vcut_ctx = contextlib.nullcontext()
        if is_terminal and _vcut.enabled("A") and getattr(self, "_rpath_visual_cols", None) is not None:
            _vcut_ctx = _vcut.install_visual_cut(
                self._backbone, self._rpath_visual_cols.to(self._backbone.device), stage="A")
        with context as realloc_probe, _acc_terminal, _pfr_ctx, _vcut_ctx:
            output = generate_terminal_json('''
assert s.count(a3) == 1 and "_vcut_ctx" not in s
s = s.replace(a3, n3)
i = s.index("            output = _ver.visual_evidence_ratio(")
j = s.index('normal_output=output, case_index=getattr(self, "_rpath_case_index", -1))', i)
j = s.index("\n", j) + 1
block = s[i:j]
ind = "\n".join(("    " + ln if ln.strip() else ln) for ln in block.split("\n"))
pre = '''            _vcut_ver = contextlib.nullcontext()
            if _vcut.enabled("A") and getattr(self, "_rpath_visual_cols", None) is not None:
                _vcut_ver = _vcut.install_visual_cut(
                    self._backbone, self._rpath_visual_cols.to(self._backbone.device), stage="A")
            with _vcut_ver:
'''
s = s[:i] + pre + ind + s[j:]
open(e, "w").write(s); print("patched", e)
