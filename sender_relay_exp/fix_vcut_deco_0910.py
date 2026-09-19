import sys
p=sys.argv[1]; s=open(p).read()
bad='''    @contextlib.contextmanager
    def _visual_cut_ctx(self, cur_vis):'''
assert s.count(bad)==1, s.count(bad)
s=s.replace(bad,'''    def _visual_cut_ctx(self, cur_vis):''')
a='''        return _vc.install_visual_cut(self, cur_vis.to(self.device), stage="R")

    def _visual_bind_step(self, past_kv, cur_vis, cur_len):'''
assert s.count(a)==1
s=s.replace(a,'''        return _vc.install_visual_cut(self, cur_vis.to(self.device), stage="R")

    @contextlib.contextmanager
    def _visual_bind_step(self, past_kv, cur_vis, cur_len):''')
open(p,"w").write(s); print("fixed",p)
