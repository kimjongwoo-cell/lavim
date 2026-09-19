"""0916 OCLD (C2 12.4) wiring for the decode copy tree engine (md5 5636f516). Backup .bak_0916_ocld. Off = unchanged."""
import hashlib, shutil, sys
p = sys.argv[1] if len(sys.argv) > 1 else "vision_text_mas/latent_qwen_engine.py"
src = open(p).read()
md5 = hashlib.md5(src.encode()).hexdigest()[:8]
assert md5 == "5636f516", f"engine md5 {md5}"
old = '''        """Append a role and retain its configured latent-KV handoff state."""
'''
new = old + '''        # C2 12.4 OCLD (memory/ocld.py, VLMAS_OCLD=1|native): the Reasoner over G retained physical
        # observations runs M = G + 1 latent steps (G support-isolated local reads + 1 integration;
        # native = same M standard steps). Off / other stage = unchanged latent_steps.
        from memory import ocld as _ocld
        if _ocld.mode() and stage == "reasoner" and images:
            _ocld_native_steps = latent_steps
            latent_steps = _ocld.steps_for(len(images))
            print(f"[OCLD] mode={_ocld.mode()} reasoner latent_steps {_ocld_native_steps} -> {latent_steps} "
                  f"(G={len(images)})", flush=True)
'''
assert src.count(old) == 1; src = src.replace(old, new)
old = '''        _srvw_r = _srvw.reasoner_probe(self._backbone, m=latent_steps, stage=stage)
'''
new = old + '''        _ocld_r = _ocld.reasoner_ocld(
            self._backbone, m=latent_steps, n_obs=len(images), stage=stage,
            seed=42 + int(getattr(self, "_rpath_case_index", 0) or 0))
'''
assert src.count(old) == 1; src = src.replace(old, new)
old = "            with _lpvh_r, _srvw_r:\n"
assert src.count(old) == 2; src = src.replace(old, "            with _lpvh_r, _srvw_r, _ocld_r:\n")
shutil.copy(p, p + ".bak_0916_ocld")
open(p, "w").write(src)
print("engine patched", hashlib.md5(src.encode()).hexdigest()[:8])
