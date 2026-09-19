"""0917 SEC-LD (C1 #19) backbone wiring for the decode copy tree. md5-guarded; off = byte-identical."""
import hashlib, shutil, sys
p = sys.argv[1] if len(sys.argv) > 1 else "backbone/qwen3vl.py"
src = open(p).read()
md5 = hashlib.md5(src.encode()).hexdigest()[:8]
assert md5 == "d6dee53f", f"backbone md5 {md5} (expected d6dee53f)"

# 1) keep-mask branch at the Pruning-B / EOVC site (no quota, same budget)
old = '''        from memory import eovc as _eovc
        if _eovc.enabled():
'''
new = '''        from memory import eovc as _eovc
        # C1 #19 SEC-LD (memory/secld_select.py, env VLMAS_SECLD=1): pixel-space spatial evidence
        # demand e_i = 1 - b_i times the same early-observation morphology direction, one support-wise
        # log-det set objective, no per-crop quota. Off = unchanged.
        from memory import secld_select as _secld
        if _secld.enabled():
            _secld_cfg = _secld.config()
            _secld_ratio = float(adaptive_keep_ratio) if _secld_cfg["keep"] is None else float(_secld_cfg["keep"])
            _secld_budget = int(round(_secld_ratio * int(pooled.shape[0])))
            _secld_pv = getattr(self, "_secld_pixel_values_fp32", None)
            if _secld_pv is None or tuple(_secld_pv.shape) != tuple(pixel_values.shape):
                _secld_pv = pixel_values
            _secld_info = _secld.select(_secld_pv.to(pooled.device), grid_thw, pooled, token_counts,
                                        _secld_budget, _secld_cfg)
            keep_groups = _secld_info["keep"].to(device=pooled.device)
            print(f"[SECLD] {_secld.summary(_secld_info)} keep_ratio={_secld_ratio:.3f}", flush=True)
        elif _eovc.enabled():
'''
assert src.count(old) == 1, ("eovc anchor", src.count(old))
src = src.replace(old, new)

# 2) keep the fp32 processor pixels for the bright-pixel rule (pv is cast to the model dtype)
old = '''        pv  = inputs["pixel_values"].to(self.device, dtype=self.dtype)
        thw = inputs["image_grid_thw"].to(self.device)

        # ── MRoPE positions for the chunk, CONTINUED from past_pos_cursor ─────────
'''
new = '''        pv  = inputs["pixel_values"].to(self.device, dtype=self.dtype)
        thw = inputs["image_grid_thw"].to(self.device)
        # SEC-LD reads the un-cast fp32 pixels (bf16 pv moves a gray level by <= 0.25); None when off.
        self._secld_pixel_values_fp32 = (
            inputs["pixel_values"].detach().to(torch.float32)
            if os.environ.get("VLMAS_SECLD", "").strip() == "1" else None
        )

        # ── MRoPE positions for the chunk, CONTINUED from past_pos_cursor ─────────
'''
assert src.count(old) == 1, ("pv anchor", src.count(old))
src = src.replace(old, new)
assert src.count("_hierarchy_pruned_vision_features(") == 2, src.count("_hierarchy_pruned_vision_features(")  # def + 1 call

shutil.copy(p, p + ".bak_0917_secld")
open(p, "w").write(src)
print("backbone patched", hashlib.md5(src.encode()).hexdigest()[:8])
