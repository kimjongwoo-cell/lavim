"""0917 NOVA (C1 #21) wiring for the decode copy tree: (1) VLMAS_C1_SELECT=nova dispatch in
memory/qasc_select.py, (2) keep the fp32 pixels on the backbone for nova as for SEC-LD. md5-guarded,
backups *.bak_0917_nova; off = byte-identical behaviour. Usage: python patch_nova.py [--check]"""
import hashlib, shutil, sys

CHECK = "--check" in sys.argv


def md5(s):
    return hashlib.md5(s.encode()).hexdigest()[:8]


def patch(path, expected, old, new, bak):
    src = open(path).read()
    m = md5(src)
    if m != expected:
        if src.count(new) == 1 and src.count(old) == 0:
            print(f"{path}: already patched (md5 {m})")
            return
        raise SystemExit(f"{path}: md5 {m} != {expected}, refusing")
    assert src.count(old) == 1, (path, "anchor count", src.count(old))
    out = src.replace(old, new)
    if CHECK:
        print(f"{path}: would patch {m} -> {md5(out)}")
        return
    shutil.copy(path, path + bak)
    open(path, "w").write(out)
    print(f"{path}: {m} -> {md5(out)} (backup {path + bak})")


# 1) dispatch (memory/qasc_select.py 17274ab8)
patch(
    "memory/qasc_select.py", "17274ab8",
    '''    if os.environ.get("VLMAS_C1_SELECT", "").strip() in ("ssda", "ntrs"):
''',
    '''    if os.environ.get("VLMAS_C1_SELECT", "").strip() == "nova":
        # C1 #21 NOVA (memory/nova_select.py): spatial-null mass e_i = 1 - b_i times the full-vision
        # morphology direction, support-wise log-det, no quota, at this same prefill site
        from memory import nova_select as _nova
        return _nova.select_vision_features(bb, pixel_values, grid_thw)
    if os.environ.get("VLMAS_C1_SELECT", "").strip() in ("ssda", "ntrs"):
''',
    ".bak_0917_nova",
)

# 2) fp32 pixels for nova too (backbone/qwen3vl.py abed5350)
patch(
    "backbone/qwen3vl.py", "abed5350",
    '''        # SEC-LD reads the un-cast fp32 pixels (bf16 pv moves a gray level by <= 0.25); None when off.
        self._secld_pixel_values_fp32 = (
            inputs["pixel_values"].detach().to(torch.float32)
            if os.environ.get("VLMAS_SECLD", "").strip() == "1" else None
        )
''',
    '''        # SEC-LD / NOVA read the un-cast fp32 pixels (bf16 pv moves a gray level by <= 0.25); None when off.
        self._secld_pixel_values_fp32 = (
            inputs["pixel_values"].detach().to(torch.float32)
            if (os.environ.get("VLMAS_SECLD", "").strip() == "1"
                or os.environ.get("VLMAS_C1_SELECT", "").strip() == "nova") else None
        )
''',
    ".bak_0917_nova",
)
