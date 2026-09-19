"""Add the VLMAS_LATDEC hook to latent_qwen_engine.py (decode an agent's latent steps).
md5-guarded, no-op if already present, atomic write with a backup into tmp_hj/.
usage: python patch_latdec_engine.py <expected_md5>
"""
import hashlib, os, shutil, sys

ROOT = "/home/users/whddn12316/wsi_latent_0915_decode_hj"
PATH = os.path.join(ROOT, "vision_text_mas/latent_qwen_engine.py")

ANCHOR = '''        full_cache_length = int(result["past_len"])
'''

BLOCK = '''        if os.environ.get("VLMAS_LATDEC", "").strip():
            # Decode this stage's latent steps straight out of the latent loop
            # (memory/latdec_diag.py). Reads result["latent_trajectory"], which the
            # backbone already returns; no extra forward, no decode change.
            from memory import latdec_diag as _latdec
            _latdec.run(self, stage=stage, result=result,
                        case_index=getattr(self, "_rpath_case_index", -1))
'''


def main(expected):
    src = open(PATH).read()
    md5 = hashlib.md5(src.encode()).hexdigest()
    if "VLMAS_LATDEC" in src:
        print(f"ALREADY_PRESENT md5={md5}"); return 0
    if md5 != expected:
        print(f"GUARD_FAIL md5={md5} expected={expected}"); return 3
    if src.count(ANCHOR) != 1:
        print(f"ANCHOR_FAIL count={src.count(ANCHOR)}"); return 4
    out = src.replace(ANCHOR, BLOCK + ANCHOR)
    shutil.copyfile(PATH, os.path.join(ROOT, f"tmp_hj/latent_qwen_engine.py.orig_{md5[:8]}"))
    tmp = PATH + ".new"
    with open(tmp, "w") as fh:
        fh.write(out)
    os.replace(tmp, PATH)
    print(f"PATCHED {md5} -> {hashlib.md5(out.encode()).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
