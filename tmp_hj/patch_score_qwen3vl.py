"""Add the SCoRe selection branch (mode == "score") to backbone/qwen3vl.py.

Guarded: applies only when the file still hashes to the expected md5 (shared file), is a
no-op if the branch is already there, backs the original up under tmp_hj/ and replaces
the file atomically.   usage: python patch_score_qwen3vl.py <expected_md5>
"""
import hashlib
import os
import shutil
import sys

ROOT = "/home/users/whddn12316/wsi_latent_0915_decode_hj"
PATH = os.path.join(ROOT, "backbone/qwen3vl.py")

ANCHOR = '''        elif mode == "strat":'''

BLOCK = '''        elif mode == "score":
            # SCoRe (Xu et al., CVPR 2026) Algorithm 1 — the literature selection arm for
            # the matched-budget accuracy table: greedy weighted k-center with cosine
            # distance on the visual states, salience^alpha as the weight
            # (VLMAS_SCORE_ALPHA, paper best 0.8). VLMAS_WSI_CONSOL_SC=1 supplies the
            # q2v salience; without it the weights are uniform (plain k-center).
            from memory.score_select import score_keep_mask
            _alpha = float(os.environ.get("VLMAS_SCORE_ALPHA", "0.8") or 0.8)
            keep, score_info = score_keep_mask(
                vis_embeddings, relevance, token_counts, budget,
                alpha=_alpha, return_info=True)
            keep = keep.to(self.device)
            print("[WSIConsol] score " + " ".join(
                f"{k}={v}" for k, v in score_info.items() if k != "per_crop")
                + f" per_crop={score_info.get('per_crop')}"
                + f" attn={os.environ.get('VLMAS_ATTN_IMPLEMENTATION', 'eager')}"
                + f" control={control or 'none'}", flush=True)
'''


def main(expected):
    src = open(PATH).read()
    md5 = hashlib.md5(src.encode()).hexdigest()
    if 'mode == "score"' in src:
        print(f"ALREADY_PRESENT md5={md5}")
        return 0
    if md5 != expected:
        print(f"GUARD_FAIL md5={md5} expected={expected}")
        return 3
    if src.count(ANCHOR) != 1:
        print(f"ANCHOR_FAIL count={src.count(ANCHOR)}")
        return 4
    out = src.replace(ANCHOR, BLOCK + ANCHOR)
    shutil.copyfile(PATH, os.path.join(ROOT, f"tmp_hj/qwen3vl.py.orig_{md5[:8]}"))
    tmp = PATH + ".new"
    with open(tmp, "w") as fh:
        fh.write(out)
    os.replace(tmp, PATH)
    print(f"PATCHED {md5} -> {hashlib.md5(out.encode()).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
