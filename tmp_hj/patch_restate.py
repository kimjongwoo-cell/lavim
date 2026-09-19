"""Opt-in LatentMAS Appendix-E mirroring: ask latent consumers to restate what they got.

Our latent-handoff note tells every consumer "do not restate it". The original LatentMAS
does the opposite: Appendix E's Critic prompt asks for "(1) original plan contents" and
Appendix D's case study shows the downstream agent verbalising the upstream latent plan.
This adds `VLMAS_LATENT_RESTATE=1` to swap that one clause. Off = byte-identical prompts.

md5-guarded, no-op if already present, atomic write with a backup into tmp_hj/.
usage: python patch_restate.py <expected_md5>
"""
import hashlib
import os
import shutil
import sys

ROOT = "/home/users/whddn12316/wsi_latent_0915_decode_hj"
PATH = os.path.join(ROOT, "vision_text_mas/latent_onepass.py")

ANCHOR = '''def announce_latent_upstream(
'''

HELPER = '''_NO_RESTATE = "so use only what helps and do not restate it."
_DO_RESTATE = (
    "so use only what helps. First restate, in text, the original contents you "
    "received, as faithfully as you can; then continue with your own task."
)


def restate_note(note: str) -> str:
    """LatentMAS Appendix-E mirroring (env VLMAS_LATENT_RESTATE=1; off = note unchanged).

    The paper hands each latent-consuming agent a prompt asking it to output "(1) original
    plan contents" before its own work, and its Appendix D case study reads the latent
    handoff by having the downstream agent verbalise it. Our note forbids exactly that, so
    this swaps the clause when the flag is on. Any note without the clause is returned as
    is, so turning the flag on can never silently rewrite an unrelated prompt.
    """
    if os.environ.get("VLMAS_LATENT_RESTATE", "").strip() != "1":
        return note
    return note.replace(_NO_RESTATE, _DO_RESTATE)


'''

OLD_BODY = '''    lines[index] = f"{note}{newline}"
'''

NEW_BODY = '''    lines[index] = f"{restate_note(note)}{newline}"
'''


def main(expected):
    src = open(PATH).read()
    md5 = hashlib.md5(src.encode()).hexdigest()
    if "VLMAS_LATENT_RESTATE" in src:
        print(f"ALREADY_PRESENT md5={md5}")
        return 0
    if md5 != expected:
        print(f"GUARD_FAIL md5={md5} expected={expected}")
        return 3
    for name, frag in (("ANCHOR", ANCHOR), ("BODY", OLD_BODY)):
        if src.count(frag) != 1:
            print(f"{name}_FAIL count={src.count(frag)}")
            return 4
    out = src.replace(ANCHOR, HELPER + ANCHOR).replace(OLD_BODY, NEW_BODY)
    shutil.copyfile(PATH, os.path.join(ROOT, f"tmp_hj/latent_onepass.py.orig_{md5[:8]}"))
    tmp = PATH + ".new"
    with open(tmp, "w") as fh:
        fh.write(out)
    os.replace(tmp, PATH)
    print(f"PATCHED {md5} -> {hashlib.md5(out.encode()).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
