"""Opt-in Reasoner task override (env VLMAS_REASONER_TASK; unset = byte-identical prompt).

The Reasoner is normally asked for a rigid 11-field-per-patch JSON schema. When its latent
handoff is read back it only yields that schema's boilerplate, so we need to test whether
the collapse is a property of the latent transfer or of the task being a fixed schema.
This lets a probe swap the task clause for a plain instruction while keeping the question
stem, the evidence-plan line and the labeled-patch list untouched.

md5-guarded, no-op if already present, atomic write with a backup into tmp_hj/.
usage: python patch_reasoner_task.py <expected_md5>
"""
import hashlib
import os
import shutil
import sys

ROOT = "/home/users/whddn12316/wsi_latent_0915_decode_hj"
PATH = os.path.join(ROOT, "vision_text_mas/prompts.py")

IMP_OLD = '''from collections.abc import Iterable
import json
'''
IMP_NEW = '''from collections.abc import Iterable
import json
import os
'''

OLD = '''    return (
        f"Question stem: {question}\\nEvidence plan: {evidence_plan.model_dump_json()}\\n"
        f"Labeled patches:\\n{patch_lines}\\n"
        "Describe every patch using patch_id, quality, architecture, cellularity, "
'''

NEW = '''    head = (
        f"Question stem: {question}\\nEvidence plan: {evidence_plan.model_dump_json()}\\n"
        f"Labeled patches:\\n{patch_lines}\\n"
    )
    # Opt-in task override (env VLMAS_REASONER_TASK). Unset -> the prompt below is
    # returned unchanged, byte for byte. Set -> the schema task is replaced by the given
    # instruction, keeping the question, the evidence-plan line and the patch list.
    _task = os.environ.get("VLMAS_REASONER_TASK", "").strip()
    if _task:
        return head + _task
    return head + (
        "Describe every patch using patch_id, quality, architecture, cellularity, "
'''


def main(expected):
    src = open(PATH).read()
    md5 = hashlib.md5(src.encode()).hexdigest()
    if "VLMAS_REASONER_TASK" in src:
        print(f"ALREADY_PRESENT md5={md5}")
        return 0
    if md5 != expected:
        print(f"GUARD_FAIL md5={md5} expected={expected}")
        return 3
    for name, frag in (("IMPORT", IMP_OLD), ("RETURN", OLD)):
        if src.count(frag) != 1:
            print(f"{name}_FAIL count={src.count(frag)}")
            return 4
    out = src.replace(IMP_OLD, IMP_NEW).replace(OLD, NEW)
    shutil.copyfile(PATH, os.path.join(ROOT, f"tmp_hj/prompts.py.orig_{md5[:8]}"))
    tmp = PATH + ".new"
    with open(tmp, "w") as fh:
        fh.write(out)
    os.replace(tmp, PATH)
    print(f"PATCHED {md5} -> {hashlib.md5(out.encode()).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
