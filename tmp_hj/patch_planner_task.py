"""Opt-in LatentMAS-style compact Planner task (env VLMAS_PLANNER_TASK=bullet).

Unset -> evidence_planner_prompt() returns the original JSON-schema prompt byte for byte.
Set to "bullet" -> the same fields as the JSON template, written as a LatentMAS-style bullet
format (role line, question, "Format your response as follows:", "Now, output ... below:").
Counts (x5 / x20 / anchors / budget) are filled from patch_budget exactly as before.
"""
import hashlib
from pathlib import Path

P = Path("/home/users/whddn12316/wsi_latent_0915_decode_hj/vision_text_mas/navigation_prompts.py")
s = P.read_text()
bak = P.with_suffix(P.suffix + ".bak_0917_planner")
if not bak.exists():
    bak.write_text(s)

old = '''    follow_up = (
        f"\\nVerifier missing evidence: {missing_evidence}"
        if missing_evidence is not None
        else ""
    )
    return ('''
new = '''    follow_up = (
        f"\\nVerifier missing evidence: {missing_evidence}"
        if missing_evidence is not None
        else ""
    )
    # Opt-in LatentMAS-style compact task (env VLMAS_PLANNER_TASK=bullet). Unset -> the
    # JSON-schema prompt below is returned unchanged, byte for byte. Same fields, bullets.
    import os as _os
    if _os.environ.get("VLMAS_PLANNER_TASK", "").strip() == "bullet":
        return (
            f"Question: {question}{follow_up}\\n\\n"
            "You are a Planner Agent. Using only the question and the whole-slide "
            "thumbnail, plan what visual evidence is needed to answer it. Do not "
            "diagnose, answer, or give locations, grid IDs, or coordinates.\\n\\n"
            "Format your response as follows:\\n"
            "- Question focus: what the question is about.\\n"
            "- Needed visual information: observable features required.\\n"
            "- Thumbnail observations: non-diagnostic overview notes.\\n"
            "- Search instruction: how to recognize useful regions.\\n"
            "- Success criteria: what makes evidence useful.\\n"
            "- Detail trigger: what requires finer scale.\\n"
            f"- x5 overview ({overview_patch_count} patches): architecture/context to look for.\\n"
            f"- x20 detail ({detail_patch_count} patches, {detail_anchor_count} anchors): "
            "cellular evidence to look for.\\n"
            "- Scale success: what must be seen at both scales.\\n\\n"
            "Now, output your plan below:"
        )
    return ('''
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new, 1)
P.write_text(s)
print("patched", P, hashlib.md5(P.read_bytes()).hexdigest()[:8])
