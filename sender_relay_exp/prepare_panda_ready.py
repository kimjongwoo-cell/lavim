#!/usr/bin/env python3
"""Wire PANDA into the ready_wsivqa layout — additive only, no existing file touched.

Creates:
  ready_wsivqa/full_no_panda/panda.json           (DatasetItem contract: Id/Question/Choice/Answer/Task)
  ready_wsivqa/full_no_panda/slides/panda__<stem>  -> datasets/MultiPathQA/slides/panda/<file_id>

Refuses to overwrite an existing panda.json (delete it manually to regenerate).
Existing symlinks are left as-is when they already point at the right target.
"""
import json
import sys
from pathlib import Path

MP = Path("/home/users/whddn12316/datasets/MultiPathQA")
SRC = MP / "metadata/beacon_932/panda.json"
SLIDES = MP / "slides/panda"
READY = MP / "ready_wsivqa/full_no_panda"
OUT = READY / "panda.json"

records = json.loads(SRC.read_text())
if OUT.exists():
    sys.exit(f"refusing to overwrite existing {OUT}")

ready_rows = []
linked = kept = 0
for rec in records:
    file_id = rec["file_id"]
    target = SLIDES / file_id
    if not target.is_file():
        sys.exit(f"missing slide: {target}")
    stem = file_id.rsplit(".", 1)[0]
    slide_id = f"panda__{stem}"
    options = rec["options"]
    if isinstance(options, str):
        options = json.loads(options)
    ready_rows.append(
        {
            "Id": slide_id,
            "Question": rec["prompt"].strip(),
            "Choice": [str(o) for o in options],
            "Answer": str(rec["answer"]),
            "Task": "panda",
        }
    )
    link = READY / "slides" / slide_id
    if link.is_symlink():
        if link.resolve() != target.resolve():
            sys.exit(f"existing symlink points elsewhere: {link}")
        kept += 1
    elif link.exists():
        sys.exit(f"non-symlink already present: {link}")
    else:
        link.symlink_to(target)
        linked += 1

OUT.write_text(json.dumps(ready_rows, ensure_ascii=False, indent=1))
print(f"wrote {OUT} ({len(ready_rows)} records); symlinks new={linked} kept={kept}")
