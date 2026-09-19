#!/usr/bin/env python3
"""Generate PANDA seed thumbnails (additive) matching the sanitized-thumbnail spec.

Writes full_no_panda/thumbnails_sanitized/panda__<stem>.png (max side 1024, RGB).
Skips files that already exist; touches nothing else.
"""
import json
from pathlib import Path

import openslide
from PIL import Image

MP = Path("/home/users/whddn12316/datasets/MultiPathQA")
SLIDES = MP / "slides/panda"
OUT = MP / "ready_wsivqa/full_no_panda/thumbnails_sanitized"
records = json.loads((MP / "ready_wsivqa/full_no_panda/panda.json").read_text())

made = kept = 0
for rec in records:
    stem = rec["Id"].split("__", 1)[1]
    target = OUT / f"{rec['Id']}.png"
    if target.exists():
        kept += 1
        continue
    slide = openslide.OpenSlide(str(SLIDES / f"{stem}.tiff"))
    thumb = slide.get_thumbnail((1024, 1024)).convert("RGB")
    slide.close()
    thumb.save(target)
    made += 1
print(f"thumbnails made={made} kept={kept}")
