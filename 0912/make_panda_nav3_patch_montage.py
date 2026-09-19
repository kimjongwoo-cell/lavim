#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import openslide
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/home/users/whddn12316/wsi_latent_0915_decode_hj")
RUNS = ROOT / "sender_relay_exp/runs"
OUT = ROOT / "0912/nav3_dashboard_assets/panda_nav3_selected_patches.jpg"
OVERVIEW_OUT = ROOT / "0912/nav3_dashboard_assets/panda_nav3_selected_overview.jpg"
CASES = [
    (0, "dbf7cc49ae2e9831448c3ca54ad92708", "1"),
    (4, "ee61886bdebe2748009da564edb587cc", "3"),
    (6, "6525e666ca51cdae0b292dcd0b8e7483", "3"),
]


def run_dir(index: int) -> Path:
    gpu = 7 if index % 2 == 0 else 8
    base = RUNS / f"nav3_prompt1_base_panda_gpu{gpu}_20260914"
    return next(base.glob(f"{index:03d}_*"))


font = ImageFont.load_default()
tile_w, tile_h = 280, 235
margin, header_h = 18, 54
canvas = Image.new("RGB", (margin * 2 + tile_w * 4, margin * 2 + (header_h + tile_h * 2) * len(CASES)), "#111827")
draw = ImageDraw.Draw(canvas)
overview = Image.new("RGB", (1156, 520 * len(CASES)), "#111827")
overview_draw = ImageDraw.Draw(overview)

for row, (index, stem, gold) in enumerate(CASES):
    case_dir = run_dir(index)
    evidence = json.loads((case_dir / "round_1/evidence.json").read_text())["patches"]
    slide_path = Path(f"/home/users/whddn12316/datasets/MultiPathQA/slides/panda/{stem}.tiff")
    slide = openslide.OpenSlide(str(slide_path))
    thumb = slide.get_thumbnail((1080, 450)).convert("RGB")
    thumb_x = 38 + (1080 - thumb.width) // 2
    thumb_y = row * 520 + 48
    overview.paste(thumb, (thumb_x, thumb_y))
    scale_x = thumb.width / slide.dimensions[0]
    scale_y = thumb.height / slide.dimensions[1]
    overview_draw.text((38, row * 520 + 18), f"PANDA ID {index}   GOLD={gold}   PRED=0", fill="#ffffff", font=font)
    for patch in evidence[:8]:
        box = patch["box"]
        color = "#7170ff" if int(patch["magnification"]) == 5 else "#fbbf24"
        x1 = thumb_x + int(box["x"] * scale_x)
        y1 = thumb_y + int(box["y"] * scale_y)
        x2 = x1 + max(3, int(box["width"] * scale_x))
        y2 = y1 + max(3, int(box["height"] * scale_y))
        overview_draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        overview_draw.text((x1 + 3, max(thumb_y, y1 - 12)), patch["patch_id"], fill=color, font=font)
    y0 = margin + row * (header_h + tile_h * 2)
    draw.text((margin, y0 + 8), f"PANDA ID {index}   GOLD={gold}   PRED=0", fill="#ffffff", font=font)
    for j, patch in enumerate(evidence[:8]):
        box = patch["box"]
        region = slide.read_region((int(box["x"]), int(box["y"])), 0, (int(box["width"]), int(box["height"]))).convert("RGB")
        region.thumbnail((tile_w - 10, tile_h - 32), Image.Resampling.LANCZOS)
        x = margin + (j % 4) * tile_w
        y = y0 + header_h + (j // 4) * tile_h
        canvas.paste(region, (x + (tile_w - region.width) // 2, y + 22))
        label = f'{patch["patch_id"]}  x{patch["magnification"]}  root={patch.get("root_id")}'
        draw.text((x + 5, y + 5), label, fill="#f9fafb", font=font)
    slide.close()

OUT.parent.mkdir(parents=True, exist_ok=True)
canvas.save(OUT, quality=92, subsampling=0)
overview.save(OVERVIEW_OUT, quality=92, subsampling=0)
print(OUT)
print(OVERVIEW_OUT)
