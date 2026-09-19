#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["openslide-python", "pillow"]
# ///
# ─── How to run ───
# uv run scripts/render_navigation_tree.py --slide /path/to/slide.svs --evidence /path/to/evidence.json --output /path/to/navigation_tree.png
"""Render one WSI thumbnail with Navigator selections and their x5-to-x20 tree."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import openslide
from PIL import Image, ImageDraw, ImageFont

MAX_THUMBNAIL_WIDTH: Final = 1_200
PANEL_WIDTH: Final = 540
PADDING: Final = 32
DETAIL_CARD_WIDTH: Final = 290
DETAIL_CARD_HEIGHT: Final = 410
X5_COLOR: Final = "#14B8A6"
X20_COLOR: Final = "#F97316"
BACKGROUND: Final = "#0B1020"
PANEL: Final = "#141C30"
TEXT: Final = "#E5E7EB"
MUTED: Final = "#A8B3CF"


@dataclass(frozen=True, slots=True)
class Box:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class Patch:
    patch_id: str
    magnification: int
    box: Box
    parent_id: str
    rank: int


def parse_patches(evidence_path: Path) -> tuple[Patch, ...]:
    """Parse the saved Navigator evidence boundary into immutable patches."""
    raw = json.loads(evidence_path.read_text(encoding="utf-8"))
    patches = raw["patches"]
    return tuple(
        Patch(
            patch_id=item["patch_id"],
            magnification=item["magnification"],
            box=Box(**item["box"]),
            parent_id=item["source_x5_anchor"],
            rank=item["rank"],
        )
        for item in patches
    )


def slide_images(slide_path: Path, patches: tuple[Patch, ...]) -> tuple[Image.Image, tuple[int, int], dict[str, Image.Image]]:
    """Read the overview and compact crop insets from one WSI open operation."""
    with openslide.OpenSlide(str(slide_path)) as slide:
        width, height = slide.dimensions
        target_height = max(1, round(MAX_THUMBNAIL_WIDTH * height / width))
        overview = slide.get_thumbnail((MAX_THUMBNAIL_WIDTH, target_height)).convert("RGB")
        insets: dict[str, Image.Image] = {}
        for patch in patches:
            region = slide.read_region((patch.box.x, patch.box.y), 0, (patch.box.width, patch.box.height)).convert("RGB")
            inset_size = (250, 150) if patch.magnification == 5 else (112, 112)
            region.thumbnail(inset_size)
            insets[patch.patch_id] = region
    return overview, (width, height), insets


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Use a broadly available sans font, with Pillow's bitmap fallback."""
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        path = Path(candidate)
        if path.is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def scaled_box(box: Box, slide_size: tuple[int, int], image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Map one level-0 crop box into thumbnail pixels."""
    slide_width, slide_height = slide_size
    image_width, image_height = image_size
    return (
        round(box.x * image_width / slide_width),
        round(box.y * image_height / slide_height),
        round((box.x + box.width) * image_width / slide_width),
        round((box.y + box.height) * image_height / slide_height),
    )


def display_labels(patches: tuple[Patch, ...]) -> dict[str, str]:
    """Create the same human-readable context/detail labels in every figure panel."""
    parents = tuple(patch for patch in patches if patch.magnification == 5)
    children = tuple(patch for patch in patches if patch.magnification == 20)
    labels = {parent.patch_id: f"Context {index + 1}" for index, parent in enumerate(parents)}
    for parent_index, parent in enumerate(parents):
        child_group = tuple(child for child in children if child.parent_id == parent.patch_id)
        for child_index, child in enumerate(child_group):
            labels[child.patch_id] = f"Detail {parent_index + 1}{chr(ord('a') + child_index)}"
    return labels


def draw_selection_overlay(
    image: Image.Image,
    patches: tuple[Patch, ...],
    slide_size: tuple[int, int],
    labels: dict[str, str],
) -> None:
    """Draw selected x5 regions and nested x20 regions with stable labels."""
    draw = ImageDraw.Draw(image)
    label_font = font(18)
    for patch in patches:
        color = X5_COLOR if patch.magnification == 5 else X20_COLOR
        bounds = scaled_box(patch.box, slide_size, image.size)
        width = 4 if patch.magnification == 5 else 3
        draw.rectangle(bounds, outline=color, width=width)
        label = labels[patch.patch_id]
        label_width = 88 if patch.magnification == 5 else 82
        draw.rounded_rectangle((bounds[0], bounds[1], bounds[0] + label_width, bounds[1] + 27), radius=5, fill=color)
        draw.text((bounds[0] + 6, bounds[1] + 4), label, fill=BACKGROUND, font=label_font)


def draw_tree(panel: Image.Image, patches: tuple[Patch, ...], labels: dict[str, str]) -> None:
    """Draw the deterministic x5 parent to x20 child selection hierarchy."""
    draw = ImageDraw.Draw(panel)
    heading = font(24)
    body = font(18)
    small = font(15)
    draw.text((28, 26), "Navigator selection tree", fill=TEXT, font=heading)
    draw.text((28, 62), "teal: x5 context   orange: x20 detail", fill=MUTED, font=small)
    parents = tuple(patch for patch in patches if patch.magnification == 5)
    children = tuple(patch for patch in patches if patch.magnification == 20)
    y = 110
    for parent in parents:
        draw.rounded_rectangle((28, y, 198, y + 36), radius=8, fill=X5_COLOR)
        draw.text((40, y + 7), f"{labels[parent.patch_id]}  · x5", fill=BACKGROUND, font=body)
        child_group = tuple(child for child in children if child.parent_id == parent.patch_id)
        branch_x = 52
        child_y = y + 54
        for offset, child in enumerate(child_group):
            line_y = child_y + offset * 35 + 16
            draw.line((branch_x, y + 36, branch_x, line_y), fill=MUTED, width=2)
            draw.line((branch_x, line_y, 90, line_y), fill=MUTED, width=2)
            draw.rounded_rectangle((94, line_y - 15, 255, line_y + 16), radius=7, fill=X20_COLOR)
            draw.text((106, line_y - 9), f"{labels[child.patch_id]}  · x20", fill=BACKGROUND, font=small)
        y = child_y + max(1, len(child_group)) * 35 + 28


def draw_detail_cards(
    canvas: Image.Image,
    patches: tuple[Patch, ...],
    insets: dict[str, Image.Image],
    labels: dict[str, str],
    y_offset: int,
) -> None:
    """Place enlarged parent-context and child-detail evidence below the overview."""
    draw = ImageDraw.Draw(canvas)
    heading = font(24)
    body = font(16)
    draw.text((PADDING, y_offset - 40), "B. Parent context and child-detail evidence", fill=TEXT, font=heading)
    parents = tuple(patch for patch in patches if patch.magnification == 5)
    children = tuple(patch for patch in patches if patch.magnification == 20)
    for index, parent in enumerate(parents):
        x_offset = PADDING + index * (DETAIL_CARD_WIDTH + 24)
        draw.rounded_rectangle(
            (x_offset, y_offset, x_offset + DETAIL_CARD_WIDTH, y_offset + DETAIL_CARD_HEIGHT),
            radius=12,
            fill=PANEL,
            outline=X5_COLOR,
            width=2,
        )
        draw.text((x_offset + 16, y_offset + 14), f"{labels[parent.patch_id]} · x5 context", fill=TEXT, font=body)
        parent_image = insets[parent.patch_id].copy()
        parent_draw = ImageDraw.Draw(parent_image)
        child_group = tuple(child for child in children if child.parent_id == parent.patch_id)
        for child in child_group:
            left = round((child.box.x - parent.box.x) * parent_image.width / parent.box.width)
            top = round((child.box.y - parent.box.y) * parent_image.height / parent.box.height)
            right = round((child.box.x + child.box.width - parent.box.x) * parent_image.width / parent.box.width)
            bottom = round((child.box.y + child.box.height - parent.box.y) * parent_image.height / parent.box.height)
            parent_draw.rectangle((left, top, right, bottom), outline=X20_COLOR, width=3)
            short_label = labels[child.patch_id].replace("Detail ", "")
            parent_draw.rounded_rectangle((left, top, left + 26, top + 20), radius=4, fill=X20_COLOR)
            parent_draw.text((left + 4, top + 2), short_label, fill=BACKGROUND, font=font(13))
        parent_x = x_offset + (DETAIL_CARD_WIDTH - parent_image.width) // 2
        canvas.paste(parent_image, (parent_x, y_offset + 44))
        child_y = y_offset + 218
        for child_index, child in enumerate(child_group):
            child_image = insets[child.patch_id]
            child_x = x_offset + 25 + child_index * 140
            draw.line((x_offset + DETAIL_CARD_WIDTH // 2, y_offset + 198, child_x + child_image.width // 2, child_y - 8), fill=MUTED, width=2)
            canvas.paste(child_image, (child_x, child_y))
            draw.text((child_x, child_y + child_image.height + 8), f"{labels[child.patch_id]} · x20", fill=X20_COLOR, font=body)


def compose(slide_path: Path, evidence_path: Path, output_path: Path) -> None:
    """Create a paper-style English figure for one observed WSI case."""
    patches = parse_patches(evidence_path)
    overview, slide_size, insets = slide_images(slide_path, patches)
    labels = display_labels(patches)
    draw_selection_overlay(overview, patches, slide_size, labels)
    canvas_height = max(overview.height + PADDING * 3 + DETAIL_CARD_HEIGHT + 56, 1_220)
    canvas_width = overview.width + PANEL_WIDTH + PADDING * 3
    canvas = Image.new("RGB", (canvas_width, canvas_height), BACKGROUND)
    canvas.paste(overview, (PADDING, PADDING))
    panel = Image.new("RGB", (PANEL_WIDTH, canvas_height - PADDING * 2), PANEL)
    draw_tree(panel, patches, labels)
    canvas.paste(panel, (overview.width + PADDING * 2, PADDING))
    draw = ImageDraw.Draw(canvas)
    draw.text((PADDING, 6), "A. Whole-slide Navigator selections", fill=TEXT, font=font(24))
    draw_detail_cards(canvas, patches, insets, labels, overview.height + PADDING * 2 + 48)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slide", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    compose(arguments.slide, arguments.evidence, arguments.output)


if __name__ == "__main__":
    main()
