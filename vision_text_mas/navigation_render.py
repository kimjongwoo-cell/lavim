"""WSI rendering helpers for deterministic discrete navigation."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from PIL import Image, ImageDraw, ImageFont

from vision_text_mas.errors import FailureCode, PipelineFailure
from vision_text_mas.geometry import Box, tissue_fraction


class SlideLike(Protocol):
    """Small OpenSlide-compatible boundary used by Navigator."""

    @property
    def dimensions(self) -> tuple[int, int]: ...

    @property
    def level_downsamples(self) -> tuple[float, ...]: ...

    def get_thumbnail(self, size: tuple[int, int]) -> Image.Image: ...

    def get_best_level_for_downsample(self, downsample: float) -> int: ...

    def read_region(
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
    ) -> Image.Image: ...


def _font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except OSError:
        return ImageFont.load_default()


def render_box(slide: SlideLike, box: Box, *, max_side: int) -> Image.Image:
    """Read a Level-0 box at a fitting pyramid level and resize it."""
    desired_downsample = max(box.width, box.height) / max_side
    try:
        level = slide.get_best_level_for_downsample(max(1.0, desired_downsample))
        downsample = float(slide.level_downsamples[level])
        read_size = (
            max(1, round(box.width / downsample)),
            max(1, round(box.height / downsample)),
        )
        image = slide.read_region((box.x, box.y), level, read_size).convert("RGB")
    except (OSError, RuntimeError) as error:
        raise PipelineFailure(
            code=FailureCode.IMAGE_IO,
            stage="navigator",
            detail=f"failed to read box {box}: {error}",
        ) from error
    scale = max_side / max(image.size)
    fitted = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(fitted, Image.Resampling.LANCZOS)


def tissue_fractions(
    slide: SlideLike,
    boxes: tuple[Box, ...],
) -> tuple[float, ...]:
    """Measure deterministic stained-tissue fractions on compact previews."""
    return tuple(tissue_fraction(render_box(slide, box, max_side=128)) for box in boxes)


def slide_thumbnail(slide: SlideLike) -> Image.Image:
    """Render the one cached whole-slide view used by root selection."""
    try:
        return slide.get_thumbnail((1_024, 1_024)).convert("RGB")
    except (OSError, RuntimeError) as error:
        raise PipelineFailure(
            code=FailureCode.IMAGE_IO,
            stage="navigator",
            detail=f"failed to render slide thumbnail: {error}",
        ) from error


def numbered_grid(
    image: Image.Image,
    *,
    parent: Box,
    boxes: tuple[Box, ...],
    eligible_ids: frozenset[int],
) -> Image.Image:
    """Overlay row-major IDs and gray out only tissue-ineligible candidates."""
    marked = image.convert("RGB")
    draw = ImageDraw.Draw(marked)
    font = _font()
    sx = marked.width / parent.width
    sy = marked.height / parent.height
    for candidate_id, box in enumerate(boxes, start=1):
        x0 = round((box.x - parent.x) * sx)
        y0 = round((box.y - parent.y) * sy)
        x1 = round((box.right - parent.x) * sx) - 1
        y1 = round((box.bottom - parent.y) * sy) - 1
        if candidate_id in eligible_ids:
            color = (220, 0, 0)
        else:
            color = (100, 100, 100)
            draw.rectangle((x0, y0, x1, y1), fill=(175, 175, 175))
        draw.rectangle((x0, y0, x1, y1), outline=color, width=2)
        draw.text(
            (x0 + 3, y0 + 2),
            str(candidate_id),
            fill=(255, 255, 0),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
            font=font,
        )
    return marked


def save_image(image: Image.Image, path: Path) -> Path:
    """Persist one RGB navigation artifact at an explicit path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        rendered = image.copy()
        rendered.info.clear()
        rendered.save(path)
    except OSError as error:
        raise PipelineFailure(
            code=FailureCode.IMAGE_IO,
            stage="navigator",
            detail=f"failed to save {path}: {error}",
        ) from error
    return path
