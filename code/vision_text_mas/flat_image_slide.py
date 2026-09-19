"""PIL-backed slide boundary for flat WSI overview images."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


class FlatImageSlide:
    """Expose a flat RGB image through the navigator's OpenSlide-like contract."""

    def __init__(self, path: Path) -> None:
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(path) as source:
            self._image = source.convert("RGB")

    @property
    def dimensions(self) -> tuple[int, int]:
        """Return the Level-0 dimensions of the image."""
        return self._image.size

    @property
    def level_downsamples(self) -> tuple[float, ...]:
        """A flat image has one Level-0 representation."""
        return (1.0,)

    def get_thumbnail(self, size: tuple[int, int]) -> Image.Image:
        """Return a fitted copy of the image."""
        thumbnail = self._image.copy()
        thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
        return thumbnail

    def get_best_level_for_downsample(self, downsample: float) -> int:
        """Always use the single available level."""
        _ = downsample
        return 0

    def read_region(
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
    ) -> Image.Image:
        """Read a Level-0 crop, padding out-of-bounds pixels with black."""
        if level != 0:
            raise ValueError(f"flat image only supports level 0, received {level}")
        x, y = location
        width, height = size
        region = Image.new("RGB", size)
        source_box = (x, y, x + width, y + height)
        overlap = self._image.crop(source_box)
        region.paste(overlap, (0, 0))
        return region

    def close(self) -> None:
        """Release the decoded image buffer."""
        self._image.close()
