"""OpenSlide-compatible reader for DICOM whole-slide series."""

from __future__ import annotations

from pathlib import Path
from typing import final

from PIL import Image
from wsidicom import WsiDicom


@final
class DicomSlide:
    """Expose a WSI DICOM series through the navigation slide contract."""

    def __init__(self, path: Path) -> None:
        self._slide = WsiDicom.open(path)
        self._dicom_levels = tuple(level.level for level in self._slide.levels)
        self._sizes = tuple((level.size.width, level.size.height) for level in self._slide.levels)

    @property
    def dimensions(self) -> tuple[int, int]:
        return self._sizes[0]

    @property
    def level_downsamples(self) -> tuple[float, ...]:
        width = self.dimensions[0]
        return tuple(width / size[0] for size in self._sizes)

    def get_best_level_for_downsample(self, downsample: float) -> int:
        return min(
            range(len(self.level_downsamples)),
            key=lambda level: abs(self.level_downsamples[level] - downsample),
        )

    def get_thumbnail(self, size: tuple[int, int]) -> Image.Image:
        requested = max(self.dimensions[0] / size[0], self.dimensions[1] / size[1])
        level = self.get_best_level_for_downsample(requested)
        image = self._slide.read_region(
            (0, 0), self._dicom_levels[level], self._sizes[level]
        ).convert("RGB")
        image.thumbnail(size, Image.Resampling.LANCZOS)
        return image

    def read_region(
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
    ) -> Image.Image:
        downsample = self.level_downsamples[level]
        level_location = (int(location[0] / downsample), int(location[1] / downsample))
        return self._slide.read_region(
            level_location, self._dicom_levels[level], size
        ).convert("RGB")

    def close(self) -> None:
        self._slide.close()
