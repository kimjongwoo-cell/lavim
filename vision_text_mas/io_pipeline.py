"""Bounded image reuse and asynchronous JSON persistence for sequential runs."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Hashable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Lock

from PIL import Image


def _image_bytes(image: Image.Image) -> int:
    return image.width * image.height * len(image.getbands())


class BoundedImageCache:
    """LRU cache of immutable image snapshots with defensive-copy reads."""

    def __init__(self, *, max_bytes: int) -> None:
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        self._max_bytes = max_bytes
        self._bytes = 0
        self._images: OrderedDict[Hashable, Image.Image] = OrderedDict()

    def get_or_load(
        self,
        key: Hashable,
        loader: Callable[[], Image.Image],
    ) -> Image.Image:
        cached = self._images.pop(key, None)
        if cached is None:
            cached = loader().copy()
            size = _image_bytes(cached)
            while self._images and self._bytes + size > self._max_bytes:
                _, evicted = self._images.popitem(last=False)
                self._bytes -= _image_bytes(evicted)
                evicted.close()
            if size <= self._max_bytes:
                self._images[key] = cached
                self._bytes += size
            else:
                return cached
        else:
            self._images[key] = cached
        return cached.copy()

    def clear(self) -> None:
        for image in self._images.values():
            image.close()
        self._images.clear()
        self._bytes = 0


class AsyncJsonWriter:
    """Single-writer queue that preserves atomic replacement and error delivery."""

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="artifact-json")
        self._futures: list[Future[None]] = []
        self._lock = Lock()
        self._closed = False

    @staticmethod
    def _write(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)

    def submit(self, path: Path, payload: bytes) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("async JSON writer is closed")
            self._futures.append(self._executor.submit(self._write, path, payload))

    def flush(self) -> None:
        with self._lock:
            pending, self._futures = self._futures, []
        for future in pending:
            future.result()

    def close(self) -> None:
        with self._lock:
            self._closed = True
        try:
            self.flush()
        finally:
            self._executor.shutdown(wait=True, cancel_futures=False)
