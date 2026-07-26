"""Entropy-aware temporal filter for dropping near-duplicate RGB frames."""

import io
from typing import Any, Optional

import numpy as np

from .Actor import Actor


def _to_grayscale(rgb_image: Any) -> np.ndarray:
    from PIL import Image

    image_bytes = Actor.rgb_to_bytes(rgb_image)
    with Image.open(io.BytesIO(image_bytes)) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def _shannon_entropy(values: np.ndarray, *, bins: int) -> float:
    """Shannon entropy (bits) of the value distribution in `values`."""
    histogram, _ = np.histogram(values, bins=bins, range=(0, 255))
    total = histogram.sum()
    if total == 0:
        return 0.0
    probabilities = histogram / total
    nonzero = probabilities[probabilities > 0]
    return float(-np.sum(nonzero * np.log2(nonzero)))


class EntropyFrameFilter:
    """Drop RGB frames that are too similar to the last frame kept.

    Plain pixel-difference thresholds are noisy under lighting flicker and camera
    jitter. Instead this measures the Shannon entropy of the *difference* image's
    histogram against the last kept frame: a near-duplicate frame produces a
    difference dominated by near-zero values (low entropy), while a genuinely new
    observation spreads the difference across many bins (high entropy). A frame is
    dropped only when its difference entropy falls below `entropy_threshold`.
    """

    def __init__(self, *, entropy_threshold: float = 3.0, bins: int = 64):
        if entropy_threshold < 0:
            raise ValueError("entropy_threshold must be non-negative.")
        if bins < 2:
            raise ValueError("bins must be at least 2.")
        self.entropy_threshold = entropy_threshold
        self.bins = bins
        self._reference: Optional[np.ndarray] = None
        self.kept_count = 0
        self.dropped_count = 0

    def reset(self) -> None:
        self._reference = None
        self.kept_count = 0
        self.dropped_count = 0

    def should_keep(self, rgb_image: Any) -> bool:
        """Return whether `rgb_image` is novel enough to keep, updating the reference frame if so."""
        gray = _to_grayscale(rgb_image)
        if self._reference is None or self._reference.shape != gray.shape:
            self._accept(gray)
            return True
        diff = np.abs(gray.astype(np.int16) - self._reference.astype(np.int16)).astype(np.uint8)
        entropy = _shannon_entropy(diff, bins=self.bins)
        if entropy < self.entropy_threshold:
            self.dropped_count += 1
            return False
        self._accept(gray)
        return True

    def _accept(self, gray: np.ndarray) -> None:
        self._reference = gray
        self.kept_count += 1
