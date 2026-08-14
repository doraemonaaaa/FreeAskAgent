"""Frame storage helpers for temporal memory."""

from __future__ import annotations

import copy
from typing import Any


def copy_image(image: Any) -> Any:
    """Copy one observation before retaining it in temporal history."""
    if image is None:
        raise ValueError("image must not be None")
    if isinstance(image, bytes):
        return image
    copier = getattr(image, "copy", None)
    return copier() if callable(copier) else copy.deepcopy(image)


__all__ = ("copy_image",)
