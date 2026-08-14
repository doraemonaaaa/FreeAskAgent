"""Preview state deliberately kept outside temporal frame history."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ...data_models import PreviewSelection


@dataclass
class PreviewStore:
    """Views and their selection for one navigation step."""

    views: tuple[Any, ...] = ()
    selection: Optional[PreviewSelection] = None
    error: Optional[str] = None

    def clear(self) -> None:
        self.views = ()
        self.selection = None
        self.error = None
