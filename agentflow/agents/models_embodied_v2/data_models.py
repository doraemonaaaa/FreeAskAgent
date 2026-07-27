"""Shared typed values used by the VLN policy and its memory modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import numpy as np


ErrorMode = Literal[
    "NONE", "WALL_STUCK", "TURN_OSCILLATION", "IN_PLACE_SPIN", "GET_NOWHERE"
]


class TemporalInputError(ValueError):
    pass


def _text(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise TemporalInputError(f"{label} must not be empty")
    return text


@dataclass(frozen=True, slots=True)
class Subgoal:
    subgoal_id: str
    description: str
    completion_criteria: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "subgoal_id", _text(self.subgoal_id, "subgoal_id"))
        object.__setattr__(self, "description", _text(self.description, "subgoal description"))
        object.__setattr__(
            self,
            "completion_criteria",
            _text(self.completion_criteria, "subgoal completion criteria"),
        )


@dataclass(frozen=True, slots=True)
class TemporalFrameInput:
    frame_id: int
    image: Any = field(repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.frame_id, bool) or not isinstance(self.frame_id, int) or self.frame_id < 1:
            raise TemporalInputError("frame_id must be a positive integer")
        if self.image is None:
            raise TemporalInputError("image must not be None")


@dataclass(frozen=True, slots=True)
class TemporalAnalysisRequest:
    subgoal: Subgoal
    frames: tuple[TemporalFrameInput, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.subgoal, Subgoal):
            raise TemporalInputError("subgoal must be a Subgoal")
        frames = tuple(self.frames)
        object.__setattr__(self, "frames", frames)
        if len(frames) != 8:
            raise TemporalInputError("request requires exactly eight frames")
        if any(not isinstance(frame, TemporalFrameInput) for frame in frames):
            raise TemporalInputError("frames must contain TemporalFrameInput values")
        ids = [frame.frame_id for frame in frames]
        if ids != sorted(set(ids)):
            raise TemporalInputError("frame IDs must be unique and increasing")


@dataclass(frozen=True, slots=True)
class CaptionResult:
    subgoal_id: str
    completed: bool
    error: bool
    error_mode: ErrorMode
    raw_response: str
    latency_ms: float

    def to_memory_text(self) -> str:
        state = "complete" if self.completed else "in progress"
        return f"Subgoal {self.subgoal_id}: {state}; error={self.error}; error_mode={self.error_mode}"


@dataclass(frozen=True, slots=True)
class TemporalCaptionerConfig:
    max_tokens: int = 48
    max_image_edge: int = 224
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if self.max_tokens < 8:
            raise ValueError("max_tokens must be at least 8")
        if self.max_image_edge < 32:
            raise ValueError("max_image_edge must be at least 32")


@dataclass(frozen=True, slots=True)
class TaskInput:
    observation: Any
    goal: str


@dataclass(frozen=True, slots=True)
class MemoryFrame:
    frame_id: int
    image: Any = field(repr=False)
    subgoal_id: str = ""


class TemporalEventKind(str, Enum):
    SUBGOAL_COMPLETED = "SUBGOAL_COMPLETED"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class TemporalEvent:
    kind: TemporalEventKind
    value: bool
    subgoal_id: str
    error_mode: ErrorMode = "NONE"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "value": self.value,
            "subgoal_id": self.subgoal_id,
            "error_mode": self.error_mode,
        }


@dataclass(frozen=True, slots=True)
class TemporalMemoryConfig:
    window_size: int = 8
    stationary_threshold: float = 0.02
    revisit_threshold: float = 0.05

    def __post_init__(self) -> None:
        if self.window_size != 8:
            raise ValueError("Temporal Memory uses a fixed eight-frame window")
        if not 0 < self.stationary_threshold < self.revisit_threshold < 1:
            raise ValueError("visual thresholds must satisfy 0 < stationary < revisit < 1")


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_matrix(cls, matrix: Any) -> "CameraIntrinsics":
        values = np.asarray(matrix, dtype=np.float64)
        if values.shape != (3, 3):
            raise ValueError("intrinsics must be a 3x3 pinhole camera matrix.")
        return cls(float(values[0, 0]), float(values[1, 1]), float(values[0, 2]), float(values[1, 2]))


@dataclass(frozen=True, slots=True)
class NavigationPoint:
    pixel_uv: tuple[int, int]
    depth_m: float
    camera_xyz: tuple[float, float, float]
    world_xyz: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class NavigationDecision:
    stop: bool
    point: NavigationPoint | None = None
    raw_response: str | None = None
