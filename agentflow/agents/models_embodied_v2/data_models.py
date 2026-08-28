"""Shared typed values used by the VLN policy and its memory modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Literal

import numpy as np


ErrorMode = Literal[
    "NONE", "WALL_STUCK", "TURN_OSCILLATION", "IN_PLACE_SPIN", "GET_NOWHERE"
]
LandmarkDirection = Literal["LEFT", "CENTER", "RIGHT", "UNKNOWN"]
LandmarkProximity = Literal["FAR", "NEAR", "AT", "UNKNOWN"]
DoorState = Literal[
    "NOT_APPLICABLE",
    "NOT_VISIBLE",
    "APPROACHING",
    "AT_THRESHOLD",
    "CROSSING",
    "CROSSED",
]
DoorCameraSide = Literal[
    "NOT_APPLICABLE",
    "UNKNOWN",
    "BEFORE_DOOR",
    "AT_DOOR",
    "AFTER_DOOR",
]
# What the actor is asking the controller to do with this step.  ``EXECUTION``
# is the pre-existing behaviour: commit to the returned waypoint.  ``PREVIEW``
# asks the runner for surrounding views before committing, and ``EXPLORATION``
# commits to a waypoint chosen to reveal the route rather than to advance along
# one already known.  A waypoint is returned in every mode, so a controller that
# cannot render surrounding views still has a usable action.
ActionMode = Literal["PREVIEW", "EXPLORATION", "EXECUTION"]
# One simulator turn primitive.  This must match Habitat's
# ``habitat.simulator.turn_angle``: a requested turn is executed as whole
# repeats of that primitive, so a mismatch would silently round every turn.
TURN_STEP_DEG = 15
# Keep explicit turns inside one short observation horizon.  The actor observes
# again after at most three Habitat turn primitives instead of committing to a
# long open-loop rotation that can pass a newly visible doorway.
MAX_TURN_DEG = 45


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
    translation_m: float = 0.0
    yaw_delta_deg: float = 0.0
    subgoal_path_length_m: float = 0.0
    landmark_visible: bool = False
    landmark_direction: LandmarkDirection = "UNKNOWN"
    landmark_proximity: LandmarkProximity = "UNKNOWN"
    landmark_passed: bool = False
    landmark_confidence: float = 0.0
    landmark_evidence: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.frame_id, bool) or not isinstance(self.frame_id, int) or self.frame_id < 1:
            raise TemporalInputError("frame_id must be a positive integer")
        if self.image is None:
            raise TemporalInputError("image must not be None")
        for name in (
            "translation_m",
            "yaw_delta_deg",
            "subgoal_path_length_m",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise TemporalInputError(f"{name} must be a finite number")
            object.__setattr__(self, name, float(value))
        if self.translation_m < 0:
            raise TemporalInputError("translation_m must not be negative")
        if self.subgoal_path_length_m < 0:
            raise TemporalInputError(
                "subgoal_path_length_m must not be negative"
            )
        for name in ("landmark_visible", "landmark_passed"):
            if not isinstance(getattr(self, name), bool):
                raise TemporalInputError(f"{name} must be a boolean")
        if self.landmark_direction not in (
            "LEFT",
            "CENTER",
            "RIGHT",
            "UNKNOWN",
        ):
            raise TemporalInputError("invalid landmark_direction")
        if self.landmark_proximity not in (
            "FAR",
            "NEAR",
            "AT",
            "UNKNOWN",
        ):
            raise TemporalInputError("invalid landmark_proximity")
        if (
            isinstance(self.landmark_confidence, bool)
            or not isinstance(self.landmark_confidence, (int, float))
            or not math.isfinite(float(self.landmark_confidence))
            or not 0.0 <= float(self.landmark_confidence) <= 1.0
        ):
            raise TemporalInputError(
                "landmark_confidence must be a number in [0, 1]"
            )
        object.__setattr__(
            self, "landmark_confidence", float(self.landmark_confidence)
        )
        if not isinstance(self.landmark_evidence, str):
            raise TemporalInputError("landmark_evidence must be a string")


@dataclass(frozen=True, slots=True)
class TemporalAnalysisRequest:
    subgoal: Subgoal
    frames: tuple[TemporalFrameInput, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.subgoal, Subgoal):
            raise TemporalInputError("subgoal must be a Subgoal")
        frames = tuple(self.frames)
        object.__setattr__(self, "frames", frames)
        if not frames:
            raise TemporalInputError("request requires at least one frame")
        if len(frames) > 8:
            raise TemporalInputError("request allows at most eight frames")
        if any(not isinstance(frame, TemporalFrameInput) for frame in frames):
            raise TemporalInputError("frames must contain TemporalFrameInput values")
        ids = [frame.frame_id for frame in frames]
        if ids != sorted(set(ids)):
            raise TemporalInputError("frame IDs must be unique and increasing")


def _confidence(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise TemporalInputError(f"{label} must be a number in [0, 1]")
    return float(value)


@dataclass(frozen=True, slots=True)
class SceneLandmark:
    """Current-frame state of the landmark named by the active subgoal."""

    visible: bool
    direction: LandmarkDirection
    proximity: LandmarkProximity
    passed: bool
    destination_dominant: bool
    confidence: float
    evidence: str
    u: int | None = None
    v: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.visible, bool) or not isinstance(self.passed, bool):
            raise TemporalInputError("landmark visibility fields must be boolean")
        if not isinstance(self.destination_dominant, bool):
            raise TemporalInputError("destination_dominant must be boolean")
        if self.direction not in ("LEFT", "CENTER", "RIGHT", "UNKNOWN"):
            raise TemporalInputError("invalid landmark direction")
        if self.proximity not in ("FAR", "NEAR", "AT", "UNKNOWN"):
            raise TemporalInputError("invalid landmark proximity")
        object.__setattr__(
            self,
            "confidence",
            _confidence(self.confidence, "landmark confidence"),
        )
        evidence = str(self.evidence or "").strip()
        if not evidence:
            raise TemporalInputError("landmark evidence must not be empty")
        object.__setattr__(self, "evidence", evidence)
        if not self.visible:
            object.__setattr__(self, "u", None)
            object.__setattr__(self, "v", None)
            if not self.passed and (
                self.direction != "UNKNOWN" or self.proximity != "UNKNOWN"
            ):
                raise TemporalInputError(
                    "invisible, unpassed landmark requires UNKNOWN state"
                )
        elif self.direction == "UNKNOWN" or self.proximity == "UNKNOWN":
            raise TemporalInputError(
                "visible landmark requires direction and proximity"
            )
        if (self.u is None) != (self.v is None):
            object.__setattr__(self, "u", None)
            object.__setattr__(self, "v", None)
        if self.u is not None and self.v is not None:
            if not 0 <= self.u <= 1000 or not 0 <= self.v <= 1000:
                object.__setattr__(self, "u", None)
                object.__setattr__(self, "v", None)


@dataclass(frozen=True, slots=True)
class FinalTargetEvidence:
    """Semantic final-destination evidence, independent of floor depth."""

    visible: bool
    proximity: LandmarkProximity
    confidence: float
    evidence: str

    def __post_init__(self) -> None:
        if not isinstance(self.visible, bool):
            raise TemporalInputError("final target visible must be boolean")
        if self.proximity not in ("FAR", "NEAR", "AT", "UNKNOWN"):
            raise TemporalInputError("invalid final target proximity")
        if not self.visible and self.proximity != "UNKNOWN":
            raise TemporalInputError(
                "invisible final target requires UNKNOWN proximity"
            )
        object.__setattr__(
            self,
            "confidence",
            _confidence(self.confidence, "final target confidence"),
        )
        object.__setattr__(self, "evidence", str(self.evidence or "").strip())


@dataclass(frozen=True, slots=True)
class SceneAnalysisRequest:
    """One bounded temporal window for a single scene-understanding call."""

    subgoal: Subgoal
    frames: tuple[TemporalFrameInput, ...]
    is_final_subgoal: bool = False
    next_subgoal: Subgoal | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.subgoal, Subgoal):
            raise TemporalInputError("subgoal must be a Subgoal")
        frames = tuple(self.frames)
        object.__setattr__(self, "frames", frames)
        if not 1 <= len(frames) <= 16:
            raise TemporalInputError(
                "scene request requires one to sixteen frames"
            )
        if any(not isinstance(frame, TemporalFrameInput) for frame in frames):
            raise TemporalInputError(
                "scene frames must contain TemporalFrameInput values"
            )
        ids = [frame.frame_id for frame in frames]
        if ids != sorted(set(ids)):
            raise TemporalInputError("scene frame IDs must be unique and increasing")
        if not isinstance(self.is_final_subgoal, bool):
            raise TemporalInputError("is_final_subgoal must be boolean")
        if self.next_subgoal is not None and not isinstance(
            self.next_subgoal, Subgoal
        ):
            raise TemporalInputError("next_subgoal must be a Subgoal or None")


@dataclass(frozen=True, slots=True)
class SceneAnalysisResult:
    """Validated perception candidates from one unified Captioner request."""

    subgoal_id: str
    landmark: SceneLandmark
    completed: bool
    completion_confidence: float
    completion_evidence: str
    door_state: DoorState
    door_camera_side: DoorCameraSide
    error: bool
    error_mode: ErrorMode
    error_confidence: float
    error_evidence: str
    final_target: FinalTargetEvidence
    raw_response: str
    latency_ms: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "subgoal_id", _text(self.subgoal_id, "subgoal_id"))
        if not isinstance(self.landmark, SceneLandmark):
            raise TemporalInputError("landmark must be SceneLandmark")
        if not isinstance(self.completed, bool) or not isinstance(self.error, bool):
            raise TemporalInputError("scene decisions must be boolean")
        if self.door_state not in (
            "NOT_APPLICABLE",
            "NOT_VISIBLE",
            "APPROACHING",
            "AT_THRESHOLD",
            "CROSSING",
            "CROSSED",
        ):
            raise TemporalInputError("invalid door state")
        if self.door_camera_side not in (
            "NOT_APPLICABLE",
            "UNKNOWN",
            "BEFORE_DOOR",
            "AT_DOOR",
            "AFTER_DOOR",
        ):
            raise TemporalInputError("invalid door camera side")
        object.__setattr__(
            self,
            "completion_confidence",
            _confidence(self.completion_confidence, "completion confidence"),
        )
        object.__setattr__(
            self,
            "error_confidence",
            _confidence(self.error_confidence, "error confidence"),
        )
        if self.error_mode not in (
            "NONE",
            "WALL_STUCK",
            "TURN_OSCILLATION",
            "IN_PLACE_SPIN",
            "GET_NOWHERE",
        ):
            raise TemporalInputError("invalid error mode")
        if self.error != (self.error_mode != "NONE"):
            raise TemporalInputError("error flag and error_mode disagree")
        if not isinstance(self.final_target, FinalTargetEvidence):
            raise TemporalInputError("final_target must be FinalTargetEvidence")
        object.__setattr__(
            self,
            "completion_evidence",
            str(self.completion_evidence or "").strip(),
        )
        object.__setattr__(
            self,
            "error_evidence",
            str(self.error_evidence or "").strip(),
        )
        if self.latency_ms < 0:
            raise TemporalInputError("latency_ms must not be negative")


@dataclass(frozen=True, slots=True)
class CaptionResult:
    subgoal_id: str
    completed: bool
    error: bool
    error_mode: ErrorMode
    raw_response: str
    latency_ms: float
    error_confidence: float = 0.0
    error_evidence: str = ""
    completion_confidence: float = 0.0
    completion_evidence: str = ""
    door_state: DoorState = "NOT_APPLICABLE"
    door_camera_side: DoorCameraSide = "NOT_APPLICABLE"
    landmark: SceneLandmark | None = None
    final_target: FinalTargetEvidence | None = None

    def to_memory_text(self) -> str:
        state = "complete" if self.completed else "in progress"
        return f"Subgoal {self.subgoal_id}: {state}; error={self.error}; error_mode={self.error_mode}"


@dataclass(frozen=True, slots=True)
class DualWindowCaptionResult:
    """Fused judgement from independent completion and error windows."""

    subgoal_id: str
    completed: bool
    error: bool
    error_mode: ErrorMode
    completion_window_size: int
    error_window_size: int
    completion_raw_response: str
    error_raw_response: str
    completion_latency_ms: float
    error_latency_ms: float
    latency_ms: float
    error_confidence: float = 0.0
    error_evidence: str = ""


@dataclass(frozen=True, slots=True)
class TemporalCaptionerConfig:
    max_tokens: int = 48
    max_image_edge: int = 224
    # Preview views are judged one at a time for a doorway's position, which
    # needs more pixels than the bounded multi-frame temporal history.
    preview_max_image_edge: int = 448
    temperature: float = 0.0
    enable_error_detection: bool = False
    min_error_detection_frames: int = 4

    def __post_init__(self) -> None:
        if self.max_tokens < 8:
            raise ValueError("max_tokens must be at least 8")
        if self.max_image_edge < 32:
            raise ValueError("max_image_edge must be at least 32")
        if self.preview_max_image_edge < 32:
            raise ValueError("preview_max_image_edge must be at least 32")
        if not isinstance(self.enable_error_detection, bool):
            raise TypeError("enable_error_detection must be a boolean")
        if not 4 <= self.min_error_detection_frames <= 8:
            raise ValueError(
                "min_error_detection_frames must be between 4 and 8"
            )


@dataclass(frozen=True, slots=True)
class TaskInput:
    observation: Any
    goal: str


@dataclass(frozen=True, slots=True)
class MemoryFrame:
    frame_id: int
    image: Any = field(repr=False)
    subgoal_id: str = ""
    translation_m: float = 0.0
    yaw_delta_deg: float = 0.0
    subgoal_path_length_m: float = 0.0
    landmark_visible: bool = False
    landmark_direction: LandmarkDirection = "UNKNOWN"
    landmark_proximity: LandmarkProximity = "UNKNOWN"
    landmark_passed: bool = False
    landmark_confidence: float = 0.0
    landmark_evidence: str = ""


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
    min_error_detection_frames: int = 4
    enable_error_detection: bool = False

    def __post_init__(self) -> None:
        if self.window_size != 8:
            raise ValueError("Temporal Memory uses a fixed eight-frame window")
        if not 0 < self.stationary_threshold < self.revisit_threshold < 1:
            raise ValueError("visual thresholds must satisfy 0 < stationary < revisit < 1")
        if not 3 <= self.min_error_detection_frames <= self.window_size:
            raise ValueError(
                "min_error_detection_frames must be between 3 and window_size"
            )
        if not isinstance(self.enable_error_detection, bool):
            raise TypeError("enable_error_detection must be a boolean")


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
    # True only when the actor verified the point against the floor plane;
    # False both for a fallback off the floor and when no camera height was
    # available to check.
    on_floor: bool = False


@dataclass(frozen=True, slots=True)
class PreviewView:
    """One surrounding view rendered in answer to a PREVIEW decision.

    ``yaw_deg`` is the heading offset from the agent's current facing.  Depth,
    intrinsics, and the camera transform belong to this view rather than to the
    forward camera, so a pixel chosen inside it back-projects in its own frame
    with no change to the RGB-D layer.
    """

    yaw_deg: float
    rgb: Any = field(repr=False)
    depth: Any = field(default=None, repr=False)
    intrinsics: Any = None
    camera_to_world: Any = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.yaw_deg, bool)
            or not isinstance(self.yaw_deg, (int, float))
            or not math.isfinite(float(self.yaw_deg))
        ):
            raise ValueError("yaw_deg must be a finite number")
        object.__setattr__(self, "yaw_deg", float(self.yaw_deg))
        if self.rgb is None:
            raise ValueError("a preview view requires an RGB image")

    @property
    def is_navigable(self) -> bool:
        """True when this view carries everything a waypoint needs."""
        return (
            self.depth is not None
            and self.intrinsics is not None
            and self.camera_to_world is not None
        )


@dataclass(frozen=True, slots=True)
class PreviewSelection:
    """Which held preview view and floor target the actor should act on.

    ``u`` and ``v`` use the same normalized 0..1000 image coordinates as the
    ordinary waypoint contract. Keeping the target in the selected camera
    view avoids silently replacing an off-centre doorway with image centre.
    """

    view_index: int
    u: int = 500
    v: int = 750
    confidence: float = 0.0
    evidence: str = ""

    def __post_init__(self) -> None:
        if (
            isinstance(self.view_index, bool)
            or not isinstance(self.view_index, int)
            or self.view_index < 0
        ):
            raise ValueError("view_index must be a non-negative integer")
        for name in ("u", "v"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= 1000
            ):
                raise ValueError(f"{name} must be an integer in [0, 1000]")
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not math.isfinite(float(self.confidence))
            or not 0.0 <= float(self.confidence) <= 1.0
        ):
            raise ValueError("confidence must be a number in [0, 1]")
        object.__setattr__(self, "confidence", float(self.confidence))
        if not isinstance(self.evidence, str):
            raise ValueError("evidence must be a string")


@dataclass(frozen=True, slots=True)
class NavigationDecision:
    stop: bool
    point: NavigationPoint | None = None
    raw_response: str | None = None
    # Defaults to EXECUTION so a controller written before action modes existed
    # keeps its behaviour of committing to ``point`` unconditionally.
    action_mode: ActionMode = "EXECUTION"
    # Signed whole turn primitives, positive to the right.  Set instead of
    # ``point`` when the step rotates in place.
    turn_deg: int | None = None

    def __post_init__(self) -> None:
        if self.turn_deg is None:
            return
        # Checked at the boundary the controller reads: it divides by the turn
        # primitive to count repeats, so a value that is not a whole multiple
        # would be silently truncated into a shorter turn than was asked for.
        if isinstance(self.turn_deg, bool) or not isinstance(
            self.turn_deg, int
        ):
            raise ValueError("turn_deg must be an integer")
        if self.turn_deg == 0:
            raise ValueError("a turn must not be zero degrees")
        if self.turn_deg % TURN_STEP_DEG:
            raise ValueError(
                f"turn_deg must be a multiple of {TURN_STEP_DEG} degrees"
            )
        if abs(self.turn_deg) > MAX_TURN_DEG:
            raise ValueError(
                f"turn_deg must be within +/-{MAX_TURN_DEG} degrees"
            )
        if self.point is not None:
            raise ValueError("a decision is either a turn or a point")
        if self.stop:
            raise ValueError("a stopping decision cannot also turn")
