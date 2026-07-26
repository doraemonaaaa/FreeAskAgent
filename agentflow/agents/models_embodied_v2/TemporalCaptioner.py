"""Timestamp-aligned video understanding for temporal VLN memory.

The model-facing input is an ordered storyboard of RGB frames with absolute
episode/video timestamps.  Navigation actions, motion evidence, topology
signals, and progress signals remain separate typed inputs so the foundation
model cannot silently treat a command as an observed effect.

`TemporalCaptioner` is deliberately stateless: callers construct one
`TemporalWindow`, call :meth:`analyze`, and decide whether/how to commit the
returned :class:`TemporalMemoryRecord` to memory.

Typical agent integration::

    record = captioner.analyze(
        TemporalWindow(
            start_seconds=window_start,
            end_seconds=window_end,
            frames=timestamped_frames,
            actions=timestamped_actions,
            motion=motion_evidence,
            topology=topology_evidence,
            progress=progress_evidence,
        )
    )
    temporal_memory.store(record.to_memory_dict())

For recorded episodes, :meth:`TemporalCaptioner.analyze_video` performs
independent sparse scene sampling and dense optical-flow sampling first.
"""

from __future__ import annotations

import io
import json
import math
import re
import statistics
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from numbers import Integral
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Optional, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field


DEFAULT_MODEL_PATH = "models/Qwen3-VL-8B-Instruct"
ACTION_TOKENS = ("FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP")
ERROR_MODES = (
    "collision",
    "repeated_visit",
    "motion_oscillation",
    "get_nowhere",
    "action_execution_mismatch",
)
ErrorMode = Literal[
    "collision",
    "repeated_visit",
    "motion_oscillation",
    "get_nowhere",
    "action_execution_mismatch",
]
ActionMatch = Literal["MATCH", "MISMATCH", "UNCERTAIN"]
MIN_STEP_WINDOW_SIZE = 2
MAX_STEP_WINDOW_SIZE = 8
SCENE_DYNAMIC_TERMS = (
    "FORWARD",
    "TURN_LEFT",
    "TURN_RIGHT",
    "STOP",
    "MOVE_FORWARD",
    "COLLISION",
    "REPEATED_VISIT",
    "MOTION_OSCILLATION",
    "GET_NOWHERE",
    "MOVING",
    "TURNING",
    "ROTATING",
    "STATIONARY",
    "STOPPED",
    "COMMAND",
    "COLLID",
    "OSCILLAT",
    "PROGRESS",
    "PANNING",
    "SWEEPING",
    "DRIVING",
    "WALKING",
    "APPROACHING",
    "ENTERING",
    "LEAVING",
    "RETURNING",
    "REVISITING",
    "HIT ",
    "前进",
    "后退",
    "左转",
    "右转",
    "转向",
    "旋转",
    "转动",
    "移动",
    "行走",
    "行进",
    "行驶",
    "驶向",
    "驶去",
    "扫向",
    "扫过",
    "走向",
    "前往",
    "进入",
    "离开",
    "穿过",
    "返回",
    "再次来到",
    "重新来到",
    "靠近",
    "接近",
    "远离",
    "越来越近",
    "越来越远",
    "到达",
    "拐弯",
    "掉头",
    "静止",
    "停下",
    "停滞",
    "振荡",
    "摆动",
    "碰撞",
    "撞到",
    "撞上",
    "受阻",
    "卡住",
    "回访",
    "进展",
    "命令",
    "执行",
    "回扫",
    "回望",
    "环视",
)


class TemporalCaptionerError(RuntimeError):
    """Base error for temporal video understanding."""


class TemporalInputError(TemporalCaptionerError, ValueError):
    """The temporal window is invalid and inference was not attempted."""


class TemporalInferenceError(TemporalCaptionerError):
    """The video-understanding backend failed."""


class TemporalOutputError(TemporalCaptionerError, ValueError):
    """The foundation model returned invalid or temporally inconsistent output."""


class CameraMotion(str, Enum):
    """Observed camera motion in the VLN action space."""

    STATIONARY = "STATIONARY"
    FORWARD = "FORWARD"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    OSCILLATING_TURN = "OSCILLATING_TURN"
    UNKNOWN = "UNKNOWN"


class MultimodalEngine(Protocol):
    def __call__(
        self,
        content: list[Any],
        *,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        response_format: Optional[type[BaseModel]] = None,
        **kwargs: Any,
    ) -> str: ...


def _finite_non_negative(value: float, field_name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise TemporalInputError(f"{field_name} must be finite and non-negative")
    return value


def _normalize_action(action: str) -> str:
    normalized = str(action).strip().upper()
    aliases = {
        "MOVE_FORWARD": "FORWARD",
        "LEFT": "TURN_LEFT",
        "RIGHT": "TURN_RIGHT",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in ACTION_TOKENS:
        raise TemporalInputError(
            f"Unsupported action {action!r}; expected one of {ACTION_TOKENS}"
        )
    return normalized


@dataclass(frozen=True, slots=True)
class TimestampedFrame:
    """One RGB observation at an absolute episode/video timestamp."""

    timestamp_seconds: float
    image: Any = field(repr=False)
    step_id: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "timestamp_seconds",
            _finite_non_negative(self.timestamp_seconds, "frame timestamp"),
        )
        if self.step_id is not None and self.step_id < 0:
            raise TemporalInputError("step_id must be non-negative")


@dataclass(frozen=True, slots=True)
class TimedAction:
    """A command issued at `timestamp_seconds`; it affects later frames only.

    For phase attribution, the commanded state lasts until the next command
    timestamp (or the temporal-window end).  Whether the command actually
    executed is determined independently from motion/odometry evidence.
    """

    timestamp_seconds: float
    action: str
    step_id: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "timestamp_seconds",
            _finite_non_negative(self.timestamp_seconds, "action timestamp"),
        )
        object.__setattr__(self, "action", _normalize_action(self.action))
        if self.step_id is not None and self.step_id < 0:
            raise TemporalInputError("step_id must be non-negative")


@dataclass(frozen=True, slots=True)
class MotionSignal:
    """Observed execution result over one absolute time interval.

    Positive `scene_flow_dx_fraction` means image content moved to the right.
    In a navigation action space without strafing this is evidence that the
    camera turned left; negative flow indicates a right turn.
    """

    start_seconds: float
    end_seconds: float
    camera_motion: CameraMotion = CameraMotion.UNKNOWN
    scene_flow_dx_fraction: Optional[float] = None
    scene_flow_magnitude_fraction: Optional[float] = None
    delta_forward_meters: Optional[float] = None
    delta_yaw_left_degrees: Optional[float] = None
    collision: Optional[bool] = None
    confidence: float = 0.0
    source: str = "unknown"
    quality_note: Optional[str] = None

    def __post_init__(self) -> None:
        start = _finite_non_negative(self.start_seconds, "motion start")
        end = _finite_non_negative(self.end_seconds, "motion end")
        if end <= start:
            raise TemporalInputError("motion end must be greater than motion start")
        object.__setattr__(self, "start_seconds", start)
        object.__setattr__(self, "end_seconds", end)
        if not isinstance(self.camera_motion, CameraMotion):
            try:
                object.__setattr__(
                    self, "camera_motion", CameraMotion(str(self.camera_motion))
                )
            except ValueError as exc:
                raise TemporalInputError(
                    f"Invalid camera motion: {self.camera_motion!r}"
                ) from exc
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise TemporalInputError("motion confidence must be in [0, 1]")
        object.__setattr__(self, "confidence", confidence)
        for name in (
            "scene_flow_dx_fraction",
            "scene_flow_magnitude_fraction",
            "delta_forward_meters",
            "delta_yaw_left_degrees",
        ):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise TemporalInputError(f"{name} must be finite when provided")
        if (
            self.camera_motion == CameraMotion.UNKNOWN
            and self.scene_flow_dx_fraction is None
            and self.scene_flow_magnitude_fraction is None
            and self.delta_forward_meters is None
            and self.delta_yaw_left_degrees is None
            and self.collision is None
            and not self.quality_note
        ):
            raise TemporalInputError(
                "motion signal must contain at least one observed measurement"
            )
        if not str(self.source).strip():
            raise TemporalInputError("motion source must not be empty")

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "camera_motion": self.camera_motion.value,
            "scene_flow_dx_fraction": self.scene_flow_dx_fraction,
            "scene_flow_magnitude_fraction": self.scene_flow_magnitude_fraction,
            "delta_forward_meters": self.delta_forward_meters,
            "delta_yaw_left_degrees": self.delta_yaw_left_degrees,
            "collision": self.collision,
            "confidence": self.confidence,
            "source": self.source,
            "quality_note": self.quality_note,
        }


@dataclass(frozen=True, slots=True)
class TopologySignal:
    timestamp_seconds: float
    node_id: Optional[str] = None
    visit_count: Optional[int] = None
    distance_to_goal_meters: Optional[float] = None
    source: str = "unknown"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "timestamp_seconds",
            _finite_non_negative(self.timestamp_seconds, "topology timestamp"),
        )
        if (
            self.node_id is None
            and self.visit_count is None
            and self.distance_to_goal_meters is None
        ):
            raise TemporalInputError(
                "topology signal must provide node_id, visit_count, or distance"
            )
        if self.visit_count is not None and self.visit_count < 1:
            raise TemporalInputError("visit_count must be at least 1")
        if (
            self.distance_to_goal_meters is not None
            and self.distance_to_goal_meters < 0
        ):
            raise TemporalInputError("distance_to_goal_meters must be non-negative")
        if not str(self.source).strip():
            raise TemporalInputError("topology source must not be empty")

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "timestamp_seconds": self.timestamp_seconds,
            "node_id": self.node_id,
            "visit_count": self.visit_count,
            "distance_to_goal_meters": self.distance_to_goal_meters,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class ProgressSignals:
    net_displacement_meters: Optional[float] = None
    new_landmarks_count: Optional[int] = None
    new_topological_nodes_count: Optional[int] = None
    completed_subgoals_count: Optional[int] = None
    no_progress_steps: Optional[int] = None

    def __post_init__(self) -> None:
        if (
            self.net_displacement_meters is not None
            and self.net_displacement_meters < 0
        ):
            raise TemporalInputError("net_displacement_meters must be non-negative")
        for name in (
            "new_landmarks_count",
            "new_topological_nodes_count",
            "completed_subgoals_count",
            "no_progress_steps",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise TemporalInputError(f"{name} must be non-negative")

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "net_displacement_meters": self.net_displacement_meters,
            "new_landmarks_count": self.new_landmarks_count,
            "new_topological_nodes_count": self.new_topological_nodes_count,
            "completed_subgoals_count": self.completed_subgoals_count,
            "no_progress_steps": self.no_progress_steps,
        }


@dataclass(frozen=True, slots=True)
class TemporalStepInput:
    """One completed VLN transition, aligned to its post-action observation.

    `image` is the observation *after* `commanded_action`.  Motion is measured
    from the pre-action observation to this image.  Keeping these fields in one
    immutable object prevents the model adapter from shifting an action onto
    the preceding frame.
    """

    step_id: int
    commanded_action: str
    post_timestamp_seconds: float
    image: Any = field(repr=False)
    motion: MotionSignal
    observed_motion: CameraMotion = CameraMotion.UNKNOWN
    action_match: ActionMatch = "UNCERTAIN"
    collision: Optional[bool] = None
    topology_node_id: Optional[str] = None
    is_new_node: Optional[bool] = None
    is_revisit: Optional[bool] = None
    distance_to_goal_meters: Optional[float] = None
    newly_completed_subgoals: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.step_id, bool)
            or not isinstance(self.step_id, Integral)
            or self.step_id < 0
        ):
            raise TemporalInputError("step_id must be non-negative")
        object.__setattr__(self, "step_id", int(self.step_id))
        object.__setattr__(
            self, "commanded_action", _normalize_action(self.commanded_action)
        )
        post_timestamp = _finite_non_negative(
            self.post_timestamp_seconds, "post-action timestamp"
        )
        object.__setattr__(self, "post_timestamp_seconds", post_timestamp)
        if self.image is None:
            raise TemporalInputError("post-action image must not be None")
        if not isinstance(self.observed_motion, CameraMotion):
            try:
                object.__setattr__(
                    self,
                    "observed_motion",
                    CameraMotion(str(self.observed_motion)),
                )
            except ValueError as exc:
                raise TemporalInputError(
                    f"Invalid observed motion: {self.observed_motion!r}"
                ) from exc
        action_match = str(self.action_match).strip().upper()
        if action_match not in {"MATCH", "MISMATCH", "UNCERTAIN"}:
            raise TemporalInputError(
                "action_match must be MATCH, MISMATCH, or UNCERTAIN"
            )
        object.__setattr__(self, "action_match", action_match)
        if not isinstance(self.motion, MotionSignal):
            raise TemporalInputError("motion must be a MotionSignal")
        if abs(self.motion.end_seconds - post_timestamp) > 0.002:
            raise TemporalInputError(
                "motion.end_seconds must equal post_timestamp_seconds"
            )
        if self.distance_to_goal_meters is not None:
            distance = float(self.distance_to_goal_meters)
            if not math.isfinite(distance) or distance < 0:
                raise TemporalInputError(
                    "distance_to_goal_meters must be finite and non-negative"
                )
            object.__setattr__(self, "distance_to_goal_meters", distance)
        if self.topology_node_id is not None:
            node_id = str(self.topology_node_id).strip()
            if not node_id:
                raise TemporalInputError("topology_node_id must not be empty")
            object.__setattr__(self, "topology_node_id", node_id)
        object.__setattr__(
            self,
            "newly_completed_subgoals",
            tuple(
                str(subgoal).strip()
                for subgoal in self.newly_completed_subgoals
                if str(subgoal).strip()
            ),
        )

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "commanded_action": self.commanded_action,
            "post_timestamp_seconds": self.post_timestamp_seconds,
            "observed_motion": self.observed_motion.value,
            "action_match": self.action_match,
            "motion": self.motion.to_prompt_dict(),
            "collision": self.collision,
            "topology_node_id": self.topology_node_id,
            "is_new_node": self.is_new_node,
            "is_revisit": self.is_revisit,
            "distance_to_goal_meters": self.distance_to_goal_meters,
            "newly_completed_subgoals": list(self.newly_completed_subgoals),
        }


@dataclass(frozen=True, slots=True)
class TemporalAnalysisRequest:
    """A short sequence of completed transitions for step-aligned understanding."""

    episode_id: Optional[str]
    goal: Optional[str]
    steps: tuple[TemporalStepInput, ...]
    progress: ProgressSignals = field(default_factory=ProgressSignals)
    reverse_retrace_similarity: Optional[float] = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(self, "notes", tuple(self.notes))
        if not MIN_STEP_WINDOW_SIZE <= len(self.steps) <= MAX_STEP_WINDOW_SIZE:
            raise TemporalInputError(
                "TemporalAnalysisRequest requires between "
                f"{MIN_STEP_WINDOW_SIZE} and {MAX_STEP_WINDOW_SIZE} "
                "completed steps"
            )
        if not isinstance(self.progress, ProgressSignals):
            raise TemporalInputError("progress must be ProgressSignals")
        previous_id: Optional[int] = None
        previous_timestamp: Optional[float] = None
        for step in self.steps:
            if not isinstance(step, TemporalStepInput):
                raise TemporalInputError(
                    "steps must contain only TemporalStepInput values"
                )
            if previous_id is not None and step.step_id <= previous_id:
                raise TemporalInputError(
                    "step_id values must be unique and strictly increasing"
                )
            if (
                previous_timestamp is not None
                and step.post_timestamp_seconds <= previous_timestamp
            ):
                raise TemporalInputError(
                    "post-action timestamps must be strictly increasing"
                )
            if (
                previous_timestamp is not None
                and step.motion.start_seconds < previous_timestamp - 0.002
            ):
                raise TemporalInputError(
                    "step transition intervals must not overlap"
                )
            previous_id = step.step_id
            previous_timestamp = step.post_timestamp_seconds
        if self.reverse_retrace_similarity is not None:
            similarity = float(self.reverse_retrace_similarity)
            if not math.isfinite(similarity) or not -1 <= similarity <= 1:
                raise TemporalInputError(
                    "reverse_retrace_similarity must be in [-1, 1]"
                )
            object.__setattr__(self, "reverse_retrace_similarity", similarity)


@dataclass(frozen=True, slots=True)
class TemporalWindow:
    """All evidence for one closed-open interval `[start, end)`.

    For rolling windows, callers should repeat a command that remains active
    from the previous window at `start_seconds`; commands outside the window
    are intentionally rejected to prevent clock-origin mistakes.
    """

    start_seconds: float
    end_seconds: float
    frames: tuple[TimestampedFrame, ...]
    actions: tuple[TimedAction, ...] = ()
    motion: tuple[MotionSignal, ...] = ()
    topology: tuple[TopologySignal, ...] = ()
    progress: ProgressSignals = field(default_factory=ProgressSignals)
    reverse_retrace_similarity: Optional[float] = None
    goal: Optional[str] = None
    episode_id: Optional[str] = None
    notes: tuple[str, ...] = ()
    timestamp_semantics: str = "episode_replay_seconds"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "start_seconds",
            _finite_non_negative(self.start_seconds, "window start"),
        )
        object.__setattr__(
            self,
            "end_seconds",
            _finite_non_negative(self.end_seconds, "window end"),
        )
        if self.end_seconds <= self.start_seconds:
            raise TemporalInputError("window end must be greater than window start")
        for name in ("frames", "actions", "motion", "topology", "notes"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.reverse_retrace_similarity is not None:
            similarity = float(self.reverse_retrace_similarity)
            if not math.isfinite(similarity) or not -1 <= similarity <= 1:
                raise TemporalInputError(
                    "reverse_retrace_similarity must be in [-1, 1]"
                )
            object.__setattr__(self, "reverse_retrace_similarity", similarity)


class TimeSpan(BaseModel):
    # Short aliases keep the model-facing JSON small while callers continue to
    # use the descriptive Python attribute names.  `populate_by_name` also
    # accepts the long names from alternative backends and tests.
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    start_seconds: float = Field(alias="start")
    end_seconds: float = Field(alias="end")


class TemporalPhase(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    interval: TimeSpan = Field(alias="time")
    scene: str
    commanded_activity: Literal[
        "FORWARD",
        "TURN_LEFT",
        "TURN_RIGHT",
        "STOP",
        "NONE",
        "MIXED",
        "UNKNOWN",
    ] = Field(default="NONE", alias="command")
    observed_motion: CameraMotion = Field(alias="motion")
    progress: Literal["PROGRESSING", "STALLED", "REGRESSING", "UNCERTAIN"]
    evidence_timestamps_seconds: list[float] = Field(alias="evidence")
    confidence: float = Field(ge=0.0, le=1.0)


class ErrorAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    verdict: Literal["PRESENT", "ABSENT", "UNCERTAIN"]
    confidence: float = Field(ge=0.0, le=1.0)
    interval: Optional[TimeSpan] = Field(default=None, alias="time")
    evidence_timestamps_seconds: list[float] = Field(
        default_factory=list,
        alias="evidence",
    )
    reason: str
    source: Literal["MODEL", "RULE", "FUSED"] = "MODEL"


def _unknown_action_execution_mismatch() -> ErrorAssessment:
    return ErrorAssessment(
        verdict="UNCERTAIN",
        confidence=0.5,
        interval=None,
        evidence_timestamps_seconds=[],
        reason="没有逐步命令—执行对齐证据。",
        source="MODEL",
    )


class ErrorAssessments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    collision: ErrorAssessment
    repeated_visit: ErrorAssessment
    motion_oscillation: ErrorAssessment
    get_nowhere: ErrorAssessment
    action_execution_mismatch: ErrorAssessment = Field(
        default_factory=_unknown_action_execution_mismatch
    )


class TemporalStepErrorAssessments(ErrorAssessments):
    """Step-mode schema requires an explicit fifth error assessment."""

    action_execution_mismatch: ErrorAssessment


class TemporalStepCaption(BaseModel):
    """One model caption grounded back onto caller-owned step evidence."""

    model_config = ConfigDict(extra="forbid")

    step_id: int = Field(ge=0)
    scene_after_action: str
    visible_landmarks: list[str] = Field(default_factory=list)
    visual_perceived_action: CameraMotion
    visual_error_clues: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    # These fields are accepted from the model only so that an untrusted
    # backend cannot smuggle conflicting alignment metadata into free text.
    # `analyze_steps` always overwrites them with TemporalStepInput values.
    commanded_action: Optional[
        Literal["FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"]
    ] = None
    post_timestamp_seconds: Optional[float] = None
    observed_motion: CameraMotion = CameraMotion.UNKNOWN
    action_match: ActionMatch = "UNCERTAIN"
    collision: Optional[bool] = None


class TemporalStepCaptionPayload(BaseModel):
    """Parsed step payload enriched with caller-owned alignment fields."""

    model_config = ConfigDict(extra="forbid")

    latest_scene: str
    scene_summary: str
    overall_progress: Literal[
        "PROGRESSING", "STALLED", "REGRESSING", "UNCERTAIN"
    ]
    step_captions: list[TemporalStepCaption]
    errors: TemporalStepErrorAssessments


class TemporalStepModelCaption(BaseModel):
    """Minimal model-facing caption; typed alignment stays caller-owned."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    step_id: int = Field(ge=0, alias="i")
    scene_after_action: str = Field(alias="c")
    visible_landmarks: list[str] = Field(default_factory=list, alias="l")
    visual_perceived_action: CameraMotion = Field(alias="m")
    visual_error_clues: list[str] = Field(default_factory=list, alias="e")


class TemporalStepModelErrorHint(BaseModel):
    """One concise, visually grounded positive error hint."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    mode: ErrorMode = Field(alias="m")
    step_ids: list[int] = Field(alias="i", min_length=1)


class TemporalStepModelPayload(BaseModel):
    """Latency-oriented wire schema supplied to the foundation model."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    overall_progress: Literal[
        "PROGRESSING", "STALLED", "REGRESSING", "UNCERTAIN"
    ] = Field(alias="p")
    step_captions: list[TemporalStepModelCaption] = Field(alias="s")
    error_hints: list[TemporalStepModelErrorHint] = Field(
        default_factory=list,
        alias="x",
    )


class TemporalCaptionPayload(BaseModel):
    """Strict schema requested from the video-understanding model."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    latest_scene: str = Field(alias="latest")
    scene_summary: str = Field(alias="scene")
    activity_summary: str = Field(alias="activity")
    overall_progress: Literal[
        "PROGRESSING", "STALLED", "REGRESSING", "UNCERTAIN"
    ] = Field(alias="progress")
    phases: list[TemporalPhase]
    errors: ErrorAssessments


class TemporalMemoryRecord(BaseModel):
    """Validated result safe to hand to a Temporal Memory implementation."""

    model_config = ConfigDict(extra="forbid")

    episode_id: Optional[str]
    window: TimeSpan
    frame_timestamps_seconds: list[float]
    action_timeline: list[dict[str, Any]]
    motion_evidence: list[dict[str, Any]]
    observed_motion_sequence: list[CameraMotion]
    reverse_retrace_similarity: Optional[float]
    latest_scene: str
    scene_summary: str
    activity_summary: str
    overall_progress: Literal["PROGRESSING", "STALLED", "REGRESSING", "UNCERTAIN"]
    phases: list[TemporalPhase]
    step_captions: list[TemporalStepCaption] = Field(default_factory=list)
    errors: ErrorAssessments
    model_latency_ms: float
    latency_budget_ms: float
    latency_budget_met: bool
    raw_response: str = Field(exclude=True)
    peak_gpu_memory_bytes: Optional[int] = None

    def to_memory_text(self) -> str:
        present = [
            mode
            for mode in ERROR_MODES
            if getattr(self.errors, mode).verdict == "PRESENT"
        ]
        errors = ", ".join(present) if present else "none confirmed"
        step_timeline = ""
        if self.step_captions:
            step_timeline = " | Step scenes: " + "; ".join(
                f"{caption.step_id}:{caption.scene_after_action}"
                for caption in self.step_captions
            )
        return (
            f"Temporal window {self.window.start_seconds:.3f}-"
            f"{self.window.end_seconds:.3f}s | "
            f"Scene: {self.scene_summary} | "
            f"Motion evidence: "
            f"{' -> '.join(motion.value for motion in self.observed_motion_sequence) or 'UNKNOWN'} | "
            f"Activity: {self.activity_summary} | "
            f"Progress: {self.overall_progress} | "
            f"Errors: {errors} | Latest: {self.latest_scene}"
            f"{step_timeline}"
        )

    def to_memory_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"raw_response"})


@dataclass(frozen=True, slots=True)
class TemporalCaptionerConfig:
    max_frames: int = 12
    # The live step path uses a terse schema so three captions normally finish
    # far below this ceiling. Legacy phase mode keeps its independent limit.
    max_tokens: int = 768
    step_max_tokens: int = 128
    max_image_edge: int = 128
    include_step_json_schema: bool = False
    temperature: float = 0.0
    caption_fps: float = 2.0
    motion_fps: float = 10.0
    max_motion_frames: int = 160
    maximum_flow_interval_seconds: float = 0.2
    stationary_flow_fraction: float = 0.0015
    turning_flow_fraction: float = 0.003
    minimum_horizontal_coherence: float = 0.55
    minimum_motion_phase_seconds: float = 0.2
    maximum_unknown_motion_gap_seconds: float = 0.8
    oscillation_retrace_threshold: float = 0.9
    minimum_motion_rule_confidence: float = 0.5
    get_nowhere_steps: int = 3
    minimum_definite_error_confidence: float = 0.5
    inference_latency_budget_ms: float = 5000.0
    goal_distance_progress_epsilon_meters: float = 0.05
    evidence_time_tolerance_seconds: float = 0.002

    def __post_init__(self) -> None:
        if self.max_frames < 2:
            raise TemporalInputError("max_frames must be at least 2")
        if self.max_tokens <= 0:
            raise TemporalInputError("max_tokens must be positive")
        if self.step_max_tokens <= 0:
            raise TemporalInputError("step_max_tokens must be positive")
        if self.max_image_edge <= 0:
            raise TemporalInputError("max_image_edge must be positive")
        if self.caption_fps <= 0 or self.motion_fps <= 0:
            raise TemporalInputError("caption_fps and motion_fps must be positive")
        if self.max_motion_frames < 2:
            raise TemporalInputError("max_motion_frames must be at least 2")
        if self.maximum_flow_interval_seconds <= 0:
            raise TemporalInputError(
                "maximum_flow_interval_seconds must be positive"
            )
        if self.maximum_unknown_motion_gap_seconds < 0:
            raise TemporalInputError(
                "maximum_unknown_motion_gap_seconds must be non-negative"
            )
        if not 0 <= self.minimum_definite_error_confidence <= 1:
            raise TemporalInputError(
                "minimum_definite_error_confidence must be in [0, 1]"
            )
        if not 0 <= self.minimum_motion_rule_confidence <= 1:
            raise TemporalInputError(
                "minimum_motion_rule_confidence must be in [0, 1]"
            )
        if self.inference_latency_budget_ms <= 0:
            raise TemporalInputError(
                "inference_latency_budget_ms must be positive"
            )
        if self.goal_distance_progress_epsilon_meters < 0:
            raise TemporalInputError(
                "goal_distance_progress_epsilon_meters must be non-negative"
            )


TEMPORAL_CAPTIONER_SYSTEM_PROMPT = """You are the Video Understanding Foundation
Model for a visual-language navigation robot. Inputs are RGB FRAME observations
in chronological order. Every FRAME marker contains an absolute episode/video
timestamp; never renumber timestamps relative to the current window.

Separate these concepts:
1. commanded_activity: what the action token requested;
2. observed_motion: what motion/odometry/optical-flow evidence says happened;
3. scene: only visible and temporally supported facts.

Never infer commanded_activity from images or observed_motion. If no input
action event overlaps a phase, commanded_activity is NONE. The caller verifies
this field deterministically from the action timeline.

Signal priority is collision sensor / odometry / topology, then action-aligned
motion evidence, then visual inference. An action is issued at its timestamp and
can only affect later frames. In this VLN action space there is no lateral
strafing: coherent scene flow to the right normally means camera TURN_LEFT, and
scene flow to the left means camera TURN_RIGHT. Never call horizontal image flow
"lateral movement". If evidence is insufficient, use UNKNOWN or UNCERTAIN.

Error definitions:
- collision: explicit contact, or repeated FORWARD commands with a nearby
  obstacle and no expected motion. Merely seeing a wall is not collision.
- repeated_visit: leaving and later returning to the same place/topological
  node. Seeing the same view during an in-place turn is not enough.
- motion_oscillation: repeated/retraced opposite turns with low net progress.
  A single small correction is not oscillation.
- get_nowhere: enough actions/steps pass without spatial, landmark,
  topological-node, subgoal, or goal-distance progress.
- action_execution_mismatch: a command and its observed execution disagree.
  This legacy window may lack one-to-one step evidence; use UNCERTAIN then.

Describe a concise phase timeline, the latest scene, activity, progress, and all
five error modes. Every PRESENT error must cite timestamps and an interval.

Keep the wire response compact so it cannot be truncated:
- output one-line JSON without Markdown or indentation;
- use the schema's short property names exactly;
- omit optional `source` and `command` properties (the caller fills them);
- latest must describe visible objects/landmarks in the final frame;
- scene must name stable visible objects/landmarks rather than only saying
  "looking around";
- activity must state the timestamped motion phases and what was being viewed;
- latest <= 45 Chinese characters, scene <= 80, activity <= 80;
- use at most four phases, at most three evidence timestamps per phase/error;
- each phase scene <= 40 characters and each error reason <= 36 characters;
- for ABSENT/UNCERTAIN use time=null and an empty evidence list unless direct
  evidence needs to be cited.
- confidence is confidence in the verdict. Missing evidence means UNCERTAIN,
  never ABSENT with confidence 0.

Return only a JSON object conforming to the supplied schema. Use Chinese for
free-text descriptions and keep enum values exactly as defined."""


STEP_ALIGNED_SYSTEM_PROMPT = """You are the Video Understanding Foundation Model
for a visual-language navigation robot. You receive a short sequence of
completed navigation steps. Every STEP marker is immediately followed by the
RGB observation captured after that step's command. Never shift a command to
the preceding or following image.

For each step, distinguish:
1. commanded_action: the caller-owned command that was sent;
2. observed_motion/action_match: caller-owned odometry or optical-flow facts;
3. visual_perceived_action: what consecutive post-action images suggest;
4. scene_after_action: visible scene and landmarks in that step's image.

Signal priority is collision sensor / odometry / topology, then optical flow,
then visual inference. Never claim that a command succeeded merely because it
was issued. UNKNOWN and UNCERTAIN are required when evidence is missing.

Return compact one-line JSON using the short fields shown in the user prompt.
Return one caption per supplied STEP, in exactly the same step_id order. `x`
contains only visually supported positive error hints; use [] when no error is
visually supported. An error hint needs only its mode and supporting step IDs.
Do not emit five verbose ABSENT assessments: caller-owned rules handle absence
and typed evidence.

Use Chinese for scene text and keep enum values exactly as defined. Do not
invent step IDs, timestamps, commands, collisions, nodes, landmarks, or
subgoal progress. Describe every image independently instead of copying one
scene across steps. The final step's `c` must describe the final post-action
image. Keep each scene within 18 Chinese characters, at most two landmarks
and one clue per step."""


class TemporalCaptioner:
    """Foundation-model adapter for timestamped VLN temporal understanding."""

    def __init__(
        self,
        *,
        engine: Optional[MultimodalEngine] = None,
        engine_factory: Optional[Callable[[], MultimodalEngine]] = None,
        model_path: str = DEFAULT_MODEL_PATH,
        config: Optional[TemporalCaptionerConfig] = None,
        use_cache: bool = False,
        debug_performance: bool = False,
        engine_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if engine is not None and engine_factory is not None:
            raise ValueError("Pass either engine or engine_factory, not both")
        self.model_path = model_path
        self.config = config or TemporalCaptionerConfig()
        self._engine = engine
        self._engine_factory = engine_factory
        self._use_cache = use_cache
        self._debug_performance = debug_performance
        self._engine_kwargs = dict(engine_kwargs or {})
        self.reset_performance_stats()

    def reset_performance_stats(self) -> None:
        """Reset episode-scoped foundation-model inference timing."""
        self._inference_count = 0
        self._inference_success_count = 0
        self._inference_failure_count = 0
        self._inference_total_ms = 0.0
        self._inference_last_ms: Optional[float] = None
        self._inference_min_ms: Optional[float] = None
        self._inference_max_ms: Optional[float] = None
        self._latency_budget_met_count = 0
        self._last_raw_response: Optional[str] = None

    @property
    def last_raw_response(self) -> Optional[str]:
        """Return the latest engine text even when output validation failed."""
        return self._last_raw_response

    def performance_summary(self) -> dict[str, Any]:
        average = (
            self._inference_total_ms / self._inference_count
            if self._inference_count
            else None
        )
        return {
            "inference_count": self._inference_count,
            "success_count": self._inference_success_count,
            "failure_count": self._inference_failure_count,
            "total_inference_ms": self._inference_total_ms,
            "average_inference_ms": average,
            "last_inference_ms": self._inference_last_ms,
            "min_inference_ms": self._inference_min_ms,
            "max_inference_ms": self._inference_max_ms,
            "latency_budget_ms": self.config.inference_latency_budget_ms,
            "latency_budget_met_count": self._latency_budget_met_count,
            "step_max_tokens": self.config.step_max_tokens,
            "max_image_edge": self.config.max_image_edge,
            "include_step_json_schema": (
                self.config.include_step_json_schema
            ),
        }

    def analyze(self, window: TemporalWindow) -> TemporalMemoryRecord:
        """Analyze one validated temporal window without mutating memory."""
        self._validate_window(window)
        prepared_images = tuple(
            self._image_to_png_bytes(frame.image) for frame in window.frames
        )
        content = self._build_model_content(window, prepared_images)
        engine = self._get_engine()

        gpu_device = self._begin_gpu_measurement()
        started = time.perf_counter()
        self._last_raw_response = None
        try:
            raw_response = engine(
                content,
                system_prompt=TEMPORAL_CAPTIONER_SYSTEM_PROMPT,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                response_format=TemporalCaptionPayload,
            )
        except Exception as exc:
            self._synchronize_gpu(gpu_device)
            latency_ms = (time.perf_counter() - started) * 1000
            self._record_inference_timing(latency_ms, success=False)
            raise TemporalInferenceError(
                "Temporal video-understanding inference failed"
            ) from exc
        self._synchronize_gpu(gpu_device)
        latency_ms = (time.perf_counter() - started) * 1000
        self._record_inference_timing(latency_ms, success=True)
        peak_gpu_memory_bytes = self._peak_gpu_memory_bytes(gpu_device)
        self._last_raw_response = str(raw_response)

        payload = self._parse_response(raw_response)
        self._validate_payload(payload, window)
        payload = self._align_payload_to_inputs(payload, window)
        payload = self._apply_rule_overrides(payload, window)
        return TemporalMemoryRecord(
            episode_id=window.episode_id,
            window=TimeSpan(
                start_seconds=window.start_seconds,
                end_seconds=window.end_seconds,
            ),
            frame_timestamps_seconds=[
                frame.timestamp_seconds for frame in window.frames
            ],
            action_timeline=[
                {
                    "issued_at_seconds": action.timestamp_seconds,
                    "active_until_seconds": (
                        window.actions[index + 1].timestamp_seconds
                        if index + 1 < len(window.actions)
                        else window.end_seconds
                    ),
                    "action": action.action,
                    "step_id": action.step_id,
                }
                for index, action in enumerate(window.actions)
            ],
            motion_evidence=[
                signal.to_prompt_dict() for signal in window.motion
            ],
            observed_motion_sequence=[
                signal.camera_motion
                for signal in window.motion
                if (
                    signal.camera_motion != CameraMotion.UNKNOWN
                    and signal.confidence
                    >= self.config.minimum_motion_rule_confidence
                )
            ],
            reverse_retrace_similarity=window.reverse_retrace_similarity,
            latest_scene=payload.latest_scene,
            scene_summary=payload.scene_summary,
            activity_summary=payload.activity_summary,
            overall_progress=payload.overall_progress,
            phases=payload.phases,
            errors=payload.errors,
            model_latency_ms=latency_ms,
            latency_budget_ms=self.config.inference_latency_budget_ms,
            latency_budget_met=(
                latency_ms <= self.config.inference_latency_budget_ms
            ),
            raw_response=str(raw_response),
            peak_gpu_memory_bytes=peak_gpu_memory_bytes,
        )

    def analyze_steps(
        self,
        request: TemporalAnalysisRequest,
    ) -> TemporalMemoryRecord:
        """Analyze a short sequence of action-aligned post-action observations.

        Unlike :meth:`analyze`, this path does not infer an action's active
        interval from an independent frame timeline.  Each immutable
        :class:`TemporalStepInput` already owns its command, transition motion,
        and post-action image, so the model sees the same one-to-one alignment
        that Temporal Memory stores.
        """
        if not isinstance(request, TemporalAnalysisRequest):
            raise TemporalInputError(
                "analyze_steps expects a TemporalAnalysisRequest"
            )
        prepared_images = tuple(
            self._image_to_png_bytes(step.image) for step in request.steps
        )
        content = self._build_step_model_content(request, prepared_images)
        engine = self._get_engine()

        gpu_device = self._begin_gpu_measurement()
        started = time.perf_counter()
        self._last_raw_response = None
        try:
            call_kwargs: dict[str, Any] = {
                "system_prompt": STEP_ALIGNED_SYSTEM_PROMPT,
                "temperature": self.config.temperature,
                "max_tokens": self.config.step_max_tokens,
            }
            # Local Qwen only appends the schema as prompt text; it does not
            # constrain decoding. The compact template below is substantially
            # cheaper. API backends that truly enforce JSON Schema can opt in.
            if self.config.include_step_json_schema:
                call_kwargs["response_format"] = TemporalStepModelPayload
            if getattr(engine, "supports_image_pixel_budget", False):
                maximum_pixels = self.config.max_image_edge**2
                call_kwargs.update(
                    image_min_pixels=min(64**2, maximum_pixels),
                    image_max_pixels=maximum_pixels,
                )
            raw_response = engine(content, **call_kwargs)
        except Exception as exc:
            self._synchronize_gpu(gpu_device)
            latency_ms = (time.perf_counter() - started) * 1000
            self._record_inference_timing(latency_ms, success=False)
            raise TemporalInferenceError(
                "Step-aligned temporal video-understanding inference failed"
            ) from exc
        self._synchronize_gpu(gpu_device)
        latency_ms = (time.perf_counter() - started) * 1000
        self._record_inference_timing(latency_ms, success=True)
        peak_gpu_memory_bytes = self._peak_gpu_memory_bytes(gpu_device)
        self._last_raw_response = str(raw_response)

        payload = self._parse_step_response(raw_response, request)
        # Step IDs must be trustworthy before zip-based alignment.  Error
        # grounding is validated after rule fusion because authoritative typed
        # evidence may replace incomplete model-authored intervals/timestamps.
        self._validate_step_payload(
            payload,
            request,
            validate_errors=False,
        )
        payload = self._align_step_payload_to_inputs(payload, request)
        window = self._step_request_as_window(request)
        errors = self._align_step_errors(payload.errors, request, window)
        self._validate_step_payload(
            payload,
            request,
            aligned_errors=errors,
        )
        overall_progress = self._reconcile_step_progress(
            payload.overall_progress,
            request,
            window,
            errors,
        )
        step_captions = payload.step_captions
        latest_scene = (
            self._static_scene_text(step_captions[-1].scene_after_action)
            or self._static_scene_text(payload.latest_scene)
            or "最新画面中的场景实体不确定"
        )
        scene_summary = (
            self._static_scene_text(payload.scene_summary)
            or latest_scene
        )
        activity_summary = "；".join(
            (
                f"Step {step.step_id}: 命令{step.commanded_action}，"
                f"感知{step.observed_motion.value}，"
                f"执行匹配={step.action_match}"
            )
            for step in request.steps
        )
        window_start = request.steps[0].motion.start_seconds
        window_end = request.steps[-1].post_timestamp_seconds
        return TemporalMemoryRecord(
            episode_id=request.episode_id,
            window=TimeSpan(
                start_seconds=window_start,
                end_seconds=window_end,
            ),
            frame_timestamps_seconds=[
                step.post_timestamp_seconds for step in request.steps
            ],
            action_timeline=[
                {
                    "step_id": step.step_id,
                    "issued_at_seconds": step.motion.start_seconds,
                    "post_timestamp_seconds": step.post_timestamp_seconds,
                    "action": step.commanded_action,
                    "observed_motion": step.observed_motion.value,
                    "action_match": step.action_match,
                    "collision": step.collision,
                }
                for step in request.steps
            ],
            motion_evidence=[
                {
                    "step_id": step.step_id,
                    **step.motion.to_prompt_dict(),
                    "camera_motion": step.observed_motion.value,
                    "collision": step.collision,
                }
                for step in request.steps
            ],
            observed_motion_sequence=[
                step.observed_motion
                for step in request.steps
                if step.observed_motion != CameraMotion.UNKNOWN
            ],
            reverse_retrace_similarity=request.reverse_retrace_similarity,
            latest_scene=latest_scene,
            scene_summary=scene_summary,
            activity_summary=activity_summary,
            overall_progress=overall_progress,
            phases=[],
            step_captions=step_captions,
            errors=errors,
            model_latency_ms=latency_ms,
            latency_budget_ms=self.config.inference_latency_budget_ms,
            latency_budget_met=(
                latency_ms <= self.config.inference_latency_budget_ms
            ),
            raw_response=str(raw_response),
            peak_gpu_memory_bytes=peak_gpu_memory_bytes,
        )

    def analyze_video(
        self,
        video_path: str | Path,
        *,
        start_seconds: float,
        end_seconds: float,
        crop: Optional[tuple[int, int, int, int]] = None,
        timestamp_offset_seconds: float = 0.0,
        actions: Sequence[TimedAction] = (),
        topology: Sequence[TopologySignal] = (),
        progress: Optional[ProgressSignals] = None,
        goal: Optional[str] = None,
        episode_id: Optional[str] = None,
        notes: Sequence[str] = (),
    ) -> TemporalMemoryRecord:
        """Decode sparse caption frames and independent high-rate motion frames."""
        caption_frames = self.sample_video_frames(
            video_path,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            fps=self.config.caption_fps,
            max_frames=self.config.max_frames,
            crop=crop,
            timestamp_offset_seconds=timestamp_offset_seconds,
        )
        motion_frames = self.sample_video_frames(
            video_path,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            fps=self.config.motion_fps,
            max_frames=self.config.max_motion_frames,
            crop=crop,
            timestamp_offset_seconds=timestamp_offset_seconds,
        )
        motion = self.estimate_motion_signals(motion_frames)
        retrace_similarity = self.reverse_retrace_similarity(
            motion_frames, motion
        )
        window = TemporalWindow(
            start_seconds=start_seconds + timestamp_offset_seconds,
            end_seconds=end_seconds + timestamp_offset_seconds,
            frames=caption_frames,
            actions=tuple(actions),
            motion=motion,
            topology=tuple(topology),
            progress=progress or ProgressSignals(),
            reverse_retrace_similarity=retrace_similarity,
            goal=goal,
            episode_id=episode_id,
            notes=tuple(notes),
        )
        return self.analyze(window)

    def sample_video_frames(
        self,
        video_path: str | Path,
        *,
        start_seconds: float,
        end_seconds: float,
        fps: float,
        max_frames: int,
        crop: Optional[tuple[int, int, int, int]] = None,
        timestamp_offset_seconds: float = 0.0,
    ) -> tuple[TimestampedFrame, ...]:
        """Sample `[start, end)` uniformly while retaining absolute timestamps."""
        try:
            import cv2
        except ImportError as exc:
            raise TemporalInputError(
                "Video sampling requires opencv-python"
            ) from exc

        path = Path(video_path).expanduser()
        if not path.is_file():
            raise TemporalInputError(f"Video does not exist: {path}")
        start = _finite_non_negative(start_seconds, "video start")
        end = _finite_non_negative(end_seconds, "video end")
        offset = _finite_non_negative(timestamp_offset_seconds, "timestamp offset")
        if end <= start:
            raise TemporalInputError("video end must be greater than video start")
        if fps <= 0:
            raise TemporalInputError("sampling fps must be positive")
        if max_frames < 2:
            raise TemporalInputError("max_frames must be at least 2")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise TemporalInputError(f"Could not open video: {path}")
        try:
            native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if native_fps <= 0 or total_frames <= 0:
                raise TemporalInputError(f"Invalid video metadata: {path}")
            duration = total_frames / native_fps
            if end > duration + 1e-6:
                raise TemporalInputError(
                    f"Requested end {end:.3f}s exceeds video duration "
                    f"{duration:.3f}s"
                )

            target_indices: list[int] = []
            sample_index = 0
            while True:
                target_time = start + sample_index / fps
                if target_time >= end - 1e-9:
                    break
                frame_index = min(
                    int(math.floor(target_time * native_fps + 0.5)),
                    total_frames - 1,
                )
                if not target_indices or frame_index != target_indices[-1]:
                    target_indices.append(frame_index)
                sample_index += 1
            if len(target_indices) < 2:
                raise TemporalInputError(
                    "Requested range/fps produced fewer than two unique frames"
                )
            target_indices = self._uniform_cap(target_indices, max_frames)

            frames: list[TimestampedFrame] = []
            for frame_index in target_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame_bgr = cap.read()
                if not ok:
                    raise TemporalInputError(
                        f"Could not decode video frame {frame_index}"
                    )
                if crop is not None:
                    x0, y0, x1, y1 = self._validate_crop(crop, frame_bgr.shape)
                    frame_bgr = frame_bgr[y0:y1, x0:x1]
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames.append(
                    TimestampedFrame(
                        timestamp_seconds=frame_index / native_fps + offset,
                        image=frame_rgb,
                        step_id=frame_index,
                    )
                )
            return tuple(frames)
        finally:
            cap.release()

    def estimate_motion_signals(
        self, frames: Sequence[TimestampedFrame]
    ) -> tuple[MotionSignal, ...]:
        """Estimate turn/stationary phases from dense adjacent optical flow.

        This should use high-rate frames (normally 10 FPS). Sparse caption
        frames are intentionally rejected as reliable motion evidence.
        """
        ordered = tuple(frames)
        self._validate_frame_order(ordered)
        if len(ordered) < 2:
            return ()
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise TemporalInputError(
                "Optical-flow estimation requires numpy and opencv-python"
            ) from exc

        gray_frames = []
        for frame in ordered:
            rgb = self._image_to_rgb_array(frame.image)
            height, width = rgb.shape[:2]
            scale = min(1.0, 240.0 / max(height, width))
            if scale < 1:
                rgb = cv2.resize(
                    rgb,
                    (max(2, round(width * scale)), max(2, round(height * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            gray_frames.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))

        raw: list[dict[str, Any]] = []
        for index, (previous, current) in enumerate(
            zip(gray_frames, gray_frames[1:])
        ):
            start = ordered[index].timestamp_seconds
            end = ordered[index + 1].timestamp_seconds
            dt = end - start
            if dt > self.config.maximum_flow_interval_seconds + 1e-9:
                raw.append(
                    {
                        "start": start,
                        "end": end,
                        "motion": CameraMotion.UNKNOWN,
                        "dx": None,
                        "magnitude": None,
                        "coherence": 0.0,
                        "confidence": 0.0,
                        "quality_note": (
                            f"frame interval {dt:.3f}s exceeds the reliable "
                            f"optical-flow limit "
                            f"{self.config.maximum_flow_interval_seconds:.3f}s"
                        ),
                    }
                )
                continue
            flow = cv2.calcOpticalFlowFarneback(
                previous,
                current,
                None,
                0.5,
                4,
                25,
                4,
                7,
                1.5,
                0,
            )
            horizontal = flow[..., 0]
            vertical = flow[..., 1]
            magnitude = np.sqrt(horizontal**2 + vertical**2)
            median_dx = float(np.median(horizontal))
            median_dy = float(np.median(vertical))
            median_magnitude = float(np.median(magnitude))
            width = previous.shape[1]
            dx_fraction = median_dx / width
            magnitude_fraction = median_magnitude / width
            direction = 1 if median_dx >= 0 else -1
            coherence = float(np.mean(horizontal * direction > 0))

            if magnitude_fraction <= self.config.stationary_flow_fraction:
                motion = CameraMotion.STATIONARY
                confidence = max(0.5, 1 - magnitude_fraction / max(
                    self.config.stationary_flow_fraction, 1e-9
                ))
            elif (
                abs(dx_fraction) >= self.config.turning_flow_fraction
                and abs(median_dx) >= abs(median_dy)
                and coherence >= self.config.minimum_horizontal_coherence
            ):
                motion = (
                    CameraMotion.TURN_LEFT
                    if median_dx > 0
                    else CameraMotion.TURN_RIGHT
                )
                confidence = min(1.0, 0.5 * coherence + 0.5)
            else:
                motion = CameraMotion.UNKNOWN
                confidence = min(0.5, coherence)

            raw.append(
                {
                    "start": start,
                    "end": end,
                    "motion": motion,
                    "dx": dx_fraction,
                    "magnitude": magnitude_fraction,
                    "coherence": coherence,
                    "confidence": confidence,
                    "quality_note": None,
                }
            )

        self._smooth_motion_labels(raw)
        return self._group_motion_samples(raw)

    def reverse_retrace_similarity(
        self,
        frames: Sequence[TimestampedFrame],
        motion: Sequence[MotionSignal],
    ) -> Optional[float]:
        """Median NCC of frames mirrored around the first turn reversal."""
        directions = [
            signal.camera_motion
            for signal in motion
            if signal.camera_motion
            in {CameraMotion.TURN_LEFT, CameraMotion.TURN_RIGHT}
        ]
        if len(directions) < 2:
            return None
        reversal_time = None
        previous = directions[0]
        for signal in motion:
            if signal.camera_motion not in {
                CameraMotion.TURN_LEFT,
                CameraMotion.TURN_RIGHT,
            }:
                continue
            if signal.camera_motion != previous:
                reversal_time = signal.start_seconds
                break
            previous = signal.camera_motion
        if reversal_time is None:
            return None

        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise TemporalInputError(
                "Retrace similarity requires numpy and opencv-python"
            ) from exc

        ordered = tuple(frames)
        center = min(
            range(len(ordered)),
            key=lambda index: abs(
                ordered[index].timestamp_seconds - reversal_time
            ),
        )
        pair_count = min(center, len(ordered) - center - 1, 32)
        if pair_count < 3:
            return None

        similarities = []
        for distance in range(1, pair_count + 1):
            first = self._image_to_rgb_array(ordered[center - distance].image)
            second = self._image_to_rgb_array(ordered[center + distance].image)
            first_gray = cv2.resize(
                cv2.cvtColor(first, cv2.COLOR_RGB2GRAY), (96, 96)
            ).astype("float32")
            second_gray = cv2.resize(
                cv2.cvtColor(second, cv2.COLOR_RGB2GRAY), (96, 96)
            ).astype("float32")
            first_gray -= float(first_gray.mean())
            second_gray -= float(second_gray.mean())
            denominator = float(
                np.linalg.norm(first_gray) * np.linalg.norm(second_gray)
            )
            if denominator > 1e-9:
                similarities.append(
                    float((first_gray * second_gray).sum() / denominator)
                )
        return statistics.median(similarities) if similarities else None

    def _get_engine(self) -> MultimodalEngine:
        if self._engine is not None:
            return self._engine
        if self._engine_factory is not None:
            self._engine = self._engine_factory()
            return self._engine

        from agentflow.agents.engine.factory import create_llm_engine

        kwargs = {
            "debug_performance": self._debug_performance,
            **self._engine_kwargs,
        }
        self._engine = create_llm_engine(
            model_string=f"local-qwen3vl-{self.model_path}",
            is_multimodal=True,
            use_cache=self._use_cache,
            **kwargs,
        )
        return self._engine

    def _record_inference_timing(
        self,
        duration_ms: float,
        *,
        success: bool,
    ) -> None:
        duration = max(0.0, float(duration_ms))
        self._inference_count += 1
        self._inference_success_count += int(success)
        self._inference_failure_count += int(not success)
        self._inference_total_ms += duration
        self._inference_last_ms = duration
        self._inference_min_ms = (
            duration
            if self._inference_min_ms is None
            else min(self._inference_min_ms, duration)
        )
        self._inference_max_ms = (
            duration
            if self._inference_max_ms is None
            else max(self._inference_max_ms, duration)
        )
        if success and duration <= self.config.inference_latency_budget_ms:
            self._latency_budget_met_count += 1

    @staticmethod
    def _begin_gpu_measurement() -> Optional[Any]:
        """Reset CUDA peak accounting after the model is resident in memory."""
        try:
            import torch

            if not torch.cuda.is_available():
                return None
            device = torch.cuda.current_device()
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            return device
        except Exception:
            # Diagnostics must never make navigation inference unavailable.
            return None

    @staticmethod
    def _synchronize_gpu(device: Optional[Any]) -> None:
        if device is None:
            return
        try:
            import torch

            torch.cuda.synchronize(device)
        except Exception:
            pass

    @staticmethod
    def _peak_gpu_memory_bytes(device: Optional[Any]) -> Optional[int]:
        if device is None:
            return None
        try:
            import torch

            return int(torch.cuda.max_memory_allocated(device))
        except Exception:
            return None

    def _validate_window(self, window: TemporalWindow) -> None:
        if len(window.frames) < 2:
            raise TemporalInputError("TemporalWindow requires at least two frames")
        if len(window.frames) > self.config.max_frames:
            raise TemporalInputError(
                f"TemporalWindow has {len(window.frames)} frames; "
                f"maximum is {self.config.max_frames}"
            )
        self._validate_frame_order(window.frames)
        for frame in window.frames:
            if not (
                window.start_seconds
                <= frame.timestamp_seconds
                < window.end_seconds
            ):
                raise TemporalInputError(
                    f"Frame timestamp {frame.timestamp_seconds} is outside "
                    f"[{window.start_seconds}, {window.end_seconds})"
                )
        self._validate_strictly_ordered(
            window.actions,
            lambda item: item.timestamp_seconds,
            "action",
        )
        self._validate_strictly_ordered(
            window.topology,
            lambda item: item.timestamp_seconds,
            "topology",
        )
        for action in window.actions:
            if not (
                window.start_seconds
                <= action.timestamp_seconds
                < window.end_seconds
            ):
                raise TemporalInputError("Action timestamp is outside the window")
        for signal in window.topology:
            if not (
                window.start_seconds
                <= signal.timestamp_seconds
                < window.end_seconds
            ):
                raise TemporalInputError("Topology timestamp is outside the window")
        self._validate_strictly_ordered(
            window.motion,
            lambda item: item.start_seconds,
            "motion",
        )
        previous_end = window.start_seconds
        for signal in window.motion:
            if (
                signal.start_seconds < window.start_seconds
                or signal.end_seconds > window.end_seconds
            ):
                raise TemporalInputError("Motion interval is outside the window")
            if signal.start_seconds < previous_end - 1e-9:
                raise TemporalInputError("Motion intervals must not overlap")
            previous_end = signal.end_seconds

    @staticmethod
    def _validate_frame_order(frames: Sequence[TimestampedFrame]) -> None:
        TemporalCaptioner._validate_strictly_ordered(
            frames,
            lambda item: item.timestamp_seconds,
            "frame",
        )

    @staticmethod
    def _validate_strictly_ordered(
        values: Sequence[Any],
        timestamp: Callable[[Any], float],
        label: str,
    ) -> None:
        previous = None
        for value in values:
            current = timestamp(value)
            if previous is not None and current <= previous:
                raise TemporalInputError(
                    f"{label} timestamps must be strictly increasing"
                )
            previous = current

    def _build_model_content(
        self,
        window: TemporalWindow,
        prepared_images: Sequence[bytes],
    ) -> list[Any]:
        content: list[Any] = [
            (
                "[BEGIN TIMESTAMPED RGB STORYBOARD]\n"
                f"timestamp_semantics={window.timestamp_semantics}\n"
                "Each image immediately follows its FRAME marker.\n"
            )
        ]
        for index, (frame, image_bytes) in enumerate(
            zip(window.frames, prepared_images)
        ):
            marker = (
                f"\n[FRAME id=f{index:03d} "
                f"absolute_t={frame.timestamp_seconds:.3f}s"
            )
            if frame.step_id is not None:
                marker += f" step_id={frame.step_id}"
            marker += "]\n"
            content.extend((marker, image_bytes))
        content.append("\n[END TIMESTAMPED RGB STORYBOARD]\n")
        content.append(self._analysis_request(window))
        return content

    def _build_step_model_content(
        self,
        request: TemporalAnalysisRequest,
        prepared_images: Sequence[bytes],
    ) -> list[Any]:
        if len(prepared_images) != len(request.steps):
            raise TemporalInputError(
                "Prepared image count does not match step count"
            )
        content: list[Any] = [
            (
                f"[BEGIN {len(request.steps)} STEP POST-ACTION STORYBOARD]\n"
                "Each image immediately follows the STEP that produced it. "
                "The image is O_(t+1), never the pre-action O_t.\n"
            )
        ]
        for step, image_bytes in zip(request.steps, prepared_images):
            collision = (
                "unknown"
                if step.collision is None
                else str(step.collision).lower()
            )
            marker = (
                f"\n[STEP step_id={step.step_id} "
                f"command={step.commanded_action} "
                f"post_t={step.post_timestamp_seconds:.3f}s "
                f"observed_action={step.observed_motion.value} "
                f"action_match={step.action_match} "
                f"collision={collision}]\n"
            )
            content.extend((marker, image_bytes))
        content.append(
            f"\n[END {len(request.steps)} STEP POST-ACTION STORYBOARD]\n"
        )
        content.append(self._step_analysis_request(request))
        return content

    def _step_analysis_request(
        self,
        request: TemporalAnalysisRequest,
    ) -> str:
        step_count = len(request.steps)
        count_text = {
            2: "两",
            3: "三",
            4: "四",
            5: "五",
            6: "六",
            7: "七",
            8: "八",
        }.get(step_count, str(step_count))
        evidence = {
            "goal": request.goal,
            "progress": request.progress.to_prompt_dict(),
            "retrace": request.reverse_retrace_similarity,
        }
        example_ids = [step.step_id for step in request.steps]
        return (
            f"\n最近走的{count_text}步发生了什么？"
            "按照每步 action 后的画面进行描述。\n"
            f"严格按照输入 step_id 顺序返回恰好{count_text}条 steps。"
            "每条只描述该 STEP marker 后紧邻的 action 后画面；"
            "区分命令、感知运动和场景事实，不要把命令当作成功执行。"
            "只报告能从画面确认的简短场景、地标、视觉运动和错误线索。"
            "x 只列视觉上疑似存在的错误及对应 step id；没有就返回空数组。"
            "只输出单行JSON，短字段格式为"
            '{"p":"PROGRESSING","s":['
            '{"i":步骤ID,"c":"场景","l":["地标"],'
            '"m":"FORWARD","e":[]}],"x":['
            '{"m":"collision","i":[步骤ID]}]}。'
            f"s 必须依次包含这些ID：{example_ids}。"
            "以下紧凑机器证据只用于判断整体进展，STEP marker 中的 action、"
            "motion、collision 和 action_match 优先于视觉模型：\n"
            + json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))
            + "\nReturn the required JSON now."
        )

    def _analysis_request(self, window: TemporalWindow) -> str:
        rule_hints = self._rule_hints(window)
        request = {
            "window": {
                "start_seconds": window.start_seconds,
                "end_seconds": window.end_seconds,
                "timestamp_semantics": window.timestamp_semantics,
            },
            "frame_timestamps_seconds": [
                frame.timestamp_seconds for frame in window.frames
            ],
            "actions": [
                {
                    "issued_at_seconds": action.timestamp_seconds,
                    "active_until_seconds": (
                        window.actions[index + 1].timestamp_seconds
                        if index + 1 < len(window.actions)
                        else window.end_seconds
                    ),
                    "action": action.action,
                    "step_id": action.step_id,
                    "semantics": (
                        "command state applies only to subsequent observations "
                        "inside this interval; observed execution is separate"
                    ),
                }
                for index, action in enumerate(window.actions)
            ],
            "motion_observations": [
                signal.to_prompt_dict() for signal in window.motion
            ],
            "topology_observations": [
                signal.to_prompt_dict() for signal in window.topology
            ],
            "progress_signals": window.progress.to_prompt_dict(),
            "reverse_retrace_similarity": window.reverse_retrace_similarity,
            "rule_hints": rule_hints,
            "goal": window.goal,
            "episode_id": window.episode_id,
            "notes": list(window.notes),
        }
        return (
            "\nAnalyze the storyboard and the machine-generated evidence below. "
            "Use at most four concise temporal phases and cite only boundary or "
            "transition evidence timestamps. Do not invent lateral "
            "strafing, timestamps, actions, collisions, landmarks, nodes, or "
            "subgoal progress. Keep the complete JSON comfortably below 700 "
            "tokens so every closing delimiter is present.\n"
            + json.dumps(request, ensure_ascii=False, separators=(",", ":"))
            + "\nReturn the required JSON now."
        )

    def _parse_response(self, raw_response: Any) -> TemporalCaptionPayload:
        if isinstance(raw_response, TemporalCaptionPayload):
            return raw_response
        if isinstance(raw_response, Mapping):
            try:
                return TemporalCaptionPayload.model_validate(raw_response)
            except Exception as exc:
                raise TemporalOutputError(
                    "Temporal model output does not match the schema"
                ) from exc
        if not isinstance(raw_response, str) or not raw_response.strip():
            raise TemporalOutputError("Temporal model returned an empty response")
        text = raw_response.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) < 3 or lines[-1].strip() != "```":
                raise TemporalOutputError("Incomplete JSON markdown fence")
            language = lines[0].strip().lower()
            if language not in {"```", "```json"}:
                raise TemporalOutputError("Only a complete JSON fence is accepted")
            text = "\n".join(lines[1:-1]).strip()
        try:
            return TemporalCaptionPayload.model_validate_json(text)
        except Exception as exc:
            raise TemporalOutputError(
                "Temporal model returned invalid structured JSON"
            ) from exc

    def _parse_step_response(
        self,
        raw_response: Any,
        request: TemporalAnalysisRequest,
    ) -> TemporalStepCaptionPayload:
        if isinstance(raw_response, TemporalStepCaptionPayload):
            return raw_response
        if isinstance(raw_response, TemporalStepModelPayload):
            return self._expand_step_model_payload(raw_response, request)
        if isinstance(raw_response, Mapping):
            broad_error: Optional[Exception] = None
            try:
                return TemporalStepCaptionPayload.model_validate(raw_response)
            except Exception as exc:
                broad_error = exc
            try:
                compact = TemporalStepModelPayload.model_validate(raw_response)
            except Exception as exc:
                detail = " ".join(str(exc).split())
                if len(detail) > 1200:
                    detail = detail[:1197] + "..."
                raise TemporalOutputError(
                    "Step-aligned model output does not match the compact "
                    "or compatibility schema"
                    + (f": {detail}" if detail else "")
                ) from (broad_error or exc)
            return self._expand_step_model_payload(compact, request)
        if not isinstance(raw_response, str) or not raw_response.strip():
            raise TemporalOutputError("Temporal model returned an empty response")
        text = raw_response.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) < 3 or lines[-1].strip() != "```":
                raise TemporalOutputError("Incomplete JSON markdown fence")
            language = lines[0].strip().lower()
            if language not in {"```", "```json"}:
                raise TemporalOutputError("Only a complete JSON fence is accepted")
            text = "\n".join(lines[1:-1]).strip()
        try:
            decoded = json.loads(text)
        except Exception as exc:
            detail = " ".join(str(exc).split())
            if len(detail) > 1200:
                detail = detail[:1197] + "..."
            raise TemporalOutputError(
                "Temporal model returned invalid step-aligned JSON"
                + (f": {detail}" if detail else "")
            ) from exc
        if not isinstance(decoded, Mapping):
            raise TemporalOutputError(
                "Temporal model step-aligned JSON must be an object"
            )
        return self._parse_step_response(decoded, request)

    def _expand_step_model_payload(
        self,
        payload: TemporalStepModelPayload,
        request: TemporalAnalysisRequest,
    ) -> TemporalStepCaptionPayload:
        """Expand the terse wire response into the stable public schema."""
        captions = [
            TemporalStepCaption(
                step_id=caption.step_id,
                scene_after_action=caption.scene_after_action,
                visible_landmarks=caption.visible_landmarks,
                visual_perceived_action=caption.visual_perceived_action,
                visual_error_clues=caption.visual_error_clues,
                confidence=0.6,
            )
            for caption in payload.step_captions
        ]
        distinct_scenes: list[str] = []
        for caption in captions:
            scene = str(caption.scene_after_action).strip()
            if scene and scene not in distinct_scenes:
                distinct_scenes.append(scene)
        latest_scene = (
            captions[-1].scene_after_action
            if captions
            else "最新场景不确定"
        )
        scene_summary = "；".join(distinct_scenes) or latest_scene

        errors: dict[str, ErrorAssessment] = {
            mode: ErrorAssessment(
                verdict="UNCERTAIN",
                confidence=0.5,
                interval=None,
                evidence_timestamps_seconds=[],
                reason="模型未报告足以确认或排除该错误的视觉线索。",
                source="MODEL",
            )
            for mode in ERROR_MODES
        }
        step_by_id = {step.step_id: step for step in request.steps}
        input_order = {step.step_id: index for index, step in enumerate(request.steps)}
        seen_modes: set[str] = set()
        for hint in payload.error_hints:
            if hint.mode in seen_modes:
                raise TemporalOutputError(
                    f"Duplicate compact error hint for {hint.mode}"
                )
            seen_modes.add(hint.mode)
            if len(set(hint.step_ids)) != len(hint.step_ids):
                raise TemporalOutputError(
                    f"Compact error hint {hint.mode} contains duplicate step IDs"
                )
            unknown_ids = [
                step_id for step_id in hint.step_ids if step_id not in step_by_id
            ]
            if unknown_ids:
                raise TemporalOutputError(
                    f"Compact error hint {hint.mode} references unknown "
                    f"step IDs {unknown_ids}"
                )
            supporting_steps = sorted(
                (step_by_id[step_id] for step_id in hint.step_ids),
                key=lambda step: input_order[step.step_id],
            )
            evidence = [
                step.post_timestamp_seconds for step in supporting_steps
            ]
            if len(evidence) > 3:
                evidence = self._uniform_cap(evidence, 3)
            errors[hint.mode] = ErrorAssessment(
                verdict="PRESENT",
                confidence=0.65,
                interval=TimeSpan(
                    start_seconds=supporting_steps[0].motion.start_seconds,
                    end_seconds=supporting_steps[-1].post_timestamp_seconds,
                ),
                evidence_timestamps_seconds=evidence,
                reason=(
                    "模型在步骤"
                    + "、".join(str(step.step_id) for step in supporting_steps)
                    + "发现视觉错误线索。"
                ),
                source="MODEL",
            )

        return TemporalStepCaptionPayload(
            latest_scene=latest_scene,
            scene_summary=scene_summary,
            overall_progress=payload.overall_progress,
            step_captions=captions,
            errors=TemporalStepErrorAssessments(**errors),
        )

    def _validate_step_payload(
        self,
        payload: TemporalStepCaptionPayload,
        request: TemporalAnalysisRequest,
        *,
        validate_errors: bool = True,
        aligned_errors: Optional[ErrorAssessments] = None,
    ) -> None:
        expected_count = len(request.steps)
        if len(payload.step_captions) != expected_count:
            raise TemporalOutputError(
                "Step-aligned model must return exactly "
                f"{expected_count} step captions"
            )
        expected_ids = [step.step_id for step in request.steps]
        returned_ids = [
            caption.step_id for caption in payload.step_captions
        ]
        if returned_ids != expected_ids:
            raise TemporalOutputError(
                "Step caption IDs must match input step IDs one-to-one and "
                f"in order; expected {expected_ids}, got {returned_ids}"
            )
        if not validate_errors:
            return
        valid_times = tuple(
            sorted(
                {
                    *(
                        step.motion.start_seconds
                        for step in request.steps
                    ),
                    *(
                        step.motion.end_seconds
                        for step in request.steps
                    ),
                    *(
                        step.post_timestamp_seconds
                        for step in request.steps
                    ),
                }
            )
        )
        window_start = request.steps[0].motion.start_seconds
        window_end = request.steps[-1].post_timestamp_seconds
        for mode in ERROR_MODES:
            assessment = getattr(
                aligned_errors if aligned_errors is not None else payload.errors,
                mode,
            )
            if assessment.interval is not None:
                span = assessment.interval
                if span.end_seconds <= span.start_seconds:
                    raise TemporalOutputError(
                        "Output interval end must exceed start"
                    )
                if (
                    span.start_seconds < window_start - 1e-9
                    or span.end_seconds > window_end + 1e-9
                ):
                    raise TemporalOutputError(
                        "Output interval is outside the step-aligned request"
                    )
            self._validate_evidence_times(
                assessment.evidence_timestamps_seconds,
                valid_times,
            )
            if assessment.verdict == "PRESENT":
                if assessment.interval is None:
                    raise TemporalOutputError(
                        f"PRESENT {mode} must contain an interval"
                    )
                if not assessment.evidence_timestamps_seconds:
                    raise TemporalOutputError(
                        f"PRESENT {mode} must cite evidence timestamps"
                    )
                if not assessment.reason.strip():
                    raise TemporalOutputError(
                        f"PRESENT {mode} must contain a reason"
                    )

    def _align_step_payload_to_inputs(
        self,
        payload: TemporalStepCaptionPayload,
        request: TemporalAnalysisRequest,
    ) -> TemporalStepCaptionPayload:
        aligned: list[TemporalStepCaption] = []
        for model_caption, step in zip(payload.step_captions, request.steps):
            scene = (
                self._static_scene_text(model_caption.scene_after_action)
                or "该步画面中的场景实体不确定"
            )
            landmarks = []
            for landmark in model_caption.visible_landmarks:
                static_landmark = self._static_scene_text(landmark)
                if static_landmark and static_landmark not in landmarks:
                    landmarks.append(static_landmark)
            aligned.append(
                model_caption.model_copy(
                    update={
                        "step_id": step.step_id,
                        "scene_after_action": scene,
                        "visible_landmarks": landmarks,
                        "commanded_action": step.commanded_action,
                        "post_timestamp_seconds": (
                            step.post_timestamp_seconds
                        ),
                        "observed_motion": step.observed_motion,
                        "action_match": step.action_match,
                        "collision": step.collision,
                    }
                )
            )
        return payload.model_copy(update={"step_captions": aligned})

    def _step_request_as_window(
        self,
        request: TemporalAnalysisRequest,
    ) -> TemporalWindow:
        topology: list[TopologySignal] = []
        motion: list[MotionSignal] = []
        for step in request.steps:
            collision = (
                step.collision
                if step.collision is not None
                else step.motion.collision
            )
            motion.append(
                replace(
                    step.motion,
                    camera_motion=step.observed_motion,
                    collision=collision,
                )
            )
            if (
                step.topology_node_id is not None
                or step.is_revisit is not None
                or step.distance_to_goal_meters is not None
            ):
                topology.append(
                    TopologySignal(
                        timestamp_seconds=step.post_timestamp_seconds,
                        node_id=step.topology_node_id,
                        visit_count=(
                            2
                            if step.is_revisit is True
                            else 1
                            if step.is_revisit is False
                            else None
                        ),
                        distance_to_goal_meters=(
                            step.distance_to_goal_meters
                        ),
                        source="temporal_memory",
                    )
                )
        # TemporalWindow uses a half-open end for legacy video frames. This
        # internal rule-evaluation view extends the final post timestamp by the
        # configured tolerance without changing the public record timestamp.
        end = (
            request.steps[-1].post_timestamp_seconds
            + self.config.evidence_time_tolerance_seconds
        )
        newly_completed_subgoals = sum(
            len(step.newly_completed_subgoals) for step in request.steps
        )
        new_nodes = sum(step.is_new_node is True for step in request.steps)
        derived_new_node_count: Optional[int] = (
            new_nodes
            if all(step.is_new_node is not None for step in request.steps)
            else None
        )
        progress = replace(
            request.progress,
            completed_subgoals_count=max(
                request.progress.completed_subgoals_count or 0,
                newly_completed_subgoals,
            ),
            new_topological_nodes_count=(
                max(
                    request.progress.new_topological_nodes_count or 0,
                    derived_new_node_count,
                )
                if derived_new_node_count is not None
                else request.progress.new_topological_nodes_count
            ),
        )
        return TemporalWindow(
            start_seconds=request.steps[0].motion.start_seconds,
            end_seconds=end,
            frames=tuple(
                TimestampedFrame(
                    timestamp_seconds=step.post_timestamp_seconds,
                    image=step.image,
                    step_id=step.step_id,
                )
                for step in request.steps
            ),
            actions=tuple(
                TimedAction(
                    timestamp_seconds=step.motion.start_seconds,
                    action=step.commanded_action,
                    step_id=step.step_id,
                )
                for step in request.steps
            ),
            motion=tuple(motion),
            topology=tuple(topology),
            progress=progress,
            reverse_retrace_similarity=request.reverse_retrace_similarity,
            goal=request.goal,
            episode_id=request.episode_id,
            notes=request.notes,
            timestamp_semantics="episode_step_post_action_seconds",
        )

    def _align_step_errors(
        self,
        model_errors: ErrorAssessments,
        request: TemporalAnalysisRequest,
        window: TemporalWindow,
    ) -> ErrorAssessments:
        errors = ErrorAssessments.model_validate(
            model_errors.model_dump(mode="python")
        )
        labels = {
            "collision": "碰撞",
            "repeated_visit": "重复访问",
            "motion_oscillation": "运动振荡",
            "get_nowhere": "无进展",
            "action_execution_mismatch": "命令与执行不一致",
        }
        goal_distances = [
            step.distance_to_goal_meters
            for step in request.steps
            if step.distance_to_goal_meters is not None
        ]
        goal_improvement = (
            goal_distances[0] - goal_distances[-1]
            if len(goal_distances) >= 2
            else None
        )
        progress = window.progress
        discoveries = (
            progress.new_landmarks_count,
            progress.new_topological_nodes_count,
            progress.completed_subgoals_count,
        )
        has_discovery_progress = any(
            value is not None and value > 0 for value in discoveries
        )
        has_step_progress = bool(
            (
                progress.net_displacement_meters is not None
                and progress.net_displacement_meters > 0.25
            )
            or has_discovery_progress
            or (
                goal_improvement is not None
                and goal_improvement
                >= self.config.goal_distance_progress_epsilon_meters
            )
        )
        step_turns = [
            step
            for step in request.steps
            if step.observed_motion
            in {CameraMotion.TURN_LEFT, CameraMotion.TURN_RIGHT}
        ]
        step_turn_reversals = sum(
            previous.observed_motion != current.observed_motion
            for previous, current in zip(step_turns, step_turns[1:])
        )
        complete_motion_coverage = all(
            step.observed_motion != CameraMotion.UNKNOWN
            for step in request.steps
        )
        threshold = self.config.minimum_definite_error_confidence
        for mode in ERROR_MODES:
            assessment = getattr(errors, mode).model_copy(
                update={"source": "MODEL"}
            )
            if mode == "action_execution_mismatch":
                negative_coverage = all(
                    step.action_match == "MATCH"
                    for step in request.steps
                )
            elif mode == "get_nowhere":
                negative_coverage = has_step_progress
            elif mode == "motion_oscillation":
                negative_coverage = (
                    has_step_progress
                    or (
                        complete_motion_coverage
                        and step_turn_reversals < 2
                    )
                )
            else:
                negative_coverage = self._has_negative_error_coverage(
                    mode, window
                )
            if assessment.verdict == "PRESENT" and negative_coverage:
                assessment = assessment.model_copy(
                    update={
                        "verdict": "UNCERTAIN",
                        "confidence": 0.5,
                        "interval": None,
                        "evidence_timestamps_seconds": [],
                        "reason": (
                            f"结构化证据与模型的{labels[mode]}判断冲突，"
                            "改为不确定。"
                        ),
                    }
                )
            elif assessment.verdict == "ABSENT" and not negative_coverage:
                assessment = assessment.model_copy(
                    update={
                        "verdict": "UNCERTAIN",
                        "confidence": 0.5,
                        "interval": None,
                        "evidence_timestamps_seconds": [],
                        "reason": (
                            f"缺少足以排除{labels[mode]}的观测覆盖，"
                            "改为不确定。"
                        ),
                    }
                )
            elif (
                assessment.verdict in {"PRESENT", "ABSENT"}
                and assessment.confidence < threshold
            ):
                assessment = assessment.model_copy(
                    update={
                        "verdict": "UNCERTAIN",
                        "confidence": 0.5,
                        "interval": None,
                        "evidence_timestamps_seconds": [],
                        "reason": (
                            f"模型对{labels[mode]}的判断置信度不足，"
                            "改为不确定。"
                        ),
                    }
                )
            elif (
                assessment.verdict == "PRESENT"
                and (
                    assessment.interval is None
                    or not assessment.evidence_timestamps_seconds
                    or not assessment.reason.strip()
                )
            ):
                # Do not fabricate a whole-window interval for an ungrounded
                # visual claim.  A typed rule override below may still replace
                # this with exact evidence; otherwise missing grounding means
                # UNCERTAIN rather than invalidating all step captions.
                assessment = assessment.model_copy(
                    update={
                        "verdict": "UNCERTAIN",
                        "confidence": 0.5,
                        "interval": None,
                        "evidence_timestamps_seconds": [],
                        "reason": (
                            f"模型的{labels[mode]}判断缺少完整时间定位或证据，"
                            "改为不确定。"
                        ),
                    }
                )
            setattr(errors, mode, assessment)

        overrides = self._rule_overrides(window)
        # Live step mode has stricter definitions than legacy replay mode.
        overrides.pop("motion_oscillation", None)
        overrides.pop("get_nowhere", None)
        retrace = request.reverse_retrace_similarity
        if (
            step_turn_reversals >= 2
            and retrace is not None
            and retrace >= self.config.oscillation_retrace_threshold
            and progress.net_displacement_meters is not None
            and progress.net_displacement_meters <= 0.25
            and not has_step_progress
        ):
            reversal_steps = [
                current
                for previous, current in zip(step_turns, step_turns[1:])
                if previous.observed_motion != current.observed_motion
            ]
            overrides["motion_oscillation"] = ErrorAssessment(
                verdict="PRESENT",
                confidence=min(1.0, max(0.9, retrace)),
                interval=TimeSpan(
                    start_seconds=step_turns[0].motion.start_seconds,
                    end_seconds=step_turns[-1].post_timestamp_seconds,
                ),
                evidence_timestamps_seconds=[
                    step_turns[0].post_timestamp_seconds,
                    reversal_steps[0].post_timestamp_seconds,
                    reversal_steps[-1].post_timestamp_seconds,
                ],
                reason=(
                    f"最近{len(request.steps)}步内发生至少两次左右反向切换，"
                    "视觉轨迹回扫且净位移很小。"
                ),
                source="RULE",
            )

        complete_discovery_coverage = all(
            value is not None for value in discoveries
        )
        if (
            progress.net_displacement_meters is not None
            and progress.net_displacement_meters <= 0.25
            and goal_improvement is not None
            and goal_improvement
            < self.config.goal_distance_progress_epsilon_meters
            and complete_discovery_coverage
            and all(value == 0 for value in discoveries)
            and progress.no_progress_steps is not None
            and progress.no_progress_steps >= self.config.get_nowhere_steps
        ):
            overrides["get_nowhere"] = ErrorAssessment(
                verdict="PRESENT",
                confidence=1.0,
                interval=TimeSpan(
                    start_seconds=request.steps[0].motion.start_seconds,
                    end_seconds=request.steps[-1].post_timestamp_seconds,
                ),
                evidence_timestamps_seconds=[
                    request.steps[0].motion.start_seconds,
                    request.steps[-1].post_timestamp_seconds,
                ],
                reason=(
                    f"最近{len(request.steps)}步净位移不超过0.25米、"
                    "目标距离无明显改善，"
                    "且没有新地标、节点或子目标。"
                ),
                source="RULE",
            )
        mismatches = [
            step for step in request.steps
            if step.action_match == "MISMATCH"
        ]
        if mismatches:
            sampled_mismatches = (
                mismatches
                if len(mismatches) <= 3
                else [
                    mismatches[0],
                    mismatches[len(mismatches) // 2],
                    mismatches[-1],
                ]
            )
            overrides["action_execution_mismatch"] = ErrorAssessment(
                verdict="PRESENT",
                confidence=max(
                    0.5,
                    max(step.motion.confidence for step in mismatches),
                ),
                interval=TimeSpan(
                    start_seconds=mismatches[0].motion.start_seconds,
                    end_seconds=mismatches[-1].post_timestamp_seconds,
                ),
                evidence_timestamps_seconds=[
                    step.post_timestamp_seconds
                    for step in sampled_mismatches
                ],
                reason=(
                    "结构化逐步证据显示至少一个环境命令与感知执行运动不一致。"
                ),
                source="RULE",
            )
        elif all(step.action_match == "MATCH" for step in request.steps):
            overrides["action_execution_mismatch"] = ErrorAssessment(
                verdict="ABSENT",
                confidence=1.0,
                interval=None,
                evidence_timestamps_seconds=[],
                reason=(
                    f"最近{len(request.steps)}个步骤的命令与"
                    "感知执行运动均匹配。"
                ),
                source="RULE",
            )

        for mode, override in overrides.items():
            model_assessment = getattr(errors, mode)
            reason = override.reason
            if model_assessment.verdict not in {
                override.verdict,
                "UNCERTAIN",
            }:
                reason = (
                    f"{reason} 模型判断为{model_assessment.verdict}，"
                    "结构化证据优先。"
                )
            confidence = override.confidence
            if model_assessment.verdict == override.verdict:
                confidence = max(confidence, model_assessment.confidence)
            setattr(
                errors,
                mode,
                override.model_copy(
                    update={
                        "source": "FUSED",
                        "reason": reason,
                        "confidence": confidence,
                    }
                ),
            )
        return errors

    def _reconcile_step_progress(
        self,
        model_progress: Literal[
            "PROGRESSING", "STALLED", "REGRESSING", "UNCERTAIN"
        ],
        request: TemporalAnalysisRequest,
        window: TemporalWindow,
        errors: ErrorAssessments,
    ) -> Literal["PROGRESSING", "STALLED", "REGRESSING", "UNCERTAIN"]:
        if errors.get_nowhere.verdict == "PRESENT":
            return (
                "REGRESSING"
                if model_progress == "REGRESSING"
                else "STALLED"
            )
        authoritative = self._authoritative_progress_state(window)
        if authoritative is not None:
            return authoritative
        if any(step.newly_completed_subgoals for step in request.steps):
            return "PROGRESSING"
        if (
            errors.motion_oscillation.verdict == "PRESENT"
            and model_progress == "PROGRESSING"
        ):
            return "UNCERTAIN"
        return model_progress

    def _validate_payload(
        self, payload: TemporalCaptionPayload, window: TemporalWindow
    ) -> None:
        if not payload.phases:
            raise TemporalOutputError("Temporal model must return at least one phase")
        if len(payload.phases) > 6:
            raise TemporalOutputError("Temporal model returned more than six phases")
        valid_times = self._evidence_timestamps(window)
        previous_end = window.start_seconds
        for phase in payload.phases:
            self._validate_span(phase.interval, window)
            if phase.interval.start_seconds < previous_end - 1e-9:
                raise TemporalOutputError(
                    "Temporal phases must be sorted and non-overlapping"
                )
            previous_end = phase.interval.end_seconds
            if not phase.evidence_timestamps_seconds:
                raise TemporalOutputError(
                    "Every temporal phase must cite evidence timestamps"
                )
            self._validate_evidence_times(
                phase.evidence_timestamps_seconds, valid_times
            )
        for mode in ERROR_MODES:
            assessment = getattr(payload.errors, mode)
            if assessment.interval is not None:
                self._validate_span(assessment.interval, window)
            self._validate_evidence_times(
                assessment.evidence_timestamps_seconds, valid_times
            )
            if assessment.verdict == "PRESENT":
                if assessment.interval is None:
                    raise TemporalOutputError(
                        f"PRESENT {mode} must contain an interval"
                    )
                if not assessment.evidence_timestamps_seconds:
                    raise TemporalOutputError(
                        f"PRESENT {mode} must cite evidence timestamps"
                    )
                if not assessment.reason.strip():
                    raise TemporalOutputError(
                        f"PRESENT {mode} must contain a reason"
                    )

    @staticmethod
    def _validate_span(span: TimeSpan, window: TemporalWindow) -> None:
        if span.end_seconds <= span.start_seconds:
            raise TemporalOutputError("Output interval end must exceed start")
        if (
            span.start_seconds < window.start_seconds - 1e-9
            or span.end_seconds > window.end_seconds + 1e-9
        ):
            raise TemporalOutputError("Output interval is outside the input window")

    def _validate_evidence_times(
        self, timestamps: Iterable[float], valid_times: Sequence[float]
    ) -> None:
        tolerance = self.config.evidence_time_tolerance_seconds
        for timestamp in timestamps:
            if not math.isfinite(float(timestamp)):
                raise TemporalOutputError("Evidence timestamp must be finite")
            if not any(
                abs(float(timestamp) - valid) <= tolerance
                for valid in valid_times
            ):
                raise TemporalOutputError(
                    f"Evidence timestamp {timestamp} does not match input evidence"
                )

    @staticmethod
    def _evidence_timestamps(window: TemporalWindow) -> tuple[float, ...]:
        timestamps = {
            window.start_seconds,
            window.end_seconds,
            *(frame.timestamp_seconds for frame in window.frames),
            *(action.timestamp_seconds for action in window.actions),
            *(signal.timestamp_seconds for signal in window.topology),
            *(signal.start_seconds for signal in window.motion),
            *(signal.end_seconds for signal in window.motion),
        }
        return tuple(sorted(timestamps))

    def _align_payload_to_inputs(
        self,
        payload: TemporalCaptionPayload,
        window: TemporalWindow,
    ) -> TemporalCaptionPayload:
        """Replace model-inferred facts that have authoritative typed inputs."""
        scene_summary = (
            self._static_scene_text(payload.scene_summary)
            or self._static_scene_text(payload.latest_scene)
            or "场景实体未知（模型输出混入动态断言）"
        )
        latest_scene = self._static_scene_text(payload.latest_scene)
        if latest_scene is None:
            latest_scene = f"{scene_summary}（最后帧细节不确定）"
        phases = [
            phase.model_copy(
                update={
                    "scene": (
                        self._static_scene_text(phase.scene)
                        or scene_summary
                    ),
                    "commanded_activity": self._command_for_phase(
                        phase.interval,
                        window,
                    ),
                    "observed_motion": self._motion_for_phase(
                        phase.interval,
                        window.motion,
                    ),
                }
            )
            for phase in payload.phases
        ]
        authoritative_progress = self._authoritative_progress_state(window)
        if (
            authoritative_progress in {"PROGRESSING", "REGRESSING"}
            and not any(
                phase.progress == authoritative_progress for phase in phases
            )
        ):
            # Aggregate progress is authoritative but not timestamped, so it
            # cannot be assigned to a specific phase.
            phases = [
                phase.model_copy(update={"progress": "UNCERTAIN"})
                for phase in phases
            ]
        overall_progress = self._reconcile_progress(
            payload.overall_progress,
            phases,
            window,
        )

        errors = payload.errors.model_copy(deep=True)
        labels = {
            "collision": "碰撞",
            "repeated_visit": "重复访问",
            "motion_oscillation": "运动振荡",
            "get_nowhere": "无进展",
            "action_execution_mismatch": "命令与执行不一致",
        }
        threshold = self.config.minimum_definite_error_confidence
        for mode in ERROR_MODES:
            # `source` is provenance owned by this adapter, not a fact the
            # foundation model is allowed to assert.
            assessment = getattr(errors, mode).model_copy(
                update={"source": "MODEL"}
            )
            if (
                mode == "motion_oscillation"
                and assessment.verdict == "PRESENT"
                and self._has_positive_progress_evidence(window)
            ):
                assessment = assessment.model_copy(
                    update={
                        "verdict": "UNCERTAIN",
                        "confidence": 0.5,
                        "reason": (
                            "视觉回扫与明确进展信号冲突，"
                            "不能确认运动振荡。"
                        ),
                    }
                )
            elif (
                assessment.verdict == "PRESENT"
                and self._has_negative_error_coverage(mode, window)
            ):
                assessment = assessment.model_copy(
                    update={
                        "verdict": "UNCERTAIN",
                        "confidence": 0.5,
                        "reason": (
                            f"结构化证据与模型的{labels[mode]}判断冲突，"
                            "改为不确定。"
                        ),
                    }
                )
            elif (
                assessment.verdict == "ABSENT"
                and not self._has_negative_error_coverage(mode, window)
            ):
                assessment = assessment.model_copy(
                    update={
                        "verdict": "UNCERTAIN",
                        "confidence": 0.5,
                        "reason": (
                            f"缺少足以排除{labels[mode]}的观测覆盖，"
                            "改为不确定。"
                        ),
                    }
                )
            elif (
                assessment.verdict in {"PRESENT", "ABSENT"}
                and assessment.confidence < threshold
            ):
                original = (
                    f"判断{labels[mode]}发生"
                    if assessment.verdict == "PRESENT"
                    else f"判断未发生{labels[mode]}"
                )
                assessment = assessment.model_copy(
                    update={
                        "verdict": "UNCERTAIN",
                        "confidence": 0.5,
                        "reason": (
                            f"原模型低置信度{original}，"
                            "证据不足，改为不确定。"
                        ),
                    }
                )
            setattr(errors, mode, assessment)

        return payload.model_copy(
            update={
                "phases": phases,
                "errors": errors,
                "overall_progress": overall_progress,
                "latest_scene": latest_scene,
                "scene_summary": scene_summary,
                "activity_summary": self._ground_activity_summary(
                    window,
                ),
            }
        )

    def _command_for_phase(
        self,
        span: TimeSpan,
        window: TemporalWindow,
    ) -> Literal[
        "FORWARD",
        "TURN_LEFT",
        "TURN_RIGHT",
        "STOP",
        "NONE",
        "MIXED",
        "UNKNOWN",
    ]:
        tolerance = self.config.evidence_time_tolerance_seconds
        phase_actions = []
        for index, action in enumerate(window.actions):
            action_end = (
                window.actions[index + 1].timestamp_seconds
                if index + 1 < len(window.actions)
                else window.end_seconds
            )
            overlap = min(span.end_seconds, action_end) - max(
                span.start_seconds,
                action.timestamp_seconds,
            )
            if overlap > tolerance:
                phase_actions.append(action.action)
        unique = tuple(dict.fromkeys(phase_actions))
        if not unique:
            return "NONE"
        if len(unique) == 1:
            return unique[0]  # type: ignore[return-value]
        return "MIXED"

    def _motion_for_phase(
        self,
        span: TimeSpan,
        motion: Sequence[MotionSignal],
    ) -> CameraMotion:
        tolerance = self.config.evidence_time_tolerance_seconds
        values = [
            signal.camera_motion
            for signal in motion
            if (
                signal.camera_motion != CameraMotion.UNKNOWN
                and signal.confidence
                >= self.config.minimum_motion_rule_confidence
                and min(span.end_seconds, signal.end_seconds)
                - max(span.start_seconds, signal.start_seconds)
                > tolerance
            )
        ]
        unique = tuple(dict.fromkeys(values))
        if not unique:
            return CameraMotion.UNKNOWN
        if len(unique) == 1:
            return unique[0]
        if set(unique) <= {
            CameraMotion.TURN_LEFT,
            CameraMotion.TURN_RIGHT,
            CameraMotion.OSCILLATING_TURN,
        }:
            return CameraMotion.OSCILLATING_TURN
        return CameraMotion.UNKNOWN

    @staticmethod
    def _static_scene_text(text: str) -> Optional[str]:
        """Keep visible entities/layout and reject dynamic/action assertions."""
        kept: list[str] = []
        for raw_clause in re.split(r"[,，;；。.!！?？\n]+", text):
            clause = raw_clause.strip(" \t:：")
            if not clause:
                continue
            # Strip neutral viewpoint wrappers without treating the entire
            # clause as a motion assertion.  This preserves facts such as
            # "画面显示前方有一扇门" while keeping memory text concise.
            clause = re.sub(
                (
                    r"^(?:当前)?(?:画面|视野|镜头)"
                    r"(?:中|内)?(?:显示|可见|呈现|看到)?"
                ),
                "",
                clause,
            ).strip(" \t:：")
            clause = re.sub(
                (
                    r"^(?:机器人|相机)(?=(?:的)?"
                    r"(?:前方|后方|左侧|右侧|附近|面前))"
                ),
                "",
                clause,
            ).strip(" \t:：")
            if not clause:
                continue
            normalized = clause.upper()
            if not any(term in normalized for term in SCENE_DYNAMIC_TERMS):
                kept.append(clause)
                continue
            for marker in ("看到", "可见", "出现"):
                if marker not in clause:
                    continue
                candidate = clause.split(marker, 1)[1].strip(" \t:：")
                candidate_normalized = candidate.upper()
                if candidate and not any(
                    term in candidate_normalized
                    for term in SCENE_DYNAMIC_TERMS
                ):
                    kept.append(candidate)
                break
        unique = tuple(dict.fromkeys(kept))
        return "、".join(unique) if unique else None

    def _ground_activity_summary(
        self,
        window: TemporalWindow,
    ) -> str:
        descriptions = {
            CameraMotion.STATIONARY: "基本停滞",
            CameraMotion.FORWARD: "前进",
            CameraMotion.TURN_LEFT: "向左转",
            CameraMotion.TURN_RIGHT: "向右转",
            CameraMotion.OSCILLATING_TURN: "往复转向",
        }
        reliable = [
            signal
            for signal in window.motion
            if (
                signal.camera_motion in descriptions
                and signal.confidence
                >= self.config.minimum_motion_rule_confidence
            )
        ]
        if not reliable:
            if window.actions:
                commands = "；".join(
                    f"{action.timestamp_seconds:.1f}秒发出{action.action}"
                    for action in window.actions
                )
                return f"{commands}；没有可靠的执行运动证据"
            return "没有可靠的运动或动作证据"
        timeline = "；".join(
            f"{signal.start_seconds:.1f}–{signal.end_seconds:.1f}秒"
            f"相机{descriptions[signal.camera_motion]}"
            for signal in reliable
        )
        return timeline

    def _reconcile_progress(
        self,
        overall: Literal[
            "PROGRESSING", "STALLED", "REGRESSING", "UNCERTAIN"
        ],
        phases: Sequence[TemporalPhase],
        window: TemporalWindow,
    ) -> Literal["PROGRESSING", "STALLED", "REGRESSING", "UNCERTAIN"]:
        authoritative = self._authoritative_progress_state(window)
        if authoritative is not None:
            return authoritative
        phase_states = {phase.progress for phase in phases}
        if len(phase_states) != 1:
            return overall
        only = next(iter(phase_states))
        if only == "UNCERTAIN" or only == overall:
            return overall
        contradictory = {
            ("PROGRESSING", "STALLED"),
            ("PROGRESSING", "REGRESSING"),
            ("STALLED", "PROGRESSING"),
            ("REGRESSING", "PROGRESSING"),
        }
        if (overall, only) in contradictory:
            return "UNCERTAIN"
        return overall

    def _has_negative_error_coverage(
        self,
        mode: str,
        window: TemporalWindow,
    ) -> bool:
        if mode == "collision":
            covered = sum(
                signal.end_seconds - signal.start_seconds
                for signal in window.motion
                if signal.collision is False
            )
            duration = window.end_seconds - window.start_seconds
            return covered >= 0.8 * duration
        if mode == "repeated_visit":
            visits = [
                signal.visit_count
                for signal in window.topology
                if signal.visit_count is not None
            ]
            return len(visits) >= 2 and all(count <= 1 for count in visits)
        if mode == "motion_oscillation":
            reliable = [
                signal
                for signal in window.motion
                if (
                    signal.camera_motion != CameraMotion.UNKNOWN
                    and signal.confidence
                    >= self.config.minimum_motion_rule_confidence
                )
            ]
            covered = sum(
                signal.end_seconds - signal.start_seconds
                for signal in reliable
            )
            duration = window.end_seconds - window.start_seconds
            directions = [
                signal.camera_motion
                for signal in reliable
                if signal.camera_motion
                in {CameraMotion.TURN_LEFT, CameraMotion.TURN_RIGHT}
            ]
            has_reversal = any(
                previous != current
                for previous, current in zip(directions, directions[1:])
            )
            retrace = window.reverse_retrace_similarity
            has_retraced_reversal = bool(
                has_reversal
                and retrace is not None
                and retrace >= self.config.oscillation_retrace_threshold
            )
            return (
                covered >= 0.8 * duration
                and not has_retraced_reversal
            )
        if mode == "get_nowhere":
            return self._has_positive_progress_evidence(window)
        if mode == "action_execution_mismatch":
            # Legacy TemporalWindow has no explicit per-step command/execution
            # match field, so it cannot prove absence of this fifth mode.
            return False
        raise ValueError(f"Unknown error mode: {mode}")

    @staticmethod
    def _has_positive_progress(progress: ProgressSignals) -> bool:
        return bool(
            (
                progress.net_displacement_meters is not None
                and progress.net_displacement_meters > 0.05
            )
            or any(
                value is not None and value > 0
                for value in (
                    progress.new_landmarks_count,
                    progress.new_topological_nodes_count,
                    progress.completed_subgoals_count,
                )
            )
        )

    def _goal_distance_progress_state(
        self,
        topology: Sequence[TopologySignal],
    ) -> Optional[Literal["PROGRESSING", "REGRESSING", "STALLED"]]:
        distances = [
            signal.distance_to_goal_meters
            for signal in topology
            if signal.distance_to_goal_meters is not None
        ]
        if len(distances) < 2:
            return None
        delta = distances[-1] - distances[0]
        epsilon = self.config.goal_distance_progress_epsilon_meters
        if delta < -epsilon:
            return "PROGRESSING"
        if delta > epsilon:
            return "REGRESSING"
        return "STALLED"

    def _has_positive_progress_evidence(
        self,
        window: TemporalWindow,
    ) -> bool:
        return (
            self._has_positive_progress(window.progress)
            or self._goal_distance_progress_state(window.topology)
            == "PROGRESSING"
        )

    def _authoritative_progress_state(
        self,
        window: TemporalWindow,
    ) -> Optional[
        Literal["PROGRESSING", "STALLED", "REGRESSING", "UNCERTAIN"]
    ]:
        topology_state = self._goal_distance_progress_state(window.topology)
        # Goal-distance deltas come directly from the environment and determine
        # whether spatial motion approached or moved away from the goal.  Net
        # displacement alone must not turn a reliable REGRESSING signal into
        # UNCERTAIN merely because the robot moved a non-zero distance.
        if topology_state in {"PROGRESSING", "REGRESSING"}:
            return topology_state
        if self._has_positive_progress(window.progress):
            return "PROGRESSING"
        return None

    def _apply_rule_overrides(
        self,
        payload: TemporalCaptionPayload,
        window: TemporalWindow,
    ) -> TemporalCaptionPayload:
        overrides = self._rule_overrides(window)
        if not overrides:
            return payload
        errors = payload.errors.model_copy(deep=True)
        for mode, override in overrides.items():
            model_assessment = getattr(errors, mode)
            reason = override.reason
            model_reason = model_assessment.reason.strip()
            if (
                model_assessment.verdict == "PRESENT"
                and model_reason
                and model_reason not in reason
                and reason not in model_reason
            ):
                reason = f"{reason} 模型补充：{model_reason}"
            if model_assessment.verdict not in {
                override.verdict,
                "UNCERTAIN",
            }:
                reason = (
                    f"{reason} 模型判断为{model_assessment.verdict}，"
                    "确定性规则证据优先。"
                )
            confidence = override.confidence
            if model_assessment.verdict == override.verdict:
                confidence = max(
                    override.confidence,
                    model_assessment.confidence,
                )
            override = override.model_copy(
                update={
                    "source": "FUSED",
                    "reason": reason,
                    "confidence": confidence,
                }
            )
            setattr(errors, mode, override)

        phases = list(payload.phases)
        overall_progress = payload.overall_progress
        if (
            "motion_oscillation" in overrides
            and overall_progress == "PROGRESSING"
        ):
            # Reverse visual retracing proves a motion pattern, but without
            # task-progress sensors it cannot prove either progress or failure.
            overall_progress = "UNCERTAIN"
        if "get_nowhere" in overrides:
            overall_progress = (
                "REGRESSING"
                if overall_progress == "REGRESSING"
                else "STALLED"
            )
            phases = [
                phase.model_copy(
                    update={
                        "progress": (
                            phase.progress
                            if phase.progress == "REGRESSING"
                            else "STALLED"
                        )
                    }
                )
                for phase in phases
            ]
        return payload.model_copy(
            update={
                "errors": errors,
                "phases": phases,
                "overall_progress": overall_progress,
            }
        )

    def _rule_hints(self, window: TemporalWindow) -> dict[str, Any]:
        overrides = self._rule_overrides(window)
        directions = [
            signal.camera_motion.value
            for signal in window.motion
            if signal.camera_motion
            in {CameraMotion.TURN_LEFT, CameraMotion.TURN_RIGHT}
        ]
        reversals = sum(
            previous != current
            for previous, current in zip(directions, directions[1:])
        )
        return {
            "turn_direction_runs": directions,
            "turn_reversal_count": reversals,
            "hard_or_high_confidence_candidates": {
                mode: assessment.model_dump(mode="json")
                for mode, assessment in overrides.items()
            },
            "warning": (
                "These are deterministic evidence checks. The model may add "
                "uncertainty but must not contradict hard sensor evidence."
            ),
        }

    def _rule_overrides(
        self, window: TemporalWindow
    ) -> dict[str, ErrorAssessment]:
        overrides: dict[str, ErrorAssessment] = {}
        collisions = [
            signal
            for signal in window.motion
            if signal.collision is True
        ]
        if collisions:
            overrides["collision"] = ErrorAssessment(
                verdict="PRESENT",
                confidence=1.0,
                interval=TimeSpan(
                    start_seconds=collisions[0].start_seconds,
                    end_seconds=collisions[-1].end_seconds,
                ),
                evidence_timestamps_seconds=[
                    collisions[0].start_seconds,
                    collisions[-1].end_seconds,
                ],
                reason="碰撞传感器或环境信号明确报告碰撞。",
                source="RULE",
            )

        repeated = [
            signal
            for signal in window.topology
            if signal.visit_count is not None and signal.visit_count > 1
        ]
        if repeated:
            timestamps = [signal.timestamp_seconds for signal in repeated]
            if (
                window.timestamp_semantics
                == "episode_step_post_action_seconds"
            ):
                first_revisit = timestamps[0]
                containing_motion = [
                    signal
                    for signal in window.motion
                    if (
                        signal.start_seconds < first_revisit
                        <= signal.end_seconds
                        + self.config.evidence_time_tolerance_seconds
                    )
                ]
                interval_start = (
                    containing_motion[0].start_seconds
                    if containing_motion
                    else window.start_seconds
                )
                interval_end = timestamps[-1]
            else:
                interval_start = max(
                    window.start_seconds,
                    timestamps[0],
                )
                interval_end = min(
                    window.end_seconds,
                    max(timestamps[-1], timestamps[0] + 1e-3),
                )
            overrides["repeated_visit"] = ErrorAssessment(
                verdict="PRESENT",
                confidence=1.0,
                interval=TimeSpan(
                    start_seconds=interval_start,
                    end_seconds=interval_end,
                ),
                evidence_timestamps_seconds=timestamps,
                reason="拓扑信号显示节点访问次数大于一次。",
                source="RULE",
            )

        directions = [
            signal
            for signal in window.motion
            if (
                signal.camera_motion
                in {CameraMotion.TURN_LEFT, CameraMotion.TURN_RIGHT}
                and signal.confidence
                >= self.config.minimum_motion_rule_confidence
                and signal.end_seconds - signal.start_seconds
                >= self.config.minimum_motion_phase_seconds
            )
        ]
        reversals = [
            (previous, current)
            for previous, current in zip(directions, directions[1:])
            if previous.camera_motion != current.camera_motion
        ]
        retrace = window.reverse_retrace_similarity
        if (
            reversals
            and retrace is not None
            and retrace >= self.config.oscillation_retrace_threshold
            and not self._has_positive_progress_evidence(window)
        ):
            evidence = [
                directions[0].start_seconds,
                reversals[0][1].start_seconds,
                directions[-1].end_seconds,
            ]
            overrides["motion_oscillation"] = ErrorAssessment(
                verdict="PRESENT",
                confidence=min(1.0, max(0.9, retrace)),
                interval=TimeSpan(
                    start_seconds=directions[0].start_seconds,
                    end_seconds=directions[-1].end_seconds,
                ),
                evidence_timestamps_seconds=evidence,
                reason=(
                    "转向方向发生反转，且反转前后的视觉轨迹高度反向重合 "
                    f"(similarity={retrace:.3f})。"
                ),
                source="RULE",
            )

        progress = window.progress
        discovery_measurements = (
            progress.new_landmarks_count,
            progress.new_topological_nodes_count,
            progress.completed_subgoals_count,
        )
        complete_discovery_coverage = all(
            value is not None for value in discovery_measurements
        )
        no_discoveries = (
            complete_discovery_coverage
            and all(value == 0 for value in discovery_measurements)
        )
        low_displacement = (
            progress.net_displacement_meters is not None
            and progress.net_displacement_meters <= 0.05
        )
        enough_steps = (
            progress.no_progress_steps is not None
            and progress.no_progress_steps >= self.config.get_nowhere_steps
        )
        if (
            complete_discovery_coverage
            and no_discoveries
            and low_displacement
            and enough_steps
            and not self._has_positive_progress_evidence(window)
        ):
            overrides["get_nowhere"] = ErrorAssessment(
                verdict="PRESENT",
                confidence=1.0,
                interval=TimeSpan(
                    start_seconds=window.start_seconds,
                    end_seconds=window.end_seconds,
                ),
                evidence_timestamps_seconds=[
                    window.start_seconds,
                    window.end_seconds,
                ],
                reason=(
                    "多个步骤没有净位移、新地标、新拓扑节点或子目标进展。"
                ),
                source="RULE",
            )
        return overrides

    def _image_to_png_bytes(self, image: Any) -> bytes:
        from PIL import Image

        rgb = self._image_to_pil(image)
        if max(rgb.size) > self.config.max_image_edge:
            scale = self.config.max_image_edge / max(rgb.size)
            rgb = rgb.resize(
                (
                    max(1, round(rgb.width * scale)),
                    max(1, round(rgb.height * scale)),
                ),
                Image.Resampling.LANCZOS,
            )
        buffer = io.BytesIO()
        rgb.save(buffer, format="PNG")
        return buffer.getvalue()

    @staticmethod
    def _image_to_pil(image: Any) -> Any:
        from PIL import Image

        try:
            if isinstance(image, bytes):
                if not image:
                    raise ValueError("empty image bytes")
                with Image.open(io.BytesIO(image)) as decoded:
                    return decoded.convert("RGB")
            if isinstance(image, (str, Path)):
                with Image.open(Path(image).expanduser()) as decoded:
                    return decoded.convert("RGB")
            if isinstance(image, Image.Image):
                return image.convert("RGB")
            if hasattr(image, "shape"):
                import numpy as np

                array = np.asarray(image)
                if array.ndim != 3 or array.shape[2] not in {3, 4}:
                    raise ValueError("expected HWC RGB/RGBA image")
                if array.dtype != np.uint8:
                    if np.issubdtype(array.dtype, np.floating):
                        maximum = float(np.nanmax(array))
                        if maximum <= 1.0:
                            array = array * 255
                    array = np.clip(array, 0, 255).astype(np.uint8)
                return Image.fromarray(array).convert("RGB")
        except Exception as exc:
            raise TemporalInputError("Could not decode RGB frame") from exc
        raise TemporalInputError(
            "Frame image must be bytes, path, PIL image, or HWC RGB numpy array"
        )

    @classmethod
    def _image_to_rgb_array(cls, image: Any) -> Any:
        import numpy as np

        return np.asarray(cls._image_to_pil(image))

    @staticmethod
    def _validate_crop(
        crop: tuple[int, int, int, int],
        frame_shape: Sequence[int],
    ) -> tuple[int, int, int, int]:
        if len(crop) != 4:
            raise TemporalInputError("crop must be (x0, y0, x1, y1)")
        x0, y0, x1, y1 = (int(value) for value in crop)
        height, width = frame_shape[:2]
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise TemporalInputError(
                f"crop {crop!r} is outside frame {width}x{height}"
            )
        return x0, y0, x1, y1

    @staticmethod
    def _uniform_cap(values: Sequence[int], maximum: int) -> list[int]:
        if len(values) <= maximum:
            return list(values)
        return [
            values[round(index * (len(values) - 1) / (maximum - 1))]
            for index in range(maximum)
        ]

    @staticmethod
    def _smooth_motion_labels(samples: list[dict[str, Any]]) -> None:
        original = [sample["motion"] for sample in samples]
        turning = {CameraMotion.TURN_LEFT, CameraMotion.TURN_RIGHT}
        for index in range(len(samples)):
            neighborhood = original[max(0, index - 2) : index + 3]
            left_count = neighborhood.count(CameraMotion.TURN_LEFT)
            right_count = neighborhood.count(CameraMotion.TURN_RIGHT)
            if max(left_count, right_count) >= 3:
                samples[index]["motion"] = (
                    CameraMotion.TURN_LEFT
                    if left_count > right_count
                    else CameraMotion.TURN_RIGHT
                )
            elif (
                original[index] == CameraMotion.UNKNOWN
                and 0 < index < len(samples) - 1
                and original[index - 1] == original[index + 1]
                and original[index - 1] in turning
            ):
                samples[index]["motion"] = original[index - 1]

    def _group_motion_samples(
        self, samples: Sequence[dict[str, Any]]
    ) -> tuple[MotionSignal, ...]:
        if not samples:
            return ()
        groups: list[list[dict[str, Any]]] = [[samples[0]]]
        for sample in samples[1:]:
            if sample["motion"] == groups[-1][-1]["motion"]:
                groups[-1].append(sample)
            else:
                groups.append([sample])

        # Fill a bounded low-confidence gap when the reliable turn direction
        # before and after it agrees. This preserves the long phase while
        # refusing to erase an actual LEFT<->RIGHT reversal.
        coalesced: list[list[dict[str, Any]]] = []
        index = 0
        turning = {CameraMotion.TURN_LEFT, CameraMotion.TURN_RIGHT}
        while index < len(groups):
            group = groups[index]
            motion = group[0]["motion"]
            duration = group[-1]["end"] - group[0]["start"]
            if (
                motion in {CameraMotion.UNKNOWN, CameraMotion.STATIONARY}
                and duration <= self.config.maximum_unknown_motion_gap_seconds
                and coalesced
                and index + 1 < len(groups)
                and coalesced[-1][0]["motion"] == groups[index + 1][0]["motion"]
                and coalesced[-1][0]["motion"] in turning
            ):
                replacement = coalesced[-1][0]["motion"]
                for sample in group:
                    sample["motion"] = replacement
                for sample in groups[index + 1]:
                    sample["motion"] = replacement
                coalesced[-1].extend(group)
                coalesced[-1].extend(groups[index + 1])
                index += 2
                continue
            if (
                motion in {CameraMotion.UNKNOWN, CameraMotion.STATIONARY}
                and duration <= self.config.minimum_motion_phase_seconds
                and not coalesced
                and index + 1 < len(groups)
                and groups[index + 1][0]["motion"] in turning
            ):
                replacement = groups[index + 1][0]["motion"]
                for sample in group:
                    sample["motion"] = replacement
                for sample in groups[index + 1]:
                    sample["motion"] = replacement
                coalesced.append(group + groups[index + 1])
                index += 2
                continue
            coalesced.append(group)
            index += 1
        groups = coalesced

        merged: list[list[dict[str, Any]]] = []
        for group in groups:
            duration = group[-1]["end"] - group[0]["start"]
            if (
                duration < self.config.minimum_motion_phase_seconds
                and merged
                and group[-1]["motion"] in {
                    CameraMotion.UNKNOWN,
                    CameraMotion.STATIONARY,
                }
            ):
                merged[-1].extend(group)
            else:
                merged.append(group)

        result = []
        for group in merged:
            dx_values = [
                sample["dx"] for sample in group if sample["dx"] is not None
            ]
            magnitude_values = [
                sample["magnitude"]
                for sample in group
                if sample["magnitude"] is not None
            ]
            result.append(
                MotionSignal(
                    start_seconds=group[0]["start"],
                    end_seconds=group[-1]["end"],
                    camera_motion=group[0]["motion"],
                    scene_flow_dx_fraction=(
                        statistics.median(dx_values) if dx_values else None
                    ),
                    scene_flow_magnitude_fraction=(
                        statistics.median(magnitude_values)
                        if magnitude_values
                        else None
                    ),
                    confidence=statistics.mean(
                        sample["confidence"] for sample in group
                    ),
                    source="opencv_farneback",
                    quality_note="; ".join(
                        dict.fromkeys(
                            sample["quality_note"]
                            for sample in group
                            if sample.get("quality_note")
                        )
                    )
                    or None,
                )
            )
        return tuple(result)


__all__ = (
    "ACTION_TOKENS",
    "ActionMatch",
    "CameraMotion",
    "DEFAULT_MODEL_PATH",
    "ErrorAssessment",
    "ErrorAssessments",
    "MotionSignal",
    "ProgressSignals",
    "TemporalCaptionPayload",
    "TemporalAnalysisRequest",
    "TemporalCaptioner",
    "TemporalCaptionerConfig",
    "TemporalCaptionerError",
    "TemporalInferenceError",
    "TemporalInputError",
    "TemporalMemoryRecord",
    "TemporalOutputError",
    "TemporalPhase",
    "TemporalStepCaption",
    "TemporalStepCaptionPayload",
    "TemporalStepErrorAssessments",
    "TemporalStepInput",
    "TemporalStepModelCaption",
    "TemporalStepModelErrorHint",
    "TemporalStepModelPayload",
    "TemporalWindow",
    "TimedAction",
    "TimestampedFrame",
    "TimeSpan",
    "TopologySignal",
)
