"""Action-aligned temporal memory for visual-language navigation.

The memory owns the transition boundary between an observation and an action.
An action staged on observation ``O_t`` is only completed when ``O_(t+1)``
arrives, producing the unambiguous transition::

    (pre_observation=O_t, commanded_action=A_t, post_observation=O_(t+1))

Only completed transitions enter the rolling video-understanding window.  This
prevents the common off-by-one error where an action is described using the
image that preceded it.
"""

from __future__ import annotations

import math
import re
import time
from collections import deque
from dataclasses import dataclass, field, replace
from enum import Enum
from numbers import Integral
from typing import Any, Deque, Literal, Optional, Sequence

from ..TemporalCaptioner import (
    ACTION_TOKENS,
    CameraMotion,
    ErrorAssessment,
    MotionSignal,
    ProgressSignals,
    TemporalCaptioner,
    TemporalMemoryRecord,
    TemporalWindow,
    TimedAction,
    TimestampedFrame,
    TimeSpan,
    TopologySignal,
)


ActionMatch = Literal["MATCH", "MISMATCH", "UNCERTAIN"]
ErrorVerdict = Literal["PRESENT", "ABSENT", "UNCERTAIN"]
ProgressVerdict = Literal["PROGRESSING", "STALLED", "REGRESSING", "UNCERTAIN"]

_ACTION_ALIASES = {
    "MOVE_FORWARD": "FORWARD",
    "LEFT": "TURN_LEFT",
    "RIGHT": "TURN_RIGHT",
}
_COMPLETE_SUBGOAL = re.compile(
    r"^\s*(?P<id>\d+)\.\s*.*?\bCompletion\s+status\s*:\s*COMPLETE\b",
    re.IGNORECASE | re.MULTILINE,
)


class TemporalMemoryError(RuntimeError):
    """Base error for invalid Temporal Memory state transitions."""


class TemporalStateError(TemporalMemoryError):
    """The caller attempted to complete or stage an invalid transition."""


class CumulativeErrorMode(str, Enum):
    """Long-horizon visual failure modes accumulated across short windows."""

    WALL_STUCK = "WALL_STUCK"
    TURN_OSCILLATION = "TURN_OSCILLATION"
    IN_PLACE_SPIN = "IN_PLACE_SPIN"


class CumulativeErrorPhase(str, Enum):
    """Lifecycle of one cumulative temporal error."""

    NORMAL = "NORMAL"
    SUSPECTED = "SUSPECTED"
    CONFIRMED = "CONFIRMED"
    RECOVERING = "RECOVERING"
    COOLDOWN = "COOLDOWN"


def _normalize_action(action: str) -> str:
    normalized = str(action).strip().upper()
    normalized = _ACTION_ALIASES.get(normalized, normalized)
    if normalized not in ACTION_TOKENS:
        raise ValueError(
            f"Unsupported action {action!r}; expected one of {ACTION_TOKENS}"
        )
    return normalized


def _finite_non_negative(value: float, label: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return normalized


def _optional_finite(value: Optional[float], label: str) -> Optional[float]:
    if value is None:
        return None
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{label} must be finite when provided")
    return normalized


@dataclass(frozen=True, slots=True)
class TemporalObservation:
    """One RGB observation plus optional authoritative environment metadata."""

    image: Any = field(repr=False)
    episode_id: str = "episode-0"
    timestamp_seconds: float = 0.0
    position_xyz: Optional[tuple[float, float, float]] = None
    yaw_degrees: Optional[float] = None
    distance_to_goal_meters: Optional[float] = None
    landmark_ids: Optional[tuple[str, ...]] = None

    def __post_init__(self) -> None:
        if self.image is None:
            raise ValueError("image must not be None")
        episode_id = str(self.episode_id).strip()
        if not episode_id:
            raise ValueError("episode_id must not be empty")
        object.__setattr__(self, "episode_id", episode_id)
        object.__setattr__(
            self,
            "timestamp_seconds",
            _finite_non_negative(self.timestamp_seconds, "timestamp_seconds"),
        )
        if self.position_xyz is not None:
            if len(self.position_xyz) != 3:
                raise ValueError("position_xyz must contain exactly three values")
            position = tuple(
                float(_optional_finite(value, "position_xyz") or 0.0)
                for value in self.position_xyz
            )
            object.__setattr__(self, "position_xyz", position)
        object.__setattr__(
            self,
            "yaw_degrees",
            _optional_finite(self.yaw_degrees, "yaw_degrees"),
        )
        distance = _optional_finite(
            self.distance_to_goal_meters,
            "distance_to_goal_meters",
        )
        if distance is not None and distance < 0:
            raise ValueError("distance_to_goal_meters must be non-negative")
        object.__setattr__(self, "distance_to_goal_meters", distance)
        if self.landmark_ids is not None:
            normalized_landmarks = tuple(
                dict.fromkeys(
                    str(landmark).strip()
                    for landmark in self.landmark_ids
                    if str(landmark).strip()
                )
            )
            object.__setattr__(self, "landmark_ids", normalized_landmarks)


@dataclass(frozen=True, slots=True)
class StepExecution:
    """The command actually sent to the environment and its outcome."""

    step_id: int
    commanded_action: str
    collision: Optional[bool] = None
    terminal: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.step_id, bool)
            or not isinstance(self.step_id, Integral)
            or self.step_id < 0
        ):
            raise ValueError("step_id must be a non-negative integer")
        object.__setattr__(self, "step_id", int(self.step_id))
        object.__setattr__(
            self,
            "commanded_action",
            _normalize_action(self.commanded_action),
        )
        if self.collision is not None and not isinstance(self.collision, bool):
            raise ValueError("collision must be bool or None")
        if not isinstance(self.terminal, bool):
            raise ValueError("terminal must be bool")


@dataclass(frozen=True, slots=True)
class TemporalStep:
    """One completed action transition safe to add to a temporal window."""

    step_id: int
    selected_action: str
    commanded_action: str
    pre_observation: TemporalObservation = field(repr=False)
    post_observation: TemporalObservation = field(repr=False)
    observed_action: str
    action_match: ActionMatch
    motion: MotionSignal
    topology_node_id: Optional[str]
    is_new_node: Optional[bool]
    is_revisit: Optional[bool]
    newly_completed_subgoals: tuple[str, ...] = ()
    newly_discovered_landmarks: tuple[str, ...] = ()
    subgoal_evidence_known: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.step_id, bool)
            or not isinstance(self.step_id, Integral)
            or self.step_id < 0
        ):
            raise ValueError("step_id must be a non-negative integer")
        object.__setattr__(self, "step_id", int(self.step_id))
        object.__setattr__(
            self, "selected_action", _normalize_action(self.selected_action)
        )
        object.__setattr__(
            self, "commanded_action", _normalize_action(self.commanded_action)
        )
        observed = str(self.observed_action).strip().upper()
        valid_observed = {motion.value for motion in CameraMotion}
        if observed not in valid_observed:
            raise ValueError(
                f"observed_action must be one of {sorted(valid_observed)}"
            )
        object.__setattr__(self, "observed_action", observed)
        if self.action_match not in {"MATCH", "MISMATCH", "UNCERTAIN"}:
            raise ValueError("action_match has an invalid value")
        object.__setattr__(
            self,
            "newly_completed_subgoals",
            tuple(self.newly_completed_subgoals),
        )
        object.__setattr__(
            self,
            "newly_discovered_landmarks",
            tuple(self.newly_discovered_landmarks),
        )
        if not isinstance(self.subgoal_evidence_known, bool):
            raise ValueError("subgoal_evidence_known must be bool")


@dataclass(frozen=True, slots=True)
class TemporalEvidenceStep:
    """Image-free episode history used for cumulative error detection.

    ``scene_descriptor`` is a small derived grayscale descriptor, never the
    original RGB observation.  Keeping this history separate from
    :class:`TemporalStep` lets the video-understanding window stay at three
    images while cumulative failures can use a longer temporal horizon.
    """

    step_id: int
    commanded_action: str
    observed_motion: str
    motion_confidence: float
    action_match: ActionMatch
    scene_descriptor: tuple[float, ...] = field(repr=False)
    frame_similarity: Optional[float]
    topology_node_id: Optional[str]
    is_new_node: Optional[bool]
    newly_completed_subgoals: tuple[str, ...] = ()
    newly_discovered_landmarks: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CumulativeErrorState:
    """Dominant cumulative failure state exposed to the memory interface."""

    mode: Optional[CumulativeErrorMode]
    phase: CumulativeErrorPhase
    score: float
    evidence_step_ids: tuple[int, ...] = ()
    first_detected_step: Optional[int] = None
    last_detected_step: Optional[int] = None
    reason: str = ""


@dataclass(frozen=True, slots=True)
class GoBackRequest:
    """Typed recovery request emitted only for a confirmed cumulative error.

    Actions are ordinary VLN primitives so a later Agent-side wiring layer can
    consume this request without adding a Habitat-specific action.
    """

    request_id: str
    trigger_step_id: int
    error_mode: CumulativeErrorMode
    safe_checkpoint_step_id: Optional[int]
    recovery_actions: tuple[str, ...]
    reason: str

    def __post_init__(self) -> None:
        normalized = tuple(_normalize_action(action) for action in self.recovery_actions)
        if any(
            action not in {"FORWARD", "TURN_LEFT", "TURN_RIGHT"}
            for action in normalized
        ):
            raise ValueError(
                "GoBackRequest actions must be FORWARD, TURN_LEFT, or TURN_RIGHT"
            )
        object.__setattr__(self, "recovery_actions", normalized)


@dataclass(frozen=True, slots=True)
class _CumulativeSignal:
    """One detector update before state-machine hysteresis is applied."""

    suspected: bool
    confirmed: bool
    score: float
    evidence_step_ids: tuple[int, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class TemporalMemoryConfig:
    """Window, geometry, optical-flow, and error-rule configuration."""

    window_size: int = 3
    # Analyze once per fresh three-step group. Deterministic rules still update
    # after every step, while the expensive foundation model is amortized.
    analysis_stride: int = 3
    inference_mode: Literal["blocking"] = "blocking"
    get_nowhere_steps: int = 3
    inference_latency_budget_ms: float = 5000.0
    forward_displacement_meters: float = 0.05
    turning_yaw_degrees: float = 5.0
    topology_radius_meters: float = 0.5
    revisit_min_step_gap: int = 3
    oscillation_retrace_min_step_gap: int = 2
    revisit_visual_similarity: float = 0.90
    oscillation_visual_similarity: float = 0.90
    low_progress_displacement_meters: float = 0.25
    goal_progress_epsilon_meters: float = 0.05
    stationary_flow_fraction: float = 0.0015
    turning_flow_fraction: float = 0.003
    forward_radial_flow_fraction: float = 0.001
    minimum_horizontal_coherence: float = 0.55
    terminal_stop_interval_seconds: float = 0.001
    cumulative_history_size: int = 32
    cumulative_clear_steps: int = 3
    cumulative_cooldown_steps: int = 3
    visual_stall_similarity: float = 0.97
    visual_stall_motion_confidence: float = 0.85
    wall_stuck_suspect_steps: int = 3
    wall_stuck_confirm_steps: int = 6
    oscillation_suspect_steps: int = 6
    oscillation_confirm_steps: int = 8
    oscillation_suspect_reversals: int = 3
    oscillation_confirm_reversals: int = 4
    turn_degrees_per_action: float = 15.0
    spin_suspect_degrees: float = 180.0
    spin_confirm_degrees: float = 300.0
    spin_observed_turn_fraction: float = 0.70
    recovery_turn_degrees: float = 180.0
    recovery_forward_steps: int = 2

    def __post_init__(self) -> None:
        if not 3 <= self.window_size <= 8:
            raise ValueError("window_size must be between 3 and 8")
        if self.analysis_stride < 1:
            raise ValueError("analysis_stride must be positive")
        if self.inference_mode != "blocking":
            raise ValueError("Only inference_mode='blocking' is supported")
        if self.get_nowhere_steps < 1:
            raise ValueError("get_nowhere_steps must be positive")
        if self.get_nowhere_steps > self.window_size:
            raise ValueError("get_nowhere_steps must not exceed window_size")
        if self.inference_latency_budget_ms <= 0:
            raise ValueError("inference_latency_budget_ms must be positive")
        for name in (
            "forward_displacement_meters",
            "turning_yaw_degrees",
            "topology_radius_meters",
            "low_progress_displacement_meters",
            "goal_progress_epsilon_meters",
            "stationary_flow_fraction",
            "turning_flow_fraction",
            "forward_radial_flow_fraction",
            "terminal_stop_interval_seconds",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.revisit_min_step_gap < 1:
            raise ValueError("revisit_min_step_gap must be positive")
        if self.oscillation_retrace_min_step_gap < 1:
            raise ValueError(
                "oscillation_retrace_min_step_gap must be positive"
            )
        if self.oscillation_retrace_min_step_gap >= self.window_size:
            raise ValueError(
                "oscillation_retrace_min_step_gap must be smaller than "
                "window_size"
            )
        if self.cumulative_history_size < 8:
            raise ValueError("cumulative_history_size must be at least 8")
        for name in (
            "cumulative_clear_steps",
            "cumulative_cooldown_steps",
            "wall_stuck_suspect_steps",
            "wall_stuck_confirm_steps",
            "oscillation_suspect_steps",
            "oscillation_confirm_steps",
            "oscillation_suspect_reversals",
            "oscillation_confirm_reversals",
            "recovery_forward_steps",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.wall_stuck_suspect_steps < 3:
            raise ValueError("wall_stuck_suspect_steps must be at least 3")
        if self.wall_stuck_confirm_steps < 2 * self.wall_stuck_suspect_steps:
            raise ValueError(
                "wall_stuck_confirm_steps must cover at least two suspect windows"
            )
        if self.oscillation_confirm_steps < self.oscillation_suspect_steps:
            raise ValueError(
                "oscillation_confirm_steps must not be smaller than "
                "oscillation_suspect_steps"
            )
        if self.cumulative_history_size < max(
            self.wall_stuck_confirm_steps,
            self.oscillation_confirm_steps,
            math.ceil(self.spin_confirm_degrees / self.turn_degrees_per_action),
        ):
            raise ValueError(
                "cumulative_history_size is too small for configured detectors"
            )
        for name in (
            "turn_degrees_per_action",
            "spin_suspect_degrees",
            "spin_confirm_degrees",
            "recovery_turn_degrees",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.spin_confirm_degrees < self.spin_suspect_degrees:
            raise ValueError(
                "spin_confirm_degrees must not be smaller than "
                "spin_suspect_degrees"
            )
        if not 0 <= self.spin_observed_turn_fraction <= 1:
            raise ValueError("spin_observed_turn_fraction must be in [0, 1]")
        for name in (
            "revisit_visual_similarity",
            "oscillation_visual_similarity",
            "minimum_horizontal_coherence",
            "visual_stall_similarity",
            "visual_stall_motion_confidence",
        ):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class TemporalRuleStatus:
    """Deterministic status retained even if model inference fails."""

    collision: ErrorVerdict
    repeated_visit: ErrorVerdict
    motion_oscillation: ErrorVerdict
    get_nowhere: ErrorVerdict
    action_execution_mismatch: ErrorVerdict
    overall_progress: ProgressVerdict
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _PendingAction:
    observation: TemporalObservation
    selected_action: str
    subgoal_snapshot: str


@dataclass(slots=True)
class _TopologyNode:
    node_id: str
    position_xyz: Optional[tuple[float, float, float]]
    visual_descriptor: Optional[Any]
    entry_count: int
    last_entry_step_id: int


class TemporalMemory:
    """The rolling temporal module for one live VLN episode."""

    def __init__(
        self,
        goal: str = "",
        *,
        episode_id: str = "episode-0",
        captioner: Optional[TemporalCaptioner] = None,
        config: Optional[TemporalMemoryConfig] = None,
    ) -> None:
        self.config = config or TemporalMemoryConfig()
        self._captioner = captioner
        self.reset(episode_id=episode_id, goal=goal)

    def reset(self, *, episode_id: str, goal: str) -> None:
        """Clear every episode-scoped item while retaining model/config objects."""
        normalized_episode = str(episode_id).strip()
        if not normalized_episode:
            raise ValueError("episode_id must not be empty")
        self.episode_id = normalized_episode
        self.goal = str(goal)
        self._steps: Deque[TemporalStep] = deque(maxlen=self.config.window_size)
        self._pending: Optional[_PendingAction] = None
        self._latest_record: Optional[TemporalMemoryRecord] = None
        self._last_analysis_error: Optional[str] = None
        self._last_failed_raw_response: Optional[str] = None
        self._last_analyzed_step_id: Optional[int] = None
        self._completed_step_count = 0
        self._subgoal_completion_status = ""
        self._known_landmarks: set[str] = set()
        self._topology_nodes: list[_TopologyNode] = []
        self._current_node_id: Optional[str] = None
        self._evidence_history: Deque[TemporalEvidenceStep] = deque(
            maxlen=self.config.cumulative_history_size
        )
        self._cumulative_states = {
            mode: CumulativeErrorState(
                mode=mode,
                phase=CumulativeErrorPhase.NORMAL,
                score=0.0,
            )
            for mode in CumulativeErrorMode
        }
        self._cumulative_clear_streak = {
            mode: 0 for mode in CumulativeErrorMode
        }
        self._cumulative_cooldown_until_step = {
            mode: 0 for mode in CumulativeErrorMode
        }
        self._pending_go_back_request: Optional[GoBackRequest] = None
        self._active_go_back_request: Optional[GoBackRequest] = None
        self._recovery_action_queue: Deque[str] = deque()
        self._recovery_actions_dispatched = 0
        self._last_safe_checkpoint_step_id: Optional[int] = None
        self._last_safe_checkpoint_descriptor: tuple[float, ...] = ()
        self._analysis_attempts = 0
        self._analysis_successes = 0
        self._analysis_failures = 0
        self._analysis_total_ms = 0.0
        self._analysis_last_ms: Optional[float] = None
        self._analysis_min_ms: Optional[float] = None
        self._analysis_max_ms: Optional[float] = None
        self._analysis_budget_met_count = 0
        self._video_inference_count = 0
        self._video_success_count = 0
        self._video_failure_count = 0
        self._video_total_ms = 0.0
        self._video_last_ms: Optional[float] = None
        self._video_min_ms: Optional[float] = None
        self._video_max_ms: Optional[float] = None
        self._video_budget_met_count = 0
        self._video_recovery_skip_count = 0
        reset_captioner_timing = getattr(
            self._captioner,
            "reset_performance_stats",
            None,
        )
        if callable(reset_captioner_timing):
            reset_captioner_timing()
        self._latest_rule_status = TemporalRuleStatus(
            collision="UNCERTAIN",
            repeated_visit="UNCERTAIN",
            motion_oscillation="UNCERTAIN",
            get_nowhere="UNCERTAIN",
            action_execution_mismatch="UNCERTAIN",
            overall_progress="UNCERTAIN",
            reasons=("尚无完整 action transition。",),
        )

    def set_captioner(self, captioner: Optional[TemporalCaptioner]) -> None:
        """Inject or replace the shared video-understanding adapter."""
        if captioner is not self._captioner and self._last_analysis_error:
            self._last_analyzed_step_id = None
        self._captioner = captioner

    @property
    def captioner(self) -> Optional[TemporalCaptioner]:
        return self._captioner

    @property
    def latest_record(self) -> Optional[TemporalMemoryRecord]:
        return self._latest_record

    @property
    def latest_rule_status(self) -> TemporalRuleStatus:
        return self._latest_rule_status

    @property
    def cumulative_error_state(self) -> CumulativeErrorState:
        """Return the dominant long-horizon failure state.

        Per-mode states remain available through
        :meth:`cumulative_error_states`; this single state is the stable
        interface intended for Task Memory and a later Go-Back controller.
        """
        return self._dominant_cumulative_state()

    def cumulative_error_states(self) -> tuple[CumulativeErrorState, ...]:
        """Return all detector states in deterministic mode order."""
        return tuple(self._cumulative_states[mode] for mode in CumulativeErrorMode)

    @property
    def pending_go_back_request(self) -> Optional[GoBackRequest]:
        """Recovery request awaiting explicit acceptance by the Agent."""
        return self._pending_go_back_request

    @property
    def active_go_back_request(self) -> Optional[GoBackRequest]:
        """Recovery request currently being consumed, if any."""
        return self._active_go_back_request

    def recent_evidence(self) -> tuple[TemporalEvidenceStep, ...]:
        """Return image-free cumulative evidence retained for this episode."""
        return tuple(self._evidence_history)

    def begin_go_back_recovery(
        self,
        request_id: Optional[str] = None,
    ) -> Optional[GoBackRequest]:
        """Accept the pending request and enter ``RECOVERING``.

        Merely confirming an error never changes an environment action.  The
        caller must explicitly accept the typed request before recovery
        primitives become available from :meth:`next_recovery_action`.
        """
        request = self._pending_go_back_request
        if request is None:
            return None
        if request_id is not None and request.request_id != request_id:
            raise TemporalStateError(
                f"Pending Go-Back request is {request.request_id!r}, "
                f"not {request_id!r}"
            )
        self._pending_go_back_request = None
        self._active_go_back_request = request
        self._recovery_action_queue = deque(request.recovery_actions)
        self._recovery_actions_dispatched = 0
        state = self._cumulative_states[request.error_mode]
        self._cumulative_states[request.error_mode] = replace(
            state,
            phase=CumulativeErrorPhase.RECOVERING,
            last_detected_step=request.trigger_step_id,
            reason=state.reason + " Go-Back recovery has started.",
        )
        return request

    def next_recovery_action(self) -> Optional[str]:
        """Peek one legal primitive only while recovery is active.

        The action remains queued until :meth:`ack_recovery_action` confirms
        that the Agent successfully staged it.  This prevents a transient
        staging failure from silently losing a recovery primitive.
        """
        request = self._active_go_back_request
        if request is None:
            return None
        state = self._cumulative_states[request.error_mode]
        if state.phase != CumulativeErrorPhase.RECOVERING:
            return None
        if not self._recovery_action_queue:
            return None
        return self._recovery_action_queue[0]

    def ack_recovery_action(self, action: str) -> None:
        """Acknowledge successful staging and consume the queued primitive."""
        expected = self.next_recovery_action()
        if expected is None:
            raise TemporalStateError("There is no recovery action awaiting ack")
        normalized = _normalize_action(action)
        if normalized != expected:
            raise TemporalStateError(
                f"Expected recovery action {expected}, got {normalized}"
            )
        self._recovery_action_queue.popleft()
        self._recovery_actions_dispatched += 1

    def finish_go_back_recovery(
        self,
        *,
        success: bool,
        note: str = "",
    ) -> None:
        """Finish the active recovery and enter cooldown or confirmed failure."""
        request = self._active_go_back_request
        if request is None:
            raise TemporalStateError("There is no active Go-Back recovery")
        self._active_go_back_request = None
        self._recovery_action_queue.clear()
        state = self._cumulative_states[request.error_mode]
        current_step = self._completed_step_count
        suffix = f" {str(note).strip()}" if str(note).strip() else ""
        if success:
            self._cumulative_cooldown_until_step[request.error_mode] = (
                current_step + self.config.cumulative_cooldown_steps
            )
            self._cumulative_clear_streak[request.error_mode] = 0
            self._cumulative_states[request.error_mode] = replace(
                state,
                phase=CumulativeErrorPhase.COOLDOWN,
                score=min(state.score, 0.5),
                reason="Go-Back recovery completed; monitoring cooldown." + suffix,
            )
        else:
            self._cumulative_cooldown_until_step[request.error_mode] = (
                current_step + self.config.cumulative_cooldown_steps
            )
            self._cumulative_clear_streak[request.error_mode] = 0
            self._cumulative_states[request.error_mode] = replace(
                state,
                phase=CumulativeErrorPhase.COOLDOWN,
                score=max(state.score, 0.9),
                reason=(
                    "Go-Back recovery failed; normal planning resumes during "
                    "a short retry cooldown."
                    + suffix
                ),
            )

    @property
    def last_analysis_error(self) -> Optional[str]:
        return self._last_analysis_error

    @property
    def known_landmarks(self) -> tuple[str, ...]:
        """Lightweight episode-level landmark labels learned so far."""
        return tuple(sorted(self._known_landmarks))

    @property
    def pending_step_id(self) -> Optional[int]:
        if self._pending is None:
            return None
        return self._completed_step_count + 1

    @property
    def pending_selected_action(self) -> Optional[str]:
        return self._pending.selected_action if self._pending is not None else None

    def update_subgoal_status(self, status: str) -> None:
        """Update the planner-owned status text included in ``context()``."""
        self._subgoal_completion_status = str(status or "")

    def stage_action(
        self,
        observation: TemporalObservation,
        selected_action: str,
        subgoal_snapshot: str,
    ) -> None:
        """Stage an action on its pre-action observation."""
        self._ensure_episode(observation)
        if self._pending is not None:
            raise TemporalStateError(
                "Cannot stage a new action before completing the pending action"
            )
        if self._steps:
            previous_timestamp = self._steps[-1].post_observation.timestamp_seconds
            if observation.timestamp_seconds < previous_timestamp - 1e-9:
                raise TemporalStateError(
                    "Pre-action observation precedes the latest completed step"
                )
        self._seed_pre_observation(observation)
        self._pending = _PendingAction(
            observation=observation,
            selected_action=_normalize_action(selected_action),
            subgoal_snapshot=str(subgoal_snapshot or ""),
        )

    def infer_pending_execution(
        self,
        *,
        collision: Optional[bool] = None,
        terminal: bool = False,
    ) -> StepExecution:
        """Construct execution state internally for ``act(rgb)`` callers."""
        if self._pending is None:
            raise TemporalStateError("There is no pending action to infer")
        step_id = self.pending_step_id
        assert step_id is not None
        return StepExecution(
            step_id=step_id,
            commanded_action=self._pending.selected_action,
            collision=collision,
            terminal=terminal,
        )

    def complete_pending_step(
        self,
        post_observation: TemporalObservation,
        execution: StepExecution,
        subgoal_snapshot: str,
    ) -> TemporalStep:
        """Close the pending action with the first observation after execution."""
        pending = self._pending
        if pending is None:
            raise TemporalStateError("There is no pending action to complete")
        self._ensure_episode(post_observation)
        self._validate_execution_order(execution)
        if (
            post_observation.timestamp_seconds
            <= pending.observation.timestamp_seconds
        ):
            raise TemporalStateError(
                "Post-action timestamp must be greater than pre-action timestamp"
            )

        motion = self._build_motion_signal(
            pending.observation,
            post_observation,
            execution,
        )
        observed_action = motion.camera_motion.value
        action_match = self._compare_action(
            execution.commanded_action,
            motion.camera_motion,
        )
        (
            node_id,
            is_new_node,
            is_revisit,
        ) = self._assign_topology_node(post_observation, execution.step_id)
        newly_completed = self._newly_completed_subgoals(
            pending.subgoal_snapshot,
            str(subgoal_snapshot or ""),
        )
        newly_discovered_landmarks = self._new_landmarks(post_observation)

        step = TemporalStep(
            step_id=execution.step_id,
            selected_action=pending.selected_action,
            commanded_action=execution.commanded_action,
            pre_observation=pending.observation,
            post_observation=post_observation,
            observed_action=observed_action,
            action_match=action_match,
            motion=motion,
            topology_node_id=node_id,
            is_new_node=is_new_node,
            is_revisit=is_revisit,
            newly_completed_subgoals=newly_completed,
            newly_discovered_landmarks=newly_discovered_landmarks,
            subgoal_evidence_known=bool(
                pending.subgoal_snapshot and str(subgoal_snapshot or "")
            ),
        )
        self._steps.append(step)
        self._completed_step_count += 1
        self._pending = None
        self._subgoal_completion_status = str(subgoal_snapshot or "")
        self._append_temporal_evidence(step)
        recovery_was_active = self._active_go_back_request is not None
        recovery_finished = self._finish_recovery_from_step_if_ready(step)
        if not recovery_was_active:
            self._update_safe_checkpoint(step)
        if not recovery_finished:
            self._update_cumulative_errors()
        if execution.terminal:
            # Keep the cumulative diagnosis for the episode artifact, but
            # never leave a recovery request that cannot be executed.
            self._pending_go_back_request = None
        self._latest_rule_status = self._derive_rule_status(tuple(self._steps))
        return step

    def analyze_if_ready(self) -> Optional[TemporalMemoryRecord]:
        """Synchronously analyze the configured recent completed transitions."""
        if len(self._steps) < self.config.window_size:
            return None
        newest_step_id = self._steps[-1].step_id
        if self._last_analyzed_step_id is not None:
            if newest_step_id - self._last_analyzed_step_id < self.config.analysis_stride:
                return self._latest_record
        if self._active_go_back_request is not None:
            # A deterministic recovery macro already owns these actions.
            # Avoid paying for a scene caption that cannot affect the queued
            # primitives; the final recovery post-frame remains eligible.
            self._video_recovery_skip_count += 1
            return self._latest_record
        if self._captioner is None:
            self._last_analysis_error = (
                "TemporalCaptioner is not configured; rule-only status is available"
            )
            return self._latest_record

        steps = tuple(self._steps)
        # Mark this window attempted. A later completed step naturally retries
        # with a new sliding window, while repeated context reads do not rerun an
        # expensive model call for the identical window.
        self._last_analyzed_step_id = newest_step_id
        analysis_started = time.perf_counter()
        try:
            if hasattr(self._captioner, "analyze_steps"):
                request = self._build_step_analysis_request(steps)
                record = self._captioner.analyze_steps(request)  # type: ignore[attr-defined]
            else:
                record = self._captioner.analyze(self._build_legacy_window(steps))
        except Exception as exc:
            self._video_failure_count += 1
            self._record_analysis_timing(
                (time.perf_counter() - analysis_started) * 1000,
                success=False,
            )
            self._last_analysis_error = f"{type(exc).__name__}: {exc}"
            failed_raw_response = getattr(
                self._captioner,
                "last_raw_response",
                None,
            )
            self._last_failed_raw_response = (
                str(failed_raw_response)
                if failed_raw_response is not None
                else None
            )
            return self._latest_record

        self._ingest_model_landmarks(record)
        self._record_video_timing(record)
        self._latest_rule_status = self._derive_rule_status(tuple(self._steps))
        record = self._fuse_rule_status_into_record(record)
        self._latest_record = record
        self._last_analysis_error = None
        self._last_failed_raw_response = None
        self._record_analysis_timing(
            (time.perf_counter() - analysis_started) * 1000,
            success=True,
        )
        return record

    def finish_episode(
        self,
        post_observation: TemporalObservation,
        execution: StepExecution,
        subgoal_snapshot: str,
    ) -> Optional[TemporalMemoryRecord]:
        """Close the final transition and perform the last eligible analysis.

        Some environments do not produce a distinct post-STOP frame.  Reusing
        the pre-frame is allowed only for STOP and receives a tiny synthetic
        timestamp increment so temporal intervals remain well formed.
        """
        if self._pending is None:
            return self._latest_record
        pending = self._pending
        # Validate ownership and order even if an unusable final observation
        # causes this transition to be discarded.
        self._ensure_episode(post_observation)
        self._validate_execution_order(execution)
        if (
            post_observation.timestamp_seconds
            <= pending.observation.timestamp_seconds
        ):
            if execution.commanded_action != "STOP":
                self._pending = None
                self._last_analysis_error = (
                    "Final non-STOP action had no later post-action observation; "
                    "the incomplete transition was discarded"
                )
                return self._latest_record
            post_observation = replace(
                post_observation,
                image=pending.observation.image,
                timestamp_seconds=(
                    pending.observation.timestamp_seconds
                    + self.config.terminal_stop_interval_seconds
                ),
            )
        self.complete_pending_step(
            post_observation,
            execution,
            subgoal_snapshot,
        )
        return self.analyze_if_ready()

    def recent_steps(self) -> tuple[TemporalStep, ...]:
        return tuple(self._steps)

    def recent_actions(self) -> tuple[str, ...]:
        completed = tuple(step.commanded_action for step in self._steps)
        if self._pending is None:
            return completed[-self.config.window_size:]
        # The pending command produced the current post-action observation, so
        # Thinker's completion-status update must see it before that transition
        # is formally closed.
        return (*completed, self._pending.selected_action)[
            -self.config.window_size:
        ]

    def diagnostics(
        self,
        *,
        include_raw_response: bool = False,
    ) -> dict[str, Any]:
        """Return a JSON-safe snapshot for agent-side runtime logging.

        Images are intentionally excluded.  The returned step IDs identify the
        exact rolling window whose post-action frames were analyzed.
        """
        rule = self._latest_rule_status
        latest_analysis: Optional[dict[str, Any]] = None
        if self._latest_record is not None:
            if hasattr(self._latest_record, "to_memory_dict"):
                latest_analysis = self._latest_record.to_memory_dict()
            elif hasattr(self._latest_record, "model_dump"):
                latest_analysis = self._latest_record.model_dump(
                    mode="json",
                    exclude={"raw_response"},
                )
            else:
                latest_analysis = {
                    "summary": self._latest_record.to_memory_text()
                }
            if include_raw_response:
                raw_response = getattr(
                    self._latest_record,
                    "raw_response",
                    None,
                )
                if raw_response is not None:
                    latest_analysis["raw_response"] = raw_response
        diagnostics = {
            "episode_id": self.episode_id,
            "goal": self.goal,
            "config": {
                "window_size": self.config.window_size,
                "analysis_stride": self.config.analysis_stride,
                "get_nowhere_steps": self.config.get_nowhere_steps,
                "cumulative_history_size": self.config.cumulative_history_size,
                "inference_mode": self.config.inference_mode,
                "latency_budget_ms": (
                    self.config.inference_latency_budget_ms
                ),
            },
            "completed_step_count": self._completed_step_count,
            "completed_step_ids": [step.step_id for step in self._steps],
            "pending_step_id": self.pending_step_id,
            "last_analyzed_step_id": self._last_analyzed_step_id,
            "known_landmarks": list(self.known_landmarks),
            "rule_status": {
                "collision": rule.collision,
                "repeated_visit": rule.repeated_visit,
                "motion_oscillation": rule.motion_oscillation,
                "get_nowhere": rule.get_nowhere,
                "action_execution_mismatch": (
                    rule.action_execution_mismatch
                ),
                "overall_progress": rule.overall_progress,
                "reasons": list(rule.reasons),
            },
            "cumulative_error": self._cumulative_state_dict(
                self.cumulative_error_state
            ),
            "cumulative_error_states": [
                self._cumulative_state_dict(state)
                for state in self.cumulative_error_states()
            ],
            "lightweight_history": {
                "size": len(self._evidence_history),
                "step_ids": [item.step_id for item in self._evidence_history],
                "contains_raw_images": False,
            },
            "pending_go_back_request": self._go_back_request_dict(
                self._pending_go_back_request
            ),
            "active_go_back_request": self._go_back_request_dict(
                self._active_go_back_request
            ),
            "recovery": {
                "dispatched_actions": self._recovery_actions_dispatched,
                "remaining_actions": list(self._recovery_action_queue),
                "video_inference_skip_count": (
                    self._video_recovery_skip_count
                ),
            },
            "last_analysis_error": self._last_analysis_error,
            "latest_analysis": latest_analysis,
            "timing": self.timing_summary(),
        }
        if include_raw_response and self._last_failed_raw_response is not None:
            diagnostics["last_failed_raw_response"] = (
                self._last_failed_raw_response
            )
        return diagnostics

    @staticmethod
    def _cumulative_state_dict(
        state: CumulativeErrorState,
    ) -> dict[str, Any]:
        return {
            "mode": state.mode.value if state.mode is not None else None,
            "phase": state.phase.value,
            "score": state.score,
            "evidence_step_ids": list(state.evidence_step_ids),
            "first_detected_step": state.first_detected_step,
            "last_detected_step": state.last_detected_step,
            "reason": state.reason,
        }

    @staticmethod
    def _go_back_request_dict(
        request: Optional[GoBackRequest],
    ) -> Optional[dict[str, Any]]:
        if request is None:
            return None
        return {
            "request_id": request.request_id,
            "trigger_step_id": request.trigger_step_id,
            "error_mode": request.error_mode.value,
            "safe_checkpoint_step_id": request.safe_checkpoint_step_id,
            "recovery_actions": list(request.recovery_actions),
            "reason": request.reason,
        }

    def timing_summary(self) -> dict[str, Any]:
        """Return weighted episode timing for memory and video inference."""
        temporal_average = (
            self._analysis_total_ms / self._analysis_attempts
            if self._analysis_attempts
            else None
        )
        captioner_summary = getattr(
            self._captioner,
            "performance_summary",
            None,
        )
        if callable(captioner_summary):
            video_understanding = captioner_summary()
        else:
            video_understanding = {
                "inference_count": self._video_inference_count,
                "success_count": self._video_success_count,
                "failure_count": self._video_failure_count,
                "total_inference_ms": self._video_total_ms,
                "average_inference_ms": (
                    self._video_total_ms / self._video_inference_count
                    if self._video_inference_count
                    else None
                ),
                "last_inference_ms": self._video_last_ms,
                "min_inference_ms": self._video_min_ms,
                "max_inference_ms": self._video_max_ms,
                "latency_budget_ms": (
                    self.config.inference_latency_budget_ms
                ),
                "latency_budget_met_count": (
                    self._video_budget_met_count
                ),
            }
        return {
            "temporal_memory": {
                "inference_count": self._analysis_attempts,
                "success_count": self._analysis_successes,
                "failure_count": self._analysis_failures,
                "total_inference_ms": self._analysis_total_ms,
                "average_inference_ms": temporal_average,
                "last_inference_ms": self._analysis_last_ms,
                "min_inference_ms": self._analysis_min_ms,
                "max_inference_ms": self._analysis_max_ms,
                "latency_budget_ms": (
                    self.config.inference_latency_budget_ms
                ),
                "latency_budget_met_count": (
                    self._analysis_budget_met_count
                ),
            },
            "video_understanding": video_understanding,
        }

    def _record_analysis_timing(
        self,
        duration_ms: float,
        *,
        success: bool,
    ) -> None:
        duration = max(0.0, float(duration_ms))
        self._analysis_attempts += 1
        self._analysis_successes += int(success)
        self._analysis_failures += int(not success)
        self._analysis_total_ms += duration
        self._analysis_last_ms = duration
        self._analysis_min_ms = (
            duration
            if self._analysis_min_ms is None
            else min(self._analysis_min_ms, duration)
        )
        self._analysis_max_ms = (
            duration
            if self._analysis_max_ms is None
            else max(self._analysis_max_ms, duration)
        )
        if success and duration <= self.config.inference_latency_budget_ms:
            self._analysis_budget_met_count += 1

    def _record_video_timing(self, record: Any) -> None:
        latency = getattr(record, "model_latency_ms", None)
        if latency is None:
            return
        duration = max(0.0, float(latency))
        self._video_inference_count += 1
        self._video_success_count += 1
        self._video_total_ms += duration
        self._video_last_ms = duration
        self._video_min_ms = (
            duration
            if self._video_min_ms is None
            else min(self._video_min_ms, duration)
        )
        self._video_max_ms = (
            duration
            if self._video_max_ms is None
            else max(self._video_max_ms, duration)
        )
        if bool(getattr(record, "latency_budget_met", False)):
            self._video_budget_met_count += 1

    def _append_temporal_evidence(self, step: TemporalStep) -> None:
        """Append one derived, image-free item to the cumulative history."""
        pre_descriptor = self._compact_scene_descriptor(
            step.pre_observation.image
        )
        post_descriptor = self._compact_scene_descriptor(
            step.post_observation.image
        )
        self._evidence_history.append(
            TemporalEvidenceStep(
                step_id=step.step_id,
                commanded_action=step.commanded_action,
                observed_motion=step.observed_action,
                motion_confidence=float(step.motion.confidence),
                action_match=step.action_match,
                scene_descriptor=post_descriptor,
                frame_similarity=self._evidence_similarity(
                    pre_descriptor,
                    post_descriptor,
                ),
                topology_node_id=step.topology_node_id,
                is_new_node=step.is_new_node,
                newly_completed_subgoals=step.newly_completed_subgoals,
                newly_discovered_landmarks=step.newly_discovered_landmarks,
            )
        )

    @classmethod
    def _compact_scene_descriptor(cls, image: Any) -> tuple[float, ...]:
        """Return an 8x8 normalized descriptor without retaining RGB data."""
        descriptor = cls._visual_descriptor(image)
        if descriptor is None:
            return ()
        try:
            import cv2
            import numpy as np

            square = np.asarray(descriptor, dtype="float32").reshape(32, 32)
            compact = cv2.resize(square, (8, 8), interpolation=cv2.INTER_AREA)
            return tuple(float(value) / 255.0 for value in compact.reshape(-1))
        except Exception:
            return ()

    @staticmethod
    def _evidence_similarity(
        first: tuple[float, ...],
        second: tuple[float, ...],
    ) -> Optional[float]:
        if not first or not second or len(first) != len(second):
            return None
        try:
            import numpy as np

            left = np.asarray(first, dtype="float32")
            right = np.asarray(second, dtype="float32")
            left -= float(left.mean())
            right -= float(right.mean())
            denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
            if denominator <= 1e-9:
                return 1.0 if bool(np.allclose(left, right)) else 0.0
            return max(
                -1.0,
                min(1.0, float(np.dot(left, right) / denominator)),
            )
        except Exception:
            return None

    def _update_safe_checkpoint(self, step: TemporalStep) -> None:
        """Remember a lightweight descriptor of the latest progressing state."""
        visual_forward = (
            step.motion.camera_motion == CameraMotion.FORWARD
            and step.motion.confidence >= 0.5
            and step.action_match == "MATCH"
        )
        structured_progress = bool(
            step.is_new_node
            or step.newly_completed_subgoals
            or step.newly_discovered_landmarks
        )
        if visual_forward or structured_progress:
            self._last_safe_checkpoint_step_id = step.step_id
            if self._evidence_history:
                self._last_safe_checkpoint_descriptor = (
                    self._evidence_history[-1].scene_descriptor
                )

    def _finish_recovery_from_step_if_ready(
        self,
        step: TemporalStep,
    ) -> bool:
        """Close RECOVERING after the final dispatched primitive completes."""
        request = self._active_go_back_request
        if request is None or self._recovery_action_queue:
            return False
        descriptor = (
            self._evidence_history[-1].scene_descriptor
            if self._evidence_history
            else ()
        )
        checkpoint_similarity = self._evidence_similarity(
            descriptor,
            self._last_safe_checkpoint_descriptor,
        )
        visibly_moved = (
            step.motion.camera_motion
            not in {CameraMotion.STATIONARY, CameraMotion.UNKNOWN}
            and step.motion.confidence >= 0.5
        )
        reached_new_view = step.is_new_node is True
        reached_checkpoint = (
            checkpoint_similarity is not None
            and checkpoint_similarity >= self.config.revisit_visual_similarity
        )
        success = visibly_moved or reached_new_view or reached_checkpoint
        evidence = []
        if visibly_moved:
            evidence.append(f"observed_motion={step.observed_action}")
        if reached_new_view:
            evidence.append("new_visual_node=true")
        if reached_checkpoint:
            evidence.append(
                f"safe_checkpoint_similarity={checkpoint_similarity:.3f}"
            )
        self.finish_go_back_recovery(
            success=success,
            note=(
                "Final recovery primitive completed; " + ", ".join(evidence)
                if success
                else "Final recovery primitive completed without visible motion."
            ),
        )
        return True

    def _update_cumulative_errors(self) -> None:
        """Update all cumulative detectors without invoking a foundation model."""
        # Recovery actions intentionally do not feed the detectors: a planned
        # half-turn would otherwise look like a new in-place spin.
        if self._active_go_back_request is not None:
            return
        signals = {
            CumulativeErrorMode.WALL_STUCK: self._detect_wall_stuck(),
            CumulativeErrorMode.TURN_OSCILLATION: (
                self._detect_turn_oscillation()
            ),
            CumulativeErrorMode.IN_PLACE_SPIN: self._detect_in_place_spin(),
        }
        for mode, signal in signals.items():
            self._transition_cumulative_state(mode, signal)
        self._maybe_create_go_back_request()

    def _detect_wall_stuck(self) -> _CumulativeSignal:
        history = tuple(self._evidence_history)
        suspect_count = self.config.wall_stuck_suspect_steps
        confirm_count = self.config.wall_stuck_confirm_steps
        suspect_steps = history[-suspect_count:]
        suspected = self._wall_stuck_window_matches(suspect_steps)
        confirm_steps = history[-confirm_count:]
        confirmed = (
            len(confirm_steps) == confirm_count
            and self._wall_stuck_window_matches(confirm_steps[:suspect_count])
            and self._wall_stuck_window_matches(confirm_steps[-suspect_count:])
        )
        evidence = confirm_steps if confirmed else suspect_steps
        stationary = sum(
            item.commanded_action == "FORWARD"
            and item.observed_motion == CameraMotion.STATIONARY.value
            and item.motion_confidence
            >= self.config.visual_stall_motion_confidence
            for item in evidence
        )
        score = (
            min(1.0, stationary / max(1, confirm_count - 1))
            if suspected or confirmed
            else 0.0
        )
        reason = (
            f"{stationary}/{len(evidence)} recent FORWARD actions produced "
            "high-confidence stationary, visually unchanged observations."
            if suspected or confirmed
            else ""
        )
        return _CumulativeSignal(
            suspected=suspected or confirmed,
            confirmed=confirmed,
            score=score,
            evidence_step_ids=tuple(item.step_id for item in evidence),
            reason=reason,
        )

    def _wall_stuck_window_matches(
        self,
        steps: Sequence[TemporalEvidenceStep],
    ) -> bool:
        required = self.config.wall_stuck_suspect_steps
        if len(steps) != required:
            return False
        stationary = sum(
            item.commanded_action == "FORWARD"
            and item.observed_motion == CameraMotion.STATIONARY.value
            and item.motion_confidence
            >= self.config.visual_stall_motion_confidence
            for item in steps
        )
        similar = sum(
            item.frame_similarity is not None
            and item.frame_similarity >= self.config.visual_stall_similarity
            for item in steps
        )
        return (
            all(item.commanded_action == "FORWARD" for item in steps)
            and stationary >= required - 1
            and similar >= required - 1
            and self._has_no_discovery(steps)
        )

    def _detect_turn_oscillation(self) -> _CumulativeSignal:
        history = tuple(self._evidence_history)
        suspect_steps = history[-self.config.oscillation_suspect_steps :]
        confirm_steps = history[-self.config.oscillation_confirm_steps :]
        suspected, suspect_score = self._oscillation_window_matches(
            suspect_steps,
            minimum_steps=self.config.oscillation_suspect_steps,
            minimum_reversals=self.config.oscillation_suspect_reversals,
        )
        confirmed, confirm_score = self._oscillation_window_matches(
            confirm_steps,
            minimum_steps=self.config.oscillation_confirm_steps,
            minimum_reversals=self.config.oscillation_confirm_reversals,
        )
        evidence = confirm_steps if confirmed else suspect_steps
        return _CumulativeSignal(
            suspected=suspected or confirmed,
            confirmed=confirmed,
            score=confirm_score if confirmed else suspect_score,
            evidence_step_ids=tuple(item.step_id for item in evidence),
            reason=(
                "Alternating left/right turns repeatedly returned to a similar "
                "view without a new node, landmark, or subgoal."
                if suspected or confirmed
                else ""
            ),
        )

    def _oscillation_window_matches(
        self,
        steps: Sequence[TemporalEvidenceStep],
        *,
        minimum_steps: int,
        minimum_reversals: int,
    ) -> tuple[bool, float]:
        if len(steps) < minimum_steps:
            return False, 0.0
        turns = [
            item.commanded_action
            for item in steps
            if item.commanded_action in {"TURN_LEFT", "TURN_RIGHT"}
        ]
        reversals = sum(
            previous != current
            for previous, current in zip(turns, turns[1:])
        )
        retrace = self._evidence_retrace_similarity(steps, minimum_gap=2)
        matches = (
            len(turns) >= minimum_steps - 1
            and reversals >= minimum_reversals
            and retrace is not None
            and retrace >= self.config.oscillation_visual_similarity
            and self._has_no_discovery(steps)
        )
        score = min(
            1.0,
            0.5 * reversals / max(1, minimum_reversals)
            + 0.5 * max(0.0, retrace or 0.0),
        )
        return matches, score if matches else 0.0

    def _detect_in_place_spin(self) -> _CumulativeSignal:
        history = tuple(self._evidence_history)
        if not history:
            return _CumulativeSignal(False, False, 0.0, (), "")
        latest_action = history[-1].commanded_action
        if latest_action not in {"TURN_LEFT", "TURN_RIGHT"}:
            return _CumulativeSignal(False, False, 0.0, (), "")
        run: list[TemporalEvidenceStep] = []
        for item in reversed(history):
            if item.commanded_action != latest_action:
                break
            run.append(item)
        run.reverse()
        suspect_count = math.ceil(
            self.config.spin_suspect_degrees
            / self.config.turn_degrees_per_action
        )
        confirm_count = math.ceil(
            self.config.spin_confirm_degrees
            / self.config.turn_degrees_per_action
        )
        observed_turn = (
            CameraMotion.TURN_LEFT.value
            if latest_action == "TURN_LEFT"
            else CameraMotion.TURN_RIGHT.value
        )
        observed_fraction = (
            sum(
                item.observed_motion == observed_turn
                and item.motion_confidence >= 0.5
                for item in run
            )
            / len(run)
            if run
            else 0.0
        )
        suspected = (
            len(run) >= suspect_count
            and observed_fraction >= self.config.spin_observed_turn_fraction
            and self._has_no_discovery(run)
        )
        retrace = self._evidence_retrace_similarity(
            run,
            minimum_gap=max(2, suspect_count // 2),
        )
        confirmed = (
            len(run) >= confirm_count
            and suspected
            and retrace is not None
            and retrace >= self.config.oscillation_visual_similarity
        )
        evidence = tuple(run[-confirm_count:])
        score = min(
            1.0,
            0.5 * len(run) / max(1, confirm_count)
            + 0.5 * observed_fraction,
        )
        return _CumulativeSignal(
            suspected=suspected,
            confirmed=confirmed,
            score=score if suspected else 0.0,
            evidence_step_ids=tuple(item.step_id for item in evidence),
            reason=(
                f"{len(run)} consecutive {latest_action} actions accumulated "
                "an in-place rotation and revisited a similar scene."
                if suspected
                else ""
            ),
        )

    @staticmethod
    def _has_no_discovery(steps: Sequence[TemporalEvidenceStep]) -> bool:
        return bool(steps) and not any(
            item.is_new_node is True
            or item.newly_completed_subgoals
            or item.newly_discovered_landmarks
            for item in steps
        )

    def _evidence_retrace_similarity(
        self,
        steps: Sequence[TemporalEvidenceStep],
        *,
        minimum_gap: int,
    ) -> Optional[float]:
        values = [
            similarity
            for left_index, left in enumerate(steps)
            for right_index, right in enumerate(steps)
            if right_index - left_index >= minimum_gap
            for similarity in [
                self._evidence_similarity(
                    left.scene_descriptor,
                    right.scene_descriptor,
                )
            ]
            if similarity is not None
        ]
        return max(values) if values else None

    def _transition_cumulative_state(
        self,
        mode: CumulativeErrorMode,
        signal: _CumulativeSignal,
    ) -> None:
        state = self._cumulative_states[mode]
        step_id = self._completed_step_count
        if state.phase == CumulativeErrorPhase.RECOVERING:
            return
        if state.phase == CumulativeErrorPhase.COOLDOWN:
            if (
                signal.confirmed
                and step_id >= self._cumulative_cooldown_until_step[mode]
            ):
                self._cumulative_states[mode] = replace(
                    state,
                    phase=CumulativeErrorPhase.CONFIRMED,
                    score=signal.score,
                    evidence_step_ids=signal.evidence_step_ids,
                    last_detected_step=step_id,
                    reason=signal.reason,
                )
            elif step_id >= self._cumulative_cooldown_until_step[mode]:
                self._cumulative_states[mode] = CumulativeErrorState(
                    mode=mode,
                    phase=CumulativeErrorPhase.NORMAL,
                    score=0.0,
                )
            return
        if state.phase == CumulativeErrorPhase.CONFIRMED:
            if signal.confirmed:
                self._cumulative_states[mode] = replace(
                    state,
                    score=max(state.score, signal.score),
                    evidence_step_ids=signal.evidence_step_ids,
                    last_detected_step=step_id,
                    reason=signal.reason or state.reason,
                )
            return
        if signal.confirmed and state.phase == CumulativeErrorPhase.SUSPECTED:
            self._cumulative_clear_streak[mode] = 0
            self._cumulative_states[mode] = replace(
                state,
                phase=CumulativeErrorPhase.CONFIRMED,
                score=max(state.score, signal.score),
                evidence_step_ids=signal.evidence_step_ids,
                last_detected_step=step_id,
                reason=signal.reason,
            )
            return
        if signal.suspected:
            self._cumulative_clear_streak[mode] = 0
            self._cumulative_states[mode] = CumulativeErrorState(
                mode=mode,
                phase=CumulativeErrorPhase.SUSPECTED,
                score=signal.score,
                evidence_step_ids=signal.evidence_step_ids,
                first_detected_step=(
                    state.first_detected_step
                    if state.first_detected_step is not None
                    else step_id
                ),
                last_detected_step=step_id,
                reason=signal.reason,
            )
            return
        if state.phase == CumulativeErrorPhase.SUSPECTED:
            self._cumulative_clear_streak[mode] += 1
            if (
                self._cumulative_clear_streak[mode]
                >= self.config.cumulative_clear_steps
            ):
                self._cumulative_states[mode] = CumulativeErrorState(
                    mode=mode,
                    phase=CumulativeErrorPhase.NORMAL,
                    score=0.0,
                )
                self._cumulative_clear_streak[mode] = 0

    def _dominant_cumulative_state(self) -> CumulativeErrorState:
        if self._active_go_back_request is not None:
            return self._cumulative_states[
                self._active_go_back_request.error_mode
            ]
        if self._pending_go_back_request is not None:
            return self._cumulative_states[
                self._pending_go_back_request.error_mode
            ]
        rank = {
            CumulativeErrorPhase.NORMAL: 0,
            CumulativeErrorPhase.COOLDOWN: 1,
            CumulativeErrorPhase.SUSPECTED: 2,
            CumulativeErrorPhase.CONFIRMED: 3,
            CumulativeErrorPhase.RECOVERING: 4,
        }
        state = max(
            self._cumulative_states.values(),
            key=lambda item: (rank[item.phase], item.score),
        )
        if state.phase == CumulativeErrorPhase.NORMAL:
            return CumulativeErrorState(
                mode=None,
                phase=CumulativeErrorPhase.NORMAL,
                score=0.0,
            )
        return state

    def _maybe_create_go_back_request(self) -> None:
        if (
            self._pending_go_back_request is not None
            or self._active_go_back_request is not None
        ):
            return
        priority = (
            CumulativeErrorMode.WALL_STUCK,
            CumulativeErrorMode.TURN_OSCILLATION,
            CumulativeErrorMode.IN_PLACE_SPIN,
        )
        state = next(
            (
                self._cumulative_states[mode]
                for mode in priority
                if self._cumulative_states[mode].phase
                == CumulativeErrorPhase.CONFIRMED
            ),
            None,
        )
        if state is None or state.mode is None:
            return
        actions = self._build_recovery_actions(state.mode)
        trigger_step_id = self._completed_step_count
        self._pending_go_back_request = GoBackRequest(
            request_id=(
                f"{self.episode_id}:{trigger_step_id}:{state.mode.value}"
            ),
            trigger_step_id=trigger_step_id,
            error_mode=state.mode,
            safe_checkpoint_step_id=self._last_safe_checkpoint_step_id,
            recovery_actions=actions,
            reason=state.reason,
        )

    def _build_recovery_actions(
        self,
        mode: CumulativeErrorMode,
    ) -> tuple[str, ...]:
        recent_turns = [
            item.commanded_action
            for item in self._evidence_history
            if item.commanded_action in {"TURN_LEFT", "TURN_RIGHT"}
        ]
        last_turn = recent_turns[-1] if recent_turns else None
        if mode == CumulativeErrorMode.TURN_OSCILLATION and last_turn:
            turn = last_turn
        elif last_turn == "TURN_LEFT":
            turn = "TURN_RIGHT"
        elif last_turn == "TURN_RIGHT":
            turn = "TURN_LEFT"
        else:
            turn = "TURN_LEFT"
        turn_count = max(
            1,
            round(
                self.config.recovery_turn_degrees
                / self.config.turn_degrees_per_action
            ),
        )
        return (turn,) * turn_count + (
            ("FORWARD",) * self.config.recovery_forward_steps
        )

    def context(self) -> str:
        """Return planner-safe text without embedding historical RGB payloads."""
        lines = [
            f"Goal: {self.goal}",
            (
                "Temporal steps: "
                f"{len(self._steps)}/{self.config.window_size} completed"
            ),
        ]
        if self._steps:
            lines.append("Recent action → observed-motion transitions:")
            for step in self._steps:
                collision = (
                    "unknown"
                    if step.motion.collision is None
                    else str(step.motion.collision).lower()
                )
                lines.append(
                    f"- step {step.step_id}: selected={step.selected_action}, "
                    f"command={step.commanded_action}, "
                    f"observed={step.observed_action}, "
                    f"match={step.action_match}, collision={collision}"
                )
        else:
            lines.append("Recent action → observed-motion transitions: None")

        rule = self._latest_rule_status
        lines.append(
            "Rule status: "
            f"collision={rule.collision}, "
            f"repeated_visit={rule.repeated_visit}, "
            f"motion_oscillation={rule.motion_oscillation}, "
            f"get_nowhere={rule.get_nowhere}, "
            f"action_execution_mismatch={rule.action_execution_mismatch}, "
            f"progress={rule.overall_progress}"
        )
        if rule.reasons:
            lines.append("Rule evidence: " + " ".join(rule.reasons))
        cumulative = self.cumulative_error_state
        lines.append(
            "Cumulative temporal error: "
            f"phase={cumulative.phase.value}, "
            f"mode={cumulative.mode.value if cumulative.mode else 'NONE'}, "
            f"score={cumulative.score:.2f}"
        )
        if cumulative.reason:
            lines.append("Cumulative evidence: " + cumulative.reason)
        if self._pending_go_back_request is not None:
            request = self._pending_go_back_request
            lines.append(
                "Go-Back action required: "
                f"request={request.request_id}, "
                f"safe_checkpoint_step={request.safe_checkpoint_step_id}, "
                f"planned_primitives={len(request.recovery_actions)}"
            )
        elif self._active_go_back_request is not None:
            lines.append(
                "Go-Back recovery: ACTIVE, "
                f"dispatched={self._recovery_actions_dispatched}, "
                f"remaining={len(self._recovery_action_queue)}"
            )
        if self._latest_record is not None:
            lines.append(
                f"Latest {self.config.window_size}-step video understanding: "
                + self._latest_record.to_memory_text()
            )
        else:
            lines.append(
                f"Latest {self.config.window_size}-step video understanding: "
                "Not available"
            )
        if self._last_analysis_error:
            lines.append(f"Video understanding status: {self._last_analysis_error}")
        lines.append(
            "Subgoal / Completion status:\n"
            + (self._subgoal_completion_status or "Not initialized")
        )
        return "\n".join(lines)

    def _build_step_analysis_request(self, steps: Sequence[TemporalStep]) -> Any:
        # Imported lazily so TemporalMemory remains importable with older
        # TemporalCaptioner versions during a rolling upgrade.
        import importlib

        module = importlib.import_module(
            "agentflow.agents.models_embodied_v2.TemporalCaptioner"
        )
        request_type = getattr(module, "TemporalAnalysisRequest", None)
        step_type = getattr(module, "TemporalStepInput", None)
        if request_type is None or step_type is None:
            return self._build_legacy_window(steps)

        progress = self._build_progress_signals(steps)
        step_inputs = tuple(
            step_type(
                step_id=step.step_id,
                commanded_action=step.commanded_action,
                post_timestamp_seconds=step.post_observation.timestamp_seconds,
                image=step.post_observation.image,
                observed_motion=step.motion.camera_motion,
                action_match=step.action_match,
                motion=step.motion,
                collision=step.motion.collision,
                topology_node_id=step.topology_node_id,
                is_new_node=step.is_new_node,
                is_revisit=step.is_revisit,
                distance_to_goal_meters=(
                    step.post_observation.distance_to_goal_meters
                ),
                newly_completed_subgoals=step.newly_completed_subgoals,
            )
            for step in steps
        )
        return request_type(
            episode_id=self.episode_id,
            goal=self.goal,
            steps=step_inputs,
            progress=progress,
            reverse_retrace_similarity=self._reverse_retrace_similarity(steps),
            notes=(
                (
                    f"最近走的{self._chinese_step_count(len(steps))}步"
                    "发生了什么？按照每步 action 后的画面进行描述。"
                ),
                (
                    "Each image is the post-action observation for the command "
                    "with the same step_id; do not shift the pairing."
                ),
            ),
        )

    def _build_legacy_window(
        self,
        steps: Sequence[TemporalStep],
    ) -> TemporalWindow:
        """Build a lossless compatibility request for older captioners."""
        if len(steps) != self.config.window_size:
            raise TemporalStateError(
                f"Expected {self.config.window_size} complete steps"
            )
        start = steps[0].pre_observation.timestamp_seconds
        latest_post = steps[-1].post_observation.timestamp_seconds
        end = max(
            latest_post + self.config.terminal_stop_interval_seconds,
            steps[-1].motion.end_seconds,
        )
        topology = tuple(
            TopologySignal(
                timestamp_seconds=step.post_observation.timestamp_seconds,
                node_id=step.topology_node_id,
                visit_count=(
                    2
                    if step.is_revisit is True
                    else (1 if step.is_revisit is False else None)
                ),
                distance_to_goal_meters=(
                    step.post_observation.distance_to_goal_meters
                ),
                source="temporal_memory",
            )
            for step in steps
            if (
                step.topology_node_id is not None
                or step.post_observation.distance_to_goal_meters is not None
            )
        )
        mapping_notes = tuple(
            (
                f"STEP step_id={step.step_id} command={step.commanded_action} "
                f"post_t={step.post_observation.timestamp_seconds:.6f}s "
                f"observed_action={step.observed_action} "
                f"action_match={step.action_match}"
            )
            for step in steps
        )
        return TemporalWindow(
            start_seconds=start,
            end_seconds=end,
            frames=tuple(
                TimestampedFrame(
                    timestamp_seconds=step.post_observation.timestamp_seconds,
                    image=step.post_observation.image,
                    step_id=step.step_id,
                )
                for step in steps
            ),
            actions=tuple(
                TimedAction(
                    timestamp_seconds=step.pre_observation.timestamp_seconds,
                    action=step.commanded_action,
                    step_id=step.step_id,
                )
                for step in steps
            ),
            motion=tuple(step.motion for step in steps),
            topology=topology,
            progress=self._build_progress_signals(steps),
            reverse_retrace_similarity=self._reverse_retrace_similarity(steps),
            goal=self.goal,
            episode_id=self.episode_id,
            notes=(
                (
                    f"最近走的{self._chinese_step_count(len(steps))}步"
                    "发生了什么？按照每步 action 后的画面进行描述。"
                ),
                *mapping_notes,
            ),
            timestamp_semantics="episode_seconds_post_action_frames",
        )

    def _build_progress_signals(
        self,
        steps: Sequence[TemporalStep],
    ) -> ProgressSignals:
        first = steps[0].pre_observation
        last = steps[-1].post_observation
        net_displacement = self._position_distance(
            first.position_xyz,
            last.position_xyz,
        )
        nodes_known = all(step.is_new_node is not None for step in steps)
        landmarks_known = all(
            step.post_observation.landmark_ids is not None for step in steps
        )
        subgoals_known = all(step.subgoal_evidence_known for step in steps)
        new_nodes = (
            sum(step.is_new_node is True for step in steps)
            if nodes_known
            else None
        )
        new_landmarks = (
            sum(len(step.newly_discovered_landmarks) for step in steps)
            if landmarks_known
            else None
        )
        completed_subgoals = (
            sum(len(step.newly_completed_subgoals) for step in steps)
            if subgoals_known
            else None
        )

        distance_progress = self._goal_distance_improvement(steps)
        coverage_complete = all(
            value is not None
            for value in (
                net_displacement,
                new_nodes,
                new_landmarks,
                completed_subgoals,
                distance_progress,
            )
        )
        no_progress = (
            coverage_complete
            and net_displacement is not None
            and net_displacement
            <= self.config.low_progress_displacement_meters
            and distance_progress is not None
            and distance_progress
            < self.config.goal_progress_epsilon_meters
            and new_nodes == 0
            and new_landmarks == 0
            and completed_subgoals == 0
        )
        return ProgressSignals(
            net_displacement_meters=net_displacement,
            new_landmarks_count=new_landmarks,
            new_topological_nodes_count=new_nodes,
            completed_subgoals_count=completed_subgoals,
            no_progress_steps=len(steps) if no_progress else (
                0 if coverage_complete else None
            ),
        )

    def _build_motion_signal(
        self,
        pre: TemporalObservation,
        post: TemporalObservation,
        execution: StepExecution,
    ) -> MotionSignal:
        displacement = self._position_distance(
            pre.position_xyz,
            post.position_xyz,
        )
        yaw_delta = self._yaw_delta(pre.yaw_degrees, post.yaw_degrees)
        camera_motion: Optional[CameraMotion] = None
        if (
            yaw_delta is not None
            and abs(yaw_delta) >= self.config.turning_yaw_degrees
        ):
            camera_motion = (
                CameraMotion.TURN_LEFT
                if yaw_delta > 0
                else CameraMotion.TURN_RIGHT
            )
        elif (
            displacement is not None
            and displacement >= self.config.forward_displacement_meters
        ):
            camera_motion = CameraMotion.FORWARD
        elif displacement is not None and yaw_delta is not None:
            # STATIONARY is authoritative only when both translation and
            # rotation were measured. A lone unchanged yaw cannot rule out
            # forward motion, and a lone unchanged position cannot rule out an
            # in-place turn.
            camera_motion = CameraMotion.STATIONARY
        if camera_motion is not None:
            return MotionSignal(
                start_seconds=pre.timestamp_seconds,
                end_seconds=post.timestamp_seconds,
                camera_motion=camera_motion,
                delta_forward_meters=displacement,
                delta_yaw_left_degrees=yaw_delta,
                collision=execution.collision,
                confidence=1.0,
                source="environment_odometry",
            )
        return self._optical_flow_motion(pre, post, execution.collision)

    def _optical_flow_motion(
        self,
        pre: TemporalObservation,
        post: TemporalObservation,
        collision: Optional[bool],
    ) -> MotionSignal:
        try:
            import cv2
            import numpy as np

            before = self._rgb_array(pre.image)
            after = self._rgb_array(post.image)
            if before.shape[:2] != after.shape[:2]:
                after = cv2.resize(
                    after,
                    (before.shape[1], before.shape[0]),
                    interpolation=cv2.INTER_AREA,
                )
            height, width = before.shape[:2]
            scale = min(1.0, 240.0 / max(height, width))
            if scale < 1:
                size = (
                    max(2, round(width * scale)),
                    max(2, round(height * scale)),
                )
                before = cv2.resize(before, size, interpolation=cv2.INTER_AREA)
                after = cv2.resize(after, size, interpolation=cv2.INTER_AREA)
            before_gray = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY)
            after_gray = cv2.cvtColor(after, cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                before_gray,
                after_gray,
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
            width = before_gray.shape[1]
            height = before_gray.shape[0]
            median_dx_fraction = float(np.median(horizontal)) / width
            magnitude_fraction = float(np.median(magnitude)) / width
            direction = 1 if median_dx_fraction >= 0 else -1
            coherence = float(np.mean(horizontal * direction > 0))

            grid_y, grid_x = np.mgrid[0:height, 0:width]
            center_x = (width - 1) / 2.0
            center_y = (height - 1) / 2.0
            radial = (
                horizontal * (grid_x - center_x) / max(width, 1)
                + vertical * (grid_y - center_y) / max(height, 1)
            )
            radial_fraction = float(np.median(radial)) / max(width, 1)

            if magnitude_fraction <= self.config.stationary_flow_fraction:
                camera_motion = CameraMotion.STATIONARY
                confidence = min(
                    1.0,
                    1.0
                    - magnitude_fraction
                    / max(self.config.stationary_flow_fraction, 1e-9),
                )
            elif (
                abs(median_dx_fraction) >= self.config.turning_flow_fraction
                and coherence >= self.config.minimum_horizontal_coherence
            ):
                camera_motion = (
                    CameraMotion.TURN_LEFT
                    if median_dx_fraction > 0
                    else CameraMotion.TURN_RIGHT
                )
                confidence = min(
                    1.0,
                    max(
                        coherence,
                        abs(median_dx_fraction)
                        / max(self.config.turning_flow_fraction * 3, 1e-9),
                    ),
                )
            elif radial_fraction >= self.config.forward_radial_flow_fraction:
                camera_motion = CameraMotion.FORWARD
                confidence = min(
                    0.8,
                    radial_fraction
                    / max(self.config.forward_radial_flow_fraction * 3, 1e-9),
                )
            else:
                camera_motion = CameraMotion.UNKNOWN
                confidence = 0.0

            return MotionSignal(
                start_seconds=pre.timestamp_seconds,
                end_seconds=post.timestamp_seconds,
                camera_motion=camera_motion,
                scene_flow_dx_fraction=median_dx_fraction,
                scene_flow_magnitude_fraction=magnitude_fraction,
                collision=collision,
                confidence=max(0.0, confidence),
                source="action_aligned_opencv_farneback",
                quality_note=(
                    f"horizontal_coherence={coherence:.3f}; "
                    f"radial_expansion={radial_fraction:.6f}"
                ),
            )
        except Exception as exc:
            return MotionSignal(
                start_seconds=pre.timestamp_seconds,
                end_seconds=post.timestamp_seconds,
                camera_motion=CameraMotion.UNKNOWN,
                collision=collision,
                confidence=0.0,
                source="optical_flow_unavailable",
                quality_note=f"{type(exc).__name__}: {exc}",
            )

    def _assign_topology_node(
        self,
        observation: TemporalObservation,
        step_id: int,
    ) -> tuple[Optional[str], Optional[bool], Optional[bool]]:
        position = observation.position_xyz
        descriptor = self._visual_descriptor(observation.image)
        if position is None:
            return self._assign_visual_topology_node(
                descriptor,
                step_id,
            )
        nearest: Optional[_TopologyNode] = None
        nearest_distance = float("inf")
        for node in self._topology_nodes:
            if node.position_xyz is None:
                continue
            distance = self._position_distance(position, node.position_xyz)
            assert distance is not None
            if (
                distance <= self.config.topology_radius_meters
                and distance < nearest_distance
            ):
                nearest = node
                nearest_distance = distance

        if nearest is None:
            node = _TopologyNode(
                node_id=f"node-{len(self._topology_nodes):04d}",
                position_xyz=position,
                visual_descriptor=descriptor,
                entry_count=1,
                last_entry_step_id=step_id,
            )
            self._topology_nodes.append(node)
            self._current_node_id = node.node_id
            return node.node_id, True, False

        if nearest.node_id == self._current_node_id:
            return nearest.node_id, False, False

        gap = step_id - nearest.last_entry_step_id
        similarity = self._descriptor_similarity(
            descriptor,
            nearest.visual_descriptor,
        )
        if gap < self.config.revisit_min_step_gap:
            is_revisit: Optional[bool] = False
        elif similarity is None:
            is_revisit = None
        else:
            is_revisit = similarity >= self.config.revisit_visual_similarity
        nearest.entry_count += 1
        nearest.last_entry_step_id = step_id
        if descriptor is not None:
            nearest.visual_descriptor = descriptor
        self._current_node_id = nearest.node_id
        return nearest.node_id, False, is_revisit

    def _assign_visual_topology_node(
        self,
        descriptor: Optional[Any],
        step_id: int,
    ) -> tuple[Optional[str], Optional[bool], Optional[bool]]:
        """Build lightweight topology when the caller supplies RGB only.

        This is deliberately a visual fallback, not metric localization.
        A high-similarity view stays in or returns to a visual node; a
        sufficiently different view creates a new node.  Missing/invalid image
        evidence remains unknown.
        """
        if descriptor is None:
            self._current_node_id = None
            return None, None, None

        current = next(
            (
                node
                for node in self._topology_nodes
                if node.node_id == self._current_node_id
            ),
            None,
        )
        if current is not None:
            similarity = self._descriptor_similarity(
                descriptor,
                current.visual_descriptor,
            )
            if (
                similarity is not None
                and similarity >= self.config.revisit_visual_similarity
            ):
                return current.node_id, False, False

        candidates = [
            (self._descriptor_similarity(descriptor, node.visual_descriptor), node)
            for node in self._topology_nodes
        ]
        candidates = [
            (similarity, node)
            for similarity, node in candidates
            if similarity is not None
        ]
        if candidates:
            best_similarity, best = max(
                candidates,
                key=lambda item: item[0],
            )
            if best_similarity >= self.config.revisit_visual_similarity:
                gap = step_id - best.last_entry_step_id
                is_revisit = gap >= self.config.revisit_min_step_gap
                best.entry_count += 1
                best.last_entry_step_id = step_id
                best.visual_descriptor = descriptor
                self._current_node_id = best.node_id
                return best.node_id, False, is_revisit

        node = _TopologyNode(
            node_id=f"visual-node-{len(self._topology_nodes):04d}",
            position_xyz=None,
            visual_descriptor=descriptor,
            entry_count=1,
            last_entry_step_id=step_id,
        )
        self._topology_nodes.append(node)
        self._current_node_id = node.node_id
        return node.node_id, True, False

    def _seed_pre_observation(self, observation: TemporalObservation) -> None:
        """Treat facts visible before an action as already known.

        Without this seed, the initial location and every landmark in ``O_0``
        would be incorrectly credited as a discovery caused by the first
        action.
        """
        if observation.landmark_ids is not None:
            self._known_landmarks.update(observation.landmark_ids)
        if observation.position_xyz is None:
            if not self._topology_nodes:
                descriptor = self._visual_descriptor(observation.image)
                if descriptor is not None:
                    node = _TopologyNode(
                        node_id="visual-node-0000",
                        position_xyz=None,
                        visual_descriptor=descriptor,
                        entry_count=1,
                        last_entry_step_id=self._completed_step_count,
                    )
                    self._topology_nodes.append(node)
                    self._current_node_id = node.node_id
            return
        spatial_nodes = [
            node
            for node in self._topology_nodes
            if node.position_xyz is not None
        ]
        nearest = min(
            spatial_nodes,
            key=lambda node: self._position_distance(
                observation.position_xyz,
                node.position_xyz,
            ),
            default=None,
        )
        if nearest is not None:
            distance = self._position_distance(
                observation.position_xyz,
                nearest.position_xyz,
            )
            if (
                distance is not None
                and distance <= self.config.topology_radius_meters
            ):
                self._current_node_id = nearest.node_id
                return
        node = _TopologyNode(
            node_id=f"node-{len(self._topology_nodes):04d}",
            position_xyz=observation.position_xyz,
            visual_descriptor=self._visual_descriptor(observation.image),
            entry_count=1,
            last_entry_step_id=self._completed_step_count,
        )
        self._topology_nodes.append(node)
        self._current_node_id = node.node_id

    def _new_landmarks(
        self,
        observation: TemporalObservation,
    ) -> tuple[str, ...]:
        if observation.landmark_ids is None:
            return ()
        new = tuple(
            landmark
            for landmark in observation.landmark_ids
            if landmark not in self._known_landmarks
        )
        self._known_landmarks.update(observation.landmark_ids)
        return new

    def _ingest_model_landmarks(self, record: Any) -> None:
        """Store captioned landmarks without retaining older RGB frames.

        R2R does not expose structured landmark IDs.  The step-aligned video
        model does, so after a validated response we attach those labels to the
        corresponding completed observations.  Overlapping sliding windows
        then distinguish genuinely new labels from landmarks already seen in
        the episode.
        """
        captions = getattr(record, "step_captions", ()) or ()
        captions_by_step = {
            int(caption.step_id): tuple(
                dict.fromkeys(
                    str(label).strip()
                    for label in getattr(caption, "visible_landmarks", ())
                    if str(label).strip()
                )
            )
            for caption in captions
        }
        if not captions_by_step:
            return

        updated_steps: list[TemporalStep] = []
        for step in self._steps:
            model_landmarks = captions_by_step.get(step.step_id)
            if model_landmarks is None:
                updated_steps.append(step)
                continue
            supplied = step.post_observation.landmark_ids or ()
            visible = tuple(dict.fromkeys((*supplied, *model_landmarks)))
            newly_discovered = tuple(
                landmark
                for landmark in visible
                if landmark not in self._known_landmarks
            )
            self._known_landmarks.update(visible)
            post_observation = replace(
                step.post_observation,
                landmark_ids=visible,
            )
            updated_steps.append(
                replace(
                    step,
                    post_observation=post_observation,
                    newly_discovered_landmarks=tuple(
                        dict.fromkeys(
                            (
                                *step.newly_discovered_landmarks,
                                *newly_discovered,
                            )
                        )
                    ),
                )
            )
        self._steps = deque(
            updated_steps,
            maxlen=self.config.window_size,
        )

    def _fuse_rule_status_into_record(self, record: Any) -> Any:
        """Keep the public model record consistent with post-model rule data."""
        if not isinstance(record, TemporalMemoryRecord):
            return record
        errors = record.errors.model_copy(deep=True)
        evidence = (
            [record.window.start_seconds, record.window.end_seconds]
            if record.window.end_seconds > record.window.start_seconds
            else []
        )
        for mode in (
            "collision",
            "repeated_visit",
            "motion_oscillation",
            "get_nowhere",
            "action_execution_mismatch",
        ):
            rule_verdict = getattr(self._latest_rule_status, mode)
            if rule_verdict == "UNCERTAIN":
                continue
            current = getattr(errors, mode)
            if current.verdict == rule_verdict:
                continue
            present = rule_verdict == "PRESENT"
            setattr(
                errors,
                mode,
                ErrorAssessment(
                    verdict=rule_verdict,
                    confidence=1.0,
                    interval=(
                        TimeSpan(
                            start_seconds=record.window.start_seconds,
                            end_seconds=record.window.end_seconds,
                        )
                        if present
                        else None
                    ),
                    evidence_timestamps_seconds=(
                        evidence if present else []
                    ),
                    reason=(
                        "Temporal Memory 的结构化逐步证据覆盖了模型判断。"
                    ),
                    source="FUSED",
                ),
            )
        progress = self._latest_rule_status.overall_progress
        return record.model_copy(
            update={
                "errors": errors,
                "overall_progress": (
                    progress
                    if progress != "UNCERTAIN"
                    else record.overall_progress
                ),
            }
        )

    @staticmethod
    def _newly_completed_subgoals(
        before: str,
        after: str,
    ) -> tuple[str, ...]:
        before_complete = {
            match.group("id") for match in _COMPLETE_SUBGOAL.finditer(before)
        }
        after_complete = {
            match.group("id") for match in _COMPLETE_SUBGOAL.finditer(after)
        }
        return tuple(
            sorted(after_complete - before_complete, key=lambda value: int(value))
        )

    def _derive_rule_status(
        self,
        steps: Sequence[TemporalStep],
    ) -> TemporalRuleStatus:
        reasons: list[str] = []
        collisions = [step.motion.collision for step in steps]
        if any(value is True for value in collisions):
            collision: ErrorVerdict = "PRESENT"
            reasons.append("环境碰撞信号报告碰撞。")
        elif collisions and all(value is False for value in collisions):
            collision = "ABSENT"
        else:
            collision = "UNCERTAIN"

        revisits = [step.is_revisit for step in steps]
        if any(value is True for value in revisits):
            repeated_visit: ErrorVerdict = "PRESENT"
            reasons.append("间隔多步后重新进入视觉一致的拓扑节点。")
        elif revisits and all(value is False for value in revisits):
            repeated_visit = "ABSENT"
        else:
            repeated_visit = "UNCERTAIN"

        matches = [step.action_match for step in steps]
        if "MISMATCH" in matches:
            action_mismatch: ErrorVerdict = "PRESENT"
            reasons.append("至少一步实际感知运动与环境命令不一致。")
        elif matches and all(value == "MATCH" for value in matches):
            action_mismatch = "ABSENT"
        else:
            action_mismatch = "UNCERTAIN"

        directions = [
            step.motion.camera_motion
            for step in steps
            if step.motion.camera_motion
            in {CameraMotion.TURN_LEFT, CameraMotion.TURN_RIGHT}
        ]
        reversals = sum(
            previous != current
            for previous, current in zip(directions, directions[1:])
        )
        retrace = self._reverse_retrace_similarity(steps)
        displacement = (
            self._position_distance(
                steps[0].pre_observation.position_xyz,
                steps[-1].post_observation.position_xyz,
            )
            if steps
            else None
        )
        if (
            reversals >= 2
            and retrace is not None
            and retrace >= self.config.oscillation_visual_similarity
            and displacement is not None
            and displacement <= self.config.low_progress_displacement_meters
        ):
            oscillation: ErrorVerdict = "PRESENT"
            reasons.append("左右转向多次反转并回扫到相似画面，净位移很小。")
        elif (
            len(steps) >= self.config.window_size
            and all(
                step.motion.camera_motion != CameraMotion.UNKNOWN for step in steps
            )
            and displacement is not None
            and (
                reversals < 2
                or (
                    retrace is not None
                    and (
                        retrace < self.config.oscillation_visual_similarity
                        or displacement
                        > self.config.low_progress_displacement_meters
                    )
                )
            )
        ):
            oscillation = "ABSENT"
        else:
            oscillation = "UNCERTAIN"

        progress = self._build_progress_signals(steps) if steps else ProgressSignals()
        distance_improvement = self._goal_distance_improvement(steps)
        full_discovery = all(
            value is not None
            for value in (
                progress.new_landmarks_count,
                progress.new_topological_nodes_count,
                progress.completed_subgoals_count,
            )
        )
        no_discovery = full_discovery and all(
            value == 0
            for value in (
                progress.new_landmarks_count,
                progress.new_topological_nodes_count,
                progress.completed_subgoals_count,
            )
        )
        enough_steps = len(steps) >= self.config.get_nowhere_steps
        low_displacement = (
            progress.net_displacement_meters is not None
            and progress.net_displacement_meters
            <= self.config.low_progress_displacement_meters
        )
        low_goal_progress = (
            distance_improvement is not None
            and distance_improvement
            < self.config.goal_progress_epsilon_meters
        )
        if enough_steps and low_displacement and low_goal_progress and no_discovery:
            get_nowhere: ErrorVerdict = "PRESENT"
            reasons.append(
                f"最近{len(steps)}步内无位移、目标距离、地标、"
                "节点或子目标进展。"
            )
        elif (
            progress.net_displacement_meters is not None
            and progress.net_displacement_meters
            > self.config.low_progress_displacement_meters
        ) or (
            distance_improvement is not None
            and distance_improvement
            >= self.config.goal_progress_epsilon_meters
        ) or (
            full_discovery
            and any(
                value is not None and value > 0
                for value in (
                    progress.new_landmarks_count,
                    progress.new_topological_nodes_count,
                    progress.completed_subgoals_count,
                )
            )
        ):
            get_nowhere = "ABSENT"
        else:
            get_nowhere = "UNCERTAIN"

        if (
            distance_improvement is not None
            and distance_improvement < -self.config.goal_progress_epsilon_meters
        ):
            overall: ProgressVerdict = "REGRESSING"
        elif get_nowhere == "PRESENT":
            overall = "STALLED"
        elif get_nowhere == "ABSENT":
            overall = "PROGRESSING"
        else:
            overall = "UNCERTAIN"
        cumulative = self.cumulative_error_state
        if cumulative.phase in {
            CumulativeErrorPhase.SUSPECTED,
            CumulativeErrorPhase.CONFIRMED,
            CumulativeErrorPhase.RECOVERING,
        }:
            overall = "STALLED"
            reasons.append(
                "累计时序错误"
                f"{cumulative.mode.value if cumulative.mode else ''}"
                f"处于{cumulative.phase.value}，覆盖短窗口进展判断。"
            )
        return TemporalRuleStatus(
            collision=collision,
            repeated_visit=repeated_visit,
            motion_oscillation=oscillation,
            get_nowhere=get_nowhere,
            action_execution_mismatch=action_mismatch,
            overall_progress=overall,
            reasons=tuple(reasons),
        )

    def _reverse_retrace_similarity(
        self,
        steps: Sequence[TemporalStep],
    ) -> Optional[float]:
        directions = [
            step.motion.camera_motion
            for step in steps
            if step.motion.camera_motion
            in {CameraMotion.TURN_LEFT, CameraMotion.TURN_RIGHT}
        ]
        if not any(
            previous != current
            for previous, current in zip(directions, directions[1:])
        ):
            return None
        descriptors = [
            self._visual_descriptor(step.post_observation.image) for step in steps
        ]
        similarities = [
            similarity
            for first_index, first in enumerate(descriptors)
            for second_index, second in enumerate(descriptors)
            if (
                second_index - first_index
                >= self.config.oscillation_retrace_min_step_gap
            )
            for similarity in [self._descriptor_similarity(first, second)]
            if similarity is not None
        ]
        return max(similarities) if similarities else None

    @staticmethod
    def _chinese_step_count(count: int) -> str:
        return {
            2: "两",
            3: "三",
            4: "四",
            5: "五",
            6: "六",
            7: "七",
            8: "八",
        }.get(count, str(count))

    @staticmethod
    def _compare_action(
        commanded_action: str,
        observed_motion: CameraMotion,
    ) -> ActionMatch:
        expected = {
            "FORWARD": CameraMotion.FORWARD,
            "TURN_LEFT": CameraMotion.TURN_LEFT,
            "TURN_RIGHT": CameraMotion.TURN_RIGHT,
            "STOP": CameraMotion.STATIONARY,
        }[commanded_action]
        if observed_motion in {
            CameraMotion.UNKNOWN,
            CameraMotion.OSCILLATING_TURN,
        }:
            return "UNCERTAIN"
        return "MATCH" if observed_motion == expected else "MISMATCH"

    def _validate_execution_order(self, execution: StepExecution) -> None:
        expected_step_id = self._completed_step_count + 1
        if execution.step_id != expected_step_id:
            raise TemporalStateError(
                f"Expected 1-based consecutive step_id {expected_step_id}, "
                f"got {execution.step_id}"
            )

    def _ensure_episode(self, observation: TemporalObservation) -> None:
        if observation.episode_id != self.episode_id:
            raise TemporalStateError(
                f"Observation belongs to episode {observation.episode_id!r}; "
                f"active episode is {self.episode_id!r}"
            )

    @staticmethod
    def _position_distance(
        first: Optional[tuple[float, float, float]],
        second: Optional[tuple[float, float, float]],
    ) -> Optional[float]:
        if first is None or second is None:
            return None
        return math.sqrt(sum((left - right) ** 2 for left, right in zip(first, second)))

    @staticmethod
    def _yaw_delta(
        first: Optional[float],
        second: Optional[float],
    ) -> Optional[float]:
        if first is None or second is None:
            return None
        return (second - first + 180.0) % 360.0 - 180.0

    @staticmethod
    def _goal_distance_improvement(
        steps: Sequence[TemporalStep],
    ) -> Optional[float]:
        if not steps:
            return None
        start = steps[0].pre_observation.distance_to_goal_meters
        end = steps[-1].post_observation.distance_to_goal_meters
        if start is None or end is None:
            return None
        return start - end

    @staticmethod
    def _rgb_array(image: Any) -> Any:
        import io

        import numpy as np
        from PIL import Image

        if isinstance(image, np.ndarray):
            array = image
            if array.ndim == 2:
                array = np.repeat(array[..., None], 3, axis=2)
            if array.ndim != 3 or array.shape[2] not in (3, 4):
                raise ValueError("numpy image must be HxWx3 or HxWx4")
            if array.shape[2] == 4:
                array = array[..., :3]
            if array.dtype != np.uint8:
                if np.issubdtype(array.dtype, np.floating):
                    scale = 255.0 if float(np.nanmax(array)) <= 1.0 else 1.0
                    array = np.clip(array * scale, 0, 255).astype(np.uint8)
                else:
                    array = np.clip(array, 0, 255).astype(np.uint8)
            return np.ascontiguousarray(array)
        if isinstance(image, Image.Image):
            return np.asarray(image.convert("RGB"))
        if isinstance(image, (bytes, bytearray, memoryview)):
            return np.asarray(
                Image.open(io.BytesIO(bytes(image))).convert("RGB")
            )
        if isinstance(image, (str, bytes)):
            return np.asarray(Image.open(image).convert("RGB"))
        raise TypeError(f"Unsupported image type: {type(image).__name__}")

    @classmethod
    def _visual_descriptor(cls, image: Any) -> Optional[Any]:
        try:
            import cv2

            rgb = cls._rgb_array(image)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            descriptor = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
            return descriptor.astype("float32").reshape(-1)
        except Exception:
            return None

    @staticmethod
    def _descriptor_similarity(
        first: Optional[Any],
        second: Optional[Any],
    ) -> Optional[float]:
        if first is None or second is None:
            return None
        try:
            import numpy as np

            left = first.astype("float32", copy=True)
            right = second.astype("float32", copy=True)
            left -= float(left.mean())
            right -= float(right.mean())
            denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
            if denominator <= 1e-9:
                # Two constant images are identical only when their values are.
                return 1.0 if bool(np.array_equal(first, second)) else 0.0
            return max(
                -1.0,
                min(1.0, float(np.dot(left, right) / denominator)),
            )
        except Exception:
            return None


__all__ = (
    "ActionMatch",
    "CumulativeErrorMode",
    "CumulativeErrorPhase",
    "CumulativeErrorState",
    "ErrorVerdict",
    "GoBackRequest",
    "ProgressVerdict",
    "StepExecution",
    "TemporalEvidenceStep",
    "TemporalMemory",
    "TemporalMemoryConfig",
    "TemporalMemoryError",
    "TemporalObservation",
    "TemporalRuleStatus",
    "TemporalStateError",
    "TemporalStep",
)
