"""Minimal eight-step Temporal Memory for standalone VLN experiments.

Temporal Memory stores executed ``action -> post-action image`` pairs, asks a
Captioner to understand the current eight-step window, and publishes two
boolean events.  It deliberately does not own topology, optical flow, task
planning, Habitat metadata, or recovery actions.
"""

from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Deque, Optional, Protocol

from ..TemporalCaptioner import (
    ACTION_TOKENS,
    CaptionResult,
    StepUnderstanding,
    Subgoal,
    TemporalAnalysisRequest,
    TemporalCaptioner,
    TemporalStepInput,
)


class TemporalMemoryError(RuntimeError):
    """Base Temporal Memory error."""


class TemporalStateError(TemporalMemoryError):
    """The memory lifecycle or input order is invalid."""


class TemporalEventKind(str, Enum):
    SUBGOAL_COMPLETED = "SUBGOAL_COMPLETED"
    GO_BACK_TO_ACTION = "GO_BACK_TO_ACTION"


@dataclass(frozen=True, slots=True)
class TemporalEvent:
    """The event payload exposed to Task Memory is intentionally one boolean."""

    kind: TemporalEventKind
    value: bool = True

    def __post_init__(self) -> None:
        if type(self.value) is not bool:
            raise TypeError("TemporalEvent.value must be bool")


class TaskMemoryPort(Protocol):
    """Small boundary used only by this standalone submodule."""

    def get_task(self) -> str: ...

    def get_task_guidance(self) -> str: ...

    def get_current_subgoal(self) -> Optional[Subgoal]: ...

    def publish_temporal_event(
        self,
        kind: TemporalEventKind,
        value: bool,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class MemoryStep:
    """One executed action paired with its first post-action observation."""

    step_id: int
    action: str
    post_image: Any
    subgoal_id: str
    timestamp_seconds: Optional[float] = None
    understanding: Optional[StepUnderstanding] = None


@dataclass(frozen=True, slots=True)
class TemporalMemoryConfig:
    window_size: int = 8
    min_error_evidence_steps: int = 3

    def __post_init__(self) -> None:
        if self.window_size != 8:
            raise ValueError("Temporal Memory uses a fixed eight-step window")
        if not 2 <= self.min_error_evidence_steps <= self.window_size:
            raise ValueError(
                "min_error_evidence_steps must be between 2 and 8"
            )


def _required_text(value: Any, label: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise TemporalStateError(f"{label} must not be empty")
    return result


def _normalize_action(action: str) -> str:
    normalized = str(action or "").strip().upper()
    normalized = {
        "MOVE_FORWARD": "FORWARD",
        "LEFT": "TURN_LEFT",
        "RIGHT": "TURN_RIGHT",
    }.get(normalized, normalized)
    if normalized not in ACTION_TOKENS:
        raise TemporalStateError(
            f"Unsupported action {action!r}; expected one of {ACTION_TOKENS}"
        )
    return normalized


def _snapshot_image(image: Any) -> Any:
    if image is None:
        raise TemporalStateError("post_image must not be None")
    if isinstance(image, bytes):
        return image
    copier = getattr(image, "copy", None)
    if callable(copier):
        try:
            return copier()
        except TypeError:
            pass
    return copy.deepcopy(image)


class TemporalMemory:
    """Keep and understand the latest eight executed navigation steps."""

    def __init__(
        self,
        *,
        captioner: Optional[TemporalCaptioner] = None,
        task_memory: Optional[TaskMemoryPort] = None,
        config: Optional[TemporalMemoryConfig] = None,
    ) -> None:
        self.captioner = captioner
        self.task_memory = task_memory
        self.config = config or TemporalMemoryConfig()
        self.task = ""
        self.task_guidance = ""
        self._steps: Deque[MemoryStep] = deque(
            maxlen=self.config.window_size
        )
        self._events: Deque[TemporalEvent] = deque()
        self._current_subgoal: Optional[Subgoal] = None
        self._latest_result: Optional[CaptionResult] = None
        self._last_analysis_error: Optional[str] = None
        self._last_analyzed_step_id: Optional[int] = None
        self._last_attempted_step_id: Optional[int] = None
        self._next_step_id = 1
        self._completed_event_subgoals: set[str] = set()
        self._error_event_latched = False

    @property
    def current_subgoal(self) -> Optional[Subgoal]:
        return self._current_subgoal

    @property
    def latest_result(self) -> Optional[CaptionResult]:
        return self._latest_result

    @property
    def latest_record(self) -> Optional[CaptionResult]:
        """Compatibility name for callers that mean the latest model result."""
        return self._latest_result

    @property
    def last_analysis_error(self) -> Optional[str]:
        return self._last_analysis_error

    def reset(
        self,
        *,
        task: Optional[str] = None,
        task_guidance: Optional[str] = None,
    ) -> None:
        """Clear all temporal state while keeping the Captioner and port."""
        if self.task_memory is not None:
            task = self.task_memory.get_task() if task is None else task
            task_guidance = (
                self.task_memory.get_task_guidance()
                if task_guidance is None
                else task_guidance
            )
            subgoal = self.task_memory.get_current_subgoal()
        else:
            subgoal = None
        self.task = _required_text(task, "task")
        self.task_guidance = _required_text(
            task_guidance, "task_guidance"
        )
        self._steps.clear()
        self._events.clear()
        self._current_subgoal = None
        self._latest_result = None
        self._last_analysis_error = None
        self._last_analyzed_step_id = None
        self._last_attempted_step_id = None
        self._next_step_id = 1
        self._completed_event_subgoals.clear()
        self._error_event_latched = False
        if subgoal is not None:
            self.set_subgoal(subgoal)

    def set_subgoal(self, subgoal: Subgoal) -> None:
        if not isinstance(subgoal, Subgoal):
            raise TypeError("subgoal must be a Subgoal")
        if (
            self._current_subgoal is not None
            and self._current_subgoal.subgoal_id != subgoal.subgoal_id
        ):
            # Images collected for the previous subgoal are not evidence for
            # the newly activated one.  Start a fresh eight-step window while
            # keeping global step IDs and the previous result for diagnostics.
            self._steps.clear()
            self._last_attempted_step_id = None
        self._current_subgoal = subgoal

    def sync_from_task_memory(self) -> Optional[Subgoal]:
        """Refresh task context and current subgoal from the configured port."""
        if self.task_memory is None:
            raise TemporalStateError("TaskMemoryPort is not configured")
        self.task = _required_text(self.task_memory.get_task(), "task")
        self.task_guidance = _required_text(
            self.task_memory.get_task_guidance(), "task_guidance"
        )
        subgoal = self.task_memory.get_current_subgoal()
        if subgoal is None:
            self._current_subgoal = None
            self._steps.clear()
            self._last_attempted_step_id = None
        else:
            self.set_subgoal(subgoal)
        return subgoal

    def append_step(
        self,
        action: str,
        post_image: Any,
        timestamp_seconds: Optional[float] = None,
    ) -> MemoryStep:
        """Push an already executed action and its post-action image."""
        if self.task_memory is not None:
            self.sync_from_task_memory()
        if self._current_subgoal is None:
            raise TemporalStateError("current subgoal is not set")
        if timestamp_seconds is not None:
            timestamp_seconds = float(timestamp_seconds)
            if (
                not math.isfinite(timestamp_seconds)
                or timestamp_seconds < 0
            ):
                raise TemporalStateError(
                    "timestamp_seconds must be finite and non-negative"
                )
            if (
                self._steps
                and self._steps[-1].timestamp_seconds is not None
                and timestamp_seconds <= self._steps[-1].timestamp_seconds
            ):
                raise TemporalStateError(
                    "timestamps must be strictly increasing"
                )
        step = MemoryStep(
            step_id=self._next_step_id,
            action=_normalize_action(action),
            post_image=_snapshot_image(post_image),
            timestamp_seconds=timestamp_seconds,
            subgoal_id=self._current_subgoal.subgoal_id,
        )
        self._steps.append(step)
        self._next_step_id += 1
        return step

    def analyze_if_ready(self) -> Optional[CaptionResult]:
        """Analyze once for each newly completed eight-step window."""
        if len(self._steps) < self.config.window_size:
            return None
        newest_id = self._steps[-1].step_id
        if newest_id == self._last_attempted_step_id:
            return None
        self._last_attempted_step_id = newest_id
        try:
            return self.analyze()
        except Exception as exc:
            self._last_analysis_error = f"{type(exc).__name__}: {exc}"
            return None

    def analyze(self) -> CaptionResult:
        """Force analysis of the current full window; errors remain observable."""
        if len(self._steps) != self.config.window_size:
            raise TemporalStateError(
                "exactly eight completed steps are required for analysis"
            )
        if self.captioner is None:
            raise TemporalStateError("TemporalCaptioner is not configured")
        if self.task_memory is not None:
            self.sync_from_task_memory()
        if self._current_subgoal is None:
            raise TemporalStateError("current subgoal is not set")
        request = TemporalAnalysisRequest(
            task=self.task,
            task_guidance=self.task_guidance,
            subgoals=(self._current_subgoal,),
            steps=tuple(
                TemporalStepInput(
                    step_id=step.step_id,
                    action=step.action,
                    image=step.post_image,
                    timestamp_seconds=step.timestamp_seconds,
                )
                for step in self._steps
            ),
        )
        try:
            result = self.captioner.analyze(request)
        except Exception as exc:
            self._last_analysis_error = f"{type(exc).__name__}: {exc}"
            raise
        self._store_result(result)
        return result

    def recent_steps(self) -> tuple[MemoryStep, ...]:
        """Return oldest-to-newest order; the newest stack item is on the right."""
        return tuple(self._steps)

    def recent_actions(self) -> tuple[str, ...]:
        return tuple(step.action for step in self._steps)

    def recent_understandings(
        self,
    ) -> tuple[Optional[StepUnderstanding], ...]:
        return tuple(step.understanding for step in self._steps)

    def drain_events(self) -> tuple[TemporalEvent, ...]:
        events = tuple(self._events)
        self._events.clear()
        return events

    def diagnostics(self, *, include_raw_response: bool = False) -> dict[str, Any]:
        result = None
        if self._latest_result is not None:
            result = self._latest_result.model_dump()
            if not include_raw_response:
                result.pop("raw_response", None)
        return {
            "task": self.task,
            "current_subgoal_id": (
                self._current_subgoal.subgoal_id
                if self._current_subgoal is not None
                else None
            ),
            "step_ids": [step.step_id for step in self._steps],
            "actions": [step.action for step in self._steps],
            "step_subgoal_ids": [
                step.subgoal_id for step in self._steps
            ],
            "analyzed_step_id": self._last_analyzed_step_id,
            "last_analysis_error": self._last_analysis_error,
            "pending_events": [
                {"kind": event.kind.value, "value": event.value}
                for event in self._events
            ],
            "latest_result": result,
        }

    def _store_result(self, result: CaptionResult) -> None:
        by_step_id = {item.step_id: item for item in result.steps}
        self._steps = deque(
            (
                replace(
                    step,
                    understanding=by_step_id.get(step.step_id),
                )
                for step in self._steps
            ),
            maxlen=self.config.window_size,
        )
        self._latest_result = result
        self._last_analysis_error = None
        self._last_analyzed_step_id = self._steps[-1].step_id
        self._last_attempted_step_id = self._last_analyzed_step_id

        assert self._current_subgoal is not None
        status = result.status_for(self._current_subgoal.subgoal_id)
        if status.completed:
            if status.subgoal_id not in self._completed_event_subgoals:
                self._completed_event_subgoals.add(status.subgoal_id)
                self._publish(
                    TemporalEventKind.SUBGOAL_COMPLETED,
                    True,
                )
            self._error_event_latched = False
            return

        # Every successful incomplete window explicitly tells Task Memory to
        # keep tracking the current subgoal.  A model failure never reaches
        # this branch, so "unknown" cannot be mistaken for "still ongoing".
        self._publish(TemporalEventKind.SUBGOAL_COMPLETED, False)

        sustained_error = (
            result.persistent_error
            and result.error_mode != "NONE"
            and len(set(result.error_evidence_step_ids))
            >= self.config.min_error_evidence_steps
        )
        if sustained_error and not self._error_event_latched:
            self._publish(TemporalEventKind.GO_BACK_TO_ACTION, True)
            self._error_event_latched = True
        elif not sustained_error:
            self._error_event_latched = False

    def _publish(
        self,
        kind: TemporalEventKind,
        value: bool = True,
    ) -> None:
        event = TemporalEvent(kind=kind, value=value)
        self._events.append(event)
        if self.task_memory is not None:
            self.task_memory.publish_temporal_event(kind, value)


__all__ = (
    "MemoryStep",
    "TaskMemoryPort",
    "TemporalEvent",
    "TemporalEventKind",
    "TemporalMemory",
    "TemporalMemoryConfig",
    "TemporalMemoryError",
    "TemporalStateError",
)
