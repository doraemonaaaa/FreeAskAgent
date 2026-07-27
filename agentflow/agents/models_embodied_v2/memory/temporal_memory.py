"""Latest eight VLN actions, their resulting images, and temporal events."""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Deque, Optional, Protocol

from ..TemporalCaptioner import (
    CaptionResult,
    ErrorMode,
    Subgoal,
    TemporalAnalysisRequest,
    TemporalCaptioner,
    TemporalInputError,
    TemporalStepInput,
    normalize_action,
)


class TemporalMemoryError(RuntimeError):
    pass


class TemporalStateError(TemporalMemoryError):
    pass


class TemporalEventKind(str, Enum):
    SUBGOAL_COMPLETED = "SUBGOAL_COMPLETED"
    GO_BACK_TO_ACTION = "GO_BACK_TO_ACTION"


@dataclass(frozen=True, slots=True)
class TemporalEvent:
    """Minimal event passed to Task Memory."""

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


class TaskMemoryPort(Protocol):
    def get_task(self) -> str: ...

    def get_current_subgoal(self) -> Optional[Subgoal]: ...

    def publish_temporal_event(self, event: TemporalEvent) -> None: ...


@dataclass(frozen=True, slots=True)
class MemoryStep:
    """One action paired with the image observed after execution."""

    step_id: int
    action: str
    post_image: Any
    subgoal_id: str


@dataclass(frozen=True, slots=True)
class TemporalMemoryConfig:
    window_size: int = 8
    stationary_threshold: float = 0.02
    revisit_threshold: float = 0.05

    def __post_init__(self) -> None:
        if self.window_size != 8:
            raise ValueError("Temporal Memory uses a fixed eight-step window")
        if not 0 < self.stationary_threshold < self.revisit_threshold < 1:
            raise ValueError(
                "visual thresholds must satisfy 0 < stationary < revisit < 1"
            )


def _image_copy(image: Any) -> Any:
    if image is None:
        raise TemporalStateError("post_image must not be None")
    if isinstance(image, bytes):
        return image
    copier = getattr(image, "copy", None)
    return copier() if callable(copier) else copy.deepcopy(image)


class TemporalMemory:
    """Build one sliding eight-step request and publish its two event types."""

    def __init__(
        self,
        *,
        captioner: TemporalCaptioner,
        task_memory: TaskMemoryPort,
        config: Optional[TemporalMemoryConfig] = None,
    ) -> None:
        self.captioner = captioner
        self.task_memory = task_memory
        self.config = config or TemporalMemoryConfig()
        self._steps: Deque[MemoryStep] = deque(maxlen=8)
        self._events: Deque[TemporalEvent] = deque()
        self._subgoal: Optional[Subgoal] = None
        self._latest_result: Optional[CaptionResult] = None
        self._last_error: Optional[str] = None
        self._last_analyzed_step: Optional[int] = None
        self._next_step_id = 1
        self._latched_error_mode: ErrorMode = "NONE"
        self.reset()

    @property
    def current_subgoal(self) -> Optional[Subgoal]:
        return self._subgoal

    @property
    def latest_result(self) -> Optional[CaptionResult]:
        return self._latest_result

    @property
    def last_analysis_error(self) -> Optional[str]:
        return self._last_error

    def reset(self) -> None:
        self._steps.clear()
        self._events.clear()
        self._subgoal = self.task_memory.get_current_subgoal()
        self._latest_result = None
        self._last_error = None
        self._last_analyzed_step = None
        self._next_step_id = 1
        self._latched_error_mode = "NONE"

    def append_step(self, action: str, post_image: Any) -> MemoryStep:
        """Append only after the action's resulting observation is available."""
        self._sync_subgoal()
        if self._subgoal is None:
            raise TemporalStateError("current subgoal is not set")
        try:
            action = normalize_action(action)
        except TemporalInputError as exc:
            raise TemporalStateError(str(exc)) from exc
        step = MemoryStep(
            step_id=self._next_step_id,
            action=action,
            post_image=_image_copy(post_image),
            subgoal_id=self._subgoal.subgoal_id,
        )
        self._steps.append(step)
        self._next_step_id += 1
        return step

    def analyze_if_ready(self) -> Optional[CaptionResult]:
        if len(self._steps) < 8:
            return None
        newest = self._steps[-1].step_id
        if newest == self._last_analyzed_step:
            return None
        try:
            return self.analyze()
        except Exception as exc:
            self._last_analyzed_step = newest
            self._last_error = f"{type(exc).__name__}: {exc}"
            return None

    def analyze(self) -> CaptionResult:
        self._sync_subgoal()
        if self._subgoal is None:
            raise TemporalStateError("current subgoal is not set")
        if len(self._steps) != 8:
            raise TemporalStateError("exactly eight steps are required")
        request = TemporalAnalysisRequest(
            subgoal=self._subgoal,
            steps=tuple(
                TemporalStepInput(step.step_id, step.action, step.post_image)
                for step in self._steps
            ),
        )
        try:
            result = self.captioner.analyze(request)
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            raise
        self._store(result)
        return result

    def recent_steps(self) -> tuple[MemoryStep, ...]:
        return tuple(self._steps)

    def recent_actions(self) -> tuple[str, ...]:
        return tuple(step.action for step in self._steps)

    def drain_events(self) -> tuple[TemporalEvent, ...]:
        events = tuple(self._events)
        self._events.clear()
        return events

    def context(self) -> str:
        return (
            self._latest_result.to_memory_text()
            if self._latest_result
            else f"Temporal window: {len(self._steps)}/8 steps"
        )

    def diagnostics(self, *, include_raw_response: bool = False) -> dict[str, Any]:
        result = self._latest_result.model_dump() if self._latest_result else None
        if result and not include_raw_response:
            result.pop("raw_response", None)
        return {
            "current_subgoal_id": self._subgoal.subgoal_id if self._subgoal else None,
            "step_ids": [step.step_id for step in self._steps],
            "actions": list(self.recent_actions()),
            "active_error_mode": self._latched_error_mode,
            "last_analysis_error": self._last_error,
            "pending_events": [event.to_dict() for event in self._events],
            "latest_result": result,
        }

    def _sync_subgoal(self) -> None:
        current = self.task_memory.get_current_subgoal()
        old_id = self._subgoal.subgoal_id if self._subgoal else None
        new_id = current.subgoal_id if current else None
        if old_id != new_id:
            self._steps.clear()
            self._latest_result = None
            self._last_analyzed_step = None
            self._latched_error_mode = "NONE"
        self._subgoal = current

    def _store(self, result: CaptionResult) -> None:
        assert self._subgoal is not None
        if result.subgoal_id != self._subgoal.subgoal_id:
            raise TemporalStateError("Captioner returned the wrong subgoal")
        self._latest_result = result
        self._last_error = None
        self._last_analyzed_step = self._steps[-1].step_id
        error_mode = "NONE" if result.completed else self._detect_error_mode()

        self._publish(
            TemporalEvent(
                kind=TemporalEventKind.SUBGOAL_COMPLETED,
                value=result.completed,
                subgoal_id=result.subgoal_id,
                error_mode=error_mode,
            )
        )
        if result.completed:
            self._latched_error_mode = "NONE"
            return

        if error_mode != "NONE" and error_mode != self._latched_error_mode:
            self._publish(
                TemporalEvent(
                    kind=TemporalEventKind.GO_BACK_TO_ACTION,
                    value=True,
                    subgoal_id=result.subgoal_id,
                    error_mode=error_mode,
                )
            )
            self._latched_error_mode = error_mode
        elif error_mode == "NONE":
            self._latched_error_mode = "NONE"

    def _detect_error_mode(self) -> ErrorMode:
        """Classify repeated failures from actions and lightweight frame change."""
        signatures = [_visual_signature(step.post_image) for step in self._steps]
        actions = self.recent_actions()
        adjacent = [
            _visual_distance(signatures[index - 1], signatures[index])
            for index in range(1, 8)
        ]

        stuck_forward = sum(
            actions[index] == "FORWARD"
            and adjacent[index - 1] <= self.config.stationary_threshold
            for index in range(1, 8)
        )
        if stuck_forward >= 3:
            return "WALL_STUCK"

        opposite = {
            ("TURN_LEFT", "TURN_RIGHT"),
            ("TURN_RIGHT", "TURN_LEFT"),
        }
        retraced_turns = sum(
            (actions[index - 1], actions[index]) in opposite
            and _visual_distance(signatures[index - 2], signatures[index])
            <= self.config.revisit_threshold
            for index in range(2, 8)
        )
        if retraced_turns >= 2:
            return "TURN_OSCILLATION"

        turn_indices = [
            index
            for index, action in enumerate(actions)
            if action in {"TURN_LEFT", "TURN_RIGHT"}
        ]
        revisited_view = any(
            later - earlier >= 3
            and _visual_distance(signatures[earlier], signatures[later])
            <= self.config.revisit_threshold
            for earlier in turn_indices
            for later in turn_indices
            if later > earlier
        )
        if len(turn_indices) >= 6 and revisited_view:
            return "IN_PLACE_SPIN"

        stationary_transitions = sum(
            distance <= self.config.stationary_threshold
            for distance in adjacent
        )
        if stationary_transitions >= 5:
            return "GET_NOWHERE"
        return "NONE"

    def _publish(self, event: TemporalEvent) -> None:
        self._events.append(event)
        self.task_memory.publish_temporal_event(event)


def _visual_signature(image: Any) -> Any:
    """Return a tiny grayscale frame used only for rule-based comparisons."""
    import io

    import numpy as np
    from PIL import Image

    if isinstance(image, bytes):
        pil = Image.open(io.BytesIO(image))
    elif isinstance(image, Image.Image):
        pil = image
    else:
        array = np.asarray(image)
        if array.dtype != np.uint8:
            if np.issubdtype(array.dtype, np.floating) and array.size:
                if float(np.nanmax(array)) <= 1:
                    array = array * 255
            array = np.clip(array, 0, 255).astype(np.uint8)
        pil = Image.fromarray(array)
    return np.asarray(
        pil.convert("L").resize((32, 32)),
        dtype=np.float32,
    ) / 255.0


def _visual_distance(first: Any, second: Any) -> float:
    import numpy as np

    return float(np.mean(np.abs(first - second)))


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
