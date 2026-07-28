"""Latest eight VLN observations and their temporal events."""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import asdict, replace
from typing import Any, Deque, Optional, Protocol, Sequence

from ..TemporalCaptioner import (
    CaptionResult,
    ErrorMode,
    Subgoal,
    TemporalAnalysisRequest,
    TemporalCaptioner,
    TemporalFrameInput,
)
from ..data_models import (
    MemoryFrame,
    TemporalEvent,
    TemporalEventKind,
    TemporalMemoryConfig,
)


class TemporalMemoryError(RuntimeError):
    pass


class TemporalStateError(TemporalMemoryError):
    pass


class TaskMemoryPort(Protocol):
    def reset(
        self,
        *,
        goal: str,
        task_guidance: str = "",
        subgoals: Sequence[Any] = (),
    ) -> None: ...

    def get_task(self) -> str: ...

    def get_current_subgoal(self) -> Optional[Subgoal]: ...

    def get_latest_observation(self) -> Any: ...

    def get_reset_generation(self) -> int: ...

    def publish_temporal_event(self, event: TemporalEvent) -> None: ...


def _image_copy(image: Any) -> Any:
    if image is None:
        raise TemporalStateError("image must not be None")
    if isinstance(image, bytes):
        return image
    copier = getattr(image, "copy", None)
    return copier() if callable(copier) else copy.deepcopy(image)


class TemporalMemory:
    """Analyze the sliding window on every new frame and publish its events.

    The window holds at most eight frames, but analysis runs as soon as Task
    Memory delivers one observation rather than waiting for a full window.
    """

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
        self._frames: Deque[MemoryFrame] = deque(maxlen=8)
        self._events: Deque[TemporalEvent] = deque()
        self._subgoal: Optional[Subgoal] = None
        self._latest_result: Optional[CaptionResult] = None
        self._last_error: Optional[str] = None
        self._last_analyzed_frame: Optional[int] = None
        self._next_frame_id = 1
        self._last_consumed_observation_count: Optional[int] = None
        self._task_reset_generation = -1
        self.reset()

    @property
    def current_subgoal(self) -> Optional[Subgoal]:
        self._sync_task_state()
        return self._subgoal

    @property
    def latest_result(self) -> Optional[CaptionResult]:
        self._sync_task_state()
        return self._latest_result

    @property
    def last_analysis_error(self) -> Optional[str]:
        self._sync_task_state()
        return self._last_error

    def reset(self) -> None:
        self._frames.clear()
        self._events.clear()
        self._subgoal = self.task_memory.get_current_subgoal()
        self._latest_result = None
        self._last_error = None
        self._last_analyzed_frame = None
        self._next_frame_id = 1
        self._last_consumed_observation_count = None
        self._task_reset_generation = (
            self.task_memory.get_reset_generation()
        )

    def reset_episode(
        self,
        *,
        goal: str,
        task_guidance: str = "",
        subgoals: Sequence[Any] = (),
    ) -> None:
        """Reset Task and Temporal Memory together for one new episode."""
        self.task_memory.reset(
            goal=goal,
            task_guidance=task_guidance,
            subgoals=subgoals,
        )
        self.reset()

    def get_latest_observation(self) -> Any:
        """Read the latest RGB observation from Task Memory."""
        rgb = self.task_memory.get_latest_observation()
        if rgb is None:
            raise TemporalStateError("Task Memory has no RGB observation")
        return rgb

    def append_latest_observation(self) -> Optional[MemoryFrame]:
        """Fetch one new RGB from Task Memory and append it to the window."""
        # A new episode may reuse the same observation count as the previous
        # one, so reset synchronization must happen before duplicate checking.
        self._sync_task_state()
        observation_count = getattr(
            self.task_memory,
            "observation_count",
            None,
        )
        if (
            observation_count is not None
            and observation_count == self._last_consumed_observation_count
        ):
            return None

        frame = self.append_observation(self.get_latest_observation())
        self._last_consumed_observation_count = observation_count
        return frame

    def append_observation(self, image: Any) -> MemoryFrame:
        """Append one RGB observation to the current subgoal's window."""
        self._sync_task_state()
        if self._subgoal is None:
            raise TemporalStateError("current subgoal is not set")
        frame = MemoryFrame(
            frame_id=self._next_frame_id,
            image=_image_copy(image),
            subgoal_id=self._subgoal.subgoal_id,
        )
        self._frames.append(frame)
        self._next_frame_id += 1
        return frame

    def update_from_task_memory(self) -> Optional[CaptionResult]:
        """Consume the latest Task Memory RGB and analyze the current window."""
        self._sync_task_state()
        # An exhausted plan leaves no subgoal to judge against. Appending would
        # raise, so report "nothing analyzed" and let the caller end the task.
        if self._subgoal is None:
            return None
        if self.append_latest_observation() is None:
            return None
        return self.analyze_if_ready()

    def analyze_if_ready(self) -> Optional[CaptionResult]:
        self._sync_task_state()
        if not self._frames:
            return None
        newest = self._frames[-1].frame_id
        if newest == self._last_analyzed_frame:
            return None
        try:
            return self.analyze()
        except Exception as exc:
            self._last_analyzed_frame = newest
            self._last_error = f"{type(exc).__name__}: {exc}"
            return None

    def analyze(self) -> CaptionResult:
        self._sync_task_state()
        if self._subgoal is None:
            raise TemporalStateError("current subgoal is not set")
        if not self._frames:
            raise TemporalStateError("at least one frame is required")
        request = TemporalAnalysisRequest(
            subgoal=self._subgoal,
            frames=tuple(
                TemporalFrameInput(frame.frame_id, frame.image)
                for frame in self._frames
            ),
        )
        try:
            result = self.captioner.analyze(request)
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            raise
        return self._store(result)

    def recent_frames(self) -> tuple[MemoryFrame, ...]:
        self._sync_task_state()
        return tuple(self._frames)

    def drain_events(self) -> tuple[TemporalEvent, ...]:
        self._sync_task_state()
        events = tuple(self._events)
        self._events.clear()
        return events

    def context(self) -> str:
        self._sync_task_state()
        return (
            self._latest_result.to_memory_text()
            if self._latest_result
            else (
                f"Temporal window: {len(self._frames)}/"
                f"{self._frames.maxlen} frames"
            )
        )

    def diagnostics(self, *, include_raw_response: bool = False) -> dict[str, Any]:
        self._sync_task_state()
        result = asdict(self._latest_result) if self._latest_result else None
        if result and not include_raw_response:
            result.pop("raw_response", None)
        return {
            "current_subgoal_id": self._subgoal.subgoal_id if self._subgoal else None,
            "frame_ids": [frame.frame_id for frame in self._frames],
            "active_error_mode": (
                self._latest_result.error_mode
                if self._latest_result and self._latest_result.error
                else "NONE"
            ),
            "last_analysis_error": self._last_error,
            "pending_events": [event.to_dict() for event in self._events],
            "latest_result": result,
        }

    def _sync_task_state(self) -> None:
        generation = self.task_memory.get_reset_generation()
        if generation != self._task_reset_generation:
            self.reset()
            return

        current = self.task_memory.get_current_subgoal()
        old_id = self._subgoal.subgoal_id if self._subgoal else None
        new_id = current.subgoal_id if current else None
        if old_id != new_id:
            self._frames.clear()
            self._latest_result = None
            self._last_analyzed_frame = None
        self._subgoal = current

    def _store(self, result: CaptionResult) -> CaptionResult:
        assert self._subgoal is not None
        if result.subgoal_id != self._subgoal.subgoal_id:
            raise TemporalStateError("Captioner returned the wrong subgoal")

        if not result.error:
            rule_mode = self._detect_error_mode()
            if rule_mode != "NONE":
                result = replace(
                    result,
                    error=True,
                    error_mode=rule_mode,
                )

        self._latest_result = result
        self._last_error = None
        self._last_analyzed_frame = self._frames[-1].frame_id

        self._publish(
            TemporalEvent(
                kind=TemporalEventKind.ERROR,
                value=result.error,
                subgoal_id=result.subgoal_id,
                error_mode=result.error_mode,
            )
        )
        self._publish(
            TemporalEvent(
                kind=TemporalEventKind.SUBGOAL_COMPLETED,
                value=result.completed,
                subgoal_id=result.subgoal_id,
                error_mode=result.error_mode,
            )
        )
        return result

    def _detect_error_mode(self) -> ErrorMode:
        """Detect cumulative errors from the visual sequence only."""
        count = len(self._frames)
        # Cumulative motion errors need a short sequence to be visible at all;
        # one or two frames carry no evidence of oscillation or lack of progress.
        if count < self.config.min_error_detection_frames:
            return "NONE"
        signatures = [_visual_signature(frame.image) for frame in self._frames]
        adjacent = [
            _visual_distance(signatures[index - 1], signatures[index])
            for index in range(1, count)
        ]
        lag_two = [
            _visual_distance(signatures[index - 2], signatures[index])
            for index in range(2, count)
        ]
        moving_threshold = max(
            self.config.stationary_threshold * 2,
            0.04,
        )

        if (
            sum(value >= moving_threshold for value in adjacent)
            >= _required_count(4, 7, len(adjacent))
            and sum(
                value <= self.config.stationary_threshold
                for value in lag_two
            )
            >= _required_count(4, 6, len(lag_two))
        ):
            return "TURN_OSCILLATION"

        revisited_view = any(
            later - earlier >= 3
            and _visual_distance(signatures[earlier], signatures[later])
            <= self.config.revisit_threshold
            for earlier in range(count)
            for later in range(earlier + 1, count)
        )
        if (
            sum(value >= moving_threshold for value in adjacent)
            >= _required_count(5, 7, len(adjacent))
            and revisited_view
        ):
            return "IN_PLACE_SPIN"

        stationary_transitions = sum(
            distance <= self.config.stationary_threshold
            for distance in adjacent
        )
        if stationary_transitions >= _required_count(5, 7, len(adjacent)):
            return "GET_NOWHERE"
        return "NONE"

    def _publish(self, event: TemporalEvent) -> None:
        self._events.append(event)
        self.task_memory.publish_temporal_event(event)


def _required_count(numerator: int, denominator: int, total: int) -> int:
    """Scale a full-window evidence threshold to a partial window.

    The full eight-frame window keeps its original absolute thresholds; a
    shorter window requires the same proportion of supporting transitions.
    """
    return max(1, -(-numerator * total // denominator))


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
    "MemoryFrame",
    "TaskMemoryPort",
    "TemporalEvent",
    "TemporalEventKind",
    "TemporalMemory",
    "TemporalMemoryConfig",
    "TemporalMemoryError",
    "TemporalStateError",
)
