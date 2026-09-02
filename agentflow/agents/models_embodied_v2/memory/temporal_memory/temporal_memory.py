"""Latest eight VLN observations and their temporal events."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict
from typing import Any, Deque, Optional

from .temporal_captioner import (
    CaptionResult,
    Subgoal,
    TemporalAnalysisRequest,
    TemporalFrameInput,
)
from .interfaces import TaskMemoryPort, TemporalCaptionerPort
from .frame_history import copy_image
from ...data_models import (
    MemoryFrame,
    TemporalEvent,
    TemporalEventKind,
    TemporalMemoryConfig,
)


class TemporalMemoryError(RuntimeError):
    pass


class TemporalStateError(TemporalMemoryError):
    pass


class _BaseTemporalMemory:
    """Analyze the sliding window on every new frame and publish its events.

    The window holds at most eight frames, but analysis runs as soon as Task
    Memory delivers one observation rather than waiting for a full window.
    """

    def __init__(
        self,
        *,
        captioner: TemporalCaptionerPort,
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
        if image is None:
            raise TemporalStateError("image must not be None")
        frame = MemoryFrame(
            frame_id=self._next_frame_id,
            image=copy_image(image),
            subgoal_id=self._subgoal.subgoal_id,
        )
        self._frames.append(frame)
        self._next_frame_id += 1
        return frame

    def update_from_task_memory(
        self,
        *,
        analyze: bool = True,
    ) -> Optional[CaptionResult]:
        """Consume the latest Task Memory RGB and optionally analyze it.

        Deferring inference still appends the frame, including its measured
        motion, so the next analysis receives the same bounded temporal
        evidence. This is used while following a stable, still-distant
        structural waypoint where the completion guard cannot yet pass.
        """
        self._sync_task_state()
        # An exhausted plan leaves no subgoal to judge against. Appending would
        # raise, so report "nothing analyzed" and let the caller end the task.
        if self._subgoal is None:
            return None
        if self.append_latest_observation() is None:
            return None
        if not analyze:
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
        from .event_publisher import store

        return store(self, result)

    def _publish(self, event: TemporalEvent) -> None:
        self._events.append(event)
        self.task_memory.publish_temporal_event(event)


from .completion_judge import CompletionMemoryMixin


class TemporalMemory(CompletionMemoryMixin, _BaseTemporalMemory):
    """Growing temporal evidence, completion judgement, and event publishing."""


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
