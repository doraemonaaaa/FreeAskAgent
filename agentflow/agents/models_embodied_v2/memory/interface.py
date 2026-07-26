"""Composable, image-only interfaces for VLN memory ablations.

The typed :class:`TemporalMemory` core deliberately works with explicit
``TemporalObservation`` and ``StepExecution`` values.  This module owns the
session lifecycle needed by an agent or a standalone ablation:

``observe(O_t) -> stage_action(A_t) -> observe(O_t+1)``.

It also composes Task and Temporal memory without moving any memory
construction into Habitat.  Habitat callers only provide an instruction at
reset and RGB observations thereafter.
"""

from __future__ import annotations

import copy
import time
from collections import deque
from collections.abc import Callable, Mapping
from typing import Any, Literal, Optional

from .task_memory import TaskMemory
from .temporal_memory import (
    StepExecution,
    TemporalMemory,
    TemporalObservation,
)


MemoryMode = Literal["none", "task", "temporal", "task+temporal"]
MEMORY_MODES: tuple[MemoryMode, ...] = (
    "none",
    "task",
    "temporal",
    "task+temporal",
)


class _OperationTimings:
    """Bounded aggregate timings; no per-frame history is retained."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._values: dict[str, dict[str, Optional[float] | int]] = {}

    def record(
        self,
        operation: str,
        duration_ms: float,
        *,
        success: bool,
    ) -> None:
        duration = max(0.0, float(duration_ms))
        item = self._values.setdefault(
            operation,
            {
                "call_count": 0,
                "success_count": 0,
                "failure_count": 0,
                "total_ms": 0.0,
                "last_ms": None,
                "min_ms": None,
                "max_ms": None,
            },
        )
        item["call_count"] = int(item["call_count"]) + 1
        key = "success_count" if success else "failure_count"
        item[key] = int(item[key]) + 1
        item["total_ms"] = float(item["total_ms"]) + duration
        item["last_ms"] = duration
        minimum = item["min_ms"]
        maximum = item["max_ms"]
        item["min_ms"] = (
            duration if minimum is None else min(float(minimum), duration)
        )
        item["max_ms"] = (
            duration if maximum is None else max(float(maximum), duration)
        )

    def summary(self) -> dict[str, dict[str, Optional[float] | int]]:
        result: dict[str, dict[str, Optional[float] | int]] = {}
        for operation, raw in self._values.items():
            item = dict(raw)
            calls = int(item["call_count"])
            total = float(item["total_ms"])
            item["average_ms"] = total / calls if calls else None
            result[operation] = item
        return result


def _timed_call(
    timings: _OperationTimings,
    operation: str,
    callback: Callable[[], Any],
) -> Any:
    started = time.perf_counter()
    try:
        result = callback()
    except Exception:
        timings.record(
            operation,
            (time.perf_counter() - started) * 1000,
            success=False,
        )
        raise
    timings.record(
        operation,
        (time.perf_counter() - started) * 1000,
        success=True,
    )
    return result


def _snapshot_image(image: Any) -> Any:
    """Detach a small observation from simulator-owned mutable storage."""
    if isinstance(image, bytes):
        return image
    copier = getattr(image, "copy", None)
    if callable(copier):
        try:
            return copier()
        except TypeError:
            pass
    cloner = getattr(image, "clone", None)
    if callable(cloner):
        return cloner()
    return copy.deepcopy(image)


class TaskMemoryInterface:
    """Uniform lifecycle and timing wrapper around :class:`TaskMemory`."""

    module_name = "task_memory"

    def __init__(
        self,
        memory: TaskMemory,
        *,
        action_history_size: int = 8,
    ) -> None:
        self.memory = memory
        self._action_history_size = int(action_history_size)
        if self._action_history_size < 1:
            raise ValueError("action_history_size must be positive")
        self._timings = _OperationTimings()
        self._actions: deque[str] = deque(maxlen=self._action_history_size)

    def reset(self, *, episode_id: str, goal: str) -> None:
        del episode_id
        self.memory.reset(goal=goal)
        self._actions.clear()
        self._timings.reset()

    def record_input(self, observation: Any) -> None:
        _timed_call(
            self._timings,
            "record_input",
            lambda: self.memory.record_input(observation),
        )

    def update_subgoal_status(self, status: str) -> None:
        _timed_call(
            self._timings,
            "update_subgoal_status",
            lambda: self.memory.update_subgoal_status(status),
        )

    def update_temporal_status(self, status: str) -> None:
        _timed_call(
            self._timings,
            "update_temporal_status",
            lambda: self.memory.update_temporal_status(status),
        )

    def stage_action(self, action: str, subgoal_snapshot: str = "") -> None:
        del subgoal_snapshot

        def update() -> None:
            normalized = str(action).strip().upper()
            self._actions.append(normalized)
            self.memory.record_event("Actor action", normalized)

        _timed_call(self._timings, "stage_action", update)

    def record_event(self, event: str, content: str) -> None:
        _timed_call(
            self._timings,
            "record_event",
            lambda: self.memory.record_event(event, content),
        )

    def recent_actions(self) -> tuple[str, ...]:
        return tuple(self._actions)

    def context(self) -> str:
        return _timed_call(
            self._timings,
            "context",
            self.memory.context,
        )

    def timing_summary(self) -> dict[str, Any]:
        operations = self._timings.summary()
        inference = operations.get("record_input", {})
        return {
            "inference_count": int(inference.get("call_count", 0)),
            "success_count": int(inference.get("success_count", 0)),
            "failure_count": int(inference.get("failure_count", 0)),
            "total_inference_ms": float(inference.get("total_ms", 0.0)),
            "average_inference_ms": inference.get("average_ms"),
            "last_inference_ms": inference.get("last_ms"),
            "operations": operations,
        }

    def diagnostics(self) -> dict[str, Any]:
        return {
            **self.memory.diagnostics(),
            "timing": self.timing_summary(),
        }


class TemporalMemoryInterface:
    """Standalone image/action interface around the typed Temporal core."""

    module_name = "temporal_memory"

    def __init__(
        self,
        memory: TemporalMemory,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.memory = memory
        self._clock = clock
        self._timings = _OperationTimings()
        self._current_observation: Optional[TemporalObservation] = None
        self._episode_started_at = self._clock()
        self._last_timestamp_seconds: Optional[float] = None

    @property
    def latest_record(self) -> Any:
        return self.memory.latest_record

    @property
    def cumulative_error_state(self) -> Any:
        return self.memory.cumulative_error_state

    @property
    def pending_go_back_request(self) -> Any:
        return self.memory.pending_go_back_request

    @property
    def active_go_back_request(self) -> Any:
        return self.memory.active_go_back_request

    def reset(self, *, episode_id: str, goal: str) -> None:
        self.memory.reset(episode_id=episode_id, goal=goal)
        self._episode_started_at = self._clock()
        self._last_timestamp_seconds = None
        self._current_observation = None
        self._timings.reset()

    def observe(
        self,
        image: Any,
        *,
        subgoal_snapshot: str = "",
        metadata: Optional[Mapping[str, Any]] = None,
        previous_execution: Optional[
            Mapping[str, Any] | StepExecution
        ] = None,
    ) -> Any:
        """Consume one observation and close the pending action, if any."""

        def update() -> Any:
            observation = self._make_observation(image, metadata)
            record = self.memory.latest_record
            if self.memory.pending_step_id is not None:
                execution = self._make_execution(previous_execution)
                self.memory.complete_pending_step(
                    observation,
                    execution,
                    subgoal_snapshot,
                )
                record = self.memory.analyze_if_ready()
            elif previous_execution is not None:
                raise ValueError(
                    "previous_execution was supplied without a pending action"
                )
            self._current_observation = observation
            return record

        return _timed_call(
            self._timings,
            "observation_update",
            update,
        )

    # A convenient standalone name: O_t is recorded before A_t is staged.
    record_input = observe
    close_previous_action = observe

    def stage_action(
        self,
        action: str,
        subgoal_snapshot: str = "",
    ) -> None:
        def update() -> None:
            if self._current_observation is None:
                raise RuntimeError(
                    "observe(image) must be called before stage_action(action)"
                )
            self.memory.stage_action(
                self._current_observation,
                action,
                subgoal_snapshot,
            )

        _timed_call(self._timings, "stage_action", update)

    def finish_episode(
        self,
        image: Any,
        *,
        subgoal_snapshot: str = "",
        metadata: Optional[Mapping[str, Any]] = None,
        previous_execution: Optional[
            Mapping[str, Any] | StepExecution
        ] = None,
    ) -> Any:
        """Close the final pending action using the final RGB observation."""

        def finish() -> Any:
            observation = self._make_observation(image, metadata)
            self._current_observation = observation
            if self.memory.pending_step_id is None:
                return self.memory.latest_record
            execution = self._make_execution(
                previous_execution,
                terminal=True,
            )
            return self.memory.finish_episode(
                observation,
                execution,
                subgoal_snapshot,
            )

        return _timed_call(self._timings, "finish_episode", finish)

    def update_subgoal_status(self, status: str) -> None:
        self.memory.update_subgoal_status(status)

    def recent_actions(self) -> tuple[str, ...]:
        return self.memory.recent_actions()

    def begin_go_back_recovery(self) -> Any:
        return self.memory.begin_go_back_recovery()

    def next_recovery_action(self) -> Optional[str]:
        return self.memory.next_recovery_action()

    def ack_recovery_action(self, action: str) -> None:
        self.memory.ack_recovery_action(action)

    def context(self) -> str:
        return _timed_call(
            self._timings,
            "context",
            self.memory.context,
        )

    def timing_summary(self) -> dict[str, Any]:
        core = self.memory.timing_summary()
        operations = self._timings.summary()
        update = operations.get("observation_update", {})
        return {
            **core,
            "interface": {
                "inference_count": int(update.get("call_count", 0)),
                "success_count": int(update.get("success_count", 0)),
                "failure_count": int(update.get("failure_count", 0)),
                "total_inference_ms": float(update.get("total_ms", 0.0)),
                "average_inference_ms": update.get("average_ms"),
                "last_inference_ms": update.get("last_ms"),
                "operations": operations,
            },
        }

    def diagnostics(
        self,
        *,
        include_raw_response: bool = False,
    ) -> dict[str, Any]:
        diagnostics = self.memory.diagnostics(
            include_raw_response=include_raw_response,
        )
        diagnostics["interface_timing"] = self.timing_summary()["interface"]
        return diagnostics

    def _make_observation(
        self,
        image: Any,
        metadata: Optional[Mapping[str, Any]],
    ) -> TemporalObservation:
        values = dict(metadata or {})
        explicit_timestamp = "timestamp_seconds" in values
        timestamp = float(
            values.get(
                "timestamp_seconds",
                self._clock() - self._episode_started_at,
            )
        )
        if (
            not explicit_timestamp
            and self._last_timestamp_seconds is not None
            and timestamp <= self._last_timestamp_seconds
        ):
            timestamp = self._last_timestamp_seconds + 1e-6
        position = values.get("position_xyz")
        landmarks = values.get("landmark_ids")
        observation = TemporalObservation(
            # Simulators are allowed to recycle an RGB array between steps.
            # Keep an independent snapshot so the rolling short-step window
            # cannot be retroactively overwritten by a later observation.
            image=_snapshot_image(image),
            episode_id=str(
                values.get("episode_id", self.memory.episode_id)
            ),
            timestamp_seconds=timestamp,
            position_xyz=(
                tuple(float(value) for value in position)
                if position is not None
                else None
            ),
            yaw_degrees=values.get("yaw_degrees"),
            distance_to_goal_meters=values.get(
                "distance_to_goal_meters"
            ),
            landmark_ids=(
                tuple(str(value) for value in landmarks)
                if landmarks is not None
                else None
            ),
        )
        self._last_timestamp_seconds = observation.timestamp_seconds
        return observation

    def _make_execution(
        self,
        execution: Optional[Mapping[str, Any] | StepExecution],
        *,
        terminal: bool = False,
    ) -> StepExecution:
        if execution is None:
            return self.memory.infer_pending_execution(terminal=terminal)
        if isinstance(execution, StepExecution):
            if terminal and not execution.terminal:
                return StepExecution(
                    step_id=execution.step_id,
                    commanded_action=execution.commanded_action,
                    collision=execution.collision,
                    terminal=True,
                )
            return execution
        return StepExecution(
            step_id=int(execution["step_id"]),
            commanded_action=str(execution["commanded_action"]),
            collision=execution.get("collision"),
            terminal=terminal or bool(execution.get("terminal", False)),
        )


class CompositeMemory:
    """Coordinate enabled memory modules behind one planner-facing API."""

    def __init__(
        self,
        goal: str,
        *,
        episode_id: str = "episode-0",
        mode: MemoryMode = "temporal",
        task: Optional[TaskMemoryInterface] = None,
        temporal: Optional[TemporalMemoryInterface] = None,
        action_history_size: int = 3,
    ) -> None:
        if mode not in MEMORY_MODES:
            raise ValueError(
                f"Unsupported memory mode {mode!r}; expected {MEMORY_MODES}"
            )
        self.mode = mode
        self.task = task
        self.temporal = temporal
        if "task" in mode and self.task is None:
            self.task = TaskMemoryInterface(TaskMemory(goal))
        if "temporal" in mode and self.temporal is None:
            self.temporal = TemporalMemoryInterface(
                TemporalMemory(goal=goal, episode_id=episode_id)
            )
        if "task" not in mode:
            self.task = None
        if "temporal" not in mode:
            self.temporal = None
        self._actions: deque[str] = deque(maxlen=action_history_size)
        self._current_image: Any = None
        self._last_temporal_status_key: Any = None
        self.episode_id = str(episode_id)
        self.goal = str(goal)

    @property
    def task_memory(self) -> Optional[TaskMemory]:
        return self.task.memory if self.task is not None else None

    @property
    def temporal_memory(self) -> Optional[TemporalMemory]:
        return (
            self.temporal.memory if self.temporal is not None else None
        )

    @property
    def latest_record(self) -> Any:
        return (
            self.temporal.latest_record
            if self.temporal is not None
            else None
        )

    def get_module(self, name: str) -> Any:
        normalized = str(name).strip().lower()
        if normalized in {"task", "task_memory"}:
            return self.task
        if normalized in {"temporal", "temporal_memory"}:
            return self.temporal
        raise KeyError(f"Unknown memory module: {name!r}")

    def reset(self, *, episode_id: str, goal: str) -> None:
        self.episode_id = str(episode_id)
        self.goal = str(goal)
        self._actions.clear()
        self._current_image = None
        self._last_temporal_status_key = None
        if self.task is not None:
            self.task.reset(episode_id=self.episode_id, goal=self.goal)
        if self.temporal is not None:
            self.temporal.reset(
                episode_id=self.episode_id,
                goal=self.goal,
            )

    def record_input(self, image: Any) -> None:
        self._current_image = image
        if self.task is not None:
            self.task.record_input(image)

    def close_previous_action(
        self,
        subgoal_snapshot: str,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
        previous_execution: Optional[
            Mapping[str, Any] | StepExecution
        ] = None,
    ) -> Any:
        if self.temporal is None:
            return None
        if self._current_image is None:
            raise RuntimeError(
                "record_input(image) must precede close_previous_action()"
            )
        record = self.temporal.observe(
            self._current_image,
            subgoal_snapshot=subgoal_snapshot,
            metadata=metadata,
            previous_execution=previous_execution,
        )
        self._sync_temporal_status()
        return record

    def update_subgoal_status(self, status: str) -> None:
        if self.task is not None:
            self.task.update_subgoal_status(status)
        if self.temporal is not None:
            self.temporal.update_subgoal_status(status)

    def stage_action(
        self,
        action: str,
        subgoal_snapshot: str = "",
    ) -> None:
        normalized = str(action).strip().upper()
        if self.temporal is not None:
            self.temporal.stage_action(normalized, subgoal_snapshot)
        if self.task is not None:
            self.task.stage_action(normalized, subgoal_snapshot)
        self._actions.append(normalized)

    def record_event(self, event: str, content: str) -> None:
        if self.task is not None:
            self.task.record_event(event, content)

    def recent_actions(self) -> tuple[str, ...]:
        return tuple(self._actions)

    def prepare_recovery_action(self) -> Optional[str]:
        """Accept a confirmed request and peek its next legal primitive."""
        if self.temporal is None:
            return None
        if (
            self.temporal.active_go_back_request is None
            and self.temporal.pending_go_back_request is not None
        ):
            self.temporal.begin_go_back_recovery()
            self._sync_temporal_status()
        return self.temporal.next_recovery_action()

    def ack_recovery_action(self, action: str) -> None:
        """Consume a recovery primitive only after it was staged successfully."""
        if self.temporal is None:
            raise RuntimeError("Temporal memory is not enabled")
        self.temporal.ack_recovery_action(action)
        self._sync_temporal_status()

    def context(self) -> str:
        sections = []
        if self.task is not None:
            sections.append("[Task Memory]\n" + self.task.context())
        if self.temporal is not None:
            sections.append(
                "[Temporal Memory]\n" + self.temporal.context()
            )
        return "\n\n".join(sections) if sections else "Memory disabled."

    def _sync_temporal_status(self) -> None:
        """Publish the latest cumulative state through Task Memory."""
        if self.temporal is None or self.task is None:
            return
        state = self.temporal.cumulative_error_state
        mode = getattr(state, "mode", None)
        phase = getattr(state, "phase", None)
        mode_text = getattr(mode, "value", mode) or "NONE"
        phase_text = getattr(phase, "value", phase) or "NORMAL"
        score = float(getattr(state, "score", 0.0))
        reason = str(getattr(state, "reason", "") or "")
        pending = self.temporal.pending_go_back_request
        active = self.temporal.active_go_back_request
        request = active or pending
        request_id = getattr(request, "request_id", None)
        request_state = (
            "ACTIVE"
            if active is not None
            else "PENDING"
            if pending is not None
            else "NONE"
        )
        text = (
            f"cumulative_error={mode_text}, phase={phase_text}, "
            f"score={score:.3f}, go_back={request_state}"
        )
        if reason:
            text += f"\nEvidence: {reason}"
        self.task.update_temporal_status(text)

        status_key = (mode_text, phase_text, request_state, request_id)
        if status_key == self._last_temporal_status_key:
            return
        self._last_temporal_status_key = status_key
        if phase_text != "NORMAL" or request_state != "NONE":
            self.task.record_event("Temporal cumulative status", text)

    def finish_episode(
        self,
        image: Any,
        *,
        subgoal_snapshot: str = "",
        metadata: Optional[Mapping[str, Any]] = None,
        previous_execution: Optional[
            Mapping[str, Any] | StepExecution
        ] = None,
    ) -> Any:
        if image is not self._current_image:
            self.record_input(image)
        if self.temporal is None:
            return None
        record = self.temporal.finish_episode(
            image,
            subgoal_snapshot=subgoal_snapshot,
            metadata=metadata,
            previous_execution=previous_execution,
        )
        self._sync_temporal_status()
        return record

    def diagnostics(
        self,
        *,
        include_raw_response: bool = False,
    ) -> dict[str, Any]:
        modules: dict[str, Any] = {}
        timing: dict[str, Any] = {}
        if self.task is not None:
            modules["task_memory"] = self.task.diagnostics()
            timing["task_memory"] = self.task.timing_summary()
        if self.temporal is not None:
            temporal_diagnostics = self.temporal.diagnostics(
                include_raw_response=include_raw_response,
            )
            modules["temporal_memory"] = temporal_diagnostics
            temporal_timing = self.temporal.timing_summary()
            timing.update(
                {
                    "temporal_memory": temporal_timing[
                        "temporal_memory"
                    ],
                    "temporal_memory_interface": temporal_timing[
                        "interface"
                    ],
                    "video_understanding": temporal_timing[
                        "video_understanding"
                    ],
                }
            )
        return {
            "mode": self.mode,
            "episode_id": self.episode_id,
            "goal": self.goal,
            "recent_actions": list(self._actions),
            "modules": modules,
            "timing": timing,
        }


__all__ = (
    "CompositeMemory",
    "MEMORY_MODES",
    "MemoryMode",
    "TaskMemoryInterface",
    "TemporalMemoryInterface",
)
