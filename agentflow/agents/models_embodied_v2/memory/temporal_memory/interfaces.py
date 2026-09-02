"""Dependency interfaces used by Temporal Memory."""

from __future__ import annotations

from typing import Any, Optional, Protocol, Sequence

from ...data_models import (
    SceneAnalysisRequest,
    SceneAnalysisResult,
    Subgoal,
    TemporalEvent,
)


class TaskMemoryPort(Protocol):
    observation_count: int

    def reset(
        self,
        *,
        goal: str,
        task_guidance: str = "",
        subgoals: Sequence[Any] = (),
    ) -> None: ...

    def get_current_subgoal(self) -> Optional[Subgoal]: ...

    def is_current_subgoal_final(self) -> bool: ...

    def get_latest_observation(self) -> Any: ...

    def get_reset_generation(self) -> int: ...

    def publish_temporal_event(self, event: TemporalEvent) -> None: ...


class TemporalCaptionerPort(Protocol):
    def analyze_scene(
        self,
        request: SceneAnalysisRequest,
    ) -> SceneAnalysisResult: ...
