"""Validate temporal results and publish their task events."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from ...data_models import CaptionResult, TemporalEvent, TemporalEventKind

if TYPE_CHECKING:
    from .temporal_memory import _BaseTemporalMemory


def store(memory: "_BaseTemporalMemory", result: CaptionResult) -> CaptionResult:
    """Persist one result and publish events in the established order."""
    from .temporal_memory import TemporalStateError

    assert memory._subgoal is not None
    if result.subgoal_id != memory._subgoal.subgoal_id:
        raise TemporalStateError("Captioner returned the wrong subgoal")
    if memory.config.enable_error_detection:
        if result.error != (result.error_mode != "NONE"):
            raise TemporalStateError("Captioner returned inconsistent error fields")
    else:
        result = replace(result, error=False, error_mode="NONE")
    memory._latest_result = result
    memory._last_error = None
    memory._last_analyzed_frame = memory._frames[-1].frame_id
    if memory.config.enable_error_detection:
        memory._publish(TemporalEvent(TemporalEventKind.ERROR, result.error, result.subgoal_id, result.error_mode))
    memory._publish(TemporalEvent(TemporalEventKind.SUBGOAL_COMPLETED, result.completed, result.subgoal_id, result.error_mode))
    return result
