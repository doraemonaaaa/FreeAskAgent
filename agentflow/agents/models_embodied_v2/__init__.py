"""VLN temporal understanding, memory, and shared data models."""

from __future__ import annotations

from importlib import import_module

from .data_models import (
    CameraIntrinsics,
    CaptionResult,
    ErrorMode,
    MemoryFrame,
    NavigationDecision,
    NavigationPoint,
    Subgoal,
    TaskInput,
    TemporalAnalysisRequest,
    TemporalCaptionerConfig,
    TemporalEvent,
    TemporalEventKind,
    TemporalFrameInput,
    TemporalInputError,
    TemporalMemoryConfig,
)

__all__ = (
    "CaptionResult",
    "CameraIntrinsics",
    "ErrorMode",
    "MemoryFrame",
    "NavigationDecision",
    "NavigationPoint",
    "Subgoal",
    "TaskInput",
    "TaskMemory",
    "TaskMemoryPort",
    "TemporalAnalysisRequest",
    "TemporalCaptioner",
    "TemporalCaptionerConfig",
    "TemporalEvent",
    "TemporalEventKind",
    "TemporalMemory",
    "TemporalMemoryConfig",
    "TemporalFrameInput",
    "TemporalInputError",
)

_LAZY_EXPORTS = {
    "TemporalCaptioner": (".TemporalCaptioner", "TemporalCaptioner"),
    "TaskMemory": (".memory", "TaskMemory"),
    "TaskMemoryPort": (".memory", "TaskMemoryPort"),
    "TemporalMemory": (".memory", "TemporalMemory"),
}


def __getattr__(name: str):
    """Load implementation modules only when their public classes are used."""
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
