"""Compact video-understanding adapter for VLN Temporal Memory.

The caller owns action/frame alignment.  This module only asks a multimodal
model to explain up to eight post-action frames, judge subgoal completion, and
identify persistent navigation errors.
"""

from __future__ import annotations

import io
import json
import math
import time
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Callable, Literal, Mapping, Optional, Protocol

from pydantic import BaseModel, ConfigDict, Field


DEFAULT_MODEL_PATH = "models/Qwen3-VL-8B-Instruct"
ACTION_TOKENS = ("FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP")
ErrorMode = Literal[
    "NONE",
    "WALL_STUCK",
    "TURN_OSCILLATION",
    "IN_PLACE_SPIN",
    "GET_NOWHERE",
]
VisualChange = Literal[
    "FORWARD",
    "TURN_LEFT",
    "TURN_RIGHT",
    "STATIONARY",
    "UNCERTAIN",
]


class TemporalCaptionerError(RuntimeError):
    """Base error raised by this adapter."""


class TemporalInputError(TemporalCaptionerError, ValueError):
    """The request is internally inconsistent."""


class TemporalInferenceError(TemporalCaptionerError):
    """The foundation-model call failed."""


class TemporalOutputError(TemporalCaptionerError, ValueError):
    """The model response is not valid temporal evidence."""


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


def _text(value: Any, label: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise TemporalInputError(f"{label} must not be empty")
    return result


def _normalize_action(action: Optional[str]) -> Optional[str]:
    if action is None:
        return None
    normalized = str(action).strip().upper()
    normalized = {
        "MOVE_FORWARD": "FORWARD",
        "LEFT": "TURN_LEFT",
        "RIGHT": "TURN_RIGHT",
    }.get(normalized, normalized)
    if normalized not in ACTION_TOKENS:
        raise TemporalInputError(
            f"Unsupported action {action!r}; expected one of {ACTION_TOKENS}"
        )
    return normalized


@dataclass(frozen=True, slots=True)
class Subgoal:
    """One visually verifiable stage of the navigation task."""

    subgoal_id: str
    description: str
    completion_criteria: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "subgoal_id", _text(self.subgoal_id, "subgoal_id")
        )
        object.__setattr__(
            self, "description", _text(self.description, "subgoal description")
        )
        object.__setattr__(
            self,
            "completion_criteria",
            _text(self.completion_criteria, "subgoal completion criteria"),
        )


@dataclass(frozen=True, slots=True)
class TemporalStepInput:
    """An executed action and the RGB observation captured after it."""

    step_id: int
    image: Any = field(repr=False)
    action: Optional[str] = None
    timestamp_seconds: Optional[float] = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.step_id, bool)
            or not isinstance(self.step_id, Integral)
            or self.step_id < 0
        ):
            raise TemporalInputError("step_id must be a non-negative integer")
        if self.image is None:
            raise TemporalInputError("post-action image must not be None")
        object.__setattr__(self, "step_id", int(self.step_id))
        object.__setattr__(self, "action", _normalize_action(self.action))
        if self.timestamp_seconds is not None:
            timestamp = float(self.timestamp_seconds)
            if not math.isfinite(timestamp) or timestamp < 0:
                raise TemporalInputError(
                    "timestamp_seconds must be finite and non-negative"
                )
            object.__setattr__(self, "timestamp_seconds", timestamp)


@dataclass(frozen=True, slots=True)
class TemporalAnalysisRequest:
    """Everything the model may use to judge one temporal window."""

    task: str
    task_guidance: str
    subgoals: tuple[Subgoal, ...]
    steps: tuple[TemporalStepInput, ...]
    episode_id: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "task", _text(self.task, "task"))
        object.__setattr__(
            self,
            "task_guidance",
            _text(self.task_guidance, "task_guidance"),
        )
        subgoals = tuple(self.subgoals)
        steps = tuple(self.steps)
        object.__setattr__(self, "subgoals", subgoals)
        object.__setattr__(self, "steps", steps)
        if not subgoals:
            raise TemporalInputError("at least one subgoal is required")
        if any(not isinstance(item, Subgoal) for item in subgoals):
            raise TemporalInputError("subgoals must contain only Subgoal values")
        subgoal_ids = [item.subgoal_id for item in subgoals]
        if len(subgoal_ids) != len(set(subgoal_ids)):
            raise TemporalInputError("subgoal IDs must be unique")
        if not 1 <= len(steps) <= 8:
            raise TemporalInputError("request requires between 1 and 8 steps")
        if any(not isinstance(item, TemporalStepInput) for item in steps):
            raise TemporalInputError(
                "steps must contain only TemporalStepInput values"
            )
        step_ids = [item.step_id for item in steps]
        if step_ids != sorted(set(step_ids)):
            raise TemporalInputError(
                "step IDs must be unique and strictly increasing"
            )
        timestamps = [
            item.timestamp_seconds
            for item in steps
            if item.timestamp_seconds is not None
        ]
        if timestamps != sorted(set(timestamps)):
            raise TemporalInputError(
                "known timestamps must be unique and strictly increasing"
            )


class StepUnderstanding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: int = Field(ge=0)
    caption: str = Field(min_length=1)
    visual_change: VisualChange
    error_clue: Optional[str] = None


class SubgoalStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subgoal_id: str = Field(min_length=1)
    completed: bool
    evidence: str = Field(min_length=1)
    evidence_step_ids: list[int] = Field(default_factory=list)


class SubgoalCompletionResult(BaseModel):
    """One locally identified subgoal plus the model's binary verdict."""

    model_config = ConfigDict(extra="forbid")

    subgoal_id: str = Field(min_length=1)
    completed: bool
    raw_response: str
    latency_ms: float = Field(ge=0.0)


class _ModelPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: list[StepUnderstanding]
    subgoals: list[SubgoalStatus]
    persistent_error: bool
    error_mode: ErrorMode
    error_evidence: str
    error_evidence_step_ids: list[int] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class CaptionResult(_ModelPayload):
    """Validated model result stored by Temporal Memory."""

    raw_response: str
    latency_ms: float = Field(ge=0.0)

    def status_for(self, subgoal_id: str) -> SubgoalStatus:
        wanted = str(subgoal_id)
        for status in self.subgoals:
            if status.subgoal_id == wanted:
                return status
        raise KeyError(f"No result for subgoal {wanted!r}")

    def to_memory_text(self) -> str:
        statuses = ", ".join(
            f"{item.subgoal_id}={'complete' if item.completed else 'incomplete'}"
            for item in self.subgoals
        )
        error = self.error_mode if self.persistent_error else "NONE"
        return f"Subgoals: {statuses}; persistent_error={error}"


@dataclass(frozen=True, slots=True)
class TemporalCaptionerConfig:
    max_tokens: int = 384
    max_image_edge: int = 448
    temperature: float = 0.0
    include_json_schema: bool = False
    latency_budget_ms: float = 5000.0

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if self.max_image_edge < 32:
            raise ValueError("max_image_edge must be at least 32")
        if self.latency_budget_ms <= 0:
            raise ValueError("latency_budget_ms must be positive")


SYSTEM_PROMPT = """You analyze a short first-person visual-navigation storyboard.
Every STEP marker is immediately followed by that action's post-action image.
Use only visible evidence. A commanded action is intent, not proof it succeeded.
Judge each supplied subgoal independently. A completed final-arrival subgoal is
not an error merely because its final frames are stationary.
The task describes a desired future, not what the camera actually sees. Never
copy a landmark, room, or motion from the task unless it is visible. Describe a
close wall as a close wall. For UNKNOWN actions, infer visual change only by
comparing adjacent images; the first image's change is UNCERTAIN.

Persistent error modes:
- WALL_STUCK: repeated attempts leave the camera at the same nearby obstacle.
- TURN_OSCILLATION: left/right corrections repeatedly retrace the same views.
- IN_PLACE_SPIN: repeated turning cycles through earlier views without progress.
- GET_NOWHERE: the window shows no meaningful scene or subgoal progress.

Report a persistent error only when it spans multiple steps. Return JSON only."""

COMPLETION_SYSTEM_PROMPT = (
    "Use only the ordered images; the subgoal text is not evidence. Reply "
    "true only if they visibly prove completion by the final image, otherwise "
    "false. The transition need not occur inside this window when the final "
    "visible state itself proves completion. Reply only true or false."
)


class TemporalCaptioner:
    """Run one compact, synchronous temporal-understanding inference."""

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

    @property
    def last_raw_response(self) -> Optional[str]:
        return self._last_raw_response

    def reset_performance_stats(self) -> None:
        self._inference_count = 0
        self._failure_count = 0
        self._total_ms = 0.0
        self._last_ms: Optional[float] = None
        self._last_raw_response: Optional[str] = None

    def performance_summary(self) -> dict[str, Any]:
        count = self._inference_count
        return {
            "inference_count": count,
            "success_count": count - self._failure_count,
            "failure_count": self._failure_count,
            "average_inference_ms": self._total_ms / count if count else None,
            "last_inference_ms": self._last_ms,
            "latency_budget_ms": self.config.latency_budget_ms,
        }

    def analyze(self, request: TemporalAnalysisRequest) -> CaptionResult:
        if not isinstance(request, TemporalAnalysisRequest):
            raise TemporalInputError("analyze expects TemporalAnalysisRequest")
        content = self._build_content(request)
        kwargs: dict[str, Any] = {
            "system_prompt": SYSTEM_PROMPT,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.include_json_schema:
            kwargs["response_format"] = _ModelPayload
        engine = self._get_engine()
        if getattr(engine, "supports_image_pixel_budget", False):
            kwargs.update(
                image_min_pixels=min(64**2, self.config.max_image_edge**2),
                image_max_pixels=self.config.max_image_edge**2,
            )

        started = time.perf_counter()
        self._last_raw_response = None
        try:
            response = engine(content, **kwargs)
            elapsed_ms = (time.perf_counter() - started) * 1000
            self._last_raw_response = str(response)
            payload = self._parse_response(response)
            self._validate_payload(request, payload)
        except TemporalOutputError:
            self._record_timing((time.perf_counter() - started) * 1000, False)
            raise
        except Exception as exc:
            self._record_timing((time.perf_counter() - started) * 1000, False)
            raise TemporalInferenceError("Temporal inference failed") from exc
        self._record_timing(elapsed_ms, True)
        return CaptionResult(
            **payload.model_dump(),
            raw_response=self._last_raw_response or "",
            latency_ms=elapsed_ms,
        )

    def evaluate_subgoal(
        self,
        request: TemporalAnalysisRequest,
    ) -> SubgoalCompletionResult:
        """Judge completion only when a video has frames but no action labels."""
        if not isinstance(request, TemporalAnalysisRequest):
            raise TemporalInputError(
                "evaluate_subgoal expects TemporalAnalysisRequest"
            )
        if len(request.subgoals) != 1:
            raise TemporalInputError(
                "completion-only evaluation requires exactly one subgoal"
            )
        if any(step.action is not None for step in request.steps):
            raise TemporalInputError(
                "completion-only evaluation requires action=None for every frame"
            )
        if len(request.steps) != 8:
            raise TemporalInputError(
                "completion-only evaluation requires exactly eight frames"
            )
        engine = self._get_engine()
        started = time.perf_counter()
        content = self._build_completion_content(request)
        kwargs: dict[str, Any] = {
            "system_prompt": COMPLETION_SYSTEM_PROMPT,
            "temperature": self.config.temperature,
            "max_tokens": 1,
        }
        if getattr(engine, "supports_image_pixel_budget", False):
            kwargs.update(
                image_min_pixels=min(64**2, self.config.max_image_edge**2),
                image_max_pixels=self.config.max_image_edge**2,
            )

        self._last_raw_response = None
        try:
            response = engine(content, **kwargs)
            self._last_raw_response = str(response)
            completed = self._parse_completion_response(response)
            elapsed_ms = (time.perf_counter() - started) * 1000
        except TemporalOutputError:
            self._record_timing((time.perf_counter() - started) * 1000, False)
            raise
        except Exception as exc:
            self._record_timing((time.perf_counter() - started) * 1000, False)
            raise TemporalInferenceError(
                "Subgoal completion inference failed"
            ) from exc
        self._record_timing(elapsed_ms, True)
        return SubgoalCompletionResult(
            subgoal_id=request.subgoals[0].subgoal_id,
            completed=completed,
            raw_response=self._last_raw_response or "",
            latency_ms=elapsed_ms,
        )

    def _build_content(self, request: TemporalAnalysisRequest) -> list[Any]:
        subgoals = "\n".join(
            f"- {item.subgoal_id}: {item.description}\n"
            f"  Completion criteria: {item.completion_criteria}"
            for item in request.subgoals
        )
        content: list[Any] = [
            f"[TASK]\n{request.task}\n\n"
            f"[TASK GUIDANCE]\n{request.task_guidance}\n\n"
            f"[SUBGOALS]\n{subgoals}\n\n"
            f"[BEGIN {len(request.steps)} STEP STORYBOARD]\n"
        ]
        for step in request.steps:
            action = step.action or "UNKNOWN"
            timestamp = (
                "unknown"
                if step.timestamp_seconds is None
                else f"{step.timestamp_seconds:.3f}s"
            )
            content.extend(
                (
                    f"\n[STEP step_id={step.step_id} action={action} "
                    f"post_t={timestamp}]\n",
                    self._image_to_png_bytes(step.image),
                )
            )
        ids = [item.step_id for item in request.steps]
        subgoal_ids = [item.subgoal_id for item in request.subgoals]
        content.append(
            f"\n[END STORYBOARD]\n"
            f"最近走的{len(ids)}步发生了什么？"
            "按照每步 action 后的画面进行描述。\n"
            f"steps 必须按顺序包含且只包含 step_id={ids}；"
            f"subgoals 必须包含且只包含 subgoal_id={subgoal_ids}。"
            "UNKNOWN action 表示没有动作真值，不得虚构命令。"
            "完成必须由画面证据支持；仅看见远处目标不算到达。"
            "持续错误必须引用至少三个步骤；成功到达后的静止不算错误。"
            "每步 caption 不超过20个汉字；visual_change 只能是 "
            "FORWARD、TURN_LEFT、TURN_RIGHT、STATIONARY 或 UNCERTAIN；"
            "error_clue 没有则为 null。subgoal evidence 和 error_evidence "
            "各不超过30个汉字。\n"
            "只输出以下结构的 JSON：\n"
            '{"steps":[{"step_id":1,"caption":"动作后场景",'
            '"visual_change":"FORWARD","error_clue":null}],'
            '"subgoals":[{"subgoal_id":"1","completed":false,'
            '"evidence":"可见证据","evidence_step_ids":[1]}],'
            '"persistent_error":false,"error_mode":"NONE",'
            '"error_evidence":"无持续错误",'
            '"error_evidence_step_ids":[],"confidence":0.8}'
        )
        return content

    def _build_completion_content(
        self,
        request: TemporalAnalysisRequest,
    ) -> list[Any]:
        subgoal = request.subgoals[0]
        content: list[Any] = [
            f"Subgoal: {subgoal.description}\n"
            f"Completion proof: {subgoal.completion_criteria}"
        ]
        content.extend(
            self._image_to_png_bytes(step.image)
            for step in request.steps
        )
        return content

    @staticmethod
    def _parse_response(response: Any) -> _ModelPayload:
        if isinstance(response, _ModelPayload):
            return response
        value = TemporalCaptioner._json_value(response)
        try:
            return _ModelPayload.model_validate(value)
        except Exception as exc:
            raise TemporalOutputError(
                "model output does not match CaptionResult schema"
            ) from exc

    @staticmethod
    def _parse_completion_response(response: Any) -> bool:
        if isinstance(response, bool):
            return response
        verdict = str(response or "").strip().lower()
        if verdict == "true":
            return True
        if verdict == "false":
            return False
        raise TemporalOutputError(
            "completion model must return exactly true or false"
        )

    @staticmethod
    def _json_value(response: Any) -> Any:
        if isinstance(response, Mapping):
            return response
        text = str(response or "").strip()
        if not text:
            raise TemporalOutputError("model returned an empty response")
        if text.startswith("```") and text.endswith("```"):
            text = "\n".join(text.splitlines()[1:-1]).strip()
        start = text.find("{")
        if start < 0:
            raise TemporalOutputError("model did not return JSON")
        try:
            value, end = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError as exc:
            raise TemporalOutputError("model returned invalid JSON") from exc
        if text[start + end :].strip():
            raise TemporalOutputError(
                "model returned trailing content after JSON"
            )
        return value

    @staticmethod
    def _validate_payload(
        request: TemporalAnalysisRequest,
        payload: _ModelPayload,
    ) -> None:
        expected_steps = [item.step_id for item in request.steps]
        returned_steps = [item.step_id for item in payload.steps]
        if returned_steps != expected_steps:
            raise TemporalOutputError(
                "step IDs must match input one-to-one and in order; "
                f"expected {expected_steps}, got {returned_steps}"
            )
        expected_subgoals = [item.subgoal_id for item in request.subgoals]
        returned_subgoals = [item.subgoal_id for item in payload.subgoals]
        if returned_subgoals != expected_subgoals:
            raise TemporalOutputError(
                "subgoal IDs must match input one-to-one and in order; "
                f"expected {expected_subgoals}, got {returned_subgoals}"
            )
        known_steps = set(expected_steps)
        evidence_lists = [
            payload.error_evidence_step_ids,
            *(item.evidence_step_ids for item in payload.subgoals),
        ]
        if any(
            len(ids) != len(set(ids)) or not set(ids).issubset(known_steps)
            for ids in evidence_lists
        ):
            raise TemporalOutputError(
                "evidence step IDs must be unique input step IDs"
            )
        if payload.persistent_error:
            if payload.error_mode == "NONE":
                raise TemporalOutputError(
                    "persistent_error=true requires a non-NONE error_mode"
                )
            if len(payload.error_evidence_step_ids) < 3:
                raise TemporalOutputError(
                    "persistent error requires evidence from at least three steps"
                )
        elif (
            payload.error_mode != "NONE"
            or payload.error_evidence_step_ids
        ):
            raise TemporalOutputError(
                "persistent_error=false requires error_mode=NONE and no IDs"
            )

    def _get_engine(self) -> MultimodalEngine:
        if self._engine is not None:
            return self._engine
        if self._engine_factory is not None:
            self._engine = self._engine_factory()
            return self._engine
        from agentflow.agents.engine.factory import create_llm_engine

        self._engine = create_llm_engine(
            model_string=f"local-qwen3vl-{self.model_path}",
            is_multimodal=True,
            use_cache=self._use_cache,
            debug_performance=self._debug_performance,
            **self._engine_kwargs,
        )
        return self._engine

    def _image_to_png_bytes(self, image: Any) -> bytes:
        try:
            from PIL import Image
            import numpy as np

            if isinstance(image, bytes):
                pil = Image.open(io.BytesIO(image)).convert("RGB")
            elif isinstance(image, Image.Image):
                pil = image.convert("RGB")
            else:
                array = np.asarray(image)
                if array.ndim != 3 or array.shape[2] not in {3, 4}:
                    raise ValueError("expected HxWx3 or HxWx4 image")
                if np.issubdtype(array.dtype, np.floating):
                    if (
                        array.size
                        and float(np.nanmin(array)) >= 0
                        and float(np.nanmax(array)) <= 1
                    ):
                        array = array * 255
                if array.dtype != np.uint8:
                    array = np.clip(array, 0, 255).astype(np.uint8)
                pil = Image.fromarray(array).convert("RGB")
            pil.thumbnail(
                (self.config.max_image_edge, self.config.max_image_edge),
                Image.Resampling.LANCZOS,
            )
            buffer = io.BytesIO()
            pil.save(buffer, format="PNG")
            return buffer.getvalue()
        except Exception as exc:
            raise TemporalInputError("unsupported post-action image") from exc

    def _record_timing(self, elapsed_ms: float, success: bool) -> None:
        self._inference_count += 1
        self._failure_count += int(not success)
        self._total_ms += elapsed_ms
        self._last_ms = elapsed_ms


__all__ = (
    "ACTION_TOKENS",
    "CaptionResult",
    "ErrorMode",
    "Subgoal",
    "SubgoalCompletionResult",
    "SubgoalStatus",
    "StepUnderstanding",
    "TemporalAnalysisRequest",
    "TemporalCaptioner",
    "TemporalCaptionerConfig",
    "TemporalCaptionerError",
    "TemporalInferenceError",
    "TemporalInputError",
    "TemporalOutputError",
    "TemporalStepInput",
    "VisualChange",
)
