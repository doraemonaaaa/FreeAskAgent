"""Compact Qwen judgement for up to eight consecutive VLN observations."""

from __future__ import annotations

import io
import json
import time
from typing import Any, Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict

from ...data_models import (
    CaptionResult,
    DualWindowCaptionResult,
    ErrorMode,
    Subgoal,
    TemporalAnalysisRequest,
    TemporalCaptionerConfig,
    TemporalFrameInput,
    TemporalInputError,
)


DEFAULT_MODEL_PATH = "models/Qwen3-VL-8B-Instruct"


class TemporalCaptionerError(RuntimeError):
    pass


class TemporalInferenceError(TemporalCaptionerError):
    pass


class TemporalOutputError(TemporalCaptionerError, ValueError):
    pass


class _ModelResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    completed: bool
    error: bool
    error_mode: ErrorMode


class _CompletionModelResult(BaseModel):
    """Strict result schema for the completion-only visual judgement."""

    model_config = ConfigDict(extra="forbid", strict=True)

    completed: bool


class _ErrorModelResult(BaseModel):
    """Strict result schema for the recent-window error judgement."""

    model_config = ConfigDict(extra="forbid", strict=True)

    error: bool
    error_mode: ErrorMode
    confidence: float
    evidence: str


SYSTEM_PROMPT = """Inspect the ordered first-person navigation images, oldest
first. There may be as few as one image early in a subgoal.
Judge whether the final image visibly proves the current subgoal is complete.
Also detect a cumulative visual error:
- WALL_STUCK: the camera stays at the same nearby obstacle.
- TURN_OSCILLATION: views repeatedly alternate left and right.
- IN_PLACE_SPIN: views rotate and return to earlier views without progress.
- GET_NOWHERE: the sequence shows no meaningful visual progress.

Set error=false and error_mode=NONE when no error is visible. A cumulative
error needs several images as evidence, so report NONE when too few are given.
Output only the requested compact JSON."""


COMPLETION_SYSTEM_PROMPT = """Inspect the selected ordered visual evidence
for one active navigation subgoal. Judge only whether the final observation,
viewed together with the ordered evidence, visibly proves that subgoal's
completion criterion. Motion and landmark fields are auxiliary evidence, not
substitutes for visual confirmation. Output only:
{"completed":false}"""


ERROR_SYSTEM_PROMPT = """Inspect the recent ordered first-person observations
and their measured motion metadata. Diagnose a cumulative execution error only
when the recent ordered evidence is clear:
- WALL_STUCK: repeated negligible translation at an obstacle;
- TURN_OSCILLATION: alternating left/right turns without progress;
- IN_PLACE_SPIN: rotation returns to earlier views without progress;
- GET_NOWHERE: no meaningful progress despite attempted navigation.

Set error=false and error_mode=NONE when there is no clear error. Confidence
must be in [0,1], and evidence must briefly ground the judgement in the recent
views and motion. Output only:
{"error":false,"error_mode":"NONE","confidence":0.0,"evidence":"..."}"""


class TemporalCaptioner:
    """Judge subgoal completion and visual error in one compact call."""

    def __init__(
        self,
        *,
        engine: Optional[Any] = None,
        model_path: str = DEFAULT_MODEL_PATH,
        config: Optional[TemporalCaptionerConfig] = None,
        use_cache: bool = False,
        debug_performance: bool = False,
        engine_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.model_path = model_path
        self.config = config or TemporalCaptionerConfig()
        self._engine = engine
        self._use_cache = use_cache
        self._debug_performance = debug_performance
        self._engine_kwargs = dict(engine_kwargs or {})
        self.last_raw_response: Optional[str] = None
        self.last_failed_raw_response: Optional[str] = None

    def analyze(self, request: TemporalAnalysisRequest) -> CaptionResult:
        if not isinstance(request, TemporalAnalysisRequest):
            raise TemporalInputError("analyze expects TemporalAnalysisRequest")
        engine = self._get_engine()
        kwargs = {
            "system_prompt": SYSTEM_PROMPT,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if getattr(engine, "supports_image_pixel_budget", False):
            kwargs.update(
                image_min_pixels=min(64**2, self.config.max_image_edge**2),
                image_max_pixels=self.config.max_image_edge**2,
            )

        started = time.perf_counter()
        try:
            response = engine(self._content(request), **kwargs)
            self.last_raw_response = str(response)
            result = self._parse_single(response)
        except TemporalOutputError:
            raise
        except Exception as exc:
            raise TemporalInferenceError("Temporal inference failed") from exc
        return CaptionResult(
            subgoal_id=request.subgoal.subgoal_id,
            completed=result.completed,
            error=result.error,
            error_mode=result.error_mode,
            raw_response=self.last_raw_response,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    def analyze_dual_window(
        self,
        *,
        subgoal: Subgoal,
        completion_frames: Sequence[TemporalFrameInput],
        error_frames: Sequence[TemporalFrameInput],
    ) -> DualWindowCaptionResult:
        """Independently judge completion and recent execution errors.

        Completion can use a compact representative history, whereas error
        diagnosis is intentionally limited to the recent suffix. This keeps
        error evidence local while retaining an anchor frame for completion.
        """
        completion = self._validate_window(
            completion_frames, label="completion"
        )
        error = self._validate_window(error_frames, label="error", max_size=8)
        error_ids = tuple(frame.frame_id for frame in error)
        completion_suffix = tuple(
            frame.frame_id for frame in completion[-len(error) :]
        )
        if error_ids != completion_suffix:
            raise TemporalInputError(
                "error frames must be a recent suffix of completion frames"
            )
        if not isinstance(subgoal, Subgoal):
            raise TemporalInputError("subgoal must be a Subgoal")

        self.last_failed_raw_response = None
        engine = self._get_engine()
        kwargs = self._inference_kwargs()
        total_started = time.perf_counter()
        completion_started = time.perf_counter()
        try:
            completion_response = engine(
                self._completion_content(subgoal, completion),
                system_prompt=COMPLETION_SYSTEM_PROMPT,
                **kwargs,
            )
            completion_raw = str(completion_response)
            completion_result = self._parse_completion(completion_response)
        except TemporalOutputError:
            self._remember_failed_response("completion", self.last_raw_response)
            raise
        except Exception as exc:
            raise TemporalInferenceError("Temporal completion inference failed") from exc
        completion_latency_ms = (time.perf_counter() - completion_started) * 1000

        error_result = None
        error_raw = "disabled"
        error_latency_ms = 0.0
        if self.config.enable_error_detection:
            if len(error) < self.config.min_error_detection_frames:
                error_raw = "insufficient_frames"
            else:
                error_started = time.perf_counter()
                error_result, error_raw = self._analyze_error_window(
                    engine=engine,
                    subgoal=subgoal,
                    frames=error,
                    kwargs=kwargs,
                )
                error_latency_ms = (time.perf_counter() - error_started) * 1000

        self.last_raw_response = error_raw if error_result is not None else completion_raw
        return DualWindowCaptionResult(
            subgoal_id=subgoal.subgoal_id,
            completed=completion_result.completed,
            error=False if error_result is None else error_result.error,
            error_mode="NONE" if error_result is None else error_result.error_mode,
            completion_window_size=len(completion),
            error_window_size=len(error),
            completion_raw_response=completion_raw,
            error_raw_response=error_raw,
            completion_latency_ms=completion_latency_ms,
            error_latency_ms=error_latency_ms,
            latency_ms=(time.perf_counter() - total_started) * 1000,
            error_confidence=(
                0.0 if error_result is None else error_result.confidence
            ),
            error_evidence=(
                "" if error_result is None else error_result.evidence.strip()
            ),
        )

    def _content(self, request: TemporalAnalysisRequest) -> list[Any]:
        content: list[Any] = [
            f"Subgoal: {request.subgoal.description}\n"
            f"Completion proof: {request.subgoal.completion_criteria}\n"
            f"Images: {len(request.frames)}, ordered oldest first."
        ]
        content.extend(self._png(frame.image) for frame in request.frames)
        content.append(
            'Return exactly: {"completed":false,"error":false,'
            '"error_mode":"NONE"}'
        )
        return content

    def _parse_single(self, response: Any) -> _ModelResult:
        value = self._json_value(response)
        if set(value) == {"completed"}:
            completion = self._validate_model(
                _CompletionModelResult,
                value,
            )
            return _ModelResult(
                completed=completion.completed,
                error=False,
                error_mode="NONE",
            )
        try:
            result = _ModelResult.model_validate(value)
        except Exception as exc:
            raise TemporalOutputError("model returned the wrong schema") from exc
        if result.error != (result.error_mode != "NONE"):
            raise TemporalOutputError("error and error_mode disagree")
        return result

    def _parse_completion(self, response: Any) -> _CompletionModelResult:
        return self._validate_model(
            _CompletionModelResult,
            self._json_value(response),
        )

    def _parse_error(self, response: Any) -> _ErrorModelResult:
        result = self._validate_model(
            _ErrorModelResult,
            self._json_value(response),
        )
        if result.error != (result.error_mode != "NONE"):
            raise TemporalOutputError("model returned the wrong schema")
        if not 0.0 <= result.confidence <= 1.0:
            raise TemporalOutputError("model returned the wrong schema")
        if not result.evidence.strip():
            raise TemporalOutputError("model returned the wrong schema")
        return result

    @staticmethod
    def _validate_model(model: type[BaseModel], value: Mapping[str, Any]) -> Any:
        try:
            return model.model_validate(value)
        except Exception as exc:
            raise TemporalOutputError("model returned the wrong schema") from exc

    @staticmethod
    def _json_value(response: Any) -> Mapping[str, Any]:
        if isinstance(response, Mapping):
            return response
        text = str(response or "").strip()
        if text.startswith("```") and text.endswith("```"):
            text = "\n".join(text.splitlines()[1:-1]).strip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end <= start:
                raise TemporalOutputError("model returned invalid JSON")
            try:
                value = json.loads(text[start : end + 1])
            except json.JSONDecodeError as exc:
                raise TemporalOutputError("model returned invalid JSON") from exc
        if not isinstance(value, Mapping):
            raise TemporalOutputError("model returned the wrong schema")
        return value

    @staticmethod
    def _validate_window(
        frames: Sequence[TemporalFrameInput],
        *,
        label: str,
        max_size: Optional[int] = None,
    ) -> tuple[TemporalFrameInput, ...]:
        values = tuple(frames)
        if not values:
            raise TemporalInputError(f"{label} frames must not be empty")
        if max_size is not None and len(values) > max_size:
            raise TemporalInputError(
                f"{label} frames allow at most {max_size} images"
            )
        if any(not isinstance(frame, TemporalFrameInput) for frame in values):
            raise TemporalInputError(
                f"{label} frames must contain TemporalFrameInput values"
            )
        ids = [frame.frame_id for frame in values]
        if ids != sorted(set(ids)):
            raise TemporalInputError(
                f"{label} frame IDs must be unique and increasing"
            )
        return values

    def _inference_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        engine = self._get_engine()
        if getattr(engine, "supports_image_pixel_budget", False):
            kwargs.update(
                image_min_pixels=min(64**2, self.config.max_image_edge**2),
                image_max_pixels=self.config.max_image_edge**2,
            )
        return kwargs

    def _completion_content(
        self,
        subgoal: Subgoal,
        frames: Sequence[TemporalFrameInput],
    ) -> list[Any]:
        content: list[Any] = [
            f"Subgoal: {subgoal.description}\n"
            f"Completion proof: {subgoal.completion_criteria}\n"
            f"Images: {len(frames)}, selected ordered visual evidence; "
            "landmark fields are auxiliary."
        ]
        for frame in frames:
            content.extend(
                (
                    "frame_id={}; translation_m={:.3f}; "
                    "yaw_delta_deg={:+.1f}; path_length_m={:.3f}; "
                    "landmark_visible={}; landmark_direction={}; "
                    "landmark_proximity={}; landmark_passed={}; "
                    "landmark_confidence={:.2f}".format(
                        frame.frame_id,
                        frame.translation_m,
                        frame.yaw_delta_deg,
                        frame.subgoal_path_length_m,
                        frame.landmark_visible,
                        frame.landmark_direction,
                        frame.landmark_proximity,
                        frame.landmark_passed,
                        frame.landmark_confidence,
                    ),
                    self._png(frame.image),
                )
            )
        return content

    def _error_content(
        self,
        subgoal: Subgoal,
        frames: Sequence[TemporalFrameInput],
    ) -> list[Any]:
        content: list[Any] = [
            f"Subgoal: {subgoal.description}\n"
            f"Completion proof: {subgoal.completion_criteria}\n"
            f"Recent ordered frames: {len(frames)}."
        ]
        for frame in frames:
            content.extend(
                (
                    "frame_id={}; translation_m={:.3f}; "
                    "yaw_delta_deg={:+.1f}; path_length_m={:.3f}; "
                    "landmark_visible={}; landmark_direction={}; "
                    "landmark_proximity={}; landmark_passed={}; "
                    "landmark_confidence={:.2f}".format(
                        frame.frame_id,
                        frame.translation_m,
                        frame.yaw_delta_deg,
                        frame.subgoal_path_length_m,
                        frame.landmark_visible,
                        frame.landmark_direction,
                        frame.landmark_proximity,
                        frame.landmark_passed,
                        frame.landmark_confidence,
                    ),
                    self._png(frame.image),
                )
            )
        return content

    def _analyze_error_window(
        self,
        *,
        engine: Any,
        subgoal: Subgoal,
        frames: Sequence[TemporalFrameInput],
        kwargs: Mapping[str, Any],
    ) -> tuple[_ErrorModelResult, str]:
        response = None
        for attempt in range(2):
            try:
                response = engine(
                    self._error_content(subgoal, frames),
                    system_prompt=ERROR_SYSTEM_PROMPT,
                    **kwargs,
                )
                raw = str(response)
                return self._parse_error(response), raw
            except TemporalOutputError as exc:
                # Retry only malformed/truncated JSON. A structurally wrong
                # response cannot be repaired by asking the identical schema
                # again and must remain visible to callers immediately.
                if "invalid JSON" not in str(exc) or attempt == 1:
                    self._remember_failed_response("error", response)
                    raise
        raise AssertionError("unreachable error-inference state")

    def _remember_failed_response(self, stage: str, response: Any) -> None:
        self.last_failed_raw_response = json.dumps(
            {"stage": stage, "response": str(response)},
            ensure_ascii=False,
        )

    def _get_engine(self) -> Any:
        if self._engine is None:
            try:
                from agentflow.agents.engine.factory import create_llm_engine

                self._engine = create_llm_engine(
                    model_string=f"local-qwen3vl-{self.model_path}",
                    is_multimodal=True,
                    use_cache=self._use_cache,
                    debug_performance=self._debug_performance,
                    **self._engine_kwargs,
                )
            except Exception as exc:
                raise TemporalInferenceError("Could not create Qwen engine") from exc
        return self._engine

    def _png(self, image: Any) -> bytes:
        try:
            import numpy as np
            from PIL import Image

            if isinstance(image, bytes):
                pil = Image.open(io.BytesIO(image)).convert("RGB")
            elif isinstance(image, Image.Image):
                pil = image.convert("RGB")
            else:
                array = np.asarray(image)
                if array.ndim != 3 or array.shape[2] not in (3, 4):
                    raise ValueError
                if np.issubdtype(array.dtype, np.floating) and array.size:
                    if float(np.nanmax(array)) <= 1:
                        array = array * 255
                array = np.clip(array, 0, 255).astype(np.uint8)
                pil = Image.fromarray(array).convert("RGB")
            pil.thumbnail(
                (self.config.max_image_edge,) * 2,
                Image.Resampling.LANCZOS,
            )
            buffer = io.BytesIO()
            pil.save(buffer, format="PNG")
            return buffer.getvalue()
        except Exception as exc:
            raise TemporalInputError("unsupported image") from exc


__all__ = (
    "CaptionResult",
    "DualWindowCaptionResult",
    "ErrorMode",
    "Subgoal",
    "TemporalAnalysisRequest",
    "TemporalCaptioner",
    "TemporalCaptionerConfig",
    "TemporalCaptionerError",
    "TemporalFrameInput",
    "TemporalInferenceError",
    "TemporalInputError",
    "TemporalOutputError",
)
