"""Compact Qwen judgement for eight consecutive VLN observations."""

from __future__ import annotations

import io
import json
import time
from typing import Any, Mapping, Optional

from pydantic import BaseModel, ConfigDict

from .data_models import (
    CaptionResult,
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


SYSTEM_PROMPT = """Inspect eight ordered first-person navigation images.
Judge whether the final image visibly proves the current subgoal is complete.
Also detect a cumulative visual error:
- WALL_STUCK: the camera stays at the same nearby obstacle.
- TURN_OSCILLATION: views repeatedly alternate left and right.
- IN_PLACE_SPIN: views rotate and return to earlier views without progress.
- GET_NOWHERE: the sequence shows no meaningful visual progress.

Set error=false and error_mode=NONE when no error is visible. Output only the
requested compact JSON."""


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
            result = self._parse(response)
            if result.error != (result.error_mode != "NONE"):
                raise TemporalOutputError("error and error_mode disagree")
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

    def _content(self, request: TemporalAnalysisRequest) -> list[Any]:
        content: list[Any] = [
            f"Subgoal: {request.subgoal.description}\n"
            f"Completion proof: {request.subgoal.completion_criteria}"
        ]
        content.extend(self._png(frame.image) for frame in request.frames)
        content.append(
            'Return exactly: {"completed":false,"error":false,'
            '"error_mode":"NONE"}'
        )
        return content

    @staticmethod
    def _parse(response: Any) -> _ModelResult:
        if isinstance(response, Mapping):
            value = response
        else:
            text = str(response or "").strip()
            if text.startswith("```") and text.endswith("```"):
                text = "\n".join(text.splitlines()[1:-1]).strip()
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise TemporalOutputError("model returned invalid JSON") from exc
        try:
            return _ModelResult.model_validate(value)
        except Exception as exc:
            raise TemporalOutputError("model returned the wrong schema") from exc

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
