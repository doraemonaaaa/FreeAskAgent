"""Single-actor RGB-D waypoint policy for Habitat navigation.

The actor asks a vision-language model for a walkable image location, validates
that location against the depth observation, and returns the corresponding
Habitat world-space waypoint or its own explicit stop decision. It has no
planner, memory, or discrete-action layer.

Habitat camera coordinates are ``x-right, y-up, z-back``: a point in front of
the camera therefore has negative ``z``.  ``camera_to_world`` must transform
from that camera frame into the Habitat world frame.
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from agentflow.agents.engine.factory import create_llm_engine
from agentflow.agents.models_embodied_v2.data_models import (
    CameraIntrinsics,
    NavigationDecision,
    NavigationPoint,
    Subgoal,
)
from agentflow.agents.models_embodied_v2.TemporalCaptioner import TemporalCaptioner
from agentflow.agents.models_embodied_v2.memory.task_memory import TaskMemory
from agentflow.agents.models_embodied_v2.memory.temporal_memory import TemporalMemory

DEFAULT_MODEL_PATH = "models/Qwen3-VL-8B-Instruct"

POINT_PROMPT = """You are an indoor navigation waypoint selector. Given the RGB
image and instruction, decide whether the navigation instruction is complete.
If complete, reply only with {\"stop\": true}. Otherwise select one visible,
obstacle-free floor point that makes progress. Do not select walls, furniture,
stairs, image borders, or the sky. Reply only with {\"stop\": false, \"u\":
integer, \"v\": integer}. Pixel coordinates use u=column from the left and
v=row from the top."""

SUBGOAL_PROMPT = """You are an indoor navigation task planner. Decompose the
given navigation instruction into a short, ordered sequence of subgoal
descriptions. Reply only with JSON in this exact shape:
{"subgoals": [{"subgoal_id": "1", "description": "...", "completion_criteria": "..."}]}

1. Preserve the instruction order and split it into the smallest visually
   verifiable navigation stages.
2. A commanded action is not completion evidence. Completion must be supported
   by the post-action visual sequence.
3. A turn subgoal is complete only when the agent reaches the named turn
   landmark, turns in the required direction, and the post-turn view is aligned
   with the next route.
4. The final arrival subgoal requires reaching and stopping beside the
   destination; merely seeing it at a distance is insufficient.
5. Give each subgoal a unique, ordered ID. For every subgoal, state concrete
   visual completion evidence in completion_criteria."""


class Actor:
    """Select and back-project one walkable Habitat RGB-D waypoint per call."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        *,
        engine: Optional[Any] = None,
        debug_performance: bool = False,
        use_cache: bool = False,
        min_depth_m: float = 0.25,
        max_depth_m: float = 10.0,
        patch_radius_px: int = 3,
        max_patch_depth_spread_m: float = 0.35,
        task_memory: Optional[TaskMemory] = None,
        temporal_memory: Optional[TemporalMemory] = None,
    ) -> None:
        if min_depth_m <= 0 or max_depth_m <= min_depth_m:
            raise ValueError("depth limits must satisfy 0 < min_depth_m < max_depth_m.")
        if patch_radius_px < 0 or max_patch_depth_spread_m < 0:
            raise ValueError("patch_radius_px and depth spread must be non-negative.")
        self.llm = engine or create_llm_engine(
            model_string=f"local-qwen3vl-{model_path}",
            is_multimodal=True,
            use_cache=use_cache,
            debug_performance=debug_performance,
        )
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m
        self.patch_radius_px = patch_radius_px
        self.max_patch_depth_spread_m = max_patch_depth_spread_m
        self.last_model_response: Optional[str] = None
        self.task_instruction: Optional[str] = None
        self.subgoals: list[Subgoal] = []
        self.last_subgoal_response: Optional[str] = None
        self.task_memory = task_memory
        if temporal_memory is not None and task_memory is None:
            raise ValueError("temporal_memory requires its associated task_memory.")
        self.temporal_memory = temporal_memory

    def prepare_task(self, instruction: str) -> tuple[Subgoal, ...]:
        """Initialize a task by decomposing its instruction and retaining subgoals.

        Call this once before navigation starts. Calling it again replaces the
        previously stored task and subgoals.
        """
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a non-empty string.")

        normalized_instruction = instruction.strip()
        response = self.llm(
            [f"Navigation instruction: {normalized_instruction}"],
            system_prompt=SUBGOAL_PROMPT,
            max_tokens=512,
            temperature=0,
        )
        self.last_subgoal_response = str(response)
        try:
            payload = self._extract_json_object(self.last_subgoal_response)
            entries = payload["subgoals"]
            if not isinstance(entries, list) or not entries:
                raise ValueError("subgoals must be a non-empty list")
            subgoals = [
                Subgoal(
                    subgoal_id=self._required_text(item["subgoal_id"]),
                    description=self._required_text(item["description"]),
                    completion_criteria=self._required_text(item["completion_criteria"]),
                )
                for item in entries
            ]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Actor returned invalid subgoal JSON: {response!r}") from exc

        self.task_instruction = normalized_instruction
        self.subgoals = subgoals

        self.reset_memory(normalized_instruction, subgoals)
        return tuple(self.subgoals)

    def reset_memory(self, instruction: str, subgoals: Sequence[Subgoal]) -> None:
        """Initialize Task and Temporal Memory for the prepared episode."""
        if self.task_memory is None:
            self.task_memory = TaskMemory(instruction, subgoals=subgoals)
        else:
            self.task_memory.reset(goal=instruction, subgoals=subgoals)

        if self.temporal_memory is None:
            self.temporal_memory = TemporalMemory(
                captioner=TemporalCaptioner(),
                task_memory=self.task_memory,
            )
        else:
            self.temporal_memory.reset()

    def act(
        self,
        rgb: Any,
        depth: Any,
        instruction: str,
        intrinsics: CameraIntrinsics | Any,
        camera_to_world: Any,
        *,
        normalized_depth: bool = False,
        depth_min_m: Optional[float] = None,
        depth_max_m: Optional[float] = None,
    ) -> NavigationDecision:
        """Return the actor's stop decision or a world-space RGB-D waypoint.

        Habitat's default depth sensor emits meters.  Set ``normalized_depth``
        only when its configuration uses normalized [0, 1] observations, and
        provide that sensor's ``depth_min_m`` and ``depth_max_m`` bounds.
        """
        image = self._as_rgb_array(rgb)
        if self.task_memory is not None:
            self.task_memory.record_input(image)
            if self.temporal_memory is not None:
                self.temporal_memory.update_from_task_memory()
        depth_m = self._depth_in_meters(
            depth,
            image.shape[:2],
            normalized=normalized_depth,
            depth_min_m=depth_min_m,
            depth_max_m=depth_max_m,
        )
        calibration = (
            intrinsics
            if isinstance(intrinsics, CameraIntrinsics)
            else CameraIntrinsics.from_matrix(intrinsics)
        )
        transform = np.asarray(camera_to_world, dtype=np.float64)
        if transform.shape != (4, 4):
            raise ValueError("camera_to_world must be a 4x4 Habitat camera-to-world matrix.")

        requested_uv = self._select_pixel(image, instruction)
        if requested_uv is None:
            return NavigationDecision(
                stop=True,
                raw_response=self.last_model_response,
            )
        u, v = self._nearest_walkable_pixel(depth_m, requested_uv)
        point_camera = self._back_project(u, v, float(depth_m[v, u]), calibration)
        point_world = transform @ np.array((*point_camera, 1.0), dtype=np.float64)
        if not np.isclose(point_world[3], 1.0) and point_world[3] != 0.0:
            point_world /= point_world[3]
        return NavigationDecision(
            stop=False,
            point=NavigationPoint(
                pixel_uv=(u, v),
                depth_m=float(depth_m[v, u]),
                camera_xyz=tuple(float(value) for value in point_camera),
                world_xyz=tuple(float(value) for value in point_world[:3]),
            ),
            raw_response=self.last_model_response,
        )

    def _select_pixel(
        self, image: np.ndarray, instruction: str
    ) -> Optional[tuple[int, int]]:
        height, width = image.shape[:2]
        prompt = (
            f"Navigation instruction: {instruction}\n"
            f"Image width: {width}; image height: {height}."
        )
        response = self.llm(
            [prompt, self._rgb_to_png(image)],
            system_prompt=POINT_PROMPT,
            max_tokens=32,
            temperature=0,
        )
        self.last_model_response = str(response)
        match = re.search(r"\{.*?\}", self.last_model_response, flags=re.DOTALL)
        if match is None:
            raise ValueError(f"Actor did not return waypoint JSON: {response!r}")
        try:
            point = json.loads(match.group(0))
            if point.get("stop") is True:
                return None
            if point.get("stop") is not False:
                raise ValueError("stop must be a boolean")
            u, v = int(point["u"]), int(point["v"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Actor returned invalid waypoint JSON: {response!r}") from exc
        return u, v

    @staticmethod
    def _extract_json_object(response: str) -> dict[str, Any]:
        """Extract the first JSON object, allowing a surrounding Markdown fence."""
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", response):
            try:
                value, _ = decoder.raw_decode(response[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        raise ValueError("response contains no JSON object")

    @staticmethod
    def _required_text(value: Any) -> str:
        text = value
        if not isinstance(text, str) or not text.strip():
            raise ValueError("each subgoal must be a non-empty string")
        return text.strip()

    def _nearest_walkable_pixel(
        self, depth_m: np.ndarray, requested_uv: tuple[int, int]
    ) -> tuple[int, int]:
        """Prefer the model point, then search nearby valid floor-image pixels."""
        height, width = depth_m.shape
        requested_u, requested_v = requested_uv
        # Candidate rows exclude the upper image, where ceilings and far walls
        # are common; the lower region is where navigable floor projects.
        min_v = int(height * 0.45)
        requested_u = int(np.clip(requested_u, 0, width - 1))
        requested_v = int(np.clip(requested_v, min_v, height - 1))
        candidates: list[tuple[float, int, int]] = []
        for radius in range(0, max(height, width), max(1, self.patch_radius_px + 1)):
            if candidates:
                break
            u0, u1 = max(0, requested_u - radius), min(width, requested_u + radius + 1)
            v0, v1 = max(min_v, requested_v - radius), min(height, requested_v + radius + 1)
            for v in range(v0, v1):
                for u in range(u0, u1):
                    if self._is_walkable_depth(depth_m, u, v):
                        distance = float((u - requested_u) ** 2 + (v - requested_v) ** 2)
                        candidates.append((distance, u, v))
        if not candidates:
            raise ValueError("Depth observation contains no valid walkable waypoint.")
        _, u, v = min(candidates)
        return u, v

    def _is_walkable_depth(self, depth_m: np.ndarray, u: int, v: int) -> bool:
        value = float(depth_m[v, u])
        if not self.min_depth_m <= value <= self.max_depth_m:
            return False
        radius = self.patch_radius_px
        patch = depth_m[max(0, v - radius):v + radius + 1, max(0, u - radius):u + radius + 1]
        valid = patch[np.isfinite(patch) & (patch >= self.min_depth_m) & (patch <= self.max_depth_m)]
        return valid.size > 0 and float(np.max(valid) - np.min(valid)) <= self.max_patch_depth_spread_m

    @staticmethod
    def _back_project(u: int, v: int, depth_m: float, intrinsics: CameraIntrinsics) -> np.ndarray:
        if intrinsics.fx <= 0 or intrinsics.fy <= 0:
            raise ValueError("Camera focal lengths must be positive.")
        # Habitat camera frame: +x right, +y up, and the camera looks along -z.
        return np.array(
            (
                (u - intrinsics.cx) * depth_m / intrinsics.fx,
                -(v - intrinsics.cy) * depth_m / intrinsics.fy,
                -depth_m,
            ),
            dtype=np.float64,
        )

    @staticmethod
    def _depth_in_meters(
        depth: Any,
        image_shape: Sequence[int],
        *,
        normalized: bool,
        depth_min_m: Optional[float],
        depth_max_m: Optional[float],
    ) -> np.ndarray:
        values = np.asarray(depth, dtype=np.float64)
        if values.ndim == 3 and values.shape[-1] == 1:
            values = values[..., 0]
        if values.ndim != 2 or tuple(values.shape) != tuple(image_shape):
            raise ValueError("depth must be HxW (or HxWx1) and match the RGB image.")
        if normalized:
            if depth_min_m is None or depth_max_m is None or depth_max_m <= depth_min_m:
                raise ValueError("normalized depth requires valid depth_min_m and depth_max_m.")
            values = depth_min_m + values * (depth_max_m - depth_min_m)
        return values

    @staticmethod
    def _as_rgb_array(rgb: Any) -> np.ndarray:
        if isinstance(rgb, (str, Path)):
            from PIL import Image

            rgb = np.asarray(Image.open(rgb).convert("RGB"))
        values = np.asarray(rgb)
        if values.ndim != 3 or values.shape[2] != 3:
            raise ValueError("rgb must be a Habitat HxWx3 RGB observation.")
        if values.dtype != np.uint8:
            values = np.clip(values, 0, 255).astype(np.uint8)
        return values

    @staticmethod
    def _rgb_to_png(rgb: np.ndarray) -> bytes:
        from PIL import Image

        buffer = io.BytesIO()
        Image.fromarray(rgb, mode="RGB").save(buffer, format="PNG")
        return buffer.getvalue()


VLNAgent = Actor

__all__ = [
    "Actor",
    "CameraIntrinsics",
    "NavigationDecision",
    "NavigationPoint",
    "Subgoal",
    "VLNAgent",
]
