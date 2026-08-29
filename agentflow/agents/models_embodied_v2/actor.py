"""Stateless RGB-D waypoint actor.

The actor is the perception and geometry layer: it validates one observation,
snaps a requested image location onto walkable floor, and back-projects it into
a Habitat world-space waypoint.  It owns no task state — no memory, no
subgoals, no navigation phase, no per-episode bookkeeping.  Deciding *which*
pixel to request, when to stop, and how the route progresses belongs to the
agent that drives it.

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
    NavigationPoint,
)
from agentflow.agents.models_embodied_v2.skiils.protocol import DEFAULT_MODEL_PATH


class Actor:
    """Validate one RGB-D observation and back-project one walkable waypoint."""

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
        camera_height_m: Optional[float] = None,
        max_floor_offset_m: float = 0.30,
    ) -> None:
        if camera_height_m is not None and camera_height_m <= 0:
            raise ValueError("camera_height_m must be positive when given.")
        if max_floor_offset_m <= 0:
            raise ValueError("max_floor_offset_m must be positive.")
        if min_depth_m <= 0 or max_depth_m <= min_depth_m:
            raise ValueError(
                "depth limits must satisfy 0 < min_depth_m < max_depth_m."
            )
        if patch_radius_px < 0 or max_patch_depth_spread_m < 0:
            raise ValueError(
                "patch_radius_px and depth spread must be non-negative."
            )
        if engine is not None:
            self.llm = engine
        elif model_path.startswith("vllm-"):
            # ``vllm-<served-model-name>`` talks to a vLLM OpenAI server
            # (``VLLM_BASE_URL``) instead of loading the checkpoint here.
            from agentflow.agents.engine.remote_qwen3vl import RemoteQwen3VL

            self.llm = RemoteQwen3VL(model_path[len("vllm-"):])
        else:
            self.llm = create_llm_engine(
                model_string=f"local-qwen3vl-{model_path}",
                is_multimodal=True,
                use_cache=use_cache,
                debug_performance=debug_performance,
            )
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m
        self.patch_radius_px = patch_radius_px
        self.max_patch_depth_spread_m = max_patch_depth_spread_m
        # Height of the camera above the floor the agent stands on. When it
        # is known, a candidate waypoint must back-project to floor level:
        # the depth-spread test alone accepts any flat surface, so a smooth
        # wall 3 m ahead passes it just as well as the floor does and the
        # resulting waypoint sits in mid-air where the agent can never arrive.
        self.camera_height_m = camera_height_m
        self.max_floor_offset_m = max_floor_offset_m
        self.last_waypoint_on_floor: Optional[bool] = None

    def waypoint_from_pixel(
        self,
        requested_uv: tuple[int, int],
        depth_m: np.ndarray,
        intrinsics: CameraIntrinsics | Any,
        camera_to_world: Any,
    ) -> NavigationPoint:
        """Snap the requested pixel onto walkable floor and back-project it."""
        calibration = (
            intrinsics
            if isinstance(intrinsics, CameraIntrinsics)
            else CameraIntrinsics.from_matrix(intrinsics)
        )
        transform = np.asarray(camera_to_world, dtype=np.float64)
        if transform.shape != (4, 4):
            raise ValueError(
                "camera_to_world must be a 4x4 Habitat camera-to-world matrix."
            )
        floor = (
            self._floor_mask(depth_m, calibration, transform)
            if self.camera_height_m is not None
            else None
        )
        u, v, on_floor = self._nearest_walkable_pixel(
            depth_m, requested_uv, floor=floor
        )
        self.last_waypoint_on_floor = on_floor
        point_camera = self._back_project(
            u, v, float(depth_m[v, u]), calibration
        )
        point_world = transform @ np.array(
            (*point_camera, 1.0), dtype=np.float64
        )
        if not np.isclose(point_world[3], 1.0) and point_world[3] != 0.0:
            point_world /= point_world[3]
        return NavigationPoint(
            pixel_uv=(u, v),
            depth_m=float(depth_m[v, u]),
            camera_xyz=tuple(float(value) for value in point_camera),
            world_xyz=tuple(float(value) for value in point_world[:3]),
            on_floor=on_floor,
        )

    def _floor_mask(
        self,
        depth_m: np.ndarray,
        intrinsics: CameraIntrinsics,
        camera_to_world: np.ndarray,
    ) -> np.ndarray:
        """Mark the pixels whose world height is within tolerance of the floor.

        Every pixel is back-projected at once. The floor is
        ``camera_height_m`` below the camera's world position, which follows
        the agent through the scene, so a step or ramp inside the tolerance
        still counts while a pool basin or a wall does not.
        """
        height, width = depth_m.shape
        us = np.arange(width, dtype=np.float64)[None, :]
        vs = np.arange(height, dtype=np.float64)[:, None]
        x_cam = (us - intrinsics.cx) * depth_m / intrinsics.fx
        y_cam = -(vs - intrinsics.cy) * depth_m / intrinsics.fy
        z_cam = -depth_m
        rotation = camera_to_world[1, :3]
        y_world = (
            rotation[0] * x_cam
            + rotation[1] * y_cam
            + rotation[2] * z_cam
            + camera_to_world[1, 3]
        )
        floor_y = camera_to_world[1, 3] - self.camera_height_m
        return np.abs(y_world - floor_y) <= self.max_floor_offset_m

    @staticmethod
    def as_rgb_array(rgb: Any) -> np.ndarray:
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
    def rgb_to_png(rgb: np.ndarray) -> bytes:
        from PIL import Image

        buffer = io.BytesIO()
        Image.fromarray(rgb, mode="RGB").save(buffer, format="PNG")
        return buffer.getvalue()

    @staticmethod
    def extract_json_object(response: str) -> dict[str, Any]:
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

    def _nearest_walkable_pixel(
        self,
        depth_m: np.ndarray,
        requested_uv: tuple[int, int],
        *,
        floor: Optional[np.ndarray] = None,
    ) -> tuple[int, int, bool]:
        """Prefer floor-level walkable pixels, then fall back by stages.

        Returns the pixel and whether it is known to lie on the floor. Without
        a floor mask that flag is False: nothing has verified it.
        """
        height, width = depth_m.shape
        requested_u, requested_v = requested_uv
        # Candidate rows exclude the upper image, where ceilings and far walls
        # are common; the lower region is where navigable floor projects.
        min_v = int(height * 0.45)
        requested_u = int(np.clip(requested_u, 0, width - 1))
        requested_v = int(np.clip(requested_v, min_v, height - 1))
        valid = (
            np.isfinite(depth_m)
            & (depth_m >= self.min_depth_m)
            & (depth_m <= self.max_depth_m)
        )
        # The patch test is morphological: a pixel is walkable when the valid
        # depths surrounding it span no more than the allowed spread. Running
        # it over the whole frame at once replaces one slice per pixel. The
        # window covers the full frame, not just the candidate rows, so the
        # topmost candidate row still sees the pixels above it.
        masked = np.where(valid, depth_m, np.nan)
        spread = (
            self._window_extreme(masked, self.patch_radius_px, np.fmax)
            - self._window_extreme(masked, self.patch_radius_px, np.fmin)
        )
        walkable = valid & (spread <= self.max_patch_depth_spread_m)

        # Habitat depth can be noisy or discontinuous near chair legs, door
        # frames, and image borders. Keep a valid-depth fallback so one bad
        # local patch does not abort the whole episode. Floor-level stages
        # come first: when the camera height is known a point that is not on
        # the floor is only ever a last resort, and it is reported as such.
        stages: list[tuple[np.ndarray, bool]] = []
        if floor is not None:
            stages.append((walkable & floor, True))
            stages.append((valid & floor, True))
        stages.append((walkable, False))
        stages.append((valid, False))
        for candidate, on_floor in stages:
            region = candidate[min_v:]
            if region.any():
                break
        else:
            raise ValueError(
                "Depth observation contains no valid walkable waypoint."
            )

        rows, columns = np.nonzero(region)
        rows = rows + min_v
        distances = (columns - requested_u) ** 2 + (rows - requested_v) ** 2
        # The scan this replaces took the smallest (distance, u, v) tuple, so
        # ties resolve towards the lower column before the lower row.
        tied = distances == distances.min()
        best_u = columns[tied].min()
        best_v = rows[tied & (columns == best_u)].min()
        return int(best_u), int(best_v), on_floor

    @staticmethod
    def _window_extreme(
        values: np.ndarray,
        radius: int,
        combine: Any,
    ) -> np.ndarray:
        """Reduce each square window of side ``2 * radius + 1``, ignoring NaN.

        The reduction is separable, so two one-dimensional passes replace the
        square window. ``np.fmax`` and ``np.fmin`` skip NaN, which is why
        padding out-of-frame positions with NaN reproduces the clamped patch
        slicing this replaces: absent pixels simply do not contribute.
        """
        if radius <= 0:
            return values
        for axis in (0, 1):
            padding = [(0, 0), (0, 0)]
            padding[axis] = (radius, radius)
            padded = np.pad(values, padding, constant_values=np.nan)
            length = values.shape[axis]
            reduced = None
            for offset in range(2 * radius + 1):
                index = [slice(None), slice(None)]
                index[axis] = slice(offset, offset + length)
                window = padded[tuple(index)]
                reduced = window if reduced is None else combine(reduced, window)
            values = reduced
        return values

    @staticmethod
    def _back_project(
        u: int,
        v: int,
        depth_m: float,
        intrinsics: CameraIntrinsics,
    ) -> np.ndarray:
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
    def depth_in_meters(
        depth: Any,
        image_shape: Sequence[int],
        *,
        normalized: bool = False,
        depth_min_m: Optional[float] = None,
        depth_max_m: Optional[float] = None,
    ) -> np.ndarray:
        """Return the depth map in meters, validated against the RGB shape.

        Habitat's default depth sensor emits meters.  Set ``normalized`` only
        when its configuration uses normalized [0, 1] observations, and provide
        that sensor's ``depth_min_m`` and ``depth_max_m`` bounds.
        """
        values = np.asarray(depth, dtype=np.float64)
        if values.ndim == 3 and values.shape[-1] == 1:
            values = values[..., 0]
        if values.ndim != 2 or tuple(values.shape) != tuple(image_shape):
            raise ValueError(
                "depth must be HxW (or HxWx1) and match the RGB image."
            )
        if normalized:
            if (
                depth_min_m is None
                or depth_max_m is None
                or depth_max_m <= depth_min_m
            ):
                raise ValueError(
                    "normalized depth requires valid depth_min_m and "
                    "depth_max_m."
                )
            values = depth_min_m + values * (depth_max_m - depth_min_m)
        return values


__all__ = ("Actor",)
