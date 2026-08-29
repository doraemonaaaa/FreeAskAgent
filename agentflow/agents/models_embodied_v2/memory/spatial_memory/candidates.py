"""Set-of-mark candidates: where the agent *could* go next, from the map.

The waypoint model is poor at inventing a pixel (it answered "straight
ahead" on 80 % of steps) but good at picking among labelled options. This
module turns the current depth frame, the occupancy grid's frontiers and the
registered landmarks into a handful of reachable floor points, draws them on
the RGB frame as numbered markers, and describes them in text, so the model
answers a multiple-choice question instead.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .occupancy_grid import Frontier


@dataclass(slots=True)
class Candidate:
    label: str                       # "1".."N" for in-view points, "L"/"R"/"B" for turns
    world_xyz: tuple[float, float, float]
    distance_m: float
    bearing_deg: float               # signed, right positive
    kind: str                        # opening | frontier | landmark | turn
    pixel_uv: Optional[tuple[int, int]] = None
    note: str = ""
    extra: dict = field(default_factory=dict)

    def describe(self) -> str:
        side = "ahead" if abs(self.bearing_deg) < 12 else (
            f"{abs(self.bearing_deg):.0f}° {'right' if self.bearing_deg > 0 else 'left'}"
        )
        if self.kind == "turn":
            return f"[{self.label}] {self.note}"
        return f"[{self.label}] {self.distance_m:.1f} m, {side}: {self.note}"


def _pose(camera_to_world: Any) -> tuple[np.ndarray, np.ndarray, float]:
    transform = np.asarray(camera_to_world, dtype=np.float64)
    position = transform[:3, 3]
    rotation = transform[:3, :3]
    forward = -transform[:3, 2]
    yaw_deg = float(np.degrees(np.arctan2(forward[0], -forward[2])))
    return position, rotation, yaw_deg


def bearing_deg(dx: float, dz: float, yaw_deg: float) -> float:
    yaw = math.radians(yaw_deg)
    ahead = dx * math.sin(yaw) + dz * (-math.cos(yaw))
    side = dx * math.cos(yaw) + dz * math.sin(yaw)
    return math.degrees(math.atan2(side, ahead))


def project_to_pixel(
    world_xyz: Sequence[float],
    intrinsics: Any,
    camera_to_world: Any,
    image_shape: tuple[int, int],
) -> Optional[tuple[int, int]]:
    """Pixel of a world point in the current view, or None when not in view."""
    position, rotation, _ = _pose(camera_to_world)
    p_cam = rotation.T @ (np.asarray(world_xyz, dtype=np.float64) - position)
    depth = -p_cam[2]
    if depth <= 0.05:
        return None
    u = float(intrinsics.fx) * p_cam[0] / depth + float(intrinsics.cx)
    v = float(intrinsics.cy) - float(intrinsics.fy) * p_cam[1] / depth
    height, width = image_shape
    if not (0 <= u < width and 0 <= v < height):
        return None
    return int(round(u)), int(round(v))


def floor_openings(
    depth_m: np.ndarray,
    floor_mask: np.ndarray,
    intrinsics: Any,
    camera_to_world: Any,
    *,
    sectors: int = 5,
    min_range_m: float = 0.8,
    max_range_m: float = 5.0,
    pull_back_m: float = 0.5,
) -> list[Candidate]:
    """The farthest visible floor in each angular sector of the view.

    A column's farthest floor pixel is where walkable ground ends (a wall, a
    drop, or the range limit); the 80th percentile over a sector's columns
    rejects single-pixel outliers. The point is pulled back a little so the
    follower does not aim at the wall itself.
    """
    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.ndim == 3:
        depth = depth[..., 0]
    mask = np.asarray(floor_mask, dtype=bool) & np.isfinite(depth) & (depth > 0.2) & (depth < max_range_m)
    height, width = depth.shape
    position, rotation, yaw = _pose(camera_to_world)
    out: list[Candidate] = []
    edges = np.linspace(0, width, sectors + 1).astype(int)
    for s in range(sectors):
        c0, c1 = edges[s], edges[s + 1]
        best: list[tuple[float, int, int]] = []
        for col in range(c0, c1):
            rows = np.nonzero(mask[:, col])[0]
            if rows.size == 0:
                continue
            row = int(rows.min())          # farthest floor pixel in this column
            best.append((float(depth[row, col]), row, col))
        if not best:
            continue
        best.sort()
        d, row, col = best[int(0.8 * (len(best) - 1))]
        if d < min_range_m:
            continue
        d_use = max(min_range_m, d - pull_back_m)
        x_cam = (col - float(intrinsics.cx)) * d_use / float(intrinsics.fx)
        y_cam = -(row - float(intrinsics.cy)) * d_use / float(intrinsics.fy)
        p = rotation @ np.array((x_cam, y_cam, -d_use)) + position
        dx, dz = float(p[0] - position[0]), float(p[2] - position[2])
        out.append(Candidate(
            label="", world_xyz=(float(p[0]), float(p[1]), float(p[2])),
            distance_m=math.hypot(dx, dz), bearing_deg=bearing_deg(dx, dz, yaw),
            kind="opening", pixel_uv=(col, row),
            note="open floor up to " + ("a wall" if d < max_range_m - 0.05 else "the range limit"),
        ))
    return out


def generate_candidates(
    *,
    depth_m: np.ndarray,
    floor_mask: Optional[np.ndarray],
    intrinsics: Any,
    camera_to_world: Any,
    image_shape: tuple[int, int],
    frontiers: Sequence[Frontier] = (),
    landmark_xyz: Optional[tuple[float, float, float]] = None,
    landmark_note: str = "",
    floor_y: Optional[float] = None,
    max_in_view: int = 5,
    dedupe_m: float = 0.75,
    fov_half_deg: float = 45.0,
) -> list[Candidate]:
    """Numbered in-view floor points plus L/R/B turn options."""
    position, _, yaw = _pose(camera_to_world)
    in_view: list[Candidate] = []
    if landmark_xyz is not None:
        pix = project_to_pixel(landmark_xyz, intrinsics, camera_to_world, image_shape)
        dx, dz = landmark_xyz[0] - position[0], landmark_xyz[2] - position[2]
        if pix is not None:
            in_view.append(Candidate("", tuple(landmark_xyz), math.hypot(dx, dz),
                                     bearing_deg(dx, dz, yaw), "landmark", pix,
                                     landmark_note or "the active subgoal's landmark"))
    if floor_mask is not None:
        in_view.extend(floor_openings(depth_m, floor_mask, intrinsics, camera_to_world))
    y = floor_y if floor_y is not None else float(position[1])
    turn_pool: dict[str, list[Frontier]] = {"L": [], "R": [], "B": []}
    for f in frontiers:
        if f.distance_m < 0.8:
            continue
        xyz = (f.centre_xz[0], y, f.centre_xz[1])
        pix = project_to_pixel(xyz, intrinsics, camera_to_world, image_shape)
        if pix is not None and abs(f.bearing_deg) <= fov_half_deg:
            in_view.append(Candidate("", xyz, f.distance_m, f.bearing_deg, "frontier", pix,
                                     "boundary of the explored map; unexplored space beyond"))
        elif abs(f.bearing_deg) > 135:
            turn_pool["B"].append(f)
        elif f.bearing_deg < 0:
            turn_pool["L"].append(f)
        else:
            turn_pool["R"].append(f)
    # Landmark first, then farther points first; drop near-duplicates.
    in_view.sort(key=lambda c: (c.kind != "landmark", -c.distance_m))
    kept: list[Candidate] = []
    for c in in_view:
        if any(math.hypot(c.world_xyz[0] - k.world_xyz[0], c.world_xyz[2] - k.world_xyz[2]) < dedupe_m
               for k in kept):
            continue
        kept.append(c)
        if len(kept) >= max_in_view:
            break
    # Number left to right so the labels read naturally on the image.
    kept.sort(key=lambda c: c.pixel_uv[0] if c.pixel_uv else 0)
    for i, c in enumerate(kept, start=1):
        c.label = str(i)
    result = kept
    for label, name in (("L", "turn left"), ("R", "turn right"), ("B", "turn around")):
        pool = turn_pool[label]
        if not pool:
            continue
        f = min(pool, key=lambda f: f.distance_m)
        xyz = (f.centre_xz[0], y, f.centre_xz[1])
        result.append(Candidate(label, xyz, f.distance_m, f.bearing_deg, "turn", None,
                                f"{name}: unexplored area {f.distance_m:.1f} m away at "
                                f"{abs(f.bearing_deg):.0f}° {'right' if f.bearing_deg > 0 else 'left'}"))
    return result


def relabel(candidates: Sequence[Candidate]) -> list[Candidate]:
    """Renumber in-view markers 1..N left to right; keep L/R/B labels."""
    in_view = [c for c in candidates if c.kind != "turn"]
    turns = [c for c in candidates if c.kind == "turn"]
    in_view.sort(key=lambda c: c.pixel_uv[0] if c.pixel_uv else 0)
    for i, c in enumerate(in_view, start=1):
        c.label = str(i)
    return in_view + turns


def annotate_image(rgb: np.ndarray, candidates: Sequence[Candidate]) -> np.ndarray:
    """Draw numbered markers on a copy of the frame."""
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image)
    radius = max(9, image.width // 40)
    try:
        font = ImageFont.load_default(size=int(radius * 1.4))
    except TypeError:  # older Pillow
        font = ImageFont.load_default()
    for c in candidates:
        if c.pixel_uv is None:
            continue
        u, v = c.pixel_uv
        colour = (255, 215, 0) if c.kind == "landmark" else (255, 255, 255)
        draw.ellipse((u - radius, v - radius, u + radius, v + radius), fill=colour, outline=(0, 0, 0), width=2)
        bbox = draw.textbbox((0, 0), c.label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((u - tw / 2, v - th / 2 - bbox[1]), c.label, fill=(0, 0, 0), font=font)
    return np.asarray(image)


def encode_png(rgb: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def describe_candidates(candidates: Sequence[Candidate]) -> str:
    return "\n".join(c.describe() for c in candidates)


__all__ = (
    "Candidate",
    "annotate_image",
    "bearing_deg",
    "describe_candidates",
    "encode_png",
    "floor_openings",
    "generate_candidates",
    "project_to_pixel",
    "relabel",
)
