"""2D occupancy grid on the floor plane, built from depth and camera pose.

World frame follows Habitat: ``y`` is up, the camera looks along its ``-z``
axis, and the floor plane is spanned by ``x`` (grid columns) and ``z`` (grid
rows). Cells are UNKNOWN until the depth camera sees either floor at that
location (FREE) or something in the obstacle band above it (OCCUPIED).
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np

UNKNOWN = 0
FREE = 1
OCCUPIED = 2


@dataclass(frozen=True, slots=True)
class Frontier:
    """A cluster of known-free cells that border unexplored space."""

    centre_xz: tuple[float, float]
    size: int
    distance_m: float
    bearing_deg: float


class OccupancyGrid:
    def __init__(
        self,
        *,
        resolution_m: float = 0.10,
        extent_m: float = 60.0,
        camera_height_m: float = 1.25,
        floor_tolerance_m: float = 0.30,
        obstacle_band_m: tuple[float, float] = (0.30, 1.80),
        max_range_m: float = 5.0,
        min_range_m: float = 0.25,
        pixel_stride: int = 3,
    ) -> None:
        if resolution_m <= 0 or extent_m <= 0:
            raise ValueError("resolution_m and extent_m must be positive")
        self.resolution_m = float(resolution_m)
        self.cells = int(round(extent_m / resolution_m))
        self.camera_height_m = float(camera_height_m)
        self.floor_tolerance_m = float(floor_tolerance_m)
        self.obstacle_band_m = (float(obstacle_band_m[0]), float(obstacle_band_m[1]))
        self.max_range_m = float(max_range_m)
        self.min_range_m = float(min_range_m)
        self.pixel_stride = max(1, int(pixel_stride))
        self.origin_xz: Optional[tuple[float, float]] = None
        self._free_hits = np.zeros((self.cells, self.cells), dtype=np.uint16)
        self._occ_hits = np.zeros((self.cells, self.cells), dtype=np.uint16)
        self.observations = 0

    # ------------------------------------------------------------ frames
    def reset(self) -> None:
        self.origin_xz = None
        self._free_hits[:] = 0
        self._occ_hits[:] = 0
        self.observations = 0

    def _ensure_origin(self, position_xz: tuple[float, float]) -> None:
        if self.origin_xz is None:
            half = self.cells * self.resolution_m / 2.0
            self.origin_xz = (position_xz[0] - half, position_xz[1] - half)

    def world_to_cell(self, x: float, z: float) -> tuple[int, int]:
        """(row, col) of a world point; may lie outside the grid."""
        if self.origin_xz is None:
            raise RuntimeError("grid has no origin until the first observation")
        col = int(math.floor((x - self.origin_xz[0]) / self.resolution_m))
        row = int(math.floor((z - self.origin_xz[1]) / self.resolution_m))
        return row, col

    def cell_to_world(self, row: int, col: int) -> tuple[float, float]:
        if self.origin_xz is None:
            raise RuntimeError("grid has no origin until the first observation")
        return (
            self.origin_xz[0] + (col + 0.5) * self.resolution_m,
            self.origin_xz[1] + (row + 0.5) * self.resolution_m,
        )

    def inside(self, row: int, col: int) -> bool:
        return 0 <= row < self.cells and 0 <= col < self.cells

    # ------------------------------------------------------------ update
    def update(
        self,
        depth_m: np.ndarray,
        intrinsics: Any,
        camera_to_world: np.ndarray,
        *,
        floor_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Fuse one depth frame. ``intrinsics`` needs fx, fy, cx, cy."""
        depth = np.asarray(depth_m, dtype=np.float64)
        if depth.ndim == 3:
            depth = depth[..., 0]
        transform = np.asarray(camera_to_world, dtype=np.float64)
        if transform.shape != (4, 4):
            raise ValueError("camera_to_world must be a 4x4 transform")
        position = transform[:3, 3]
        self._ensure_origin((float(position[0]), float(position[2])))
        self.observations += 1

        stride = self.pixel_stride
        sub = depth[::stride, ::stride]
        height, width = depth.shape
        vs = np.arange(0, height, stride, dtype=np.float64)[:, None]
        us = np.arange(0, width, stride, dtype=np.float64)[None, :]
        valid = np.isfinite(sub) & (sub > self.min_range_m) & (sub < self.max_range_m)
        if floor_mask is not None:
            floor_sub = np.asarray(floor_mask, dtype=bool)[::stride, ::stride]
        else:
            floor_sub = None
        if valid.any():
            d = sub[valid]
            x_cam = ((us - float(intrinsics.cx)) * sub / float(intrinsics.fx))[valid]
            y_cam = (-(vs - float(intrinsics.cy)) * sub / float(intrinsics.fy))[valid]
            z_cam = -d
            rotation = transform[:3, :3]
            points = rotation @ np.vstack((x_cam, y_cam, z_cam)) + position[:, None]
            floor_y = float(position[1]) - self.camera_height_m
            height_above = points[1] - floor_y
            if floor_sub is not None:
                is_floor = floor_sub[valid]
            else:
                is_floor = np.abs(height_above) <= self.floor_tolerance_m
            low, high = self.obstacle_band_m
            is_obstacle = (height_above > low) & (height_above <= high)
            self._accumulate(points[0][is_floor], points[2][is_floor], self._free_hits)
            self._accumulate(points[0][is_obstacle], points[2][is_obstacle], self._occ_hits)
        # The agent stands where it stands: its own footprint is free.
        self._stamp_free_disc(float(position[0]), float(position[2]), radius_m=0.25)

    def _accumulate(self, xs: np.ndarray, zs: np.ndarray, hits: np.ndarray) -> None:
        if xs.size == 0:
            return
        cols = np.floor((xs - self.origin_xz[0]) / self.resolution_m).astype(np.int64)
        rows = np.floor((zs - self.origin_xz[1]) / self.resolution_m).astype(np.int64)
        keep = (rows >= 0) & (rows < self.cells) & (cols >= 0) & (cols < self.cells)
        if not keep.any():
            return
        flat = rows[keep] * self.cells + cols[keep]
        counts = np.bincount(flat, minlength=self.cells * self.cells)
        counts = np.minimum(counts, 8).astype(np.uint16)
        hits += counts.reshape(self.cells, self.cells)
        np.minimum(hits, 60000, out=hits)

    def _stamp_free_disc(self, x: float, z: float, *, radius_m: float) -> None:
        row, col = self.world_to_cell(x, z)
        r = int(math.ceil(radius_m / self.resolution_m))
        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                if dr * dr + dc * dc > r * r:
                    continue
                rr, cc = row + dr, col + dc
                if self.inside(rr, cc):
                    self._free_hits[rr, cc] = min(int(self._free_hits[rr, cc]) + 2, 60000)

    # ------------------------------------------------------------ queries
    def state_map(self) -> np.ndarray:
        """UNKNOWN / FREE / OCCUPIED per cell."""
        occ = (self._occ_hits >= 2) & (self._occ_hits.astype(np.int64) * 4 >= self._free_hits)
        free = (~occ) & (self._free_hits >= 1)
        state = np.zeros_like(self._free_hits, dtype=np.uint8)
        state[free] = FREE
        state[occ] = OCCUPIED
        return state

    def state_at(self, x: float, z: float) -> int:
        row, col = self.world_to_cell(x, z)
        if not self.inside(row, col):
            return UNKNOWN
        return int(self.state_map()[row, col])

    def explored_area_m2(self) -> float:
        state = self.state_map()
        return float(np.count_nonzero(state != UNKNOWN)) * self.resolution_m ** 2

    def _blocked_map(self, state: np.ndarray, inflate_m: float) -> np.ndarray:
        blocked = state == OCCUPIED
        r = int(math.ceil(inflate_m / self.resolution_m))
        if r <= 0:
            return blocked
        out = blocked.copy()
        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                if dr * dr + dc * dc > r * r or (dr == 0 and dc == 0):
                    continue
                shifted = np.zeros_like(blocked)
                rs = slice(max(dr, 0), self.cells + min(dr, 0))
                rd = slice(max(-dr, 0), self.cells + min(-dr, 0))
                cs = slice(max(dc, 0), self.cells + min(dc, 0))
                cd = slice(max(-dc, 0), self.cells + min(-dc, 0))
                shifted[rd, cd] = blocked[rs, cs]
                out |= shifted
        return out

    def frontiers(
        self,
        position_xz: tuple[float, float],
        yaw_deg: float,
        *,
        min_size: int = 3,
    ) -> list[Frontier]:
        """Clusters of FREE cells with an UNKNOWN 4-neighbour."""
        state = self.state_map()
        free = state == FREE
        unknown = state == UNKNOWN
        neighbour_unknown = np.zeros_like(unknown)
        neighbour_unknown[1:, :] |= unknown[:-1, :]
        neighbour_unknown[:-1, :] |= unknown[1:, :]
        neighbour_unknown[:, 1:] |= unknown[:, :-1]
        neighbour_unknown[:, :-1] |= unknown[:, 1:]
        frontier = free & neighbour_unknown
        cells = list(zip(*np.nonzero(frontier)))
        if not cells:
            return []
        remaining = set(cells)
        clusters: list[list[tuple[int, int]]] = []
        while remaining:
            seed = remaining.pop()
            stack = [seed]
            cluster = [seed]
            while stack:
                r, c = stack.pop()
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        n = (r + dr, c + dc)
                        if n in remaining:
                            remaining.remove(n)
                            stack.append(n)
                            cluster.append(n)
            if len(cluster) >= min_size:
                clusters.append(cluster)
        result = []
        for cluster in clusters:
            rows = np.array([c[0] for c in cluster], dtype=np.float64)
            cols = np.array([c[1] for c in cluster], dtype=np.float64)
            # Use the member nearest the centroid so the target is a real cell.
            centroid = (rows.mean(), cols.mean())
            idx = int(np.argmin((rows - centroid[0]) ** 2 + (cols - centroid[1]) ** 2))
            x, z = self.cell_to_world(int(rows[idx]), int(cols[idx]))
            dx, dz = x - position_xz[0], z - position_xz[1]
            distance = math.hypot(dx, dz)
            bearing = _bearing_deg(dx, dz, yaw_deg)
            result.append(Frontier((x, z), len(cluster), distance, bearing))
        result.sort(key=lambda f: f.distance_m)
        return result

    def plan(
        self,
        start_xz: tuple[float, float],
        goal_xz: tuple[float, float],
        *,
        inflate_m: float = 0.25,
        unknown_cost: float = 2.5,
        margin_m: float = 3.0,
        max_expansions: int = 60000,
    ) -> Optional[list[tuple[float, float]]]:
        """A* over FREE and (at a premium) UNKNOWN cells; None if no path."""
        if self.origin_xz is None:
            return None
        state = self.state_map()
        blocked = self._blocked_map(state, inflate_m)
        sr, sc = self.world_to_cell(*start_xz)
        gr, gc = self.world_to_cell(*goal_xz)
        if not self.inside(sr, sc):
            return None
        gr, gc = self._nearest_unblocked(blocked, gr, gc, radius_cells=int(1.0 / self.resolution_m))
        if gr is None:
            return None
        # The start cell is never blocked: the agent is standing on it.
        blocked[sr, sc] = False
        m = int(margin_m / self.resolution_m)
        r0, r1 = max(min(sr, gr) - m, 0), min(max(sr, gr) + m, self.cells - 1)
        c0, c1 = max(min(sc, gc) - m, 0), min(max(sc, gc) + m, self.cells - 1)
        cost_map = np.where(state == UNKNOWN, unknown_cost, 1.0)

        def h(r: int, c: int) -> float:
            return math.hypot(r - gr, c - gc)

        start = (sr, sc)
        goal = (gr, gc)
        best = {start: 0.0}
        parent: dict[tuple[int, int], tuple[int, int]] = {}
        heap = [(h(sr, sc), 0.0, start)]
        expansions = 0
        found = False
        while heap:
            _, g, node = heapq.heappop(heap)
            if node == goal:
                found = True
                break
            if g > best.get(node, math.inf):
                continue
            expansions += 1
            if expansions > max_expansions:
                break
            r, c = node
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if not (r0 <= nr <= r1 and c0 <= nc <= c1):
                        continue
                    if blocked[nr, nc]:
                        continue
                    if dr and dc and (blocked[r + dr, c] or blocked[r, c + dc]):
                        continue  # no corner cutting through walls
                    step = math.sqrt(2.0) if dr and dc else 1.0
                    ng = g + step * float(cost_map[nr, nc])
                    nxt = (nr, nc)
                    if ng < best.get(nxt, math.inf):
                        best[nxt] = ng
                        parent[nxt] = node
                        heapq.heappush(heap, (ng + h(nr, nc), ng, nxt))
        if not found:
            return None
        cells = [goal]
        while cells[-1] != start:
            cells.append(parent[cells[-1]])
        cells.reverse()
        return [self.cell_to_world(r, c) for r, c in cells]

    def _nearest_unblocked(
        self, blocked: np.ndarray, row: int, col: int, *, radius_cells: int
    ) -> tuple[Optional[int], Optional[int]]:
        if self.inside(row, col) and not blocked[row, col]:
            return row, col
        best = None
        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                rr, cc = row + dr, col + dc
                if not self.inside(rr, cc) or blocked[rr, cc]:
                    continue
                d = dr * dr + dc * dc
                if best is None or d < best[0]:
                    best = (d, rr, cc)
        if best is None:
            return None, None
        return best[1], best[2]

    def render(self, *, px_per_cell: int = 1) -> np.ndarray:
        """RGB image: unknown grey, free white, occupied black."""
        state = self.state_map()
        image = np.full((self.cells, self.cells, 3), 128, dtype=np.uint8)
        image[state == FREE] = 255
        image[state == OCCUPIED] = 0
        if px_per_cell > 1:
            image = np.repeat(np.repeat(image, px_per_cell, axis=0), px_per_cell, axis=1)
        return image


def _bearing_deg(dx: float, dz: float, yaw_deg: float) -> float:
    """Signed angle from the heading to (dx, dz); right is positive."""
    yaw = math.radians(yaw_deg)
    forward = (math.sin(yaw), -math.cos(yaw))
    right = (math.cos(yaw), math.sin(yaw))
    ahead = dx * forward[0] + dz * forward[1]
    side = dx * right[0] + dz * right[1]
    return math.degrees(math.atan2(side, ahead))


def path_length_m(path: Iterable[tuple[float, float]]) -> float:
    total = 0.0
    previous = None
    for point in path:
        if previous is not None:
            total += math.hypot(point[0] - previous[0], point[1] - previous[1])
        previous = point
    return total


__all__ = (
    "FREE",
    "Frontier",
    "OCCUPIED",
    "OccupancyGrid",
    "UNKNOWN",
    "path_length_m",
)
