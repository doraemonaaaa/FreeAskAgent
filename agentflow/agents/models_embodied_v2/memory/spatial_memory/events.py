"""Deterministic spatial events read off the occupancy grid and the trail.

The first event kind is the doorway crossing: the agent moved through a
narrow constriction (obstacles close on both sides) that separates two
sizeable free regions.  Detection is pure geometry — no model call — so the
completion judge can treat it as measurement, the way it already treats net
yaw for turns and height change for stairs.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .occupancy_grid import FREE, OCCUPIED, OccupancyGrid

DOORWAY_MAX_WIDTH_M = 1.2
DOORWAY_SIDE_MAX_M = 0.9
DOORWAY_MIN_THROUGH_M = 0.45
DOORWAY_MIN_REGION_M2 = 3.0
DOORWAY_EVENT_COOLDOWN_M = 2.0
_SCAN_MAX_M = 1.6
_WINDOW_HALF_M = 4.0
_LOOKBACK_STEPS = 3


@dataclass(frozen=True)
class DoorwayCrossing:
    step: int
    position_xz: tuple[float, float]
    width_m: float


class DoorwayCrossingDetector:
    """Fire once each time the trail passes through a door-width constriction."""

    def __init__(
        self,
        *,
        max_width_m: float = DOORWAY_MAX_WIDTH_M,
        side_max_m: float = DOORWAY_SIDE_MAX_M,
        min_through_m: float = DOORWAY_MIN_THROUGH_M,
        min_region_m2: float = DOORWAY_MIN_REGION_M2,
        cooldown_m: float = DOORWAY_EVENT_COOLDOWN_M,
    ) -> None:
        self.max_width_m = float(max_width_m)
        self.side_max_m = float(side_max_m)
        self.min_through_m = float(min_through_m)
        self.min_region_m2 = float(min_region_m2)
        self.cooldown_m = float(cooldown_m)
        self._history: deque = deque(maxlen=16)  # (step, xz, width, left, right)
        self._heading_rad: Optional[float] = None
        self._last_event_xz: Optional[tuple[float, float]] = None

    def reset(self) -> None:
        self._history.clear()
        self._heading_rad = None
        self._last_event_xz = None

    # -- geometry helpers ---------------------------------------------------
    def _side_clearance_m(
        self, state: np.ndarray, grid: OccupancyGrid, xz, direction
    ) -> float:
        """Distance to the first OCCUPIED cell along ``direction``; inf if none."""
        step = grid.resolution_m
        distance = step
        while distance <= _SCAN_MAX_M:
            row, col = grid.world_to_cell(
                xz[0] + direction[0] * distance, xz[1] + direction[1] * distance
            )
            if grid.inside(row, col) and state[row, col] == OCCUPIED:
                return distance
            distance += step
        return float("inf")

    def _regions_separated(self, state, grid, gate_xz, before_xz, after_xz) -> bool:
        """Seal the gate; ``before`` and ``after`` must fall into two free regions,
        each at least ``min_region_m2`` of seen floor inside the local window."""
        rows, cols = state.shape
        half = int(_WINDOW_HALF_M / grid.resolution_m)
        gate = grid.world_to_cell(*gate_xz)
        r0, r1 = max(0, gate[0] - half), min(rows, gate[0] + half + 1)
        c0, c1 = max(0, gate[1] - half), min(cols, gate[1] + half + 1)
        window = state[r0:r1, c0:c1].copy()
        seal = int(round(0.8 / grid.resolution_m))
        gr, gc = gate[0] - r0, gate[1] - c0
        window[
            max(0, gr - seal):gr + seal + 1, max(0, gc - seal):gc + seal + 1
        ] = OCCUPIED
        min_cells = int(self.min_region_m2 / (grid.resolution_m ** 2))

        def flood(xz) -> Optional[frozenset]:
            row, col = grid.world_to_cell(*xz)
            row, col = row - r0, col - c0
            # The pose cell itself can be UNKNOWN (the camera never sees its
            # own feet); take the nearest FREE cell within a small radius.
            seed = None
            for radius in range(0, 6):
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        rr, cc = row + dr, col + dc
                        if 0 <= rr < window.shape[0] and 0 <= cc < window.shape[1] and window[rr, cc] == FREE:
                            seed = (rr, cc)
                            break
                    if seed:
                        break
                if seed:
                    break
            if seed is None:
                return None
            seen = {seed}
            queue = deque([seed])
            while queue:
                rr, cc = queue.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < window.shape[0]
                        and 0 <= nc < window.shape[1]
                        and (nr, nc) not in seen
                        and window[nr, nc] == FREE
                    ):
                        seen.add((nr, nc))
                        queue.append((nr, nc))
            return frozenset(seen)

        region_before = flood(before_xz)
        region_after = flood(after_xz)
        if region_before is None or region_after is None:
            return False
        if len(region_before) < min_cells or len(region_after) < min_cells:
            return False
        return not (region_before & region_after)

    # -- main entry ----------------------------------------------------------
    def update(self, *, step: int, grid: OccupancyGrid, trail) -> Optional[DoorwayCrossing]:
        if len(trail) < 2 or grid.origin_xz is None if hasattr(grid, "origin_xz") else False:
            return None
        if len(trail) < 2:
            return None
        xz = trail[-1]
        prev = trail[-2]
        move = (xz[0] - prev[0], xz[1] - prev[1])
        norm = float(np.hypot(*move))
        if norm >= 0.05:
            self._heading_rad = float(np.arctan2(move[1], move[0]))
        if self._heading_rad is None:
            return None
        state = grid.state_map()
        perp = (-np.sin(self._heading_rad), np.cos(self._heading_rad))
        left = self._side_clearance_m(state, grid, xz, perp)
        right = self._side_clearance_m(state, grid, xz, (-perp[0], -perp[1]))
        self._history.append((int(step), tuple(xz), left + right, left, right))
        if len(self._history) < 2 * _LOOKBACK_STEPS + 1:
            return None
        candidate = self._history[-1 - _LOOKBACK_STEPS]
        c_step, c_xz, c_width, c_left, c_right = candidate
        if c_width > self.max_width_m or max(c_left, c_right) > self.side_max_m:
            return None
        before_xz = self._history[-1 - 2 * _LOOKBACK_STEPS][1]
        after_xz = self._history[-1][1]
        into = (c_xz[0] - before_xz[0], c_xz[1] - before_xz[1])
        out = (after_xz[0] - c_xz[0], after_xz[1] - c_xz[1])
        if float(np.hypot(*into)) < self.min_through_m or float(np.hypot(*out)) < self.min_through_m:
            return None
        if into[0] * out[0] + into[1] * out[1] <= 0.0:
            return None  # lingered or backed out rather than passed through
        if self._last_event_xz is not None and float(
            np.hypot(c_xz[0] - self._last_event_xz[0], c_xz[1] - self._last_event_xz[1])
        ) < self.cooldown_m:
            return None
        if not self._regions_separated(state, grid, c_xz, before_xz, after_xz):
            return None
        self._last_event_xz = c_xz
        return DoorwayCrossing(step=c_step, position_xz=c_xz, width_m=float(c_width))
