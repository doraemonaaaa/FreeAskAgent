"""Spatial Memory: the agent's map of where it has been and what it found.

Owns three things the other memories do not:

* an :class:`OccupancyGrid` of the floor plane fused from every depth frame,
  with explored/unexplored bookkeeping and an A* planner;
* a :class:`LandmarkRegistry` of Captioner sightings anchored in world
  coordinates, so "the doorway on the left" survives the frames that no
  longer show it;
* one :class:`CommittedTarget` at a time -- the world point the agent is
  walking to -- so the waypoint model is consulted at decision points rather
  than every 0.25 m.

It never issues actions itself. The agent asks it for the next waypoint along
the planned path, for the status of the committed target, or for a frontier
to explore when nothing is in view.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from .candidates import Candidate, relabel
from .landmarks import LandmarkRegistry, SpatialLandmark
from .occupancy_grid import FREE, OCCUPIED, UNKNOWN, Frontier, OccupancyGrid, path_length_m
from .targets import CommittedTarget


class SpatialMemory:
    def __init__(
        self,
        *,
        camera_height_m: Optional[float] = None,
        resolution_m: float = 0.10,
        extent_m: float = 60.0,
        max_range_m: float = 5.0,
        lookahead_m: float = 1.5,
    ) -> None:
        self.camera_height_m = float(camera_height_m) if camera_height_m else 1.25
        self.grid = OccupancyGrid(
            resolution_m=resolution_m,
            extent_m=extent_m,
            camera_height_m=self.camera_height_m,
            max_range_m=max_range_m,
        )
        self.landmarks = LandmarkRegistry()
        self.lookahead_m = float(lookahead_m)
        self.target: Optional[CommittedTarget] = None
        self.step: int = -1
        self.position: Optional[np.ndarray] = None
        self.yaw_deg: float = 0.0
        self.trail: list[tuple[float, float]] = []
        self.released_targets: list[CommittedTarget] = []
        self._visited_frontiers: list[tuple[float, float]] = []
        self.last_path: Optional[list[tuple[float, float]]] = None
        self.last_release_reason: Optional[str] = None
        # Controller traversability window (navmesh) around the agent, when
        # the runner provides one: origin (x, z), resolution, bool mask.
        self._nav_origin: Optional[tuple[float, float]] = None
        self._nav_resolution: float = 0.25
        self._nav_mask: Optional[np.ndarray] = None

    # ------------------------------------------------------------ traversability
    def set_traversability(
        self,
        *,
        origin_xz: tuple[float, float],
        resolution_m: float,
        mask: np.ndarray,
    ) -> None:
        self._nav_origin = (float(origin_xz[0]), float(origin_xz[1]))
        self._nav_resolution = float(resolution_m)
        self._nav_mask = np.asarray(mask, dtype=bool)

    @property
    def has_traversability(self) -> bool:
        return self._nav_mask is not None

    def is_navigable(self, x: float, z: float) -> Optional[bool]:
        """True/False inside the window, None when unknown."""
        if self._nav_mask is None or self._nav_origin is None:
            return None
        col = int(math.floor((x - self._nav_origin[0]) / self._nav_resolution))
        row = int(math.floor((z - self._nav_origin[1]) / self._nav_resolution))
        rows, cols = self._nav_mask.shape
        if not (0 <= row < rows and 0 <= col < cols):
            return None
        return bool(self._nav_mask[row, col])

    def snap_navigable(
        self, x: float, z: float, *, radius_m: float = 1.0
    ) -> Optional[tuple[float, float]]:
        """Nearest navigable cell centre within ``radius_m``; the point
        itself when it is navigable or when no window is known."""
        known = self.is_navigable(x, z)
        if known is None or known:
            return (x, z)
        res = self._nav_resolution
        col = int(math.floor((x - self._nav_origin[0]) / res))
        row = int(math.floor((z - self._nav_origin[1]) / res))
        r = int(math.ceil(radius_m / res))
        rows, cols = self._nav_mask.shape
        best = None
        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                rr, cc = row + dr, col + dc
                if not (0 <= rr < rows and 0 <= cc < cols) or not self._nav_mask[rr, cc]:
                    continue
                cx = self._nav_origin[0] + (cc + 0.5) * res
                cz = self._nav_origin[1] + (rr + 0.5) * res
                d = math.hypot(cx - x, cz - z)
                if d <= radius_m and (best is None or d < best[0]):
                    best = (d, cx, cz)
        return None if best is None else (best[1], best[2])

    def filter_navigable(
        self, candidates: list[Candidate], *, radius_m: float = 0.75
    ) -> list[Candidate]:
        """Drop candidates the controller cannot reach; snap the rest."""
        if not self.has_traversability:
            return candidates
        kept: list[Candidate] = []
        for c in candidates:
            snapped = self.snap_navigable(c.world_xyz[0], c.world_xyz[2], radius_m=radius_m)
            if snapped is None:
                continue
            c.world_xyz = (snapped[0], c.world_xyz[1], snapped[1])
            kept.append(c)
        return relabel(kept)

    # ------------------------------------------------------------ episode
    def reset(self) -> None:
        self.grid.reset()
        self.landmarks.reset()
        self.target = None
        self.step = -1
        self.position = None
        self.yaw_deg = 0.0
        self.trail.clear()
        self.released_targets.clear()
        self._visited_frontiers.clear()
        self.last_path = None
        self.last_release_reason = None
        self._nav_origin = None
        self._nav_mask = None

    # ------------------------------------------------------------ observe
    @staticmethod
    def pose_from_transform(camera_to_world: Any) -> tuple[np.ndarray, float]:
        transform = np.asarray(camera_to_world, dtype=np.float64)
        if transform.shape != (4, 4):
            raise ValueError("camera_to_world must be a 4x4 transform")
        position = transform[:3, 3].copy()
        forward = -transform[:3, 2]
        yaw_deg = float(np.degrees(np.arctan2(forward[0], -forward[2])))
        return position, yaw_deg

    def observe(
        self,
        *,
        step: int,
        depth_m: np.ndarray,
        intrinsics: Any,
        camera_to_world: Any,
        floor_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Fuse one RGB-D observation and advance the committed target."""
        self.position, self.yaw_deg = self.pose_from_transform(camera_to_world)
        self.step = int(step)
        self.trail.append(self.position_xz)
        self.grid.update(depth_m, intrinsics, camera_to_world, floor_mask=floor_mask)
        if self.target is not None:
            self.target.update(
                step=self.step, position_xz=self.position_xz, yaw_deg=self.yaw_deg
            )

    @property
    def position_xz(self) -> tuple[float, float]:
        if self.position is None:
            raise RuntimeError("no observation yet")
        return (float(self.position[0]), float(self.position[2]))

    @property
    def floor_y(self) -> float:
        if self.position is None:
            raise RuntimeError("no observation yet")
        return float(self.position[1]) - self.camera_height_m

    # ------------------------------------------------------------ landmarks
    def register_landmark(
        self,
        name: str,
        world_xyz: tuple[float, float, float],
        *,
        subgoal_id: Optional[str] = None,
        confidence: float = 0.0,
        kind: str = "landmark",
    ) -> SpatialLandmark:
        return self.landmarks.register(
            name,
            world_xyz,
            step=max(self.step, 0),
            subgoal_id=subgoal_id,
            confidence=confidence,
            kind=kind,
        )

    def landmark_for_subgoal(self, subgoal_id: Optional[str]) -> Optional[SpatialLandmark]:
        return self.landmarks.for_subgoal(subgoal_id)

    # ------------------------------------------------------------ targets
    def commit_target(
        self,
        world_xyz: tuple[float, float, float],
        *,
        kind: str,
        subgoal_id: Optional[str],
        reason: str = "",
        tolerance_m: float = 0.5,
        max_age_steps: int = 12,
        stagnation_steps: int = 6,
        snap: bool = True,
    ) -> Optional[CommittedTarget]:
        """Commit to a world point; None when the controller cannot reach it."""
        world = tuple(float(v) for v in world_xyz)
        if snap:
            # The follower needs a navigable goal. The controller's
            # traversability window is authoritative when known; otherwise
            # at least move the point off the wall clearance band.
            if self.has_traversability:
                nav = self.snap_navigable(world[0], world[2], radius_m=1.0)
                if nav is None:
                    self.last_release_reason = "unreachable target"
                    return None
                world = (nav[0], world[1], nav[1])
            else:
                snapped = self.grid.snap_to_clear_free(world[0], world[2])
                if snapped is not None:
                    world = (snapped[0], world[1], snapped[1])
        if self.target is not None and self.target.status == "active":
            self.release_target("replaced")
        self.target = CommittedTarget(
            world_xyz=world,
            kind=kind,
            subgoal_id=subgoal_id,
            created_step=max(self.step, 0),
            tolerance_m=float(tolerance_m),
            max_age_steps=int(max_age_steps),
            stagnation_steps=int(stagnation_steps),
            reason=reason,
        )
        if self.position is not None:
            self.target.update(step=self.step, position_xz=self.position_xz, yaw_deg=self.yaw_deg)
        return self.target

    def release_target(self, reason: str) -> None:
        if self.target is None:
            return
        if self.target.status == "active":
            self.target.status = f"released:{reason}"
        if self.target.kind == "frontier":
            self._visited_frontiers.append(self.target.xz())
        self.released_targets.append(self.target)
        del self.released_targets[:-16]
        self.last_release_reason = reason
        self.target = None

    def active_target(self, subgoal_id: Optional[str]) -> Optional[CommittedTarget]:
        """The committed target if it is still worth walking to."""
        target = self.target
        if target is None:
            return None
        if target.subgoal_id != subgoal_id:
            self.release_target("subgoal changed")
            return None
        if target.status != "active":
            self.release_target(target.status)
            return None
        return target

    def next_waypoint(self) -> Optional[tuple[tuple[float, float, float], float, str]]:
        """World point to walk to next along the planned path to the target.

        Returns ``(xyz, remaining_path_m, how)`` where ``how`` is ``"path"``
        when the grid found a route and ``"direct"`` when it did not and the
        agent should simply head toward the target.
        """
        if self.target is None or self.position is None:
            return None
        goal_xz = self.target.xz()
        path = self.grid.plan(self.position_xz, goal_xz)
        self.last_path = path
        y = float(self.target.world_xyz[1])
        if not path or len(path) < 2:
            return (self.target.world_xyz, math.dist(self.position_xz, goal_xz), "direct")
        remaining = path_length_m(path)
        walked = 0.0
        previous = path[0]
        chosen = path[-1]
        for point in path[1:]:
            walked += math.hypot(point[0] - previous[0], point[1] - previous[1])
            previous = point
            if walked >= self.lookahead_m:
                chosen = point
                break
        return ((chosen[0], y, chosen[1]), remaining, "path")

    # ------------------------------------------------------------ frontiers
    def frontiers(self, *, min_size: int = 3) -> list[Frontier]:
        if self.position is None:
            return []
        return self.grid.frontiers(self.position_xz, self.yaw_deg, min_size=min_size)

    def choose_frontier(
        self,
        *,
        preferred_bearing_deg: Optional[float] = None,
        min_distance_m: float = 0.8,
        ideal_distance_m: float = 3.0,
    ) -> Optional[tuple[float, float, float]]:
        """Pick the unexplored boundary best worth walking to.

        Nearer clusters and clusters ahead (or toward ``preferred_bearing``)
        score better; boundaries already walked to are skipped so the agent
        does not bounce between the same two openings.
        """
        best = None
        for frontier in self.frontiers():
            if frontier.distance_m < min_distance_m:
                continue
            if self.is_navigable(*frontier.centre_xz) is False and self.snap_navigable(
                *frontier.centre_xz, radius_m=0.5
            ) is None:
                continue
            if any(
                math.hypot(frontier.centre_xz[0] - vx, frontier.centre_xz[1] - vz) < 0.75
                for vx, vz in self._visited_frontiers
            ):
                continue
            bearing = frontier.bearing_deg
            if preferred_bearing_deg is not None:
                bearing = (bearing - preferred_bearing_deg + 180.0) % 360.0 - 180.0
            score = (
                abs(frontier.distance_m - ideal_distance_m) / ideal_distance_m
                + (1.0 - math.cos(math.radians(bearing)))
                - min(frontier.size, 30) / 60.0
            )
            if best is None or score < best[0]:
                best = (score, frontier)
        if best is None:
            return None
        x, z = best[1].centre_xz
        return (x, self.floor_y, z)

    # ------------------------------------------------------------ reporting
    def diagnostics(self) -> dict[str, Any]:
        target = self.target
        state = {
            "step": self.step,
            "observations": self.grid.observations,
            "explored_m2": round(self.grid.explored_area_m2(), 2),
            "landmarks": len(self.landmarks),
            "trail_m": round(path_length_m(self.trail), 2),
            "frontiers": len(self.frontiers()) if self.position is not None else 0,
            "last_release": self.last_release_reason,
            "target": None,
        }
        if target is not None:
            state["target"] = {
                "kind": target.kind,
                "subgoal_id": target.subgoal_id,
                "status": target.status,
                "age": target.age(self.step),
                "distance_m": (
                    round(target.current_distance_m(), 2)
                    if target.current_distance_m() is not None
                    else None
                ),
                "best_distance_m": (
                    round(target.best_distance_m, 2)
                    if target.best_distance_m is not None
                    else None
                ),
                "stagnant_steps": target.stagnant_steps,
                "reason": target.reason,
                "world_xyz": [round(v, 2) for v in target.world_xyz],
            }
        return state

    def summary(self) -> str:
        """One token for the step log: ``sp=kind:status d=1.2 age=3``."""
        target = self.target
        if target is None:
            return "sp=-"
        distance = target.current_distance_m()
        return "sp={}:{} d={} age={}".format(
            target.kind,
            target.status,
            f"{distance:.1f}" if distance is not None else "-",
            target.age(self.step),
        )

    def render_topdown(self, *, px_per_cell: int = 2) -> np.ndarray:
        """Grid image with the trail (blue), landmarks (green), target (red)."""
        image = self.grid.render(px_per_cell=px_per_cell)

        def paint(xz: tuple[float, float], colour: tuple[int, int, int], radius: int = 1) -> None:
            row, col = self.grid.world_to_cell(*xz)
            r0, c0 = row * px_per_cell, col * px_per_cell
            image[max(r0 - radius, 0): r0 + radius + 1, max(c0 - radius, 0): c0 + radius + 1] = colour

        if self.grid.origin_xz is not None:
            for xz in self.trail:
                paint(xz, (40, 90, 255), 0)
            for landmark in self.landmarks.all():
                paint((landmark.world_xyz[0], landmark.world_xyz[2]), (0, 170, 0), 2)
            if self.target is not None:
                paint(self.target.xz(), (230, 30, 30), 2)
            if self.position is not None:
                paint(self.position_xz, (255, 140, 0), 2)
        return image


__all__ = ("SpatialMemory", "FREE", "OCCUPIED", "UNKNOWN")
