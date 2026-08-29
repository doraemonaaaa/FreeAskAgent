"""A committed navigation target and its lifecycle.

This generalises the agent's doorway lock: once a world point is chosen (from
the waypoint model, a located landmark, a preview heading or a frontier) the
agent walks to it without asking the model again, until the point is reached,
walked past, too old, or provably not getting closer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass(slots=True)
class CommittedTarget:
    world_xyz: tuple[float, float, float]
    kind: str  # model_waypoint | landmark | preview | frontier
    subgoal_id: Optional[str]
    created_step: int
    tolerance_m: float = 0.5
    max_age_steps: int = 12
    stagnation_steps: int = 6
    reason: str = ""
    best_distance_m: Optional[float] = None
    stagnant_steps: int = 0
    updates: int = 0
    status: str = "active"
    distances: list[float] = field(default_factory=list)
    # Signed bearing from the heading to the target at each update; right is
    # positive. A large recent bearing means the agent is still turning to
    # face the target, which looks like an in-place spin to motion-only
    # error detection but is the route.
    bearings: list[float] = field(default_factory=list)

    def xz(self) -> tuple[float, float]:
        return (self.world_xyz[0], self.world_xyz[2])

    def age(self, step: int) -> int:
        return int(step - self.created_step)

    def update(
        self,
        *,
        step: int,
        position_xz: tuple[float, float],
        yaw_deg: float,
    ) -> str:
        """Advance the lifecycle for the current pose; returns the status."""
        if self.status != "active":
            return self.status
        dx = self.world_xyz[0] - position_xz[0]
        dz = self.world_xyz[2] - position_xz[1]
        distance = math.hypot(dx, dz)
        self.distances.append(distance)
        yaw = math.radians(yaw_deg)
        forward = (math.sin(yaw), -math.cos(yaw))
        right = (math.cos(yaw), math.sin(yaw))
        self.bearings.append(math.degrees(math.atan2(
            dx * right[0] + dz * right[1], dx * forward[0] + dz * forward[1]
        )))
        del self.bearings[:-8]
        self.updates += 1
        if distance <= self.tolerance_m:
            self.status = "reached"
            return self.status
        # Just behind the camera within a metre: served its purpose.
        ahead = dx * forward[0] + dz * forward[1]
        if distance <= 1.0 and ahead < -0.15:
            self.status = "passed"
            return self.status
        if self.best_distance_m is None or distance < self.best_distance_m - 0.10:
            self.best_distance_m = distance
            self.stagnant_steps = 0
        else:
            self.stagnant_steps += 1
        if self.stagnant_steps >= self.stagnation_steps:
            self.status = "stagnant"
            return self.status
        if self.age(step) >= self.max_age_steps:
            self.status = "stale"
            return self.status
        return self.status

    def current_distance_m(self) -> Optional[float]:
        return self.distances[-1] if self.distances else None

    def still_aligning(self, *, tolerance_deg: float, window: int = 6) -> bool:
        """True while any recent bearing was well off the camera axis."""
        return any(abs(b) > tolerance_deg for b in self.bearings[-window:])


__all__ = ("CommittedTarget",)
