"""World-anchored landmarks recognised by the Captioner."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass(slots=True)
class SpatialLandmark:
    name: str
    subgoal_id: Optional[str]
    world_xyz: tuple[float, float, float]
    first_step: int
    last_step: int
    observations: int = 1
    confidence: float = 0.0
    kind: str = "landmark"
    history: list[tuple[float, float, float]] = field(default_factory=list)

    def distance_xz(self, position_xz: tuple[float, float]) -> float:
        return math.hypot(
            self.world_xyz[0] - position_xz[0], self.world_xyz[2] - position_xz[1]
        )


class LandmarkRegistry:
    """Merge repeated sightings of the same landmark into one world point."""

    def __init__(self, *, merge_radius_m: float = 1.0, max_history: int = 8) -> None:
        self.merge_radius_m = float(merge_radius_m)
        self.max_history = int(max_history)
        self._items: list[SpatialLandmark] = []

    def reset(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

    def all(self) -> tuple[SpatialLandmark, ...]:
        return tuple(self._items)

    def register(
        self,
        name: str,
        world_xyz: tuple[float, float, float],
        *,
        step: int,
        subgoal_id: Optional[str] = None,
        confidence: float = 0.0,
        kind: str = "landmark",
    ) -> SpatialLandmark:
        """Add a sighting; a nearby same-named landmark absorbs it.

        The stored point is the running mean of its sightings, so a single
        mislocalised frame moves an established landmark only a little.
        """
        world = tuple(float(v) for v in world_xyz)
        for item in self._items:
            if item.name != name:
                continue
            if math.hypot(item.world_xyz[0] - world[0], item.world_xyz[2] - world[2]) > self.merge_radius_m:
                continue
            n = item.observations
            item.world_xyz = tuple(
                (item.world_xyz[i] * n + world[i]) / (n + 1) for i in range(3)
            )
            item.observations = n + 1
            item.last_step = step
            item.confidence = max(item.confidence, float(confidence))
            if subgoal_id is not None:
                item.subgoal_id = subgoal_id
            item.history.append(world)
            del item.history[: -self.max_history]
            return item
        item = SpatialLandmark(
            name=name,
            subgoal_id=subgoal_id,
            world_xyz=world,
            first_step=step,
            last_step=step,
            confidence=float(confidence),
            kind=kind,
            history=[world],
        )
        self._items.append(item)
        return item

    def for_subgoal(self, subgoal_id: Optional[str]) -> Optional[SpatialLandmark]:
        """The best-supported landmark registered for a subgoal."""
        if subgoal_id is None:
            return None
        candidates = [item for item in self._items if item.subgoal_id == subgoal_id]
        if not candidates:
            return None
        return max(candidates, key=lambda item: (item.observations, item.last_step))

    def nearest(self, position_xz: tuple[float, float]) -> Optional[SpatialLandmark]:
        if not self._items:
            return None
        return min(self._items, key=lambda item: item.distance_xz(position_xz))


__all__ = ("LandmarkRegistry", "SpatialLandmark")
