from dataclasses import dataclass
from typing import Any

@dataclass
class MotionGoal:
    command: str              # Forward / TurnLeft / TurnRight / Wait / Ask
    value: float | str | None
    subgoal_text: str
    verifier_hint: str | None
    frame: str = "ego"

@dataclass
class Observation:
    rgbd: Any
    pose: Any | None = None
    local_obstacles: Any | None = None
    timestamp: float | None = None

@dataclass
class Trajectory:
    poses: list[tuple[float, float, float]]
    score: float | None = None

@dataclass
class ExecutionResult:
    success: bool
    trajectory: Trajectory | None
    message: str
    blocked: bool = False
