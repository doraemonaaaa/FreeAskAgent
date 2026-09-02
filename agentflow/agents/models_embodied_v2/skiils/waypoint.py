"""Motion-grounded loop detection shared by the VLN agent's target locks."""

from __future__ import annotations

from typing import Optional

from agentflow.agents.models_embodied_v2.memory.temporal_memory import (
    TemporalMemory,
)
from .protocol import (
    LANDMARK_STEER_MIN_CONFIDENCE,
    STALL_EVIDENCE_FRAMES,
    STALL_TRANSLATION_LIMIT_M,
    TURN_EVIDENCE_DEG,
)


class WaypointPolicyMixin:
    """Measured-motion evidence for releasing a held target."""

    def _landmark_located_for_current_subgoal(self, current) -> bool:
        landmark = getattr(self, "last_landmark", None)
        return bool(
            current is not None
            and landmark is not None
            and getattr(self, "_landmark_subgoal_id", None) == current.subgoal_id
            and landmark.visible
            and landmark.u is not None
            and landmark.confidence >= LANDMARK_STEER_MIN_CONFIDENCE
        )

    
    def _motion_grounded_error_candidate(
        self,
        error_mode: str,
    ) -> tuple[Optional[str], str]:
        if not isinstance(self.temporal_memory, TemporalMemory):
            return None, "motion evidence unavailable"
        recent = self.temporal_memory.recent_frames()[
            -STALL_EVIDENCE_FRAMES:
        ]
        if len(recent) < STALL_EVIDENCE_FRAMES:
            return None, "fewer than four measured motion intervals"
        translation = sum(frame.translation_m for frame in recent)
        if translation > STALL_TRANSLATION_LIMIT_M:
            return (
                None,
                f"rejected: recent translation {translation:.2f}m "
                f"exceeds {STALL_TRANSLATION_LIMIT_M:.2f}m",
            )
    
        turns = [
            frame.yaw_delta_deg
            for frame in recent
            if abs(frame.yaw_delta_deg) >= TURN_EVIDENCE_DEG
        ]
        positive = sum(value > 0 for value in turns)
        negative = sum(value < 0 for value in turns)
        if positive > 0 and negative > 0:
            return (
                "TURN_OSCILLATION",
                "motion-grounded as TURN_OSCILLATION from stalled "
                "translation and both yaw signs",
            )
        if max(positive, negative) >= 3:
            return (
                "IN_PLACE_SPIN",
                "motion-grounded as IN_PLACE_SPIN from stalled translation "
                "and same-sign yaw",
            )
        if len(turns) <= 1 and error_mode in ("WALL_STUCK", "GET_NOWHERE"):
            return (
                error_mode,
                f"motion-grounded as {error_mode} from stalled translation "
                "with little rotation",
            )
        return None, "rejected: measured yaw pattern does not support error"
    
    
    @staticmethod
    def _scale_normalized(value: int, size: int) -> int:
        if size < 1:
            raise ValueError("image dimensions must be positive")
        return int(value * (size - 1) / 1000.0 + 0.5)
    
