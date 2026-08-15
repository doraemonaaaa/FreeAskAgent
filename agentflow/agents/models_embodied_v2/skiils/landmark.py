"""Expose unified temporal-scene landmarks to the waypoint policy."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from agentflow.agents.models_embodied_v2.data_models import Subgoal
from agentflow.agents.models_embodied_v2.memory.temporal_memory import (
    TemporalMemory,
)
from .protocol import LandmarkOutput as _LandmarkOutput


class LandmarkTrackerMixin:
    """Adapt Captioner landmark state for navigation and debug output."""

    def _record_scene_landmark(
        self,
        image: np.ndarray,
        caption: Any,
    ) -> _LandmarkOutput:
        """Install the unified Captioner's landmark without another VLM call."""
        analyzed_id = getattr(caption, "subgoal_id", None)
        scene_landmark = getattr(caption, "landmark", None)
        self.last_landmark_normalized = None
        self.last_landmark_pixel = None
        if scene_landmark is None or analyzed_id is None:
            landmark = TemporalMemory._unknown_landmark(
                "scene captioner landmark unavailable"
            )
            self.last_landmark = landmark
            self.last_landmark_raw_response = None
            self.last_landmark_error = None
            return landmark

        if self._landmark_subgoal_id != str(analyzed_id):
            self._landmark_history.clear()
            self._landmark_subgoal_id = str(analyzed_id)
        landmark = _LandmarkOutput.model_validate(
            {
                "visible": scene_landmark.visible,
                "direction": scene_landmark.direction,
                "proximity": scene_landmark.proximity,
                "passed": scene_landmark.passed,
                "destination_dominant": (
                    scene_landmark.destination_dominant
                ),
                "u": scene_landmark.u,
                "v": scene_landmark.v,
                "confidence": scene_landmark.confidence,
                "evidence": scene_landmark.evidence,
            }
        )
        self.last_landmark = landmark
        self.last_landmark_raw_response = getattr(
            caption, "raw_response", None
        )
        self.last_landmark_error = None
        if landmark.u is not None and landmark.v is not None:
            height, width = image.shape[:2]
            self.last_landmark_normalized = (landmark.u, landmark.v)
            self.last_landmark_pixel = (
                self._scale_normalized(landmark.u, width),
                self._scale_normalized(landmark.v, height),
            )
        history_item = landmark.model_dump()
        history_item.pop("u", None)
        history_item.pop("v", None)
        recent = (
            self.temporal_memory.recent_frames()
            if isinstance(self.temporal_memory, TemporalMemory)
            else ()
        )
        if recent:
            frame = recent[-1]
            history_item.update(
                {
                    "translation_m": frame.translation_m,
                    "yaw_delta_deg": frame.yaw_delta_deg,
                    "subgoal_path_length_m": frame.subgoal_path_length_m,
                }
            )
        self._landmark_history.append(history_item)
        return landmark

    def _landmark_context_for_waypoint(
        self,
        current: Optional[Subgoal],
    ) -> str:
        if (
            current is None
            or self.last_landmark is None
            or self._landmark_subgoal_id != current.subgoal_id
        ):
            return "Current landmark state: unavailable for active subgoal."
        landmark = self.last_landmark
        return (
            "Current landmark state: "
            f"visible={landmark.visible}; direction={landmark.direction}; "
            f"proximity={landmark.proximity}; passed={landmark.passed}; "
            f"destination_dominant={landmark.destination_dominant}; "
            f"confidence={landmark.confidence:.2f}; "
            f"evidence={landmark.evidence}"
        )
