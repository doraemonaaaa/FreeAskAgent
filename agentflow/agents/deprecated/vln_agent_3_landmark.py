"""Visual landmark tracking policy used by VLN agent v3."""

from __future__ import annotations

import json
import re
from typing import Any, Optional

import numpy as np

from agentflow.agents.models_embodied_v2.data_models import Subgoal
from agentflow.agents.models_embodied_v2.memory.growing_completion_memory import (
    GrowingCompletionMemory,
)
from agentflow.agents.vln_agent_3_protocol import (
    DOORWAY_CROSSING_M,
    DOORWAY_LANDMARK_CONFIDENCE,
    DOORWAY_MAX_CROSSING_YAW_DEG,
    DOORWAY_NEAR_CROSSING_M,
    DOORWAY_SIDE_CROSSING_M,
    LANDMARK_GENERATION_ATTEMPTS,
    LANDMARK_PROMPT,
    LandmarkOutput as _LandmarkOutput,
    STRUCTURED_VLM_MAX_TOKENS,
    VLM_IMAGE_MAX_PIXELS,
    VLM_IMAGE_MIN_PIXELS,
)


class LandmarkTrackerMixin:
    """Model-backed landmark state plus motion-grounded crossing checks."""

    def _track_landmark(
        self,
        image: np.ndarray,
        subgoal: Optional[Subgoal],
        *,
        translation_m: float,
        yaw_delta_deg: float,
    ) -> _LandmarkOutput:
        if subgoal is None:
            landmark = GrowingCompletionMemory._unknown_landmark(
                "no active subgoal"
            )
            self.last_landmark = landmark
            self.last_landmark_raw_response = None
            self.last_landmark_error = None
            return landmark
    
        if self._landmark_subgoal_id != subgoal.subgoal_id:
            self._landmark_history.clear()
            self._landmark_subgoal_id = subgoal.subgoal_id
        history = (
            json.dumps(
                list(self._landmark_history),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            if self._landmark_history
            else "none"
        )
        next_subgoal = self._next_subgoal(subgoal)
        next_stage = (
            "none (active subgoal is final)"
            if next_subgoal is None
            else (
                f"{next_subgoal.description}; completion criterion: "
                f"{next_subgoal.completion_criteria}"
            )
        )
        path_length_m = translation_m
        if isinstance(self.temporal_memory, GrowingCompletionMemory):
            prior_frames = self.temporal_memory.recent_frames()
            if prior_frames:
                path_length_m += prior_frames[-1].subgoal_path_length_m
        prompt = (
            f"Active subgoal: {subgoal.description}\n"
            f"Completion criterion: {subgoal.completion_criteria}\n"
            f"Next route stage: {next_stage}\n"
            f"Measured path length in active subgoal: "
            f"{path_length_m:.3f} m\n"
            f"Current measured motion: translation_m={translation_m:.3f}, "
            f"yaw_delta_deg={yaw_delta_deg:+.1f}\n"
            f"Recent landmark states: {history}\n"
            f"{self._behavior_context()}"
        )
        last_error: Optional[Exception] = None
        response: Any = ""
        for attempt in range(LANDMARK_GENERATION_ATTEMPTS):
            attempt_prompt = prompt
            if attempt:
                attempt_prompt += (
                    "\nThe previous response failed strict validation. "
                    "Return every required field with the exact schema; an "
                    "invisible unpassed target must use UNKNOWN direction "
                    "and UNKNOWN proximity."
                )
            response = self.llm(
                [attempt_prompt, self._rgb_to_png(image)],
                system_prompt=LANDMARK_PROMPT,
                image_min_pixels=VLM_IMAGE_MIN_PIXELS,
                image_max_pixels=VLM_IMAGE_MAX_PIXELS,
                max_tokens=STRUCTURED_VLM_MAX_TOKENS,
                temperature=0,
            )
            self.last_landmark_raw_response = str(response)
            try:
                landmark = _LandmarkOutput.model_validate(
                    self._extract_json_object(
                        self.last_landmark_raw_response
                    )
                )
                doorway_stage = bool(
                    re.search(
                        r"\b(?:exit|leave|doorway|threshold|"
                        r"cross(?:ed|ing)?)\b",
                        (
                            f"{subgoal.description} "
                            f"{subgoal.completion_criteria}"
                        ),
                        flags=re.IGNORECASE,
                    )
                )
                if doorway_stage:
                    crossing_evidence = self._verified_doorway_crossing(
                        landmark,
                        translation_m=translation_m,
                        yaw_delta_deg=yaw_delta_deg,
                    )
                    if crossing_evidence is not None:
                        landmark = landmark.model_copy(
                            update={
                                "passed": True,
                                "confidence": max(
                                    landmark.confidence,
                                    DOORWAY_LANDMARK_CONFIDENCE,
                                ),
                                "evidence": (
                                    f"{landmark.evidence}; "
                                    f"{crossing_evidence}"
                                ),
                            }
                        )
                    elif landmark.passed:
                        landmark = landmark.model_copy(
                            update={
                                "passed": False,
                                "evidence": (
                                    f"{landmark.evidence}; rejected passed "
                                    "claim because measured doorway-crossing "
                                    "sequence is absent"
                                ),
                            }
                        )
                self.last_landmark_error = None
                break
            except Exception as exc:
                last_error = exc
        else:
            assert last_error is not None
            self.last_landmark_error = (
                f"{type(last_error).__name__}: {last_error}"
            )
            landmark = GrowingCompletionMemory._unknown_landmark(
                "tracker output unavailable"
            )
        self.last_landmark = landmark
        history_item = landmark.model_dump()
        history_item.update(
            {
                "translation_m": translation_m,
                "yaw_delta_deg": yaw_delta_deg,
                "subgoal_path_length_m": path_length_m,
            }
        )
        self._landmark_history.append(history_item)
        return landmark
    
    def _verified_doorway_crossing(
        self,
        landmark: _LandmarkOutput,
        *,
        translation_m: float,
        yaw_delta_deg: float,
    ) -> Optional[str]:
        """Require ordered threshold motion instead of repeated frame votes."""
        if not landmark.destination_dominant:
            return None
    
        history = tuple(self._landmark_history)
        for index in range(len(history) - 1, -1, -1):
            approach = history[index]
            if (
                not bool(approach.get("visible"))
                or approach.get("direction") != "CENTER"
                or approach.get("proximity") not in {"NEAR", "AT"}
                or float(approach.get("confidence", 0.0))
                < DOORWAY_LANDMARK_CONFIDENCE
            ):
                continue
    
            motion = history[index + 1 :]
            crossing_translation = sum(
                float(item.get("translation_m", 0.0))
                for item in motion
            ) + translation_m
            crossing_yaw = sum(
                abs(float(item.get("yaw_delta_deg", 0.0)))
                for item in motion
            ) + abs(yaw_delta_deg)
            required_translation = (
                DOORWAY_CROSSING_M
                if approach.get("proximity") == "AT"
                else DOORWAY_NEAR_CROSSING_M
            )
            if (
                crossing_translation >= required_translation
                and crossing_yaw <= DOORWAY_MAX_CROSSING_YAW_DEG
            ):
                return (
                    "verified doorway crossing from "
                    f"{approach['proximity']} after advancing "
                    f"{crossing_translation:.2f}m with "
                    f"{crossing_yaw:.1f}deg accumulated yaw into a "
                    "destination-dominant view"
                )
    
        # A doorway can remain off-center while the low-level controller
        # follows an oblique but valid path through it. In that case, require
        # the destination to remain dominant while the doorway changes sides
        # across substantial measured translation. Pure camera rotation does
        # not satisfy this branch.
        if landmark.direction in {"LEFT", "RIGHT"}:
            opposite = "RIGHT" if landmark.direction == "LEFT" else "LEFT"
            for index in range(len(history) - 1, -1, -1):
                approach = history[index]
                if (
                    not bool(approach.get("visible"))
                    or not bool(approach.get("destination_dominant"))
                    or approach.get("direction") != opposite
                    or float(approach.get("confidence", 0.0))
                    < DOORWAY_LANDMARK_CONFIDENCE
                ):
                    continue
                motion = history[index + 1 :]
                crossing_translation = sum(
                    float(item.get("translation_m", 0.0))
                    for item in motion
                ) + translation_m
                crossing_yaw = sum(
                    abs(float(item.get("yaw_delta_deg", 0.0)))
                    for item in motion
                ) + abs(yaw_delta_deg)
                if (
                    crossing_translation >= DOORWAY_SIDE_CROSSING_M
                    and crossing_yaw <= DOORWAY_MAX_CROSSING_YAW_DEG
                ):
                    return (
                        "verified oblique doorway crossing from side "
                        f"transition {opposite}->{landmark.direction} after "
                        f"advancing {crossing_translation:.2f}m with "
                        f"{crossing_yaw:.1f}deg accumulated yaw while the "
                        "destination remained dominant"
                    )
        return None
    
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
    
