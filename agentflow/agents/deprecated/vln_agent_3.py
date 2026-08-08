"""Version-3 VLN RGB-D waypoint actor.

Version 3 keeps the proven waypoint selection and RGB-D back-projection from
``vln_agent_2`` while changing temporal understanding in two ways:

* every observation is analyzed for subgoal completion;
* the completion window contains every frame since the current subgoal began;
* a separate recent eight-frame window asks the model to diagnose error modes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import replace
import re
from typing import Any, Deque, Optional, Sequence

import numpy as np

from agentflow.agents.models_embodied_v2.TemporalCaptioner import (
    TemporalCaptioner,
)
from agentflow.agents.models_embodied_v2.data_models import (
    Subgoal,
    TemporalCaptionerConfig,
)
from agentflow.agents.models_embodied_v2.memory.task_memory import TaskMemory
from agentflow.agents.vln_agent_2 import (
    Actor as V2Actor,
    CameraIntrinsics,
    NavigationDecision,
    NavigationPoint,
)
from agentflow.agents.vln_agent_3_protocol import (
    BEHAVIOR_HISTORY_SIZE,
    CORRIDOR_LOCK_FORWARD_STEPS,
    DEFAULT_MODEL_PATH,
    ERROR_CONFIRMATION_WINDOW,
    FINAL_TARGET_EVIDENCE_VOTES,
    FINAL_TARGET_EVIDENCE_WINDOW,
    FINAL_TARGET_MAX_WAYPOINT_DEPTH_M,
    FINAL_TARGET_MIN_PATH_M,
    LANDMARK_HISTORY_SIZE,
    NavigationIntent,
    POINT_PROMPT,
    RECOVERY_LATERAL_DISTANCE_M,
    SUBGOAL_GENERATION_ATTEMPTS,
    STRUCTURED_VLM_MAX_TOKENS,
    TEMPORAL_MAX_IMAGE_EDGE,
    SUBGOAL_PROMPT,
    TURN_ALIGNMENT_DEG,
    TURN_EVIDENCE_DEG,
    LandmarkOutput as _LandmarkOutput,
)
from agentflow.agents.models_embodied_v2.memory.growing_completion_memory import (
    GrowingCompletionMemory,
)
from agentflow.agents.vln_agent_3_planning import parse_subgoal_plan
from agentflow.agents.vln_agent_3_landmark import LandmarkTrackerMixin
from agentflow.agents.vln_agent_3_waypoint import WaypointPolicyMixin


class Actor(LandmarkTrackerMixin, WaypointPolicyMixin, V2Actor):
    """Version 3 actor used by the Habitat waypoint worker."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_path=model_path, **kwargs)
        self._landmark_history: Deque[dict[str, Any]] = deque(
            maxlen=LANDMARK_HISTORY_SIZE
        )
        self._behavior_history: Deque[dict[str, Any]] = deque(
            maxlen=BEHAVIOR_HISTORY_SIZE
        )
        self._error_candidates: Deque[str] = deque(
            maxlen=ERROR_CONFIRMATION_WINDOW
        )
        self._final_target_evidence: Deque[bool] = deque(
            maxlen=FINAL_TARGET_EVIDENCE_WINDOW
        )
        self._reset_runtime_state()

    def _reset_runtime_state(self) -> None:
        """Clear all episode-local navigation and debug state."""
        self.last_subgoal_before: Optional[str] = None
        self.last_subgoal_after: Optional[str] = None
        self.last_requested_pixel: Optional[tuple[int, int]] = None
        self.last_requested_normalized: Optional[tuple[int, int]] = None
        self.last_waypoint_raw_response: Optional[str] = None
        self.last_waypoint_stop_disposition: Optional[str] = None
        self.last_waypoint_model_intent: Optional[str] = None
        self.last_waypoint_applied_intent: Optional[str] = None
        self.last_waypoint_guard_reason: Optional[str] = None
        self.last_waypoint_evidence: Optional[str] = None
        self.last_error_candidate: Optional[str] = None
        self.last_error_guard_reason: Optional[str] = None
        self.last_recovery_mode: Optional[str] = None
        self.last_landmark: Optional[_LandmarkOutput] = None
        self.last_landmark_raw_response: Optional[str] = None
        self.last_landmark_error: Optional[str] = None
        self._landmark_subgoal_id: Optional[str] = None
        self._error_candidate_subgoal: Optional[str] = None
        self._active_recovery_mode: Optional[str] = None
        self._recovery_steps_remaining = 0
        self._force_forward_this_step = False
        self._force_left_turn_this_step = False
        self._previous_position: Optional[np.ndarray] = None
        self._previous_yaw_deg: Optional[float] = None
        self._navigation_subgoal_id: Optional[str] = None
        self._navigation_phase: NavigationIntent = "FOLLOW_CORRIDOR"
        self._corridor_forward_streak = 0
        self._corridor_heading_yaw_deg: Optional[float] = None
        self._subgoal_net_yaw_deg = 0.0
        self._turn_follow_phase_started = False
        self._landmark_history.clear()
        self._behavior_history.clear()
        self._final_target_evidence.clear()
        self._error_candidates.clear()

    def act(
        self,
        rgb: Any,
        depth: Any,
        instruction: str,
        intrinsics: CameraIntrinsics | Any,
        camera_to_world: Any,
        *,
        normalized_depth: bool = False,
        depth_min_m: Optional[float] = None,
        depth_max_m: Optional[float] = None,
    ) -> NavigationDecision:
        """Run one step while retaining state transitions for debug output."""
        # These flags describe only the waypoint selected during this call.
        # A recovery selection may set one of them below; clear both before
        # delegating to V2Actor so a completed recovery cannot leak into later
        # model-selected waypoints.
        self._force_forward_this_step = False
        self._force_left_turn_this_step = False
        current = (
            self.task_memory.get_current_subgoal()
            if self.task_memory is not None
            else None
        )
        self.last_subgoal_before = (
            current.subgoal_id if current is not None else None
        )
        self._sync_navigation_phase(current)
        had_previous_pose = self._previous_position is not None
        previous_waypoint = self.last_requested_normalized
        translation_m, yaw_delta_deg = self._measure_motion(
            camera_to_world
        )
        if had_previous_pose:
            self._record_behavior(
                subgoal_id=self.last_subgoal_before,
                translation_m=translation_m,
                yaw_delta_deg=yaw_delta_deg,
                requested_waypoint=previous_waypoint,
            )
            self._update_navigation_progress(
                current,
                yaw_delta_deg=yaw_delta_deg,
            )
            self._update_corridor_lock(
                translation_m=translation_m,
                yaw_delta_deg=yaw_delta_deg,
            )
        landmark = self._track_landmark(
            self._as_rgb_array(rgb),
            current,
            translation_m=translation_m,
            yaw_delta_deg=yaw_delta_deg,
        )
        if isinstance(self.temporal_memory, GrowingCompletionMemory):
            self.temporal_memory.set_motion_evidence(
                translation_m=translation_m,
                yaw_delta_deg=yaw_delta_deg,
            )
            self.temporal_memory.set_landmark_evidence(landmark)
        try:
            decision = super().act(
                rgb,
                depth,
                instruction,
                intrinsics,
                camera_to_world,
                normalized_depth=normalized_depth,
                depth_min_m=depth_min_m,
                depth_max_m=depth_max_m,
            )
        except ValueError as exc:
            if str(exc) != (
                "Depth observation contains no valid walkable waypoint."
            ):
                raise
            # At very close range every lower-image depth value can be
            # invalid. That is an action-level recovery condition, not a
            # reason to abort the episode. Return a synthetic lateral target;
            # the Habitat runner turns when its follower rejects this point.
            image = self._as_rgb_array(rgb)
            transform = np.asarray(camera_to_world, dtype=np.float64)
            position = transform[:3, 3]
            self._force_left_turn_this_step = True
            self.last_recovery_mode = "NO_VALID_DEPTH"
            self.last_waypoint_guard_reason = (
                "forced turn because depth has no valid walkable waypoint"
            )
            self.last_waypoint_evidence = (
                "action-level NO_VALID_DEPTH recovery: turn away from the "
                "near-field obstacle"
            )
            self.last_requested_normalized = (250, 500)
            self.last_requested_pixel = (
                int(image.shape[1] * 0.25),
                int(image.shape[0] * 0.50),
            )
            decision = NavigationDecision(
                stop=False,
                point=NavigationPoint(
                    pixel_uv=self.last_requested_pixel,
                    depth_m=0.0,
                    camera_xyz=(0.0, 0.0, -1.0),
                    world_xyz=tuple(float(value) for value in position),
                ),
                raw_response=(
                    f"{self.last_model_response}; no-valid-depth recovery"
                ),
            )
        if (
            self._force_forward_this_step
            and not decision.stop
            and decision.point is not None
        ):
            transform = np.asarray(camera_to_world, dtype=np.float64)
            decision = replace(
                decision,
                point=replace(
                    decision.point,
                    world_xyz=(
                        float(transform[0, 3]),
                        decision.point.world_xyz[1],
                        float(transform[2, 3]),
                    ),
                ),
                raw_response=(
                    f"{decision.raw_response}; force-forward waypoint"
                ),
            )
        elif (
            self._force_left_turn_this_step
            and not decision.stop
            and decision.point is not None
        ):
            transform = np.asarray(camera_to_world, dtype=np.float64)
            camera_right = transform[:3, 0]
            lateral = camera_right[[0, 2]]
            lateral_norm = float(np.linalg.norm(lateral))
            if lateral_norm > 1e-6:
                lateral = lateral / lateral_norm
            position = transform[:3, 3]
            decision = replace(
                decision,
                point=replace(
                    decision.point,
                    world_xyz=(
                        float(
                            position[0]
                            - RECOVERY_LATERAL_DISTANCE_M * lateral[0]
                        ),
                        decision.point.world_xyz[1],
                        float(
                            position[2]
                            - RECOVERY_LATERAL_DISTANCE_M * lateral[1]
                        ),
                    ),
                ),
                raw_response=(
                    f"{decision.raw_response}; force-left-turn waypoint"
                ),
            )
        current = (
            self.task_memory.get_current_subgoal()
            if self.task_memory is not None
            else None
        )
        self.last_subgoal_after = (
            current.subgoal_id if current is not None else None
        )
        if self._should_stop_from_final_target_evidence(
            decision,
            current,
        ):
            self.last_waypoint_stop_disposition = (
                "accepted_final_evidence_guard"
            )
            return NavigationDecision(
                stop=True,
                raw_response=(
                    f"{decision.raw_response}; final target evidence guard"
                ),
            )
        return decision

    def _should_stop_from_final_target_evidence(
        self,
        decision: NavigationDecision,
        current: Optional[Subgoal],
    ) -> bool:
        """Stop after repeated model-grounded near-field target evidence."""
        if (
            decision.stop
            or decision.point is None
            or current is None
            or not self.subgoals
            or current.subgoal_id != self.subgoals[-1].subgoal_id
            or self._navigation_phase != "FINAL_APPROACH"
            or self.last_waypoint_applied_intent != "FINAL_APPROACH"
        ):
            self._final_target_evidence.clear()
            return False
        evidence = (self.last_waypoint_evidence or "").lower()
        final_subgoal = self.subgoals[-1]
        final_text = (
            f"{final_subgoal.description} "
            f"{final_subgoal.completion_criteria}"
        ).lower()
        ignored_words = {
            "agent", "and", "before", "camera", "complete", "completion",
            "destination", "final", "has", "have", "into", "positioned",
            "reach", "reached", "stop", "stopped", "stopping", "target",
            "that", "the", "this", "with",
        }
        goal_words = {
            word
            for word in re.findall(r"[a-z][a-z-]+", final_text)
            if len(word) >= 4 and word not in ignored_words
        }
        evidence_words = set(re.findall(r"[a-z][a-z-]+", evidence))
        model_identifies_target = bool(
            (
                re.search(r"\b(target|destination)\b", evidence)
                or goal_words.intersection(evidence_words)
            )
            and re.search(
                r"\b(visible|appears?|near|close|beside|reached|inside|"
                r"on|in front of|next to)\b",
                evidence,
            )
        )
        near_field = (
            decision.point.depth_m
            <= FINAL_TARGET_MAX_WAYPOINT_DEPTH_M
        )
        path_length_m = 0.0
        if isinstance(self.temporal_memory, GrowingCompletionMemory):
            frames = self.temporal_memory.recent_frames()
            if frames:
                path_length_m = frames[-1].subgoal_path_length_m
        supported = (
            model_identifies_target
            and near_field
            and path_length_m >= FINAL_TARGET_MIN_PATH_M
        )
        self._final_target_evidence.append(supported)
        return (
            self._final_target_evidence.count(True)
            >= FINAL_TARGET_EVIDENCE_VOTES
        )

    def _measure_motion(
        self,
        camera_to_world: Any,
    ) -> tuple[float, float]:
        transform = np.asarray(camera_to_world, dtype=np.float64)
        if transform.shape != (4, 4):
            raise ValueError(
                "camera_to_world must be a 4x4 camera transform."
            )
        position = transform[:3, 3].copy()
        forward = -transform[:3, 2]
        yaw_deg = float(
            np.degrees(np.arctan2(forward[0], -forward[2]))
        )
        if self._previous_position is None:
            translation_m = 0.0
            yaw_delta_deg = 0.0
        else:
            translation_m = float(
                np.linalg.norm(
                    position[[0, 2]]
                    - self._previous_position[[0, 2]]
                )
            )
            assert self._previous_yaw_deg is not None
            yaw_delta_deg = (
                yaw_deg - self._previous_yaw_deg + 180.0
            ) % 360.0 - 180.0
        self._previous_position = position
        self._previous_yaw_deg = yaw_deg
        return translation_m, yaw_delta_deg

    def _phase_for_subgoal(
        self,
        subgoal: Optional[Subgoal],
    ) -> NavigationIntent:
        if subgoal is None:
            return "STOP"
        description = subgoal.description.lower()
        if re.search(r"\bturn\s+left\b", description):
            return "TURN_LEFT"
        if re.search(r"\bturn\s+right\b", description):
            return "TURN_RIGHT"
        arrival_text = (
            f"{subgoal.description} {subgoal.completion_criteria}"
        ).lower()
        if re.search(
            r"\b(stop|final|destination|arrive|arrival|beside)\b|"
            r"\b(?:next to|in front of)\b",
            arrival_text,
        ):
            return "FINAL_APPROACH"
        if re.search(r"\b(straight|hallway|corridor)\b", description):
            return "FOLLOW_CORRIDOR"
        return "APPROACH_LANDMARK"

    def _sync_navigation_phase(
        self,
        subgoal: Optional[Subgoal],
    ) -> None:
        subgoal_id = subgoal.subgoal_id if subgoal is not None else None
        if subgoal_id != self._navigation_subgoal_id:
            self._navigation_subgoal_id = subgoal_id
            self._corridor_forward_streak = 0
            self._corridor_heading_yaw_deg = None
            self._subgoal_net_yaw_deg = 0.0
            self._turn_follow_phase_started = False
        self._navigation_phase = (
            "FOLLOW_CORRIDOR"
            if self._turn_follow_phase_started
            else self._phase_for_subgoal(subgoal)
        )

    def _update_navigation_progress(
        self,
        subgoal: Optional[Subgoal],
        *,
        yaw_delta_deg: float,
    ) -> None:
        if subgoal is None:
            return
        self._subgoal_net_yaw_deg += yaw_delta_deg
        initial_phase = self._phase_for_subgoal(subgoal)
        if initial_phase not in ("TURN_LEFT", "TURN_RIGHT"):
            return
        if not re.search(
            r"\bnext\s+(?:left|right)-turn\s+decision\s+point\b",
            subgoal.description,
            flags=re.IGNORECASE,
        ):
            return
        turn_progress = (
            -self._subgoal_net_yaw_deg
            if initial_phase == "TURN_LEFT"
            else self._subgoal_net_yaw_deg
        )
        if turn_progress < TURN_ALIGNMENT_DEG:
            return
        if not self._turn_follow_phase_started:
            self._turn_follow_phase_started = True
            self._corridor_forward_streak = 0
            self._corridor_heading_yaw_deg = None
        self._navigation_phase = "FOLLOW_CORRIDOR"

    def _update_corridor_lock(
        self,
        *,
        translation_m: float,
        yaw_delta_deg: float,
    ) -> None:
        if self._navigation_phase != "FOLLOW_CORRIDOR":
            self._corridor_forward_streak = 0
            return
        if (
            translation_m >= 0.10
            and abs(yaw_delta_deg) < TURN_EVIDENCE_DEG
        ):
            self._corridor_forward_streak += 1
        elif self._corridor_heading_yaw_deg is None:
            self._corridor_forward_streak = 0
        if (
            self._corridor_heading_yaw_deg is None
            and self._corridor_forward_streak
            >= CORRIDOR_LOCK_FORWARD_STEPS
            and self._previous_yaw_deg is not None
        ):
            self._corridor_heading_yaw_deg = self._previous_yaw_deg

    def _next_subgoal(
        self,
        current: Optional[Subgoal],
    ) -> Optional[Subgoal]:
        if current is None:
            return None
        for index, item in enumerate(self.subgoals):
            if item.subgoal_id == current.subgoal_id:
                next_index = index + 1
                return (
                    self.subgoals[next_index]
                    if next_index < len(self.subgoals)
                    else None
                )
        return None

    def _landmark_is_near_for_current_subgoal(
        self,
        current: Optional[Subgoal],
    ) -> bool:
        return bool(
            current is not None
            and self.last_landmark is not None
            and self._landmark_subgoal_id == current.subgoal_id
            and self.last_landmark.visible
            and self.last_landmark.proximity in ("NEAR", "AT")
            and self.last_landmark.confidence >= 0.6
        )

    def _record_behavior(
        self,
        *,
        subgoal_id: Optional[str],
        translation_m: float,
        yaw_delta_deg: float,
        requested_waypoint: Optional[tuple[int, int]],
    ) -> None:
        if translation_m >= 0.10:
            behavior = "MOVE_FORWARD"
        elif yaw_delta_deg <= -TURN_EVIDENCE_DEG:
            behavior = "TURN_LEFT"
        elif yaw_delta_deg >= TURN_EVIDENCE_DEG:
            behavior = "TURN_RIGHT"
        else:
            behavior = "NO_MOTION"
        self._behavior_history.append(
            {
                "subgoal_id": subgoal_id,
                "behavior": behavior,
                "translation_m": translation_m,
                "yaw_delta_deg": yaw_delta_deg,
                "requested_waypoint": requested_waypoint,
            }
        )

    def behavior_history(self) -> tuple[dict[str, Any], ...]:
        """Return recent measured outcomes of previously requested waypoints."""
        return tuple(dict(item) for item in self._behavior_history)

    def _behavior_context(self) -> str:
        if not self._behavior_history:
            return "Recent waypoint behavior: none (first observation)."
        lines = ["Recent waypoint behavior, oldest first:"]
        for item in self._behavior_history:
            waypoint = item["requested_waypoint"]
            waypoint_text = (
                "none"
                if waypoint is None
                else f"({waypoint[0]},{waypoint[1]})"
            )
            lines.append(
                f"- subgoal={item['subgoal_id']}; "
                f"behavior={item['behavior']}; "
                f"translation_m={item['translation_m']:.3f}; "
                f"yaw_delta_deg={item['yaw_delta_deg']:+.1f}; "
                f"previous_normalized_waypoint={waypoint_text}"
            )
        return "\n".join(lines)

    def prepare_task(self, instruction: str) -> tuple[Subgoal, ...]:
        """Generate and strictly validate an ordered subgoal plan."""
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a non-empty string.")

        normalized_instruction = instruction.strip()
        last_error: Optional[Exception] = None
        response_text = ""
        for attempt in range(SUBGOAL_GENERATION_ATTEMPTS):
            if attempt == 0:
                content = [
                    f"Navigation instruction: {normalized_instruction}"
                ]
            else:
                content = [
                    f"Navigation instruction: {normalized_instruction}\n"
                    "The previous response failed strict validation. Return "
                    "only the exact requested JSON schema with unique "
                    "consecutive string IDs starting at \"1\"."
                ]
            response = self.llm(
                content,
                system_prompt=SUBGOAL_PROMPT,
                max_tokens=512,
                temperature=0,
            )
            response_text = str(response)
            self.last_subgoal_response = response_text
            try:
                subgoals = parse_subgoal_plan(
                    response_text,
                    extract_json=self._extract_json_object,
                    instruction=normalized_instruction,
                )
                break
            except Exception as exc:
                last_error = exc
        else:
            raise ValueError(
                "Actor returned invalid subgoal JSON after "
                f"{SUBGOAL_GENERATION_ATTEMPTS} attempts: {response_text!r}"
            ) from last_error

        self.task_instruction = normalized_instruction
        self.subgoals = subgoals
        self.reset_memory(normalized_instruction, subgoals)
        return tuple(self.subgoals)

    def reset_memory(
        self,
        instruction: str,
        subgoals: Sequence[Subgoal],
    ) -> None:
        """Initialize task state and the growing completion memory."""
        if self.task_memory is None:
            self.task_memory = TaskMemory(instruction, subgoals=subgoals)
        else:
            self.task_memory.reset(goal=instruction, subgoals=subgoals)

        if self.temporal_memory is None:
            self.temporal_memory = GrowingCompletionMemory(
                captioner=TemporalCaptioner(
                    engine=self.llm,
                    config=TemporalCaptionerConfig(
                        enable_error_detection=True,
                        min_error_detection_frames=8,
                        max_image_edge=TEMPORAL_MAX_IMAGE_EDGE,
                        # The error schema includes free-text evidence.  The
                        # shared 64-token structured budget can truncate that
                        # JSON before its closing brace.
                        max_tokens=max(128, STRUCTURED_VLM_MAX_TOKENS),
                    ),
                ),
                task_memory=self.task_memory,
            )
        else:
            self.temporal_memory.reset()
        self._reset_runtime_state()


VLNAgent = Actor

__all__ = (
    "Actor",
    "CameraIntrinsics",
    "DEFAULT_MODEL_PATH",
    "GrowingCompletionMemory",
    "NavigationDecision",
    "NavigationPoint",
    "POINT_PROMPT",
    "Subgoal",
    "SUBGOAL_PROMPT",
    "VLNAgent",
)
