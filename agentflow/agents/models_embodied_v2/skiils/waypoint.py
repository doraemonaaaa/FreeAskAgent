"""Waypoint selection and motion-grounded recovery for VLN agent v3."""

from __future__ import annotations

import re
from typing import Any, Optional

import numpy as np

from agentflow.agents.models_embodied_v2.memory.growing_completion_memory import (
    GrowingCompletionMemory,
)
from .protocol import (
    CORRIDOR_WAYPOINT_DEVIATION,
    ERROR_CONFIDENCE_THRESHOLD,
    ERROR_CONFIRMATION_VOTES,
    POINT_PROMPT,
    RECOVERY_FORWARD_U,
    RECOVERY_FORWARD_V,
    RECOVERY_HOLD_STEPS,
    RECOVERY_TURN_U,
    RECOVERY_TURN_V,
    STALL_EVIDENCE_FRAMES,
    STALL_TRANSLATION_LIMIT_M,
    STRUCTURED_VLM_MAX_TOKENS,
    TURN_EVIDENCE_DEG,
    VLM_IMAGE_MAX_PIXELS,
    VLM_IMAGE_MIN_PIXELS,
    WAYPOINT_GENERATION_ATTEMPTS,
    NavigationIntent,
    WaypointOutput as _WaypointOutput,
)


class WaypointPolicyMixin:
    """Validated waypoint inference, guards, and deterministic recovery."""

    def _recovery_mode_for_step(self) -> Optional[str]:
        caption = self.last_caption
        if caption is None:
            self.last_error_candidate = None
            self.last_error_guard_reason = "captioner did not return a result"
            return self._consume_active_recovery()
        if caption.completed:
            self._error_candidates.clear()
            self._active_recovery_mode = None
            self._recovery_steps_remaining = 0
            self.last_error_candidate = None
            self.last_error_guard_reason = "subgoal completed"
            return None
    
        if self._error_candidate_subgoal != caption.subgoal_id:
            self._error_candidates.clear()
            self._active_recovery_mode = None
            self._recovery_steps_remaining = 0
            self._error_candidate_subgoal = caption.subgoal_id
        if self._recovery_steps_remaining > 0:
            self.last_error_candidate = self._active_recovery_mode
            self.last_error_guard_reason = "continuing confirmed recovery"
            return self._consume_active_recovery()
    
        candidate = "NONE"
        if not caption.error:
            self.last_error_guard_reason = "model reported no error"
        elif caption.error_confidence < ERROR_CONFIDENCE_THRESHOLD:
            self.last_error_guard_reason = (
                f"confidence {caption.error_confidence:.2f} below "
                f"{ERROR_CONFIDENCE_THRESHOLD:.2f}"
            )
        else:
            grounded_mode, reason = self._motion_grounded_error_candidate(
                caption.error_mode
            )
            self.last_error_guard_reason = reason
            if grounded_mode is not None:
                candidate = grounded_mode
        self.last_error_candidate = candidate
        self._error_candidates.append(candidate)
        modes = (
            "WALL_STUCK",
            "TURN_OSCILLATION",
            "IN_PLACE_SPIN",
            "GET_NOWHERE",
        )
        confirmed = next(
            (
                mode
                for mode in modes
                if self._error_candidates.count(mode)
                >= ERROR_CONFIRMATION_VOTES
            ),
            None,
        )
        if confirmed is not None:
            self._active_recovery_mode = confirmed
            self._recovery_steps_remaining = RECOVERY_HOLD_STEPS
    
        return self._consume_active_recovery()
    
    def _consume_active_recovery(self) -> Optional[str]:
        if self._recovery_steps_remaining <= 0:
            return None
        mode = self._active_recovery_mode
        self._recovery_steps_remaining -= 1
        if self._recovery_steps_remaining == 0:
            self._active_recovery_mode = None
            self._error_candidates.clear()
        return mode
    
    def _motion_grounded_error_candidate(
        self,
        error_mode: str,
    ) -> tuple[Optional[str], str]:
        if not isinstance(self.temporal_memory, GrowingCompletionMemory):
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
    
    def _select_pixel(
        self,
        image: np.ndarray,
        instruction: str,
        *,
        subgoal_context: str = "",
    ) -> Optional[tuple[int, int]]:
        """Defer model STOP until memory or repeated near-target evidence."""
        self.last_requested_pixel = None
        self.last_requested_normalized = None
        self.last_waypoint_raw_response = None
        self.last_waypoint_stop_disposition = None
        self.last_waypoint_model_intent = None
        self.last_waypoint_applied_intent = None
        self.last_waypoint_guard_reason = None
        self.last_waypoint_evidence = None
        recovery_mode = self._recovery_mode_for_step()
        self.last_recovery_mode = recovery_mode
        current = (
            self.task_memory.get_current_subgoal()
            if self.task_memory is not None
            else None
        )
        self._sync_navigation_phase(current)
        height, width = image.shape[:2]
        if recovery_mode is not None:
            intent: NavigationIntent = (
                self._navigation_phase
                if self._navigation_phase != "STOP"
                else "APPROACH_LANDMARK"
            )
            fixed_mode = recovery_mode
            force_forward = recovery_mode in (
                "TURN_OSCILLATION",
                "IN_PLACE_SPIN",
            )
            evidence = (
                f"action-level {fixed_mode} recovery: "
                + (
                    "hold a stable lower-center forward waypoint"
                    if force_forward
                    else "hold a stable left-side turn waypoint"
                )
            )
            synthetic = _WaypointOutput(
                stop=False,
                intent=intent,
                u=RECOVERY_FORWARD_U if force_forward else RECOVERY_TURN_U,
                v=RECOVERY_FORWARD_V if force_forward else RECOVERY_TURN_V,
                confidence=1.0,
                evidence=evidence,
            )
            self.last_model_response = synthetic.model_dump_json()
            self.last_waypoint_raw_response = self.last_model_response
            self.last_waypoint_model_intent = intent
            self.last_waypoint_applied_intent = intent
            self.last_waypoint_evidence = evidence
            self.last_waypoint_guard_reason = (
                "forced deterministic action waypoint for "
                f"{fixed_mode}"
            )
            self.last_requested_normalized = (
                synthetic.u,
                synthetic.v,
            )
            self.last_requested_pixel = (
                self._scale_normalized(synthetic.u, width),
                self._scale_normalized(synthetic.v, height),
            )
            self._force_forward_this_step = force_forward
            self._force_left_turn_this_step = not force_forward
            return self.last_requested_pixel
        prompt = "\n".join(
            (
                subgoal_context
                or f"Navigation instruction: {instruction}",
                f"Full route instruction: {instruction}",
                f"Required navigation phase: {self._navigation_phase}.",
                (
                    "Corridor heading lock: active; preserve the established "
                    "forward direction."
                    if self._corridor_heading_yaw_deg is not None
                    and self._navigation_phase == "FOLLOW_CORRIDOR"
                    else "Corridor heading lock: not active."
                ),
                self._landmark_context_for_waypoint(current),
                self._behavior_context(),
                "Use the current image as the source of truth. Landmark and "
                "behavior histories are supporting context.",
                "Return normalized coordinates in the fixed 0..1000 "
                "coordinate system.",
                f"Displayed image width: {width}; height: {height}.",
            )
        )
        last_error: Optional[Exception] = None
        response: Any = ""
        for attempt in range(WAYPOINT_GENERATION_ATTEMPTS):
            attempt_prompt = prompt
            if attempt:
                attempt_prompt += (
                    "\nThe previous response failed strict validation. "
                    "Return every required field with the exact schema."
                )
            response = self.llm(
                [attempt_prompt, self._rgb_to_png(image)],
                system_prompt=POINT_PROMPT,
                image_min_pixels=VLM_IMAGE_MIN_PIXELS,
                image_max_pixels=VLM_IMAGE_MAX_PIXELS,
                max_tokens=STRUCTURED_VLM_MAX_TOKENS,
                temperature=0,
            )
            self.last_model_response = str(response)
            self.last_waypoint_raw_response = self.last_model_response
            try:
                payload = self._extract_json_object(
                    self.last_model_response
                )
                waypoint = _WaypointOutput.model_validate(payload)
                break
            except Exception as exc:
                last_error = exc
        else:
            fallback_intent: NavigationIntent = (
                self._navigation_phase
                if self._navigation_phase != "STOP"
                else "FINAL_APPROACH"
            )
            waypoint = _WaypointOutput(
                stop=False,
                intent=fallback_intent,
                u=500,
                v=750,
                confidence=0.0,
                evidence=(
                    "safe lower-center fallback after invalid waypoint "
                    "model output"
                ),
            )
            self.last_waypoint_guard_reason = (
                "invalid waypoint JSON after "
                f"{WAYPOINT_GENERATION_ATTEMPTS} attempts; applied safe "
                f"fallback ({type(last_error).__name__}: {last_error})"
            )
            self.last_model_response = (
                f"{response!s} [invalid waypoint JSON; safe fallback]"
            )
    
        self.last_waypoint_model_intent = waypoint.intent
        self.last_waypoint_applied_intent = waypoint.intent
        self.last_waypoint_evidence = waypoint.evidence
        required_turn = None
        if current is not None:
            turn_match = re.search(
                r"\bnext\s+(left|right)-turn\s+decision\s+point\b",
                current.description,
                flags=re.IGNORECASE,
            )
            if turn_match is not None:
                required_turn = f"TURN_{turn_match.group(1).upper()}"
        evidence_marks_decision = bool(
            re.search(
                r"\b(junction|decision point|turn point)\b",
                waypoint.evidence,
                flags=re.IGNORECASE,
            )
            and re.search(
                r"\b(visible|near|at|reached|arrived)\b",
                waypoint.evidence,
                flags=re.IGNORECASE,
            )
        )
        allow_required_turn = bool(
            waypoint.intent == required_turn
            and (
                self._landmark_is_near_for_current_subgoal(current)
                or evidence_marks_decision
            )
        )
        if (
            not waypoint.stop
            and self._navigation_phase == "FOLLOW_CORRIDOR"
            and (
                self._corridor_heading_yaw_deg is not None
                or self._turn_follow_phase_started
            )
            and recovery_mode is None
        ):
            assert waypoint.u is not None
            if (
                waypoint.intent != "FOLLOW_CORRIDOR"
                or abs(waypoint.u - 500)
                > CORRIDOR_WAYPOINT_DEVIATION
            ):
                if allow_required_turn:
                    self.last_waypoint_guard_reason = (
                        "released corridor lock at the measured next-turn "
                        "decision point"
                    )
                else:
                    self.last_waypoint_guard_reason = (
                        "blocked side turn while the active corridor-follow "
                        "phase must remain centered"
                    )
                    waypoint = waypoint.model_copy(
                        update={
                            "intent": "FOLLOW_CORRIDOR",
                            "u": 500,
                        }
                    )
                    self.last_waypoint_applied_intent = "FOLLOW_CORRIDOR"
    
        if waypoint.stop:
            on_final_subgoal = bool(
                current is not None
                and self.subgoals
                and current.subgoal_id == self.subgoals[-1].subgoal_id
            )
            self.last_waypoint_stop_disposition = (
                "deferred_unverified_final"
                if on_final_subgoal
                else "ignored_nonfinal"
            )
            self.last_waypoint_guard_reason = (
                "model STOP deferred until task completion or repeated "
                "motion-grounded near-target evidence"
            )
            self.last_waypoint_applied_intent = (
                "FINAL_APPROACH"
                if on_final_subgoal
                else self._navigation_phase
            )
            waypoint = waypoint.model_copy(
                update={
                    "stop": False,
                    "intent": self.last_waypoint_applied_intent,
                    "u": 500,
                    "v": 750,
                }
            )
            self.last_model_response = (
                f"{self.last_model_response} [STOP deferred]"
            )

        assert waypoint.u is not None and waypoint.v is not None
        self.last_requested_normalized = (waypoint.u, waypoint.v)
        self.last_requested_pixel = (
            self._scale_normalized(waypoint.u, width),
            self._scale_normalized(waypoint.v, height),
        )
        self.last_waypoint_raw_response = self.last_waypoint_raw_response or (
            self.last_model_response
        )
        return self.last_requested_pixel
    
    @staticmethod
    def _scale_normalized(value: int, size: int) -> int:
        if size < 1:
            raise ValueError("image dimensions must be positive")
        return int(value * (size - 1) / 1000.0 + 0.5)
    
