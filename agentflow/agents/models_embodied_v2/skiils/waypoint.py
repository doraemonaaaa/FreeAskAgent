"""Waypoint selection and motion-grounded recovery for VLN agent v3."""

from __future__ import annotations

import re
from typing import Any, Optional

import numpy as np

from agentflow.agents.models_embodied_v2.memory.temporal_memory import (
    TemporalMemory,
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
    parse_actor_output,
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
    
    def _select_pixel(
        self,
        image: np.ndarray,
        instruction: str,
        *,
        subgoal_context: str = "",
        evaluate_recovery: bool = True,
        allow_preview: bool = True,
        allow_turn: bool = True,
    ) -> _WaypointOutput:
        """Return this step's decision, with model STOP deferred.

        The decision itself is returned rather than a pixel, because a step can
        resolve three different ways — a waypoint, an in-place turn, or a
        request to preview — and only the first has coordinates.  When it is a
        waypoint, ``last_requested_pixel`` holds the pixel it maps to.

        ``evaluate_recovery`` is False on the second call within one step, as a
        previewed step makes: the error-vote window and the recovery hold
        counter are both per-step, so evaluating twice would cast two votes for
        one observation and burn a hold step early.

        ``allow_preview`` is False once the surrounding views are already in
        hand, or a previewed step could ask to preview again and loop the
        controller.

        ``allow_turn`` is False in that same pass: a turn executes against the
        agent's real facing, but the decision was made while looking at a
        rotated view, so the two would mean different headings.
        """
        self.last_requested_pixel = None
        self.last_requested_normalized = None
        self.last_requested_turn_deg = None
        self.last_waypoint_raw_response = None
        self.last_waypoint_stop_disposition = None
        self.last_waypoint_model_intent = None
        self.last_waypoint_applied_intent = None
        self.last_waypoint_model_action_mode = None
        self.last_waypoint_applied_action_mode = None
        self.last_waypoint_guard_reason = None
        self.last_waypoint_evidence = None
        self.last_waypoint_confidence = None
        recovery_mode = (
            self._recovery_mode_for_step() if evaluate_recovery else None
        )
        if evaluate_recovery:
            self.last_recovery_mode = recovery_mode
            # Only the first call of a step clears these, so the view a
            # previewed step commits to survives into the debug payload.
            self.last_preview_view_index = None
            self.last_preview_yaw_deg = None
            self.last_preview_selection = None
            self.last_preview_guard_reason = None
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
                # A confirmed recovery is a forced deterministic action; there
                # is nothing for the controller to preview or explore first.
                action_mode="EXECUTION",
                u=RECOVERY_FORWARD_U if force_forward else RECOVERY_TURN_U,
                v=RECOVERY_FORWARD_V if force_forward else RECOVERY_TURN_V,
                confidence=1.0,
                evidence=evidence,
            )
            self.last_model_response = synthetic.model_dump_json()
            self.last_waypoint_raw_response = self.last_model_response
            self.last_waypoint_model_intent = intent
            self.last_waypoint_applied_intent = intent
            self.last_waypoint_model_action_mode = synthetic.action_mode
            self.last_waypoint_applied_action_mode = synthetic.action_mode
            self.last_waypoint_evidence = evidence
            self.last_waypoint_confidence = synthetic.confidence
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
            return synthetic
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
        # A PREVIEW reply carries no navigation intent, and the flat internal
        # form still needs one; the required phase is the honest answer, except
        # that STOP is not a steering intent.
        safe_phase: NavigationIntent = (
            self._navigation_phase
            if self._navigation_phase != "STOP"
            else "FINAL_APPROACH"
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
                waypoint = parse_actor_output(
                    payload,
                    preview_intent=safe_phase,
                )
                break
            except Exception as exc:
                last_error = exc
        else:
            waypoint = _WaypointOutput(
                stop=False,
                intent=safe_phase,
                # The reply that would have carried a mode is the thing that
                # failed validation, so fall back to committing to the point.
                action_mode="EXECUTION",
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
        # The corridor lock and the STOP deferral below rewrite where the agent
        # goes, not whether it should look around first, so the applied mode
        # only diverges from the model's when a guard forces an action.
        self.last_waypoint_model_action_mode = waypoint.action_mode
        self.last_waypoint_applied_action_mode = waypoint.action_mode
        self.last_waypoint_evidence = waypoint.evidence
        self.last_waypoint_confidence = waypoint.confidence
        if waypoint.action_mode == "PREVIEW":
            if allow_preview:
                # PREVIEW deliberately carries no action: the controller renders
                # the surrounding views and calls back. The guards below rewrite
                # where the agent goes, so they have nothing to act on here.
                self.last_waypoint_guard_reason = (
                    "preview requested; no action produced this step"
                )
                return waypoint
            # The views are already in hand, so asking again would loop the
            # controller. Commit to a safe waypoint in the chosen view instead.
            waypoint = waypoint.model_copy(
                update={"action_mode": "EXECUTION", "u": 500, "v": 750}
            )
            self.last_waypoint_applied_action_mode = "EXECUTION"
            self.last_waypoint_guard_reason = (
                "preview requested again after the views were provided; "
                "committed to a safe waypoint instead"
            )
        if waypoint.is_turn and not allow_turn:
            # Committing to a point inside the chosen view is the unambiguous
            # way to face that heading; the follower turns to reach it.
            safe_intent: NavigationIntent = (
                self._navigation_phase
                if self._navigation_phase != "STOP"
                else "FINAL_APPROACH"
            )
            waypoint = waypoint.model_copy(
                update={
                    "intent": safe_intent,
                    "turn_deg": None,
                    "u": 500,
                    "v": 750,
                }
            )
            self.last_waypoint_applied_intent = safe_intent
            self.last_waypoint_guard_reason = (
                "turn requested while resolving a preview; committed to a "
                "waypoint inside the chosen view instead"
            )
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
            # Resolving a preview releases the lock. Asking to look around is
            # the agent saying the corridor heading no longer resolves the
            # route, and re-centring inside a rotated view would command the
            # very turn this guard exists to block.
            and allow_preview
        ):
            # An explicit turn is a side turn by definition, so it faces the
            # same guard as a side waypoint. Ordering matters: a turn carries
            # no ``u`` to compare.
            deviates = (
                waypoint.is_turn
                or waypoint.intent != "FOLLOW_CORRIDOR"
                or abs(waypoint.u - 500) > CORRIDOR_WAYPOINT_DEVIATION
            )
            if deviates:
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
                            "turn_deg": None,
                            "u": 500,
                            # A blocked turn has no v of its own, so the safe
                            # lower-centre row stands in for it.
                            "v": 750 if waypoint.v is None else waypoint.v,
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

        if waypoint.is_turn:
            # An in-place turn has no image coordinates: the controller repeats
            # the simulator's turn primitive instead of steering to a point.
            self.last_requested_turn_deg = waypoint.turn_deg
            self.last_waypoint_raw_response = (
                self.last_waypoint_raw_response or self.last_model_response
            )
            return waypoint

        assert waypoint.u is not None and waypoint.v is not None
        self.last_requested_normalized = (waypoint.u, waypoint.v)
        self.last_requested_pixel = (
            self._scale_normalized(waypoint.u, width),
            self._scale_normalized(waypoint.v, height),
        )
        self.last_waypoint_raw_response = self.last_waypoint_raw_response or (
            self.last_model_response
        )
        return waypoint
    
    @staticmethod
    def _scale_normalized(value: int, size: int) -> int:
        if size < 1:
            raise ValueError("image dimensions must be positive")
        return int(value * (size - 1) / 1000.0 + 0.5)
    
