"""VLN RGB-D waypoint actor with unified temporal scene understanding."""


from __future__ import annotations

from collections import deque
from dataclasses import replace
import os
import re
import time
from typing import Any, Deque, Optional, Sequence

import math
import numpy as np

from agentflow.agents.models_embodied_v2.actor import Actor
from agentflow.agents.models_embodied_v2.memory.temporal_memory import (
    TemporalCaptioner,
)
from agentflow.agents.models_embodied_v2.data_models import (
    ActionMode,
    Subgoal,
    TemporalCaptionerConfig,
    CameraIntrinsics,
    NavigationDecision,
    NavigationPoint,
    PreviewSelection,
    PreviewView,
)

from agentflow.agents.models_embodied_v2.memory.task_memory import TaskMemory
from agentflow.agents.models_embodied_v2.memory.spatial_memory import (
    SpatialMemory,
    annotate_image,
    describe_candidates,
    generate_candidates,
)
from agentflow.agents.models_embodied_v2.memory.spatial_memory.candidates import (
    Candidate,
    encode_png,
    project_to_pixel,
)

from agentflow.agents.models_embodied_v2.skiils.protocol import (
    STAGE_PATH_OVERRUN_M,
    stage_is_doorway,
    BEHAVIOR_HISTORY_SIZE,
    COMMITTED_TARGET_REACHED_M,
    COMMITTED_TARGET_TOLERANCE_FRACTION,
    COMMITTED_TARGET_TOLERANCE_MAX_M,
    CORRIDOR_LOCK_FORWARD_STEPS,
    DEFAULT_MODEL_PATH,
    ERROR_CONFIRMATION_WINDOW,
    FINAL_STOP_EVIDENCE_WINDOW,
    LANDMARK_HISTORY_SIZE,
    LANDMARK_STEER_MIN_CONFIDENCE,
    NavigationIntent,
    POINT_PROMPT,
    LOCKED_TARGET_TURN_TOLERANCE_DEG,
    PREVIEW_REARM_TRANSLATION_M,
    PREVIEW_SELECTION_MIN_CONFIDENCE,
    STALL_EVIDENCE_FRAMES,
    RECOVERY_LATERAL_DISTANCE_M,
    SUBGOAL_GENERATION_ATTEMPTS,
    STRUCTURED_VLM_MAX_TOKENS,
    VLM_IMAGE_MAX_PIXELS,
    VLM_IMAGE_MIN_PIXELS,
    CAPTIONER_ANALYSIS_INTERVAL_STEPS,
    CAPTIONER_MAX_TOKENS,
    SPATIAL_LOOKAHEAD_M,
    SOM_MAX_CANDIDATES,
    SOM_MAX_TARGET_DISTANCE_M,
    SOM_PROMPT,
    SOM_TARGET_MAX_AGE_STEPS,
    SOM_TURN_DEG,
    SPATIAL_TARGET_MAX_AGE_STEPS,
    SPATIAL_TARGET_MIN_COMMIT_M,
    SPATIAL_TARGET_STAGNATION_STEPS,
    TEMPORAL_MAX_IMAGE_EDGE,
    SUBGOAL_PROMPT,
    TURN_ABANDON_DEG,
    TURN_AROUND_MIN_PROGRESS_DEG,
    TURN_AROUND_PATTERN,
    TURN_ALIGNMENT_DEG,
    TURN_EVIDENCE_DEG,
    TURN_TARGET_CENTRED_U,
    TURN_TARGET_MIN_PROGRESS_DEG,
    LandmarkOutput as _LandmarkOutput,
    WaypointOutput as _WaypointOutput,
)
from agentflow.agents.models_embodied_v2.skiils.planning import parse_subgoal_plan
from agentflow.agents.models_embodied_v2.skiils.landmark import LandmarkTrackerMixin
from agentflow.agents.models_embodied_v2.skiils.waypoint import WaypointPolicyMixin

from agentflow.agents.models_embodied_v2.memory.temporal_memory import (
    TemporalMemory,
)
class VLNAgent(LandmarkTrackerMixin, WaypointPolicyMixin):
    """Version 3 actor used by the Habitat waypoint worker."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        *,
        engine=None,
        debug_performance=False,
        use_cache=False,
        min_depth_m=0.25,
        max_depth_m=10.0,
        patch_radius_px=3,
        max_patch_depth_spread_m=0.35,
        camera_height_m=None,
        max_floor_offset_m=0.30,
        task_memory=None,
        temporal_memory=None,
        use_spatial_memory: Optional[bool] = None,
        **kwargs,
    ) -> None:
        # The actor owns observation validation, walkable-pixel snapping, and
        # back-projection.  This agent owns everything with task state: the
        # plan, both memories, the navigation phase, and the guards.
        self.actor = Actor(
            model_path,
            engine=engine,
            debug_performance=debug_performance,
            use_cache=use_cache,
            min_depth_m=min_depth_m,
            max_depth_m=max_depth_m,
            patch_radius_px=patch_radius_px,
            max_patch_depth_spread_m=max_patch_depth_spread_m,
            camera_height_m=camera_height_m,
            max_floor_offset_m=max_floor_offset_m,
        )
        # The landmark and waypoint policies prompt the same checkpoint.
        self.llm = self.actor.llm
        # Where the agent has been and what it found: the occupancy map,
        # world-anchored landmarks and the point it is currently walking to.
        self.spatial_memory = SpatialMemory(
            camera_height_m=camera_height_m,
            lookahead_m=SPATIAL_LOOKAHEAD_M,
        )
        # ``VLN_SPATIAL_MEMORY=0`` runs the identical agent without the map
        # in the loop, for A/B evaluation.
        if use_spatial_memory is None:
            use_spatial_memory = os.environ.get(
                "VLN_SPATIAL_MEMORY", "1"
            ).strip().lower() not in ("0", "false", "off", "no")
        self.use_spatial_memory = bool(use_spatial_memory)
        # Set-of-mark selection rides on Spatial Memory; VLN_SOM=0 keeps the
        # pixel-proposal path for A/B comparison.
        self.use_som = self.use_spatial_memory and os.environ.get(
            "VLN_SOM", "1"
        ).strip().lower() not in ("0", "false", "off", "no")
        self.last_som_candidates: list[dict[str, Any]] = []
        self.last_som_choice: Optional[str] = None
        # The marker-annotated frame the model was shown this step, for video.
        self.last_som_image: Optional[np.ndarray] = None
        self.last_som_raw_response: Optional[str] = None
        self.last_som_error: Optional[str] = None
        self.last_spatial_summary: str = "sp=-"
        self.last_spatial_error: Optional[str] = None

        self.last_model_response: Optional[str] = None
        # Per-``act`` wall-clock breakdown, in milliseconds, for latency
        # debugging.
        self.last_timings: dict[str, float] = {}
        # The analysis produced during the last ``act``, or None when Temporal
        # Memory did not analyze. Held here because reading Temporal Memory's
        # own ``latest_result`` clears it as soon as the subgoal advances.
        self.last_caption: Optional[Any] = None

        self.task_instruction: Optional[str] = None
        self.subgoals: list[Subgoal] = []
        self.last_subgoal_response: Optional[str] = None

        self.task_memory = task_memory
        if temporal_memory is not None and task_memory is None:
            raise ValueError(
                "temporal_memory requires its associated task_memory."
            )
        self.temporal_memory = temporal_memory

        self._landmark_history: Deque[dict[str, Any]] = deque(
            maxlen=LANDMARK_HISTORY_SIZE
        )
        self._behavior_history: Deque[dict[str, Any]] = deque(
            maxlen=BEHAVIOR_HISTORY_SIZE
        )
        self._error_candidates: Deque[str] = deque(
            maxlen=ERROR_CONFIRMATION_WINDOW
        )
        self._final_stop_evidence: Deque[bool] = deque(
            maxlen=FINAL_STOP_EVIDENCE_WINDOW
        )
        self._reset_runtime_state()

    def _reset_runtime_state(self) -> None:
        """Clear all episode-local navigation and debug state."""
        self._external_candidates: Optional[list] = None
        self.last_subgoal_before: Optional[str] = None
        self.last_subgoal_after: Optional[str] = None
        self.last_requested_pixel: Optional[tuple[int, int]] = None
        self.last_requested_normalized: Optional[tuple[int, int]] = None
        # Set instead of the pixel pair when a step rotates in place.
        self.last_requested_turn_deg: Optional[int] = None
        self.last_waypoint_raw_response: Optional[str] = None
        self.last_waypoint_stop_disposition: Optional[str] = None
        self.last_waypoint_model_intent: Optional[str] = None
        self.last_waypoint_applied_intent: Optional[str] = None
        self.last_waypoint_model_action_mode: Optional[ActionMode] = None
        self.last_waypoint_applied_action_mode: Optional[ActionMode] = None
        # Which surrounding view resolved the last PREVIEW, its heading offset,
        # the Captioner's own judgement, and why a fallback was used instead.
        # All stay None on a step that never previewed.
        self.last_preview_view_index: Optional[int] = None
        self.last_preview_yaw_deg: Optional[float] = None
        self.last_preview_selection: Optional[PreviewSelection] = None
        self.last_preview_guard_reason: Optional[str] = None
        # A preview stays consumed at this physical position until the camera
        # has made real translational progress. Rotation alone must not re-arm
        # another relative-heading preview.
        self._preview_requires_progress = False
        self._preview_anchor_position_xz: Optional[np.ndarray] = None
        self.last_waypoint_guard_reason: Optional[str] = None
        self.last_waypoint_evidence: Optional[str] = None
        self.last_waypoint_confidence: Optional[float] = None
        self.last_error_candidate: Optional[str] = None
        self.last_error_guard_reason: Optional[str] = None
        self.last_recovery_mode: Optional[str] = None
        self.last_landmark: Optional[_LandmarkOutput] = None
        self.last_landmark_raw_response: Optional[str] = None
        self.last_landmark_error: Optional[str] = None
        self.last_landmark_normalized: Optional[tuple[int, int]] = None
        self.last_landmark_pixel: Optional[tuple[int, int]] = None
        self._landmark_subgoal_id: Optional[str] = None
        self._error_candidate_subgoal: Optional[str] = None
        self._active_recovery_mode: Optional[str] = None
        self._recovery_progress_m = 0.0
        self._recovery_attempt_steps = 0
        self._recovery_anchor_position_xz: Optional[np.ndarray] = None
        self._force_forward_this_step = False
        self._force_left_turn_this_step = False
        self._previous_position: Optional[np.ndarray] = None
        self._previous_yaw_deg: Optional[float] = None
        self._navigation_subgoal_id: Optional[str] = None
        self._navigation_phase: NavigationIntent = "FOLLOW_CORRIDOR"
        self._corridor_forward_streak = 0
        self._corridor_heading_yaw_deg: Optional[float] = None
        self._subgoal_net_yaw_deg = 0.0
        self._subgoal_walked_m = 0.0
        self._overrun_fired_at_m = 0.0
        self._force_preview_this_step = False
        self._overrun_count = 0
        self._turn_follow_phase_started = False
        # A model-localized doorway is a stable physical target. Keep its
        # world-space waypoint across steps instead of letting independent VLM
        # calls move the target around while the follower is routing around
        # furniture toward the opening.
        self._doorway_waypoint: Optional[NavigationPoint] = None
        self._doorway_waypoint_subgoal_id: Optional[str] = None
        # Stage whose located final target has already been committed to
        # once before a STOP vote was allowed.
        self._final_commit_subgoal_id: Optional[str] = None
        # Bearing to the locked target over the same window the motion
        # grounding inspects, so a same-sign rotation can be told apart from a
        # spin by whether the target was still off-axis while it happened.
        self._doorway_waypoint_bearing_history: Deque[float] = deque(
            maxlen=STALL_EVIDENCE_FRAMES
        )
        self._doorway_waypoint_best_distance_m: Optional[float] = None
        self._doorway_waypoint_stagnant_steps = 0
        self._doorway_waypoint_reach_tolerance_m = COMMITTED_TARGET_REACHED_M
        self._steps_since_scene_analysis = 0
        self._scene_analysis_subgoal_id: Optional[str] = None
        self._spatial_step = -1
        self._current_depth_m: Optional[np.ndarray] = None
        self._current_floor_mask: Optional[np.ndarray] = None
        self._oracle_goal_xyz: Optional[tuple[float, float, float]] = None
        self._judge_target_point: Optional[NavigationPoint] = None
        self._judge_target_subgoal_id: Optional[str] = None
        self._judge_target_tolerance_m = COMMITTED_TARGET_REACHED_M
        self._elevation_subgoal_id: Optional[str] = None
        self._subgoal_start_y: Optional[float] = None
        self._landmark_history.clear()
        self._behavior_history.clear()
        self._error_candidates.clear()
        self._final_stop_evidence.clear()

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
        navigable_window: Optional[dict[str, Any]] = None,
        oracle_goal_xyz: Optional[Sequence[float]] = None,
        cwp_candidates: Optional[Sequence[dict]] = None,
    ) -> NavigationDecision:
        """Run one step while retaining state transitions for debug output.

        ``navigable_window`` is the controller's traversability around the
        agent (origin_xz, resolution_m, mask); Spatial Memory uses it to keep
        targets and candidates where the follower can actually go.
        """
        act_started = time.perf_counter()
        # Diagnostic upper bound for set-of-mark: never set in evaluation.
        self._oracle_goal_xyz = (
            tuple(float(v) for v in oracle_goal_xyz)
            if oracle_goal_xyz is not None else None
        )
        # Externally supplied (runner-side CWP) waypoint candidates for this
        # step; when present they replace the floor-openings generator.
        self._external_candidates = list(cwp_candidates) if cwp_candidates else None
        if navigable_window is not None and self.use_spatial_memory:
            try:
                self.spatial_memory.set_traversability(
                    origin_xz=tuple(navigable_window["origin_xz"]),
                    resolution_m=float(navigable_window["resolution_m"]),
                    mask=np.asarray(navigable_window["mask"], dtype=bool),
                )
            except Exception as exc:
                self.last_spatial_error = f"{type(exc).__name__}: {exc}"
        # These flags describe only the waypoint selected during this call.
        # A recovery selection may set one of them below; clear both before
        # delegating to the RGB-D layer so a completed recovery cannot leak
        # into later model-selected waypoints.
        self._force_forward_this_step = False
        self._force_left_turn_this_step = False
        self.last_som_image = None
        # Only the step resolved from surrounding views reports a previewed
        # view; leaving these set would tag every later step (and video
        # frame) with a stale PREVIEW marker.
        self.last_preview_view_index = None
        self.last_preview_yaw_deg = None
        # Preview views belong to the step that asked for them. Holding them
        # past it would keep several images alive for the rest of the episode
        # and let a later reader mistake them for this step's.
        if isinstance(self.temporal_memory, TemporalMemory):
            self.temporal_memory.clear_preview()
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
            self._update_preview_progress()
            self._update_recovery_progress(translation_m)
            self._subgoal_walked_m += float(translation_m)
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
        self._spatial_step += 1
        self._current_depth_m = self.actor.depth_in_meters(
            depth,
            self.actor.as_rgb_array(rgb).shape[:2],
            normalized=normalized_depth,
            depth_min_m=depth_min_m,
            depth_max_m=depth_max_m,
        )
        self._observe_spatial(intrinsics, camera_to_world)
        if isinstance(self.temporal_memory, TemporalMemory):
            self.temporal_memory.set_motion_evidence(
                translation_m=translation_m,
                yaw_delta_deg=yaw_delta_deg,
            )
            self.temporal_memory.set_doorway_target_distance(
                self._doorway_target_distance(
                    current,
                    camera_to_world=camera_to_world,
                ),
                reach_tolerance_m=self._judge_target_tolerance(current),
            )
            self.temporal_memory.set_doorway_crossing(
                self._spatial_crossing_this_step()
            )
            self.temporal_memory.set_depth_observation(self._current_depth_m)
            self.temporal_memory.set_turn_progress(
                self._turn_progress_deg(current)
            )
            # Still the previous step's value here: ``_select_pixel`` clears
            # it later in this same step.
            self.temporal_memory.set_stop_proposed(
                self.last_waypoint_stop_disposition is not None
            )
            self.temporal_memory.set_elevation_progress(
                self._elevation_progress_m(current, camera_to_world)
            )
        try:
            decision = self._waypoint_decision(
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
            image = self.actor.as_rgb_array(rgb)
            transform = np.asarray(camera_to_world, dtype=np.float64)
            position = transform[:3, 3]
            self._force_left_turn_this_step = True
            self.last_recovery_mode = "NO_VALID_DEPTH"
            # The model may have asked to preview, but the near-field obstacle
            # has to be cleared before any surrounding view is worth taking.
            self.last_waypoint_applied_action_mode = "EXECUTION"
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
                action_mode="EXECUTION",
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
        self.last_timings["total_ms"] = (
            time.perf_counter() - act_started
        ) * 1000
        return decision

    def act_on_preview(
        self,
        views: Sequence[PreviewView],
        instruction: str,
        *,
        normalized_depth: bool = False,
        depth_min_m: Optional[float] = None,
        depth_max_m: Optional[float] = None,
    ) -> NavigationDecision:
        """Act on the surrounding views the controller rendered for a PREVIEW.

        This is the second half of the step ``act`` began, not a new one.
        Motion, the landmark tracker, and Temporal Memory were all advanced
        there and deliberately do not run again: re-entering ``act`` would count
        the same frame twice and record a stationary step as a stall.

        The views are deposited in working memory, which is where the Captioner
        judges which heading to take.  This agent only picks the floor point
        inside whichever view comes back, with its ordinary waypoint policy.
        """
        started = time.perf_counter()
        views = tuple(views)
        if not views:
            raise ValueError("resolving a preview requires at least one view")
        unusable = [
            index
            for index, view in enumerate(views)
            if not view.is_navigable
        ]
        if unusable:
            raise ValueError(
                "preview views {} lack depth, intrinsics, or a camera "
                "transform, so a waypoint chosen in them could not be "
                "back-projected".format(unusable)
            )
        # A previewed step selects a fresh action, so no forced primitive from
        # the first half may leak into it.
        self._force_forward_this_step = False
        self._force_left_turn_this_step = False
        view_transform = np.asarray(views[0].camera_to_world, dtype=np.float64)
        self._preview_requires_progress = True
        self._preview_anchor_position_xz = view_transform[[0, 2], 3].copy()

        preview_select_started = time.perf_counter()
        view_index = self._selected_preview_view(views)
        timings = {
            "preview_select_ms": (
                time.perf_counter() - preview_select_started
            ) * 1000
        }
        view = views[view_index]
        self.last_preview_view_index = view_index
        self.last_preview_yaw_deg = view.yaw_deg

        image = self.actor.as_rgb_array(view.rgb)
        select_started = time.perf_counter()
        selection = self.last_preview_selection
        if (
            selection is not None
            and selection.confidence >= PREVIEW_SELECTION_MIN_CONFIDENCE
        ):
            # The selector already spent the one VLM call needed to resolve
            # this PREVIEW. Commit to its selected floor point instead of
            # asking the waypoint model to interpret the same heading again
            # (which could request PREVIEW in a loop).
            self._commit_selected_preview(
                selection,
                width=image.shape[1],
                height=image.shape[0],
            )
        else:
            self._select_pixel(
                image,
                instruction,
                subgoal_context=(
                    self.task_memory.current_subgoal_context()
                    if self.task_memory is not None
                    else ""
                ),
                # All three constrain this second pass within one step: the
                # error votes and recovery hold were already consumed by
                # ``act``, the surrounding views are already in hand, and a
                # turn would be measured against the agent's real facing
                # rather than the rotated preview view.
                evaluate_recovery=False,
                allow_preview=False,
                allow_turn=False,
            )
        timings["select_pixel_ms"] = (
            time.perf_counter() - select_started
        ) * 1000
        # With previews and turns both refused above, the only outcome left is
        # a waypoint inside the chosen view.
        requested_uv = self.last_requested_pixel
        if requested_uv is None:
            # A STOP or malformed selection in this pass leaves no pixel;
            # the conventional lower-centre stand-in keeps the episode
            # alive instead of killing the whole rank with an assertion
            # (observed once in 200 episodes, in act_on_preview).
            requested_uv = (image.shape[1] // 2, int(image.shape[0] * 0.75))

        depth_m = self.actor.depth_in_meters(
            view.depth,
            image.shape[:2],
            normalized=normalized_depth,
            depth_min_m=depth_min_m,
            depth_max_m=depth_max_m,
        )
        waypoint_started = time.perf_counter()
        try:
            point = self.actor.waypoint_from_pixel(
                requested_uv,
                depth_m,
                view.intrinsics,
                view.camera_to_world,
            )
        except ValueError as exc:
            if str(exc) != (
                "Depth observation contains no valid walkable waypoint."
            ):
                raise
            # The chosen heading is blocked at very close range. Returning no
            # point leaves the controller on its own recovery primitive, which
            # is the same degradation a PREVIEW with no renderer gets.
            self.last_recovery_mode = "NO_VALID_DEPTH"
            self.last_waypoint_guard_reason = (
                "previewed view {} has no valid walkable waypoint".format(
                    view_index
                )
            )
            self._record_timings(timings, started)
            return NavigationDecision(
                stop=False,
                raw_response=(
                    f"{self.last_model_response}; previewed view has no "
                    "valid depth"
                ),
                action_mode=self._applied_action_mode(),
            )
        self._maybe_lock_doorway_waypoint(
            point,
            camera_to_world=view.camera_to_world,
        )
        self._commit_spatial_target(
            point,
            camera_to_world=view.camera_to_world,
            kind="preview",
        )
        timings["waypoint_ms"] = (
            time.perf_counter() - waypoint_started
        ) * 1000
        self._record_timings(timings, started)
        return NavigationDecision(
            stop=False,
            point=point,
            raw_response=self.last_model_response,
            action_mode=self._applied_action_mode(),
        )

    def _commit_selected_preview(
        self,
        selection: PreviewSelection,
        *,
        width: int,
        height: int,
    ) -> None:
        """Turn one high-confidence preview target into a floor waypoint."""
        intent: NavigationIntent = (
            self._navigation_phase
            if self._navigation_phase != "STOP"
            else "FINAL_APPROACH"
        )
        raw_response = getattr(
            getattr(self.temporal_memory, "captioner", None),
            "last_preview_raw_response",
            None,
        )
        self.last_model_response = raw_response or selection.evidence
        self.last_waypoint_raw_response = self.last_model_response
        self.last_waypoint_model_intent = intent
        self.last_waypoint_applied_intent = intent
        self.last_waypoint_model_action_mode = "EXECUTION"
        self.last_waypoint_applied_action_mode = "EXECUTION"
        self.last_waypoint_evidence = selection.evidence
        self.last_waypoint_confidence = selection.confidence
        self.last_waypoint_guard_reason = (
            "committed directly to the Captioner-selected preview heading; "
            "skipped redundant waypoint VLM"
        )
        self.last_requested_turn_deg = None
        self.last_requested_normalized = (selection.u, selection.v)
        self.last_requested_pixel = (
            self._scale_normalized(selection.u, width),
            self._scale_normalized(selection.v, height),
        )

    def _waypoint_decision(
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
        """Return the stop decision or the world-space RGB-D waypoint.

        Habitat's default depth sensor emits meters.  Set ``normalized_depth``
        only when its configuration uses normalized [0, 1] observations, and
        provide that sensor's ``depth_min_m`` and ``depth_max_m`` bounds.
        """
        started = time.perf_counter()
        image = self.actor.as_rgb_array(rgb)
        timings = {"rgb_ms": (time.perf_counter() - started) * 1000}

        memory_started = time.perf_counter()
        caption = None
        if self.task_memory is not None:
            self.task_memory.record_input(image)
            if self.temporal_memory is not None:
                current_before_analysis = (
                    self.task_memory.get_current_subgoal()
                )
                doorway_distance = self._doorway_target_distance(
                    current_before_analysis,
                    camera_to_world=camera_to_world,
                )
                # Completion cannot pass the structural-door guard while the
                # camera is still far from the model-localized doorway. Keep
                # every RGB/motion frame but avoid an expensive, guaranteed
                # in-progress Captioner call until the threshold is near.
                # A turn stage is judged on measured rotation and the
                # landmark's position in the image, both of which live in
                # the analysis, so it is never deferred.
                phase_before_analysis = self._phase_for_subgoal(
                    current_before_analysis
                )
                defer_scene_analysis = bool(
                    doorway_distance is not None
                    and doorway_distance > 1.25
                    and phase_before_analysis
                    not in ("TURN_LEFT", "TURN_RIGHT")
                )
                # A plain walk/approach stage completes on measured arrival
                # at its committed point, which the frames cannot change
                # from one step to the next; judging it every other step
                # halves the Captioner cost. Doorway, turn and final stages
                # keep the per-step judgement their crossing/arrival
                # streaks depend on.
                throttle_scene_analysis = bool(
                    CAPTIONER_ANALYSIS_INTERVAL_STEPS > 1
                    and phase_before_analysis
                    in ("FOLLOW_CORRIDOR", "APPROACH_LANDMARK")
                    and not self._is_final_subgoal(current_before_analysis)
                    and not self._doorway_subgoal(current_before_analysis)
                    # The first observation of a stage is always judged so
                    # its landmark is localized before any steering.
                    and current_before_analysis is not None
                    and self._scene_analysis_subgoal_id
                    == current_before_analysis.subgoal_id
                    and self._steps_since_scene_analysis + 1
                    < CAPTIONER_ANALYSIS_INTERVAL_STEPS
                )
                caption = self.temporal_memory.update_from_task_memory(
                    analyze=not (
                        defer_scene_analysis or throttle_scene_analysis
                    ),
                )
                if caption is not None:
                    self._steps_since_scene_analysis = 0
                    self._scene_analysis_subgoal_id = caption.subgoal_id
                else:
                    self._steps_since_scene_analysis += 1
        self.last_caption = caption
        if caption is not None:
            self._record_scene_landmark(image, caption)
            self._register_spatial_landmark(intrinsics, camera_to_world)
        timings.update({
            # Whole memory phase: frame copy, rule-based error detection, and
            # the Captioner call when Temporal Memory analyzes this step.
            "memory_ms": (time.perf_counter() - memory_started) * 1000,
            # The Captioner's own model time, so the rule-based visual
            # bookkeeping around it can be read as the difference.
            "captioner_ms": (
                caption.latency_ms if caption is not None else 0.0
            ),
        })

        # Every planned subgoal is done, so the task itself is done. Stop here
        # rather than asking the model to steer towards a subgoal that no
        # longer exists.
        if self.task_memory is not None and self.task_memory.is_task_complete():
            self.last_model_response = "all subgoals complete"
            # This returns before ``_select_pixel`` clears the per-step waypoint
            # state, so set the mode here rather than reporting the previous
            # step's value alongside a terminal decision.
            self.last_waypoint_applied_action_mode = "EXECUTION"
            self._record_timings(timings, started)
            return NavigationDecision(
                stop=True,
                raw_response=self.last_model_response,
            )

        current = (
            self.task_memory.get_current_subgoal()
            if self.task_memory is not None
            else None
        )
        self._release_doorway_waypoint_for_motion(
            current,
            camera_to_world=camera_to_world,
        )
        self._stage_overrun_reorient(current)
        locked = self._locked_doorway_decision(
            current,
            camera_to_world=camera_to_world,
        )
        if locked is not None:
            timings["depth_ms"] = 0.0
            timings["select_pixel_ms"] = 0.0
            timings["waypoint_ms"] = 0.0
            self._record_timings(timings, started)
            return locked
        spatial = self._spatial_target_decision(
            current,
            image_shape=image.shape[:2],
        )
        if spatial is not None:
            timings["depth_ms"] = 0.0
            timings["select_pixel_ms"] = 0.0
            timings["waypoint_ms"] = 0.0
            self._record_timings(timings, started)
            return spatial
        if self._force_preview_this_step:
            # Stage-overrun reorientation: rendering costs no episode step
            # and the preview handler re-chooses a heading from all views.
            self._force_preview_this_step = False
            timings["depth_ms"] = 0.0
            timings["select_pixel_ms"] = 0.0
            timings["waypoint_ms"] = 0.0
            self._record_timings(timings, started)
            return NavigationDecision(
                stop=False,
                raw_response="stage overrun: requesting surrounding views",
                action_mode="PREVIEW",
            )

        depth_started = time.perf_counter()
        depth_m = self.actor.depth_in_meters(
            depth,
            image.shape[:2],
            normalized=normalized_depth,
            depth_min_m=depth_min_m,
            depth_max_m=depth_max_m,
        )
        timings["depth_ms"] = (time.perf_counter() - depth_started) * 1000

        som_started = time.perf_counter()
        som = self._som_decision(
            current,
            image=image,
            depth_m=depth_m,
            intrinsics=intrinsics,
            camera_to_world=camera_to_world,
        )
        if som is not None:
            timings["select_pixel_ms"] = (time.perf_counter() - som_started) * 1000
            timings["waypoint_ms"] = 0.0
            self._record_timings(timings, started)
            return som

        # Read Task Memory only after Temporal Memory has published this
        # step's analysis, so waypoint selection steers by the subgoal this
        # step established rather than the previous step's.
        select_started = time.perf_counter()
        selected = self._select_pixel(
            image,
            instruction,
            subgoal_context=(
                self.task_memory.current_subgoal_context()
                if self.task_memory is not None
                else ""
            ),
            depth_m=depth_m,
        )
        timings["select_pixel_ms"] = (
            time.perf_counter() - select_started
        ) * 1000
        selected = self._steer_to_visible_landmark(
            selected,
            current,
            width=image.shape[1],
            height=image.shape[0],
        )
        if selected.action_mode == "PREVIEW":
            # PREVIEW carries no action by design: the controller renders the
            # surrounding views and calls back, so this is not a stop.
            self._record_timings(timings, started)
            return NavigationDecision(
                stop=False,
                raw_response=self.last_model_response,
                action_mode="PREVIEW",
            )
        if selected.stop:
            self._record_timings(timings, started)
            return NavigationDecision(
                stop=True,
                raw_response=self.last_model_response,
                action_mode="EXECUTION",
            )
        if selected.is_turn:
            # An in-place turn never reaches the RGB-D layer: there is no pixel
            # to snap and no point to back-project.
            self._record_timings(timings, started)
            return NavigationDecision(
                stop=False,
                raw_response=self.last_model_response,
                action_mode=self._applied_action_mode(),
                turn_deg=selected.turn_deg,
            )
        frontier = self._spatial_frontier_decision(
            current,
            image_shape=image.shape[:2],
        )
        if frontier is not None:
            timings["waypoint_ms"] = 0.0
            self._record_timings(timings, started)
            return frontier
        requested_uv = self.last_requested_pixel
        if requested_uv is None:
            # A STOP or malformed selection in this pass leaves no pixel;
            # the conventional lower-centre stand-in keeps the episode
            # alive instead of killing the whole rank with an assertion
            # (observed once in 200 episodes, in act_on_preview).
            requested_uv = (image.shape[1] // 2, int(image.shape[0] * 0.75))
        waypoint_started = time.perf_counter()
        point = self.actor.waypoint_from_pixel(
            requested_uv,
            depth_m,
            intrinsics,
            camera_to_world,
        )
        self._maybe_lock_doorway_waypoint(
            point,
            camera_to_world=camera_to_world,
        )
        self._commit_spatial_target(point, camera_to_world=camera_to_world)
        timings["waypoint_ms"] = (time.perf_counter() - waypoint_started) * 1000
        self._record_timings(timings, started)
        return NavigationDecision(
            stop=False,
            point=point,
            raw_response=self.last_model_response,
            action_mode=self._applied_action_mode(),
        )

    def _steer_to_visible_landmark(
        self,
        selected: _WaypointOutput,
        current: Optional[Subgoal],
        *,
        width: int,
        height: int,
    ) -> _WaypointOutput:
        """Walk toward a landmark the Captioner has already located.

        The waypoint model reliably asks to TURN or PREVIEW when the doorway
        sits off-centre in the image, even though its prompt tells it to
        steer to the floor beneath a visible landmark. When the Captioner
        reported that landmark in this same image, the geometry is already
        known: the requested pixel becomes the landmark's own position, and
        the RGB-D layer snaps it down to the floor in front of it. A waypoint
        or STOP the model produced itself is left alone.
        """
        if selected.stop or (
            selected.action_mode != "PREVIEW" and not selected.is_turn
        ):
            return selected
        landmark = self.last_landmark
        if (
            current is None
            or landmark is None
            or self._landmark_subgoal_id != current.subgoal_id
            or not landmark.visible
            or landmark.u is None
            or landmark.v is None
            or landmark.confidence < LANDMARK_STEER_MIN_CONFIDENCE
        ):
            return selected
        intent: NavigationIntent = (
            self._navigation_phase
            if self._navigation_phase != "STOP"
            else "APPROACH_LANDMARK"
        )
        asked = "PREVIEW" if selected.action_mode == "PREVIEW" else "TURN"
        evidence = (
            "steered to the floor beneath the Captioner-localized landmark "
            f"at ({landmark.u},{landmark.v}) instead of the model's {asked}: "
            f"{landmark.evidence}"
        )
        override = _WaypointOutput(
            stop=False,
            intent=intent,
            action_mode="EXECUTION",
            u=landmark.u,
            v=landmark.v,
            confidence=landmark.confidence,
            evidence=evidence,
        )
        self.last_waypoint_guard_reason = (
            f"overrode model {asked}: landmark is localized in the current "
            "image"
        )
        self.last_waypoint_applied_intent = intent
        self.last_waypoint_applied_action_mode = "EXECUTION"
        self.last_waypoint_evidence = evidence
        self.last_waypoint_confidence = landmark.confidence
        self.last_requested_turn_deg = None
        self.last_requested_normalized = (landmark.u, landmark.v)
        self.last_requested_pixel = (
            self._scale_normalized(landmark.u, width),
            self._scale_normalized(landmark.v, height),
        )
        self._force_forward_this_step = False
        self._force_left_turn_this_step = False
        return override

    @staticmethod
    def _doorway_subgoal(subgoal: Optional[Subgoal]) -> bool:
        if subgoal is None:
            return False
        return bool(
            stage_is_doorway(
                f"{subgoal.description} {subgoal.completion_criteria}"
            )
        )

    def _maybe_lock_doorway_waypoint(
        self,
        point: NavigationPoint,
        *,
        camera_to_world: Any,
    ) -> None:
        current = (
            self.task_memory.get_current_subgoal()
            if self.task_memory is not None
            else None
        )
        # A deferred STOP answered with the safe lower-centre filler pixel
        # carries the model's arrival confidence but none of its
        # localization; there is nothing to lock. A deferred STOP steered
        # to the located target is a localized point and locks normally.
        if (
            self.last_waypoint_stop_disposition is not None
            and self.last_requested_normalized == (500, 750)
        ):
            return
        evidence = self.last_waypoint_evidence or ""
        # The model's confidence is advisory (see protocol.py): a doorway
        # named in the evidence, a Captioner-located landmark or a
        # Captioner-selected heading is the structural claim, and the
        # on-floor check plus stagnation release guard the lock itself.
        doorway_target = bool(
            self._doorway_subgoal(current)
            and re.search(
                r"\b(?:door|doorway|threshold|opening|exit)\b",
                evidence,
                flags=re.IGNORECASE,
            )
        )
        # A landmark the Captioner located for an approach stage is committed
        # to the same way as a doorway: it is the point whose arrival decides
        # the stage, and the judge defers completion until it is reached.
        # The final stage is excluded because a reused point would bypass the
        # waypoint model that proposes STOP.
        landmark_target = bool(
            current is not None
            and not self._is_final_subgoal(current)
            and self._navigation_phase
            in ("APPROACH_LANDMARK", "TURN_LEFT", "TURN_RIGHT")
            and re.search(
                r"overrode model|beneath the located landmark",
                self.last_waypoint_guard_reason or "",
            )
        )
        # A heading the Captioner picked out of the surrounding views is
        # the one target the agent has for a subgoal whose landmark is not
        # in the forward view at all. Without a lock the follower turns one
        # primitive toward it, the next step's PREVIEW is refused (the views
        # were just inspected) and the fallback walks straight ahead: the
        # measured route was one 15-degree turn followed by a forward step
        # in the wrong direction, repeated until the agent hit a wall.
        preview_target = bool(
            current is not None
            and not self._is_final_subgoal(current)
            and "Captioner-selected preview heading"
            in (self.last_waypoint_guard_reason or "")
        )
        if not (doorway_target or landmark_target or preview_target):
            return
        # A point the actor could not place on the floor is a wall, a far
        # window, or a clamped depth reading. Locking it would make the agent
        # chase a target it can never reach and would also hold the doorway
        # completion guard shut, so it steers this step only.
        if self.actor.camera_height_m is not None and not point.on_floor:
            self.last_waypoint_guard_reason = (
                (self.last_waypoint_guard_reason or "")
                + "; doorway waypoint not locked: point is off the floor"
            ).lstrip("; ")
            return
        # Do not replace a still-active structural target with a fresh VLM
        # projection. Near a doorway, small view changes can move the chosen
        # pixel onto the far wall and produce a plausible but very distant
        # world point. The existing target is released only by measured
        # stagnation or by advancing to another subgoal.
        if (
            self._doorway_waypoint is not None
            and self._doorway_waypoint_subgoal_id == current.subgoal_id
        ):
            return
        self._doorway_waypoint = point
        self._doorway_waypoint_subgoal_id = current.subgoal_id
        self._doorway_waypoint_bearing_history.clear()
        distance_m = self._planar_waypoint_distance(
            point,
            camera_to_world=camera_to_world,
        )
        self._doorway_waypoint_best_distance_m = distance_m
        self._doorway_waypoint_stagnant_steps = 0
        # The point's error grows with the range it was localized from.
        self._doorway_waypoint_reach_tolerance_m = float(
            np.clip(
                COMMITTED_TARGET_TOLERANCE_FRACTION * distance_m,
                COMMITTED_TARGET_REACHED_M,
                COMMITTED_TARGET_TOLERANCE_MAX_M,
            )
        )

    def _clear_doorway_waypoint(self, *, keep_for_judge: bool = False) -> None:
        """Drop the committed point; optionally keep it as judge evidence.

        A point released because the follower looped is no longer steered
        to, but the stage's endpoint has not moved: the judge keeps holding
        completion until the camera actually gets there or walks past it.
        A point released for stagnation was probably wrong and is dropped
        outright.
        """
        if keep_for_judge and self._doorway_waypoint is not None:
            self._judge_target_point = self._doorway_waypoint
            self._judge_target_subgoal_id = self._doorway_waypoint_subgoal_id
            self._judge_target_tolerance_m = (
                self._doorway_waypoint_reach_tolerance_m
            )
        else:
            self._judge_target_point = None
            self._judge_target_subgoal_id = None
            self._judge_target_tolerance_m = COMMITTED_TARGET_REACHED_M
        self._doorway_waypoint = None
        self._doorway_waypoint_subgoal_id = None
        self._doorway_waypoint_best_distance_m = None
        self._doorway_waypoint_stagnant_steps = 0
        self._doorway_waypoint_reach_tolerance_m = COMMITTED_TARGET_REACHED_M
        self._doorway_waypoint_bearing_history.clear()

    def _elevation_progress_m(
        self,
        current: Optional[Subgoal],
        camera_to_world: Any,
    ) -> float:
        """Camera height change since the active subgoal began."""
        current_id = current.subgoal_id if current is not None else None
        y = float(np.asarray(camera_to_world, dtype=np.float64)[1, 3])
        if current_id != self._elevation_subgoal_id or self._subgoal_start_y is None:
            self._elevation_subgoal_id = current_id
            self._subgoal_start_y = y
        return y - self._subgoal_start_y

    def _release_doorway_waypoint_for_motion(
        self,
        current: Optional[Subgoal],
        *,
        camera_to_world: Any,
    ) -> None:
        """Drop a locked door target as soon as measured motion loops.

        A locked waypoint bypasses the waypoint VLM, so its motion recovery
        must run before the lock's early return. The ordinary waypoint path
        then consumes the same grounded mode and emits its deterministic
        recovery action during this step.
        """
        if self._doorway_target_distance(
            current,
            camera_to_world=camera_to_world,
        ) is None:
            return
        if self._doorway_waypoint is not None:
            self._doorway_waypoint_bearing_history.append(
                self._waypoint_bearing_deg(
                    self._doorway_waypoint,
                    camera_to_world=camera_to_world,
                )
            )
        mode, reason = self._motion_grounded_error_candidate("NONE")
        # Four stalled motion intervals distinguish a legitimate short doorway
        # alignment from a same-sign spin, so both loop modes invalidate the
        # locked target.
        if mode not in {"TURN_OSCILLATION", "IN_PLACE_SPIN"}:
            return
        # A target behind the agent takes six or more same-sign turn
        # primitives before the follower's first forward step, which is
        # exactly the signature of IN_PLACE_SPIN. If the target was well off
        # the camera axis at any point in the grounding window, the rotation
        # in that window was the route, not a loop; this also covers the step
        # right after alignment, when the target is centred but the window
        # still holds only turns.
        if (
            mode == "IN_PLACE_SPIN"
            and self._doorway_waypoint is not None
            and any(
                abs(bearing) > LOCKED_TARGET_TURN_TOLERANCE_DEG
                for bearing in self._doorway_waypoint_bearing_history
            )
        ):
            self.last_error_guard_reason = (
                "kept locked doorway waypoint: same-sign rotation is still "
                "aligning with the off-axis target"
            )
            return
        self._clear_doorway_waypoint(keep_for_judge=True)
        self.last_error_candidate = mode
        self.last_error_guard_reason = (
            f"released locked doorway waypoint: {reason}"
        )

    @staticmethod
    def _planar_waypoint_distance(
        point: NavigationPoint,
        *,
        camera_to_world: Any,
    ) -> float:
        transform = np.asarray(camera_to_world, dtype=np.float64)
        position_xz = transform[[0, 2], 3]
        target_xz = np.asarray(point.world_xyz, dtype=np.float64)[[0, 2]]
        return float(np.linalg.norm(target_xz - position_xz))

    @staticmethod
    def _waypoint_bearing_deg(
        point: NavigationPoint,
        *,
        camera_to_world: Any,
    ) -> float:
        """Signed planar angle from the camera axis to the point; right > 0."""
        transform = np.asarray(camera_to_world, dtype=np.float64)
        position_xz = transform[[0, 2], 3]
        forward_xz = -transform[[0, 2], 2]
        right_xz = transform[[0, 2], 0]
        offset = (
            np.asarray(point.world_xyz, dtype=np.float64)[[0, 2]]
            - position_xz
        )
        return float(np.degrees(np.arctan2(
            float(np.dot(offset, right_xz)),
            float(np.dot(offset, forward_xz)),
        )))

    @staticmethod
    def _waypoint_passed(
        point: NavigationPoint,
        *,
        camera_to_world: Any,
    ) -> bool:
        """True when the point is close behind the camera.

        A waypoint just inside a doorway is rarely stepped on exactly: the
        agent cuts the corner and walks on. Once the point lies behind the
        camera within a metre it has served its purpose, so neither the
        follower should turn back for it nor the judge wait for it.
        """
        transform = np.asarray(camera_to_world, dtype=np.float64)
        position_xz = transform[[0, 2], 3]
        forward_xz = -transform[[0, 2], 2]
        offset = np.asarray(point.world_xyz, dtype=np.float64)[[0, 2]] - position_xz
        distance = float(np.linalg.norm(offset))
        if distance > 1.0 or distance == 0.0:
            return False
        return float(np.dot(offset, forward_xz)) < 0.0

    def _doorway_target_distance(
        self,
        current: Optional[Subgoal],
        *,
        camera_to_world: Any,
    ) -> Optional[float]:
        current_id = current.subgoal_id if current is not None else None
        point = self._doorway_waypoint
        if point is None or current_id != self._doorway_waypoint_subgoal_id:
            point = self._judge_target_point
            if point is None or current_id != self._judge_target_subgoal_id:
                # Third source: the spatial memory's model-located landmark
                # point for this stage. Without it the judge never learns
                # that the agent physically arrived, and 17/28 stalled
                # episodes were exactly "landmark point reached, stage never
                # credited" (docs 2026-08-31 §15). som/frontier targets stay
                # excluded: they are mid-route waypoints, not endpoints.
                landmark_target = self._spatial_landmark_target(current_id)
                if landmark_target is not None:
                    distance = landmark_target.current_distance_m()
                    if distance is not None:
                        return float(distance)
                return None
        if self._waypoint_passed(point, camera_to_world=camera_to_world):
            return 0.0
        return self._planar_waypoint_distance(
            point,
            camera_to_world=camera_to_world,
        )

    def _spatial_landmark_target(self, current_id: Optional[str]):
        """The active model-located landmark target for this stage, if any."""
        if not self.use_spatial_memory or current_id is None:
            return None
        target = self.spatial_memory.target
        if (
            target is not None
            and target.status == "active"
            and target.kind == "landmark"
            and target.subgoal_id == current_id
        ):
            return target
        return None

    def _judge_target_tolerance(self, current: Optional[Subgoal]) -> float:
        current_id = current.subgoal_id if current is not None else None
        if (
            self._doorway_waypoint is not None
            and current_id == self._doorway_waypoint_subgoal_id
        ):
            return self._doorway_waypoint_reach_tolerance_m
        if (
            self._judge_target_point is not None
            and current_id == self._judge_target_subgoal_id
        ):
            return self._judge_target_tolerance_m
        landmark_target = self._spatial_landmark_target(current_id)
        if landmark_target is not None:
            return float(landmark_target.tolerance_m)
        return COMMITTED_TARGET_REACHED_M

    def _locked_doorway_decision(
        self,
        current: Optional[Subgoal],
        *,
        camera_to_world: Any,
    ) -> Optional[NavigationDecision]:
        point = self._doorway_waypoint
        current_id = current.subgoal_id if current is not None else None
        if point is None or current_id != self._doorway_waypoint_subgoal_id:
            if point is not None:
                self._clear_doorway_waypoint()
            return None
        distance_m = self._planar_waypoint_distance(
            point,
            camera_to_world=camera_to_world,
        )
        # Reaching or walking past the localized threshold releases control
        # to the policy, but keeps the target as completion evidence until
        # Task Memory advances.
        if distance_m <= max(
            0.35, self._doorway_waypoint_reach_tolerance_m
        ) or self._waypoint_passed(point, camera_to_world=camera_to_world):
            # Hand over for good: a point left armed re-engages as soon as
            # the policy's next step carries the camera past the tolerance
            # again, and drags the agent back to a target it already met.
            self._clear_doorway_waypoint(keep_for_judge=True)
            return None
        best = self._doorway_waypoint_best_distance_m
        if best is None or distance_m < best - 0.10:
            self._doorway_waypoint_best_distance_m = distance_m
            self._doorway_waypoint_stagnant_steps = 0
        else:
            self._doorway_waypoint_stagnant_steps += 1
        if self._doorway_waypoint_stagnant_steps >= 16:
            self._clear_doorway_waypoint()
            return None
        intent: NavigationIntent = (
            self._navigation_phase
            if self._navigation_phase != "STOP"
            else "APPROACH_LANDMARK"
        )
        self.last_model_response = "reusing locked doorway waypoint"
        self.last_waypoint_raw_response = self.last_model_response
        self.last_waypoint_model_intent = intent
        self.last_waypoint_applied_intent = intent
        self.last_waypoint_model_action_mode = "EXECUTION"
        self.last_waypoint_applied_action_mode = "EXECUTION"
        self.last_waypoint_confidence = 1.0
        self.last_waypoint_evidence = (
            "continuing toward the previously localized structural doorway"
        )
        self.last_waypoint_guard_reason = (
            "reused stable doorway world waypoint; skipped waypoint VLM"
        )
        self.last_requested_turn_deg = None
        return NavigationDecision(
            stop=False,
            point=point,
            raw_response=self.last_model_response,
            action_mode="EXECUTION",
        )

    def _applied_action_mode(self) -> ActionMode:
        """Return the mode this step actually asks the controller to take."""
        return self.last_waypoint_applied_action_mode or "EXECUTION"

    # ------------------------------------------------------------------
    # Spatial Memory: map fusion, landmark anchoring and committed targets.
    # The doorway lock above keeps precedence for doorway stages; Spatial
    # Memory generalises the same "walk to a known world point without
    # asking the model again" to every other stage that walks somewhere.

    def _observe_spatial(self, intrinsics: Any, camera_to_world: Any) -> None:
        """Fuse this step's depth into the map; never breaks the step."""
        self.last_spatial_error = None
        depth_m = self._current_depth_m
        if depth_m is None or not self.use_spatial_memory:
            self.last_spatial_summary = "sp=off" if not self.use_spatial_memory else "sp=-"
            return
        try:
            calibration = (
                intrinsics
                if isinstance(intrinsics, CameraIntrinsics)
                else CameraIntrinsics.from_matrix(intrinsics)
            )
            floor_mask = None
            if self.actor.camera_height_m is not None:
                floor_mask = self.actor._floor_mask(
                    depth_m,
                    calibration,
                    np.asarray(camera_to_world, dtype=np.float64),
                )
            self._current_floor_mask = floor_mask
            self.spatial_memory.observe(
                step=self._spatial_step,
                depth_m=depth_m,
                intrinsics=calibration,
                camera_to_world=camera_to_world,
                floor_mask=floor_mask,
            )
        except Exception as exc:  # the map is an aid, not a dependency
            self.last_spatial_error = f"{type(exc).__name__}: {exc}"
        summary = self.spatial_memory.summary()
        if getattr(self, "_overrun_count", 0):
            summary += " ovr={}".format(self._overrun_count)
        crossings = getattr(self.spatial_memory, "crossings", ())
        if crossings:
            summary += " door_evts={}".format(len(crossings))
            if self.spatial_memory.crossing_detected_at(self._spatial_step):
                event = crossings[-1]
                summary += " EVT=door@({:.1f},{:.1f})w{:.2f}m@s{}".format(
                    event.position_xz[0], event.position_xz[1],
                    event.width_m, event.step,
                )
        self.last_spatial_summary = summary

    def _spatial_crossing_this_step(self) -> bool:
        """Map-measured doorway crossing confirmed by this step's observation."""
        if not self.use_spatial_memory:
            return False
        try:
            return bool(
                self.spatial_memory.crossing_detected_at(self._spatial_step)
            )
        except Exception:
            return False

    def _spatial_stage_eligible(self, current: Optional[Subgoal]) -> bool:
        """Stages that walk to a point: not turns, not the final approach.

        The final stage keeps the waypoint model in the loop every step so
        its STOP proposals are not skipped; a turn stage is a rotation.
        """
        return bool(
            current is not None
            and self._navigation_phase in ("FOLLOW_CORRIDOR", "APPROACH_LANDMARK")
            and not self._is_final_subgoal(current)
        )

    def _register_spatial_landmark(
        self, intrinsics: Any, camera_to_world: Any
    ) -> None:
        """Anchor the Captioner's located landmark in world coordinates."""
        landmark = self.last_landmark
        pixel = self.last_landmark_pixel
        depth_m = self._current_depth_m
        if (
            not self.use_spatial_memory
            or landmark is None
            or pixel is None
            or not landmark.visible
            or depth_m is None
            or self.task_memory is None
        ):
            return
        current = self.task_memory.get_current_subgoal()
        if current is None or self._landmark_subgoal_id != current.subgoal_id:
            return
        try:
            point = self.actor.waypoint_from_pixel(
                pixel, depth_m, intrinsics, camera_to_world
            )
        except ValueError:
            return
        try:
            self.spatial_memory.register_landmark(
                current.description,
                point.world_xyz,
                subgoal_id=current.subgoal_id,
                confidence=float(landmark.confidence),
                kind="doorway" if self._doorway_subgoal(current) else "landmark",
            )
        except Exception as exc:
            self.last_spatial_error = f"{type(exc).__name__}: {exc}"

    def _spatial_point(
        self,
        world_xyz: tuple[float, float, float],
        distance_m: float,
        *,
        image_shape: tuple[int, int],
    ) -> NavigationPoint:
        height, width = image_shape
        # The point is a world coordinate, not something in this frame; the
        # pixel is the conventional lower-centre stand-in the controller and
        # the video overlay expect.
        self.last_requested_normalized = (500, 750)
        self.last_requested_pixel = (width // 2, int(height * 0.75))
        self.last_requested_turn_deg = None
        return NavigationPoint(
            pixel_uv=self.last_requested_pixel,
            depth_m=float(distance_m),
            camera_xyz=(0.0, 0.0, -float(distance_m)),
            world_xyz=tuple(float(v) for v in world_xyz),
            on_floor=True,
        )

    def _stage_overrun_reorient(self, current: Optional[Subgoal]) -> bool:
        """Fire once per STAGE_PATH_OVERRUN_M walked on one unfinished stage.

        Runs before every waypoint-reuse path (doorway lock included): more
        walking than any single stage should need means the pursued point is
        wrong, so every held target is dropped and one PREVIEW look-around is
        forced to re-choose the heading from all views.
        """
        if current is None:
            return False
        # The final stage is included: a route whose last stage runs away
        # (ep 67: 20+ m on one held target) needs the same look-around, and
        # a PREVIEW does not interfere with the stop protocol.
        if self._subgoal_walked_m - self._overrun_fired_at_m <= STAGE_PATH_OVERRUN_M:
            return False
        self._overrun_fired_at_m = self._subgoal_walked_m
        self._force_preview_this_step = True
        self._overrun_count += 1
        self._clear_doorway_waypoint()
        if self.use_spatial_memory and self.spatial_memory.target is not None:
            self.spatial_memory.release_target("stage overrun")
        self.last_error_guard_reason = (
            "stage overrun: {:.1f} m walked on one stage; dropped held "
            "targets and requesting surrounding views".format(
                self._subgoal_walked_m
            )
        )
        return True

    def _spatial_target_decision(
        self,
        current: Optional[Subgoal],
        *,
        image_shape: tuple[int, int],
    ) -> Optional[NavigationDecision]:
        """Keep walking to the committed target; skip the waypoint VLM."""
        memory = self.spatial_memory
        if not self.use_spatial_memory:
            return None
        if not self._spatial_stage_eligible(current):
            if memory.target is not None:
                memory.release_target("stage not eligible")
            return None
        target = memory.active_target(current.subgoal_id)
        if target is None:
            return None
        mode, reason = self._motion_grounded_error_candidate("NONE")
        if mode is not None:
            if mode in ("IN_PLACE_SPIN", "TURN_OSCILLATION") and target.still_aligning(
                tolerance_deg=LOCKED_TARGET_TURN_TOLERANCE_DEG
            ):
                # The follower turns in place to face an off-axis target
                # before its first forward step: that rotation is the route,
                # not a loop (same exemption as the doorway lock).
                self.last_error_guard_reason = (
                    "kept spatial target: rotation is still aligning with "
                    "the off-axis target"
                )
            else:
                # A stalled or looping agent needs the policy's recovery,
                # which only runs on the model path.
                memory.release_target(f"motion error {mode}")
                self.last_error_guard_reason = f"released spatial target: {reason}"
                return None
        step = memory.next_waypoint()
        if step is None:
            return None
        world_xyz, remaining_m, how = step
        distance = target.current_distance_m()
        if distance is None:
            distance = remaining_m
        intent: NavigationIntent = (
            self._navigation_phase
            if self._navigation_phase != "STOP"
            else "APPROACH_LANDMARK"
        )
        self.last_model_response = f"reusing spatial target ({target.kind})"
        self.last_waypoint_raw_response = self.last_model_response
        self.last_waypoint_model_intent = intent
        self.last_waypoint_applied_intent = intent
        self.last_waypoint_model_action_mode = "EXECUTION"
        self.last_waypoint_applied_action_mode = "EXECUTION"
        self.last_waypoint_confidence = 1.0
        self.last_waypoint_evidence = (
            f"continuing to the committed {target.kind} target "
            f"{distance:.1f} m away ({how} route, {remaining_m:.1f} m)"
        )
        self.last_waypoint_guard_reason = (
            f"reused spatial target ({target.kind}, age {target.age(memory.step)}); "
            "skipped waypoint VLM"
        )
        self.last_waypoint_stop_disposition = None
        point = self._spatial_point(world_xyz, distance, image_shape=image_shape)
        self.last_spatial_summary = memory.summary()
        return NavigationDecision(
            stop=False,
            point=point,
            raw_response=self.last_model_response,
            action_mode="EXECUTION",
        )

    def _som_engine(self):
        """The engine for set-of-mark choices: a fine-tuned adapter when
        ``VLN_SOM_MODEL`` names one on the same vLLM server, else the main
        engine. Lets a LoRA trained for the choice serve only that call."""
        name = os.environ.get("VLN_SOM_MODEL")
        if not name:
            return self.llm
        if getattr(self, "_som_llm_name", None) != name:
            from agentflow.agents.engine.remote_qwen3vl import RemoteQwen3VL

            self._som_llm = RemoteQwen3VL(name)
            self._som_llm_name = name
        return self._som_llm

    def _candidates_from_external(
        self,
        external: Sequence[dict],
        *,
        intrinsics: CameraIntrinsics,
        camera_to_world: Any,
        image_shape: tuple[int, int],
    ) -> list:
        """Build Candidate objects from runner-supplied CWP waypoints.

        Keeps the exact contract generate_candidates has downstream: non-turn
        kinds count as in-view candidates, get relabeled 1..N left-to-right,
        and the L/R/B turn escape hatches are appended unchanged.
        """
        cam = np.asarray(camera_to_world, dtype=np.float64)
        cam_pos = cam[:3, 3]
        out = []
        for entry in list(external)[:SOM_MAX_CANDIDATES]:
            world = tuple(float(v) for v in entry["world_xyz"])
            dist = float(math.hypot(world[0] - cam_pos[0], world[2] - cam_pos[2]))
            uv = entry.get("pixel_uv")
            if uv is None:
                try:
                    uv = project_to_pixel(world, intrinsics, camera_to_world, image_shape)
                except Exception:
                    uv = None
            out.append(Candidate(
                "",
                world,
                dist,
                float(entry.get("bearing_deg", 0.0)),
                "opening",
                tuple(int(v) for v in uv) if uv is not None else None,
                entry.get("note") or "walkable opening about {:.1f} m away".format(dist),
            ))
        for lab, note in (("L", "turn left: area outside the current view"),
                          ("R", "turn right: area outside the current view"),
                          ("B", "turn around: the way back")):
            out.append(Candidate(lab, (0.0, 0.0, 0.0), 0.0,
                                 {"L": -90.0, "R": 90.0, "B": 180.0}[lab],
                                 "turn", None, note))
        return out

    def _som_decision(
        self,
        current: Optional[Subgoal],
        *,
        image: np.ndarray,
        depth_m: np.ndarray,
        intrinsics: Any,
        camera_to_world: Any,
    ) -> Optional[NavigationDecision]:
        """Let the model choose among map-generated markers, then commit.

        Returns None to fall back to the pixel-proposal path: SoM is off,
        the stage is a turn or the final approach, recovery is active, no
        candidate could be generated, or the reply was unusable twice.
        """
        self.last_som_choice = None
        self.last_som_candidates = []
        self.last_som_error = None
        self.last_som_image = None
        if not self.use_som or not self._spatial_stage_eligible(current):
            return None
        # The same recovery evaluation the pixel path runs; an active
        # recovery takes the deterministic path and must not be voted twice.
        if self._recovery_mode_for_step() is not None:
            return None
        located = self._located_landmark_decision(
            current,
            image_shape=image.shape[:2],
            depth_m=depth_m,
            intrinsics=intrinsics,
            camera_to_world=camera_to_world,
        )
        if located is not None:
            return located
        memory = self.spatial_memory
        calibration = (
            intrinsics
            if isinstance(intrinsics, CameraIntrinsics)
            else CameraIntrinsics.from_matrix(intrinsics)
        )
        try:
            if self._external_candidates:
                # Runner-side CWP candidates replace floor-openings; the
                # navmesh filter, relabeling, SoM drawing, prompting and
                # target commitment downstream stay unchanged.
                candidates = self._candidates_from_external(
                    self._external_candidates,
                    intrinsics=calibration,
                    camera_to_world=camera_to_world,
                    image_shape=image.shape[:2],
                )
            else:
                landmark = memory.landmark_for_subgoal(current.subgoal_id)
                candidates = generate_candidates(
                    depth_m=depth_m,
                    floor_mask=self._current_floor_mask,
                    intrinsics=calibration,
                    camera_to_world=camera_to_world,
                    image_shape=image.shape[:2],
                    frontiers=memory.frontiers(),
                    landmark_xyz=landmark.world_xyz if landmark is not None else None,
                    landmark_note=(
                        f"the active subgoal's landmark ({current.description})"
                        if landmark is not None else ""
                    ),
                    floor_y=memory.floor_y if memory.position is not None else None,
                    max_in_view=SOM_MAX_CANDIDATES,
                )
        except Exception as exc:
            self.last_som_error = f"{type(exc).__name__}: {exc}"
            return None
        candidates = memory.filter_navigable(candidates)
        if not any(c.kind != "turn" for c in candidates):
            self.last_som_error = "no reachable in-view candidates"
            return None
        self.last_som_candidates = [
            {
                "label": c.label, "kind": c.kind, "distance_m": round(c.distance_m, 2),
                "bearing_deg": round(c.bearing_deg, 1), "pixel_uv": c.pixel_uv,
                "world_xyz": [round(v, 2) for v in c.world_xyz],
            }
            for c in candidates
        ]
        annotated = annotate_image(image, candidates)
        self.last_som_image = annotated
        listing = describe_candidates(candidates)
        text = "\n".join((
            self.task_memory.current_subgoal_context() if self.task_memory else "",
            f"Full route instruction: {self.task_instruction}",
            f"Required navigation phase: {self._navigation_phase}.",
            self._landmark_context_for_waypoint(current),
            self._behavior_context(),
            "Options:",
            listing,
            "Choose one option label.",
        ))
        labels = {c.label: c for c in candidates}
        chosen = None
        response_text = ""
        if self._oracle_goal_xyz is not None:
            # DIAGNOSTIC: the candidate nearest the goal (planar), no model.
            gx, gz = self._oracle_goal_xyz[0], self._oracle_goal_xyz[2]
            chosen = min(
                candidates,
                key=lambda c: (c.world_xyz[0] - gx) ** 2 + (c.world_xyz[2] - gz) ** 2,
            )
            response_text = '{"choice":"%s","confidence":1.0,"evidence":"oracle"}' % chosen.label
            self.last_waypoint_confidence = 1.0
            self.last_waypoint_evidence = "oracle choice (diagnostic)"
        for attempt in range(2 if chosen is None else 0):
            prompt = text if attempt == 0 else (
                text + "\nThe previous reply was not a valid option label; "
                "reply with the exact JSON shape and one listed label."
            )
            try:
                response = self._som_engine()(
                    [prompt, encode_png(annotated)],
                    system_prompt=SOM_PROMPT,
                    image_min_pixels=VLM_IMAGE_MIN_PIXELS,
                    image_max_pixels=VLM_IMAGE_MAX_PIXELS,
                    max_tokens=STRUCTURED_VLM_MAX_TOKENS,
                    temperature=0,
                )
                response_text = str(response)
                payload = self._extract_json_object(response_text)
                choice = str(payload.get("choice", "")).strip().upper()
                if choice in labels:
                    chosen = labels[choice]
                    self.last_waypoint_confidence = float(payload.get("confidence") or 0.0)
                    self.last_waypoint_evidence = str(payload.get("evidence") or "")
                    break
            except Exception as exc:  # malformed reply: retry once, then fall back
                self.last_som_error = f"{type(exc).__name__}: {exc}"
        self.last_som_raw_response = response_text
        self.last_model_response = response_text
        self.last_waypoint_raw_response = response_text
        if chosen is None:
            self.last_som_error = self.last_som_error or "reply was not a listed label"
            return None
        self.last_som_choice = chosen.label
        self.last_recovery_mode = None
        self.last_waypoint_model_action_mode = "EXECUTION"
        self.last_waypoint_applied_action_mode = "EXECUTION"
        self.last_waypoint_model_intent = self._navigation_phase
        self.last_waypoint_applied_intent = self._navigation_phase
        self.last_waypoint_stop_disposition = None
        self._final_stop_evidence.append(False)
        if chosen.kind == "turn" and chosen.world_xyz is None:
            return None
        # Walk to the chosen point: a marker, or the unexplored area behind
        # a turn option. Age scales with distance so a far marker is not
        # abandoned half-way; stagnation still releases a blocked one.
        target_xyz = chosen.world_xyz
        if chosen.kind != "turn" and chosen.distance_m > SOM_MAX_TARGET_DISTANCE_M:
            # Walk part of the way toward a far marker, then decide again.
            position = np.asarray(camera_to_world, dtype=np.float64)[:3, 3]
            direction = np.asarray(chosen.world_xyz) - position
            direction[1] = 0.0
            norm = float(np.linalg.norm(direction))
            if norm > 1e-6:
                scaled = position + direction / norm * SOM_MAX_TARGET_DISTANCE_M
                target_xyz = (float(scaled[0]), float(chosen.world_xyz[1]), float(scaled[2]))
        max_age = int(min(SOM_TARGET_MAX_AGE_STEPS, chosen.distance_m / 0.2 + 6))
        committed = memory.commit_target(
            target_xyz,
            kind="som" if chosen.kind != "turn" else "frontier",
            subgoal_id=current.subgoal_id,
            reason=f"marker {chosen.label} ({chosen.kind}): {self.last_waypoint_evidence[:60]}",
            tolerance_m=COMMITTED_TARGET_REACHED_M,
            max_age_steps=max(8, max_age),
            stagnation_steps=SPATIAL_TARGET_STAGNATION_STEPS,
        )
        if committed is None:
            self.last_som_error = "chosen marker is not reachable"
            return None
        decision = self._spatial_target_decision(current, image_shape=image.shape[:2])
        if decision is None:
            return None
        if chosen.pixel_uv is not None:
            self.last_requested_pixel = chosen.pixel_uv
            self.last_requested_normalized = (
                int(chosen.pixel_uv[0] * 1000 / max(image.shape[1] - 1, 1)),
                int(chosen.pixel_uv[1] * 1000 / max(image.shape[0] - 1, 1)),
            )
        self.last_waypoint_guard_reason = (
            f"set-of-mark: model chose marker {chosen.label} ({chosen.kind}, "
            f"{chosen.distance_m:.1f} m, {chosen.bearing_deg:+.0f}°) of "
            f"{len(candidates)} options"
        )
        return decision

    def _located_landmark_decision(
        self,
        current: Optional[Subgoal],
        *,
        image_shape: tuple[int, int],
        depth_m: np.ndarray,
        intrinsics: Any,
        camera_to_world: Any,
    ) -> Optional[NavigationDecision]:
        """Walk to the Captioner-located landmark and lock it as the stage's
        committed point, exactly as the pixel path does through
        ``_steer_to_visible_landmark`` and ``_maybe_lock_doorway_waypoint``.

        Subgoal completion for doorway and landmark stages is judged on
        arrival at that committed point, so a set-of-mark choice must not
        bypass it when the landmark is in view.
        """
        landmark = self.last_landmark
        pixel = self.last_landmark_pixel
        if pixel is None or not self._landmark_located_for_current_subgoal(current):
            return None
        try:
            point = self.actor.waypoint_from_pixel(
                pixel, depth_m, intrinsics, camera_to_world
            )
        except ValueError:
            return None
        if self.actor.camera_height_m is not None and not point.on_floor:
            return None
        if self._planar_waypoint_distance(point, camera_to_world=camera_to_world) < 0.4:
            return None  # already beside it; the judge finishes the stage
        committed = self.spatial_memory.commit_target(
            point.world_xyz,
            kind="landmark",
            subgoal_id=current.subgoal_id,
            reason=f"located landmark: {landmark.evidence[:60]}",
            tolerance_m=COMMITTED_TARGET_REACHED_M,
            max_age_steps=SOM_TARGET_MAX_AGE_STEPS,
            stagnation_steps=SPATIAL_TARGET_STAGNATION_STEPS,
        )
        if committed is None:
            return None
        intent: NavigationIntent = (
            self._navigation_phase
            if self._navigation_phase != "STOP"
            else "APPROACH_LANDMARK"
        )
        self.last_model_response = "steering to the located landmark"
        self.last_waypoint_raw_response = self.last_model_response
        self.last_waypoint_model_intent = intent
        self.last_waypoint_applied_intent = intent
        self.last_waypoint_model_action_mode = "EXECUTION"
        self.last_waypoint_applied_action_mode = "EXECUTION"
        self.last_waypoint_confidence = float(landmark.confidence)
        self.last_waypoint_evidence = (
            "steered to the floor beneath the Captioner-localized landmark: "
            f"{landmark.evidence}"
        )
        self.last_waypoint_guard_reason = (
            "overrode model: landmark is localized in the current image; "
            "walking to the floor beneath the located landmark"
        )
        self.last_waypoint_stop_disposition = None
        self.last_recovery_mode = None
        self._final_stop_evidence.append(False)
        # Lock the navmesh-snapped point so the completion judge measures
        # arrival against the same target the follower is given.
        self._maybe_lock_doorway_waypoint(
            replace(point, world_xyz=committed.world_xyz, on_floor=True),
            camera_to_world=camera_to_world,
        )
        decision = self._spatial_target_decision(current, image_shape=image_shape)
        if decision is None:
            return None
        self.last_requested_pixel = pixel
        self.last_requested_normalized = self.last_landmark_normalized or (500, 750)
        self.last_waypoint_guard_reason = (
            "overrode model: landmark is localized in the current image; "
            "walking to the floor beneath the located landmark"
        )
        return decision

    def _spatial_frontier_decision(
        self,
        current: Optional[Subgoal],
        *,
        image_shape: tuple[int, int],
    ) -> Optional[NavigationDecision]:
        """Walk to the nearest unexplored boundary instead of straight ahead.

        Only when the model could not decide and the policy fell back to the
        lower-centre "walk straight" pixel: a refused second PREVIEW or an
        invalid reply. A frontier is a place the map has not seen, which is
        where the route continuation has to be.
        """
        reason = self.last_waypoint_guard_reason or ""
        if not self.use_spatial_memory:
            return None
        if not re.search(r"safe waypoint instead|safe fallback", reason):
            return None
        if not self._spatial_stage_eligible(current):
            return None
        memory = self.spatial_memory
        try:
            frontier = memory.choose_frontier()
        except Exception as exc:
            self.last_spatial_error = f"{type(exc).__name__}: {exc}"
            return None
        if frontier is None:
            return None
        if memory.commit_target(
            frontier,
            kind="frontier",
            subgoal_id=current.subgoal_id,
            reason="model undecided; nearest unexplored boundary",
            tolerance_m=COMMITTED_TARGET_REACHED_M,
            max_age_steps=SPATIAL_TARGET_MAX_AGE_STEPS,
            stagnation_steps=SPATIAL_TARGET_STAGNATION_STEPS,
        ) is None:
            return None
        decision = self._spatial_target_decision(current, image_shape=image_shape)
        if decision is not None:
            self.last_waypoint_guard_reason = (
                f"{reason}; redirected to a Spatial Memory frontier"
            )
        return decision

    def _commit_spatial_target(
        self,
        point: NavigationPoint,
        *,
        camera_to_world: Any,
        kind: Optional[str] = None,
    ) -> None:
        """Remember this step's waypoint as the point to keep walking to."""
        current = (
            self.task_memory.get_current_subgoal()
            if self.task_memory is not None
            else None
        )
        if not self.use_spatial_memory or not self._spatial_stage_eligible(current):
            return
        if (
            self._doorway_waypoint is not None
            and self._doorway_waypoint_subgoal_id == current.subgoal_id
        ):
            return  # the doorway lock owns this stage's target
        if self.actor.camera_height_m is not None and not point.on_floor:
            return
        # A recovery waypoint is re-evaluated every step by design; holding
        # it would freeze the agent on the first escape direction it tried.
        if self.last_recovery_mode is not None or re.search(
            r"recovery", self.last_waypoint_guard_reason or "", flags=re.IGNORECASE
        ):
            return
        distance = self._planar_waypoint_distance(
            point, camera_to_world=camera_to_world
        )
        if distance < SPATIAL_TARGET_MIN_COMMIT_M:
            return
        if kind is None:
            reason = self.last_waypoint_guard_reason or ""
            kind = (
                "landmark"
                if re.search(r"overrode model|beneath the located landmark", reason)
                else "model_waypoint"
            )
        self.spatial_memory.commit_target(
            point.world_xyz,
            kind=kind,
            subgoal_id=current.subgoal_id,
            reason=(self.last_waypoint_evidence or "")[:80],
            tolerance_m=COMMITTED_TARGET_REACHED_M,
            max_age_steps=SPATIAL_TARGET_MAX_AGE_STEPS,
            stagnation_steps=SPATIAL_TARGET_STAGNATION_STEPS,
        )
        self.last_spatial_summary = self.spatial_memory.summary()

    def _selected_preview_view(
        self,
        views: Sequence[PreviewView],
    ) -> int:
        """Deposit the views in working memory and read back the chosen one.

        Depositing triggers the Captioner, so the selection is read
        immediately afterwards. If a replacement selector declines, fall back
        to the most forward view and record that fallback in the guard reason.
        """
        forward_index = min(
            range(len(views)),
            key=lambda index: abs(views[index].yaw_deg),
        )
        self.last_preview_selection = None
        # Kept apart from ``last_waypoint_guard_reason``: ``_select_pixel``
        # clears that one on entry, and these are different guards anyway.
        self.last_preview_guard_reason = None
        if not isinstance(self.temporal_memory, TemporalMemory):
            self.last_preview_guard_reason = (
                "preview views discarded: working memory cannot hold them"
            )
            return forward_index

        self.temporal_memory.set_preview_views(views)
        selection = self.temporal_memory.preview_selection()
        self.last_preview_selection = selection
        if selection is not None:
            # Direction is semantic: the widest free space may be the room we
            # are meant to leave. Depth is still validated when the selected
            # floor point is back-projected, but it must not replace the
            # Captioner's chosen doorway or landmark view.
            return selection.view_index

        error = self.temporal_memory.last_preview_error()
        self.last_preview_guard_reason = (
            "preview view selector failed ({}); used the most forward "
            "view".format(error)
            if error is not None
            else "preview view selector declined; used the most forward view"
        )
        return forward_index

    def _record_timings(
        self,
        timings: dict[str, float],
        started: float,
    ) -> None:
        timings["total_ms"] = (time.perf_counter() - started) * 1000
        self.last_timings = timings

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

    def _update_preview_progress(self) -> None:
        """Re-arm PREVIEW only after net displacement from its anchor."""
        if (
            not self._preview_requires_progress
            or self._preview_anchor_position_xz is None
            or self._previous_position is None
        ):
            return
        displacement = float(np.linalg.norm(
            self._previous_position[[0, 2]] - self._preview_anchor_position_xz
        ))
        if displacement < PREVIEW_REARM_TRANSLATION_M:
            return
        self._preview_requires_progress = False
        self._preview_anchor_position_xz = None

    def _is_final_subgoal(self, subgoal: Optional[Subgoal]) -> bool:
        return bool(
            subgoal is not None
            and self.subgoals
            and subgoal.subgoal_id == self.subgoals[-1].subgoal_id
        )

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
        if re.search(TURN_AROUND_PATTERN, description):
            # A half turn in either direction; it is driven and measured
            # like a left turn, against the larger target below.
            return "TURN_LEFT"
        # The last stage is the destination whatever words the planner chose:
        # only FINAL_APPROACH keeps the waypoint model, and with it STOP, in
        # the loop every step.
        if self._is_final_subgoal(subgoal):
            return "FINAL_APPROACH"
        # "beside the bed" reads like an arrival, but an intermediate stage
        # approaches its landmark: only the final stage may run the STOP
        # protocol, and only APPROACH_LANDMARK commits to a landmark point.
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
            # A new stage can introduce a new turn or landmark at the same
            # physical position, so an earlier stage's preview must not block
            # it from looking around.
            self._preview_requires_progress = False
            self._preview_anchor_position_xz = None
            self._final_stop_evidence.clear()
            self._corridor_forward_streak = 0
            self._corridor_heading_yaw_deg = None
            self._subgoal_net_yaw_deg = 0.0
            self._subgoal_walked_m = 0.0
            self._overrun_fired_at_m = 0.0
            self._turn_follow_phase_started = False
        self._navigation_phase = (
            self._post_turn_phase(subgoal)
            if self._turn_follow_phase_started and subgoal is not None
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
        # The requested rotation is measured, not judged from frames: the
        # model asks for the same 15-degree turn every step because a single
        # image cannot show how far it has already turned. Once the camera has
        # rotated far enough the stage's movement part takes over, and past a
        # half circle it does so unconditionally rather than spin in place.
        turn_progress = self._turn_progress_deg(subgoal)
        around = self._is_turn_around(subgoal)
        alignment = TURN_AROUND_MIN_PROGRESS_DEG if around else TURN_ALIGNMENT_DEG
        abandon = TURN_ABANDON_DEG + 90.0 if around else TURN_ABANDON_DEG
        # The turn exists to bring the stage's landmark into view. Once the
        # Captioner has it near the image centre, turning on to the nominal
        # angle would only turn away from it again.
        if (
            turn_progress < alignment
            and abs(self._subgoal_net_yaw_deg) < abandon
            and not (
                turn_progress >= TURN_TARGET_MIN_PROGRESS_DEG
                and self._landmark_centred_for_current_subgoal(subgoal)
            )
        ):
            return
        if not self._turn_follow_phase_started:
            self._turn_follow_phase_started = True
            self._corridor_forward_streak = 0
            self._corridor_heading_yaw_deg = None
        self._navigation_phase = self._post_turn_phase(subgoal)

    def _landmark_centred_for_current_subgoal(
        self,
        subgoal: Optional[Subgoal],
    ) -> bool:
        landmark = self.last_landmark
        return bool(
            subgoal is not None
            and landmark is not None
            and self._landmark_subgoal_id == subgoal.subgoal_id
            and landmark.visible
            and landmark.u is not None
            and landmark.confidence >= LANDMARK_STEER_MIN_CONFIDENCE
            and abs(landmark.u - 500) <= TURN_TARGET_CENTRED_U
        )

    @staticmethod
    def _is_turn_around(subgoal: Optional[Subgoal]) -> bool:
        return bool(
            subgoal is not None
            and re.search(TURN_AROUND_PATTERN, subgoal.description, re.IGNORECASE)
        )

    def _turn_progress_deg(self, subgoal: Optional[Subgoal]) -> float:
        """Rotation so far in the direction the subgoal asks for, in degrees."""
        if self._is_turn_around(subgoal):
            return abs(self._subgoal_net_yaw_deg)  # either way round counts
        initial_phase = self._phase_for_subgoal(subgoal)
        if initial_phase == "TURN_LEFT":
            return -self._subgoal_net_yaw_deg
        if initial_phase == "TURN_RIGHT":
            return self._subgoal_net_yaw_deg
        return 0.0

    def _post_turn_phase(self, subgoal: Subgoal) -> NavigationIntent:
        """Phase for the rest of a turn stage once the turn itself is done."""
        remainder = re.sub(
            r"\bturn\s+(?:to\s+the\s+)?(?:left|right)\b[\s,]*(?:and|then)?",
            " ",
            subgoal.description,
            flags=re.IGNORECASE,
        ).strip()
        remainder = re.sub(
            TURN_AROUND_PATTERN + r"[\s,]*(?:and|then)?", " ", remainder, flags=re.IGNORECASE
        ).strip()
        if not remainder:
            return "FOLLOW_CORRIDOR"
        phase = self._phase_for_subgoal(
            replace(subgoal, description=remainder)
        )
        if phase in ("TURN_LEFT", "TURN_RIGHT", "STOP"):
            return "FOLLOW_CORRIDOR"
        if phase == "FINAL_APPROACH" and not self._is_final_subgoal(subgoal):
            # "beside the bed" reads as an arrival, but only the last stage
            # may approach with STOP on the table.
            return "APPROACH_LANDMARK"
        return phase

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
            and self.last_landmark.confidence >= LANDMARK_STEER_MIN_CONFIDENCE
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
                    "The previous response failed strict validation "
                    f"({last_error}). Write one stage per line as "
                    "id|description|completion criterion, with IDs counting "
                    "1, 2, 3 and no gaps. Output only those lines: no JSON, "
                    "no brackets, no commentary."
                ]
            response = self.llm(
                content,
                system_prompt=SUBGOAL_PROMPT,
                # A plan that nests its stages outgrows the budget it would
                # need when flat, and a response truncated before its closing
                # brackets cannot be parsed at all.
                max_tokens=1024,
                # Greedy decoding first; a retry has to be allowed to diverge,
                # or it reproduces the response that just failed.
                temperature=0.0 if attempt == 0 else 0.7,
            )
            response_text = str(response)
            self.last_subgoal_response = response_text
            try:
                subgoals = parse_subgoal_plan(
                    response_text,
                    instruction=normalized_instruction,
                )
                break
            except Exception as exc:
                last_error = exc
        else:
            raise ValueError(
                "Actor returned invalid subgoal JSON after "
                f"{SUBGOAL_GENERATION_ATTEMPTS} attempts "
                # Without the underlying exception a plain AttributeError or
                # NameError in the parsing path is indistinguishable from a
                # genuinely malformed model response.
                f"({type(last_error).__name__}: {last_error}): "
                f"{response_text!r}"
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
            self.temporal_memory = TemporalMemory(
                captioner=TemporalCaptioner(
                    engine=self.llm,
                    config=TemporalCaptionerConfig(
                        enable_error_detection=True,
                        min_error_detection_frames=8,
                        max_image_edge=TEMPORAL_MAX_IMAGE_EDGE,
                        # One compact response now carries landmark,
                        # completion, error, and final-target evidence.
                        max_tokens=max(
                            CAPTIONER_MAX_TOKENS, STRUCTURED_VLM_MAX_TOKENS
                        ),
                    ),
                ),
                task_memory=self.task_memory,
            )
        else:
            self.temporal_memory.reset()
        self.spatial_memory.reset()
        self._reset_runtime_state()

    # The landmark and waypoint policies prompt the shared checkpoint through
    # the agent, so both model helpers stay reachable under their mixin names.
    def _rgb_to_png(self, rgb: np.ndarray) -> bytes:
        return self.actor.rgb_to_png(rgb)

    def _as_rgb_array(self, rgb: Any) -> np.ndarray:
        return self.actor.as_rgb_array(rgb)

    def _extract_json_object(self, response: str) -> dict[str, Any]:
        return self.actor.extract_json_object(response)


__all__ = (
    "CameraIntrinsics",
    "DEFAULT_MODEL_PATH",
    "TemporalMemory",
    "NavigationDecision",
    "NavigationPoint",
    "POINT_PROMPT",
    "PreviewView",
    "Subgoal",
    "SUBGOAL_PROMPT",
    "VLNAgent",
)
