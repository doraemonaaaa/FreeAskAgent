"""Configuration, prompts, and validated model outputs for VLN agent v3."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


DEFAULT_MODEL_PATH = "models/JoyAI-VL-Interaction"
# Planning runs once per episode and its retries are sampled, so a third
# attempt is cheap insurance against one malformed plan aborting the run.
SUBGOAL_GENERATION_ATTEMPTS = 3
WAYPOINT_GENERATION_ATTEMPTS = 2
LANDMARK_GENERATION_ATTEMPTS = 2
ERROR_CONFIDENCE_THRESHOLD = 0.9
ERROR_CONFIRMATION_WINDOW = 5
ERROR_CONFIRMATION_VOTES = 4
RECOVERY_HOLD_STEPS = 2
RECOVERY_FORWARD_U = 500
RECOVERY_FORWARD_V = 750
RECOVERY_TURN_U = 250
RECOVERY_TURN_V = 500
RECOVERY_LATERAL_DISTANCE_M = 1.5
STALL_EVIDENCE_FRAMES = 4
STALL_TRANSLATION_LIMIT_M = 0.20
TURN_EVIDENCE_DEG = 5.0
DOORWAY_MIN_PATH_M = 0.50
DOORWAY_CROSSING_M = 0.25
DOORWAY_NEAR_CROSSING_M = 0.50
DOORWAY_SIDE_CROSSING_M = 0.75
DOORWAY_MAX_CROSSING_YAW_DEG = 30.0
DOORWAY_LANDMARK_CONFIDENCE = 0.70
# Retain an anchor frame plus the eight-frame recent window used by error
# detection.  A larger completion window substantially increases the peak
# visual-token allocation once dual-window inference becomes active.
MAX_COMPLETION_EVIDENCE_FRAMES = 9
RECENT_COMPLETION_EVIDENCE_FRAMES = 8
# Bound every online visual request.  Waypoint and landmark requests otherwise
# retain the source RGB resolution, whereas temporal-memory requests are
# resized by TemporalCaptioner.  Keeping one shared budget avoids an abrupt
# allocation increase when temporal error detection starts at frame eight.
VLM_IMAGE_MIN_PIXELS = 64**2
VLM_IMAGE_MAX_PIXELS = 224**2
TEMPORAL_MAX_IMAGE_EDGE = 160
STRUCTURED_VLM_MAX_TOKENS = 64
# The landmark schema is the longest of the structured outputs, and adding the
# u/v pair pushes a typical reply past the shared 64-token budget, which would
# truncate the JSON before its closing brace and fail validation.
LANDMARK_MAX_TOKENS = 128
BEHAVIOR_HISTORY_SIZE = 8
LANDMARK_HISTORY_SIZE = 6
CORRIDOR_LOCK_FORWARD_STEPS = 2
CORRIDOR_WAYPOINT_DEVIATION = 75
DECISION_POINT_MIN_PATH_M = 2.0
DECISION_POINT_TURN_EVIDENCE_DEG = 10.0
TURN_ALIGNMENT_DEG = 60.0
FINAL_TARGET_EVIDENCE_WINDOW = 3
FINAL_TARGET_EVIDENCE_VOTES = 2
FINAL_TARGET_MAX_WAYPOINT_DEPTH_M = 2.75
FINAL_TARGET_MIN_PATH_M = 0.25

NavigationIntent = Literal[
    "FOLLOW_CORRIDOR",
    "APPROACH_LANDMARK",
    "TURN_LEFT",
    "TURN_RIGHT",
    "FINAL_APPROACH",
    "STOP",
]

POINT_PROMPT = """You are an indoor navigation waypoint selector. Steer only
toward the active navigation subgoal. Select one visible, obstacle-free floor
point that makes progress. Never select walls, furniture, stairs, or an image
border.

First choose a navigation intent, then a waypoint consistent with it:
- FOLLOW_CORRIDOR: continue through the main corridor; do not enter side rooms.
- APPROACH_LANDMARK: approach the active subgoal's visible landmark.
- TURN_LEFT / TURN_RIGHT: execute the active subgoal's requested turn.
- FINAL_APPROACH: approach the final destination.
- STOP: stop only at the final destination.

Coordinates are normalized integers from 0 to 1000, independent of image
resolution: u=0 is the left edge, u=1000 is the right edge, v=0 is the top,
and v=1000 is the bottom. Reply only with one exact JSON object:
{"stop":false,"intent":"FOLLOW_CORRIDOR","u":integer,"v":integer,"confidence":0.0,"evidence":"brief visual reason"}.

The target may be one subgoal of a longer route, numbered "(n of m)". Reply
{"stop":true,"intent":"STOP","confidence":0.0,"evidence":"brief visual reason"}
only when the active subgoal is final and the destination is at immediate
stopping distance. A STOP response is a proposal: the controller requires
repeated near-field evidence or completed Task Memory before issuing Habitat
STOP. Never propose STOP during an intermediate subgoal. The required
navigation phase in the user prompt is otherwise authoritative. When it is
FOLLOW_CORRIDOR and the target landmark is not near, keep the main corridor
ahead and reject attractive side doorways or rooms. On FINAL_APPROACH, treat
ordinary visual aliases as the same named target and explicitly name the
visible target plus its near-field relation in evidence. If further motion
would circle or pass a target already beside the camera, propose STOP."""

SUBGOAL_PROMPT = """You are an indoor navigation task planner. Decompose the
instruction into a short ordered list of stages. Write one stage per line in
exactly this form, and output nothing else — no JSON, no brackets, no bullets,
no commentary, no blank lines:
id|description|completion criterion

Example of a complete two-stage answer:
1|Walk down the hall to the marked doorway|The camera is at the doorway where the next left turn can be executed
2|Turn left and enter the kitchen|The camera has crossed the threshold and the kitchen interior is central in the view

Rules:
1. One stage per line, in instruction order, with IDs counting 1, 2, 3 with no
   gaps. Each line holds exactly three fields separated by two "|" characters;
   never write "|" inside a description or a criterion.
2. Every completion criterion must describe a visually verifiable endpoint,
   never an action merely being attempted or currently in progress.
3. A movement clause immediately followed by a turn at a landmark completes
   only when the agent reaches that named landmark or decision point. For
   example, "walk down the hall; turn left at the marked doorway" means the
   walking stage completes upon reaching that doorway, not while moving.
   Do not require stopping at an intermediate landmark. Its criterion must not
   claim that the agent has already passed, turned at, or aligned beyond that
   landmark; those belong to the following turn stage.
4. A doorway-exit stage completes when the camera reaches or crosses the
   doorway threshold and the destination room becomes the central/dominant
   view. Brief peripheral visibility of the starting room does not invalidate
   a crossing; do not require it to disappear completely.
5. A turn completes only after reaching its landmark, executing the requested
   turn, and aligning with the next route.
6. Final arrival requires reaching and stopping beside the destination;
   merely seeing it at a distance is insufficient.
7. Never add route geometry absent from the instruction. In particular,
   "around" an object does not mean a full circuit, returning to the start,
   or completing a loop unless the instruction explicitly says so."""

LANDMARK_PROMPT = """You are a strict visual landmark tracker for indoor
navigation. Inspect the current first-person image for the landmark or endpoint
named by the active subgoal and completion criterion. Recent tracker states,
measured motion, the next route stage, and behavior history are context.

Return only this exact JSON:
{"visible":false,"direction":"UNKNOWN","proximity":"UNKNOWN","passed":false,"destination_dominant":false,"u":null,"v":null,"confidence":0.0,"evidence":"brief visual reason"}

Rules:
1. direction is LEFT, CENTER, RIGHT, or UNKNOWN relative to the current image.
1b. u and v locate the center of the landmark itself in the current image, as
   normalized integers from 0 to 1000 independent of resolution: u=0 is the
   left edge, u=1000 the right edge, v=0 the top, v=1000 the bottom. Give both
   whenever visible is true, and make them agree with direction. Use null for
   both when the landmark is not visible in this image.
2. proximity is FAR, NEAR, AT, or UNKNOWN. AT means the camera is at the
   landmark/threshold, not merely that the landmark fills part of the image.
   For an intermediate movement stage followed by a turn, the tracked target
   is the doorway/junction decision point where that next turn can be executed.
   Indoor scans may not render material labels clearly; a spatially matching
   decision point may be AT when its material is uncertain. Require ordered
   forward progress and the correct turn geometry.
3. passed is true only when the ordered history plus the current image shows
   that the camera crossed or moved beyond the landmark. For a doorway, a
   sequence that approaches the opening, moves forward through its plane, and
   ends with the destination-room interior central/dominant is sufficient
   even if the old room remains visible at the image edge. Do not infer passed
   from disappearance alone.
4. destination_dominant is true only when the destination room itself and its
   defining contents occupy the center or most of the current image while the
   old doorway or starting room is only at an edge. Seeing the destination
   only through a doorway opening is not dominant.
5. If visible=false and passed=false, direction and proximity must be UNKNOWN.
6. Track only the active subgoal's landmark. Do not report unrelated objects.
7. Use the next route stage only to disambiguate the correct decision point.
   Do not match an earlier unrelated side doorway, especially one on the
   opposite side from the requested next turn.
8. On a final arrival stage, ordinary visual or category aliases count as the
   named target when their spatial relation matches the instruction.
9. confidence is from 0 to 1 and evidence must cite what is visible now."""


class SubgoalOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    subgoal_id: str
    description: str
    completion_criteria: str

    @field_validator("subgoal_id", "description", "completion_criteria")
    @classmethod
    def non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("field must not be empty")
        return value


class SubgoalPlanOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    subgoals: list[SubgoalOutput]

    @model_validator(mode="after")
    def ordered_ids(self) -> "SubgoalPlanOutput":
        if not self.subgoals:
            raise ValueError("subgoals must not be empty")
        actual = [item.subgoal_id for item in self.subgoals]
        expected = [str(index) for index in range(1, len(actual) + 1)]
        if actual != expected:
            raise ValueError(
                "subgoal IDs must be unique consecutive strings starting at 1"
            )
        return self


class WaypointOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    stop: bool
    intent: NavigationIntent
    u: Optional[int] = None
    v: Optional[int] = None
    confidence: float
    evidence: str

    @model_validator(mode="after")
    def valid_stop_or_point(self) -> "WaypointOutput":
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        self.evidence = self.evidence.strip()
        if not self.evidence:
            raise ValueError("evidence must not be empty")
        if self.stop:
            if self.intent != "STOP":
                raise ValueError("STOP output requires intent=STOP")
            for value in (self.u, self.v):
                if value is not None and not 0 <= value <= 1000:
                    raise ValueError(
                        "normalized coordinates must be in [0, 1000]"
                    )
            return self
        if self.intent == "STOP":
            raise ValueError("non-STOP output cannot use intent=STOP")
        if self.u is None or self.v is None:
            raise ValueError("non-STOP output requires u and v")
        if not 0 <= self.u <= 1000 or not 0 <= self.v <= 1000:
            raise ValueError("normalized coordinates must be in [0, 1000]")
        return self


class LandmarkOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    visible: bool
    direction: Literal["LEFT", "CENTER", "RIGHT", "UNKNOWN"]
    proximity: Literal["FAR", "NEAR", "AT", "UNKNOWN"]
    passed: bool
    destination_dominant: bool = False
    # Normalized image location of the landmark, used to draw it on the frame.
    # Deliberately optional and never required: a missing or out-of-range pair
    # costs a marker on the visualization, whereas failing validation here
    # would retry the call and then discard an otherwise usable tracker state.
    u: Optional[int] = None
    v: Optional[int] = None
    confidence: float
    evidence: str

    @field_validator("u", "v", mode="before")
    @classmethod
    def tolerant_pixel(cls, value: object) -> Optional[int]:
        """Accept any plausible number and drop anything else.

        Runs before strict typing, which would otherwise reject a reply of
        512.0 and discard an entire usable tracker state over a coordinate
        that only the visualization consumes.
        """
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            try:
                return int(float(value.strip()))
            except ValueError:
                return None
        return None

    @model_validator(mode="after")
    def consistent_landmark(self) -> "LandmarkOutput":
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.u is None or self.v is None or not self.visible:
            self.u = None
            self.v = None
        elif not (0 <= self.u <= 1000 and 0 <= self.v <= 1000):
            self.u = None
            self.v = None
        self.evidence = self.evidence.strip()
        if not self.evidence:
            raise ValueError("evidence must not be empty")
        if not self.visible and not self.passed:
            if self.direction != "UNKNOWN" or self.proximity != "UNKNOWN":
                raise ValueError(
                    "invisible, unpassed landmark requires UNKNOWN state"
                )
        if self.visible:
            if self.direction == "UNKNOWN" or self.proximity == "UNKNOWN":
                raise ValueError(
                    "visible landmark requires direction and proximity"
                )
        return self
