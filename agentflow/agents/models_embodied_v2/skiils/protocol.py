"""Configuration, prompts, and validated model outputs for VLN agent v3."""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

from agentflow.agents.models_embodied_v2.data_models import (
    ActionMode,
    MAX_TURN_DEG,
    TURN_STEP_DEG,
)


DEFAULT_MODEL_PATH = "models/JoyAI-VL-Interaction"
# Planning runs once per episode and its retries are sampled, so a third
# attempt is cheap insurance against one malformed plan aborting the run.
SUBGOAL_GENERATION_ATTEMPTS = 3
WAYPOINT_GENERATION_ATTEMPTS = 2
ERROR_CONFIDENCE_THRESHOLD = 0.9
ERROR_CONFIRMATION_WINDOW = 5
ERROR_CONFIRMATION_VOTES = 4
# A final STOP is accepted independently of Temporal Memory only after the
# waypoint model repeats strong, visually grounded near-target evidence.  This
# keeps one hallucinated STOP from ending an episode while avoiding a deadlock
# when the scene Captioner disagrees with repeated current-frame observations.
FINAL_STOP_EVIDENCE_WINDOW = 5
FINAL_STOP_EVIDENCE_VOTES = 2
FINAL_STOP_MIN_CONFIDENCE = 0.90
RECOVERY_FORWARD_U = 500
RECOVERY_FORWARD_V = 750
RECOVERY_TURN_U = 250
RECOVERY_TURN_V = 500
RECOVERY_LATERAL_DISTANCE_M = 1.5
RECOVERY_SUCCESS_TRANSLATION_M = 0.20
PREVIEW_REARM_TRANSLATION_M = 0.25
PREVIEW_SELECTION_MIN_CONFIDENCE = 0.60
STALL_EVIDENCE_FRAMES = 4
STALL_TRANSLATION_LIMIT_M = 0.20
TURN_EVIDENCE_DEG = 5.0
# Bound the unified temporal scene request. A larger window increases visual
# token allocation on every step after the window fills.
MAX_COMPLETION_EVIDENCE_FRAMES = 16
# A doorway completion is normally released by geometry (the camera reached
# the localized doorway) or by the model having reported the approach. When
# neither happens -- the doorway point was mislocalized, or the model jumped
# straight from NOT_VISIBLE to CROSSED -- this many consecutive confident
# CROSSED/AFTER_DOOR judgements, after at least this much measured walking,
# are accepted instead of holding the subgoal shut for the whole episode.
DOORWAY_CROSSED_STREAK_ACCEPT = 4
DOORWAY_CROSSED_MIN_PATH_M = 1.0
DOORWAY_CROSSED_MIN_CONFIDENCE = 0.90
# Geometry outranks the model. Once the camera has come within
# DOORWAY_REACHED_M of the localized doorway point that fact is latched for
# the subgoal, so a crossing reported a few steps later is still accepted.
# While the point is farther than DOORWAY_STILL_AHEAD_M and was never
# reached, a CROSSED claim is contradicted by measurement and rejected no
# matter how confident or repeated it is.
DOORWAY_REACHED_M = 0.50
DOORWAY_STILL_AHEAD_M = 1.00
# Bound every online visual request. Temporal-memory requests are also resized
# by TemporalCaptioner before reaching the engine.
VLM_IMAGE_MIN_PIXELS = 64**2
# 224 px left a 640x480 frame as ~35 visual tokens, too few to place a
# doorway; 448 px is ~140 tokens, still cheap now that inference is bf16.
VLM_IMAGE_MAX_PIXELS = 448**2
# 16 frames x 320 px is ~1200 visual tokens per scene call. At 160 px the
# landmark position the Captioner reports was unusable.
TEMPORAL_MAX_IMAGE_EDGE = 320
# Below this Captioner confidence a localized landmark is not trusted to
# override the waypoint model's own TURN/PREVIEW request.
LANDMARK_STEER_MIN_CONFIDENCE = 0.60
# The actor schema is now a nested discriminated union, which costs roughly
# twenty-five more tokens per reply than the flat waypoint shape.  At the
# previous 64-token budget a typical reply would truncate before its closing
# braces, and a truncated reply fails validation and silently degrades into the
# safe fallback waypoint.  A PREVIEW reply is far shorter and ends early.
STRUCTURED_VLM_MAX_TOKENS = 96
BEHAVIOR_HISTORY_SIZE = 8
LANDMARK_HISTORY_SIZE = 6
CORRIDOR_LOCK_FORWARD_STEPS = 2
CORRIDOR_WAYPOINT_DEVIATION = 75
TURN_ALIGNMENT_DEG = 60.0

NavigationIntent = Literal[
    "FOLLOW_CORRIDOR",
    "APPROACH_LANDMARK",
    "TURN_LEFT",
    "TURN_RIGHT",
    "FINAL_APPROACH",
    "STOP",
]

POINT_PROMPT = """You are an indoor navigation actor. Steer only toward the
active navigation subgoal.

First decide this step's action mode, and write it as the first field:
- EXECUTION: the current image resolves where to go; commit to a waypoint.
- EXPLORATION: the direction is unresolved, but a visible floor path would
  reveal it; move to gather information rather than to advance a known route.
- PREVIEW: the current image cannot resolve the direction and the answer is
  likely outside this field of view; ask to inspect the surrounding views.
PREVIEW is active perception, not an error fallback. Choose it proactively at
a reached blind corner, corridor end, T/L junction, doorway threshold, or
multi-branch decision point when the required continuation may be outside the
current field of view or an occluding wall hides what is around the corner.
Also choose it when two or more route branches are plausible and the current
image cannot distinguish which one matches the instruction. Do not preview in
a single unambiguous corridor, when the required opening or landmark and a
floor path to it are already visible, or merely because the target is distant.
One preview inspects the surrounding headings simultaneously, so do not ask
for repeated previews without first making navigational progress.

EXECUTION and EXPLORATION also need a navigation intent:
- FOLLOW_CORRIDOR: continue through the main corridor; do not enter side rooms.
- APPROACH_LANDMARK: approach the active subgoal's visible landmark.
- TURN_LEFT / TURN_RIGHT: execute the active subgoal's requested turn.
- FINAL_APPROACH: approach the final destination.
- STOP: EXECUTION only, and only at the final destination.

An EXECUTION is either a waypoint or a turn, never both. Give a waypoint
whenever a usable floor point is visible. Ask for a turn only when the way
forward is not visible from here and rotating will bring it into view.

Select one visible, obstacle-free floor point that makes progress. Never select
walls, furniture, stairs, or an image border. Coordinates are normalized
integers from 0 to 1000, independent of image resolution: u=0 is the left edge,
u=1000 is the right edge, v=0 is the top, and v=1000 is the bottom.

turn_deg is a multiple of 15: positive turns right, negative turns left. It must
agree with the intent, so TURN_RIGHT takes a positive value and TURN_LEFT a
negative one. Never request more than 45 degrees in one decision. The camera
will observe again after that short turn, so ask only for the smallest turn that
brings the route into view.

Reply only with one exact JSON object, in one of these four shapes:
{"action_mode":"EXECUTION","execution":{"stop":false,"intent":"FOLLOW_CORRIDOR","u":integer,"v":integer},"confidence":0.0,"evidence":"brief visual reason"}
{"action_mode":"EXECUTION","execution":{"stop":false,"intent":"TURN_RIGHT","turn_deg":30},"confidence":0.0,"evidence":"brief visual reason"}
{"action_mode":"EXPLORATION","exploration":{"intent":"APPROACH_LANDMARK","u":integer,"v":integer},"confidence":0.0,"evidence":"brief visual reason"}
{"action_mode":"PREVIEW","preview":true,"confidence":0.0,"evidence":"brief visual reason"}

The target may be one subgoal of a longer route, numbered "(n of m)". Use
{"action_mode":"EXECUTION","execution":{"stop":true,"intent":"STOP"},"confidence":0.0,"evidence":"brief visual reason"}
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
exactly this form, and output nothing else 鈥?no JSON, no brackets, no bullets,
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
    """Flat internal form of one actor decision.

    The wire format is the nested ``ActorOutput`` union below.  Keeping this
    flat shape as the internal one means the guards, the recovery path, and the
    debug state are unaffected by the nesting.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    stop: bool
    intent: NavigationIntent
    action_mode: ActionMode = "EXECUTION"
    u: Optional[int] = None
    v: Optional[int] = None
    turn_deg: Optional[int] = None
    confidence: float
    evidence: str

    @property
    def is_turn(self) -> bool:
        """True when this decision rotates in place instead of moving to a point."""
        return self.turn_deg is not None

    @model_validator(mode="after")
    def valid_stop_or_point(self) -> "WaypointOutput":
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        self.evidence = self.evidence.strip()
        if not self.evidence:
            raise ValueError("evidence must not be empty")
        for value in (self.u, self.v):
            if value is not None and not 0 <= value <= 1000:
                raise ValueError(
                    "normalized coordinates must be in [0, 1000]"
                )
        if self.stop:
            if self.intent != "STOP":
                raise ValueError("STOP output requires intent=STOP")
            # Stopping is terminal, so there is nothing left to preview or
            # explore towards.
            if self.action_mode != "EXECUTION":
                raise ValueError("STOP output requires action_mode=EXECUTION")
            if self.turn_deg is not None:
                raise ValueError("a stopping output cannot also turn")
            return self
        if self.intent == "STOP":
            raise ValueError("non-STOP output cannot use intent=STOP")
        if self.action_mode == "PREVIEW":
            # PREVIEW asks the controller for surrounding views and carries no
            # action of its own, neither a waypoint nor a turn.
            if self.turn_deg is not None:
                raise ValueError("a PREVIEW output cannot carry a turn")
            return self
        if self.turn_deg is not None:
            if self.u is not None or self.v is not None:
                raise ValueError(
                    "an output is either a turn or a waypoint, not both"
                )
            _validate_turn(self.turn_deg, self.intent)
            return self
        if self.u is None or self.v is None:
            raise ValueError("a committing output requires u and v")
        return self


def _validate_normalized(u: int, v: int) -> None:
    for value in (u, v):
        if not 0 <= value <= 1000:
            raise ValueError("normalized coordinates must be in [0, 1000]")


def _validate_turn(turn_deg: int, intent: NavigationIntent) -> None:
    """Bound a turn to whole turn primitives and to its stated direction.

    The direction check is not redundant with the sign: the guards read
    ``intent`` while the controller executes ``turn_deg``, so a reply that
    disagrees with itself would turn one way and be judged as the other.
    """
    if turn_deg == 0:
        raise ValueError("a turn must not be zero degrees")
    if turn_deg % TURN_STEP_DEG:
        raise ValueError(
            f"turn_deg must be a multiple of {TURN_STEP_DEG} degrees"
        )
    if abs(turn_deg) > MAX_TURN_DEG:
        raise ValueError(
            f"turn_deg must be within +/-{MAX_TURN_DEG} degrees"
        )
    expected = "TURN_RIGHT" if turn_deg > 0 else "TURN_LEFT"
    if intent != expected:
        raise ValueError(f"turn_deg {turn_deg:+d} requires intent={expected}")


class ExecutionProposal(BaseModel):
    """Commit to a waypoint, or stop at the final destination."""

    model_config = ConfigDict(extra="forbid", strict=True)

    stop: bool
    intent: NavigationIntent
    u: Optional[int] = None
    v: Optional[int] = None
    # Signed whole turn primitives: positive turns right, negative left.  A
    # turn and a waypoint are alternatives, never a pair, because the
    # controller can only execute one of them per decision.
    turn_deg: Optional[int] = None

    @model_validator(mode="after")
    def valid_proposal(self) -> "ExecutionProposal":
        if self.stop:
            if self.intent != "STOP":
                raise ValueError("a stopping execution requires intent=STOP")
            if self.turn_deg is not None:
                raise ValueError("a stopping execution cannot also turn")
            return self
        if self.intent == "STOP":
            raise ValueError("a moving execution cannot use intent=STOP")
        if self.turn_deg is not None:
            if self.u is not None or self.v is not None:
                raise ValueError(
                    "an execution is either a turn or a waypoint, not both"
                )
            _validate_turn(self.turn_deg, self.intent)
            return self
        if self.u is None or self.v is None:
            raise ValueError("a moving execution requires u and v")
        _validate_normalized(self.u, self.v)
        return self


class ExplorationProposal(BaseModel):
    """Move to reveal the route rather than to advance a known one."""

    model_config = ConfigDict(extra="forbid", strict=True)

    intent: NavigationIntent
    u: int
    v: int

    @model_validator(mode="after")
    def valid_proposal(self) -> "ExplorationProposal":
        # Exploration is never terminal; stopping belongs to EXECUTION.
        if self.intent == "STOP":
            raise ValueError("exploration cannot use intent=STOP")
        _validate_normalized(self.u, self.v)
        return self


class _ActorOutputBase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    confidence: float
    evidence: str

    @model_validator(mode="after")
    def valid_common(self) -> "_ActorOutputBase":
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        self.evidence = self.evidence.strip()
        if not self.evidence:
            raise ValueError("evidence must not be empty")
        return self


class ExecutionOutput(_ActorOutputBase):
    action_mode: Literal["EXECUTION"]
    execution: ExecutionProposal


class ExplorationOutput(_ActorOutputBase):
    action_mode: Literal["EXPLORATION"]
    exploration: ExplorationProposal


class PreviewOutput(_ActorOutputBase):
    action_mode: Literal["PREVIEW"]
    preview: bool

    @model_validator(mode="after")
    def valid_request(self) -> "PreviewOutput":
        if not self.preview:
            raise ValueError("PREVIEW output requires preview=true")
        return self


# ``action_mode`` is the discriminator and is written first in the prompt's
# templates on purpose: decoding is autoregressive, so the mode is committed
# before any waypoint is generated, and a PREVIEW reply ends early instead of
# emitting coordinates nobody will use.
ActorOutput = Annotated[
    Union[ExecutionOutput, ExplorationOutput, PreviewOutput],
    Field(discriminator="action_mode"),
]

ACTOR_OUTPUT_ADAPTER: TypeAdapter = TypeAdapter(ActorOutput)


def parse_actor_output(
    payload: dict,
    *,
    preview_intent: NavigationIntent,
) -> WaypointOutput:
    """Validate the nested wire format and flatten it for the policy code.

    ``preview_intent`` supplies the navigation intent a PREVIEW reply does not
    carry, so the flat form stays well-formed for the debug state and guards.
    """
    output = ACTOR_OUTPUT_ADAPTER.validate_python(payload)
    if isinstance(output, PreviewOutput):
        if preview_intent == "STOP":
            raise ValueError("preview_intent must not be STOP")
        return WaypointOutput(
            stop=False,
            intent=preview_intent,
            action_mode="PREVIEW",
            confidence=output.confidence,
            evidence=output.evidence,
        )
    return _flatten_proposal(output)


def _flatten_proposal(
    output: "ExecutionOutput | ExplorationOutput",
) -> WaypointOutput:
    proposal = (
        output.execution
        if isinstance(output, ExecutionOutput)
        else output.exploration
    )
    return WaypointOutput(
        stop=getattr(proposal, "stop", False),
        intent=proposal.intent,
        action_mode=output.action_mode,
        u=proposal.u,
        v=proposal.v,
        # Exploration is waypoint-only, so only an execution can carry a turn.
        turn_deg=getattr(proposal, "turn_deg", None),
        confidence=output.confidence,
        evidence=output.evidence,
    )


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
