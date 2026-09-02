"""Configuration, prompts, and validated model outputs for the VLN agent."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    field_validator,
    model_validator,
)

from agentflow.agents.models_embodied_v2.data_models import (
    ActionMode,
    MAX_TURN_DEG,
)


import os


def _env_int(name: str, default: int) -> int:
    """Deployment override (set by the runner's config.yaml); default otherwise."""
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


DEFAULT_MODEL_PATH = os.environ.get("VLN_MODEL_PATH", "models/JoyAI-VL-Interaction")
# Planning runs once per episode and its retries are sampled, so a third
# attempt is cheap insurance against one malformed plan aborting the run.
SUBGOAL_GENERATION_ATTEMPTS = 3
RECOVERY_LATERAL_DISTANCE_M = 1.5
# A locked target further off the camera axis than this is still being
# turned toward; same-sign rotation without translation is then the follower
# aligning, not an in-place spin.
LOCKED_TARGET_TURN_TOLERANCE_DEG = 20.0
PREVIEW_SELECTION_MIN_CONFIDENCE = 0.60
STALL_EVIDENCE_FRAMES = 4
STALL_TRANSLATION_LIMIT_M = 0.20
TURN_EVIDENCE_DEG = 5.0
# Bound the unified temporal scene request. A larger window increases visual
# token allocation on every step after the window fills.
# 16 frames at 320 px cost 6.6 s per Captioner call and raised its malformed
# JSON rate from 9 % (one frame) to 36 %; eight frames keep the crossing
# evidence while halving both.
MAX_COMPLETION_EVIDENCE_FRAMES = _env_int("VLN_COMPLETION_EVIDENCE_FRAMES", 8)
# Judge a plain walk/approach stage every N observations instead of every
# step. Doorway, turn and final stages are still judged on every step: their
# completion depends on the exact frame in which the jambs pass or the target
# is beside the camera.
CAPTIONER_ANALYSIS_INTERVAL_STEPS = _env_int("VLN_CAPTIONER_INTERVAL_STEPS", 2)
# Full single-line scene JSON is ~150 tokens; the model sometimes
# pretty-prints it, which roughly doubles that, so leave headroom and let the
# closing brace end generation early.
CAPTIONER_MAX_TOKENS = _env_int("VLN_CAPTIONER_MAX_TOKENS", 512)
# Spatial Memory committed targets (memory/spatial_memory): a world point the
# agent walks to without re-querying the waypoint model. Released when
# reached, walked past, not closer for STAGNATION steps, or older than
# MAX_AGE steps. Points nearer than MIN_COMMIT are one step away and are
# not worth committing to.
SPATIAL_TARGET_MAX_AGE_STEPS = 12
SPATIAL_TARGET_STAGNATION_STEPS = 6
SPATIAL_TARGET_MIN_COMMIT_M = 0.6
SPATIAL_LOOKAHEAD_M = 1.5
# Set-of-mark waypoint selection (memory/spatial_memory/candidates.py): the
# map proposes numbered floor points, the model picks one. Enabled with
# Spatial Memory unless VLN_SOM=0.
SOM_MAX_CANDIDATES = 5
SOM_TARGET_MAX_AGE_STEPS = 24
# A far marker is walked only this far before the model is asked again, so
# a stage that should have turned or stopped half-way is not overshot.
SOM_MAX_TARGET_DISTANCE_M = 3.0

SOM_PROMPT = """You are choosing where an indoor navigation agent walks next.
The image is the agent's current view with numbered markers drawn on
reachable floor points; the text lists each marker's distance and direction
and what lies there (open floor, the edge of the explored map, or the active
subgoal's landmark located by the scene observer). Letters L, R and B, when
listed, are places outside the current view that the map has not explored
yet, reached by turning left, right or around.

Pick the option that best continues the ACTIVE subgoal within the full route
instruction:
- prefer a marker on the route toward the named landmark, doorway or room;
- in a corridor or hallway stage keep the corridor, do not enter side rooms;
- choose L, R or B only when no marker in view can lead where the route
  goes (a blind corner, a turn the instruction asks for, a dead end);
- never choose a marker the instruction tells you to move away from.

Reply with exactly one single-line JSON object; angle brackets are values you
fill in, not placeholders to copy:
{"choice":"<marker number, or L, R, B>","confidence":<float 0-1>,"evidence":"<at most 20 words>"}
confidence is your own probability that this choice follows the route: about
0.9 when obvious, 0.5 when two options compete, 0.2 when guessing."""
# A doorway completion is normally released by geometry (the camera reached
# the localized doorway) or by the model having reported the approach. When
# neither happens -- the doorway point was mislocalized, or the model jumped
# straight from NOT_VISIBLE to CROSSED -- this many consecutive confident
# CROSSED/AFTER_DOOR judgements, after at least this much measured walking,
# are accepted instead of holding the subgoal shut for the whole episode.
DOORWAY_CROSSED_STREAK_ACCEPT = 4
DOORWAY_CROSSED_MIN_PATH_M = 1.0
# Geometry outranks the model. A subgoal is judged on arrival, not en route:
# while the actor still holds a committed waypoint for the subgoal (a
# localized doorway or landmark point) that is farther than
# COMMITTED_TARGET_REACHED_M and was never reached, any completion the model
# reports is deferred. Coming within that distance is latched for the
# subgoal, so a completion reported a few steps later is still accepted.
COMMITTED_TARGET_REACHED_M = 0.50
# A point localized from far away carries a proportionally larger error
# (pixel error times depth), so its arrival tolerance grows with the distance
# it was localized from, between the two bounds.
COMMITTED_TARGET_TOLERANCE_FRACTION = 0.25
COMMITTED_TARGET_TOLERANCE_MAX_M = 1.00
# Skipping ahead to the final stage. The destination must be reported AT with
# high confidence for several consecutive observations, and the plan must
# plausibly be near its end: either only a couple of stages remain, or the
# active stage has been stuck for many observations. A minimum walked
# distance rules out a look-alike seen right after the start.
STAGE_SKIP_AT_STREAK = 3
STAGE_SKIP_MIN_CONFIDENCE = 0.80
STAGE_SKIP_MAX_REMAINING_STAGES = 2
STAGE_SKIP_STALL_OBSERVATIONS = 20
STAGE_SKIP_MIN_TASK_PATH_M = 3.0
# The model's own landmark for the active stage, measured through the depth
# map at the pixel it reported: a doorway "crossed" or a destination "AT"
# that is still farther than this is contradicted by measurement.
LANDMARK_RANGE_VETO_M = 1.5
# A stairs stage is complete when the camera has risen or dropped this much
# in the requested direction and its height has levelled off again.
STAIRS_MIN_RISE_M = 0.30
STAIRS_LEVEL_TOLERANCE_M = 0.05
# A turn subgoal cannot be complete before the camera has measurably turned
# this far in the requested direction; past TURN_ABANDON_DEG it has turned a
# half circle and the turn phase ends regardless of what the model asks for.
TURN_MIN_PROGRESS_DEG = 60.0
# A stage that has cost this many walked metres without completing is being
# pursued in the wrong direction: drop the committed target and look around.
STAGE_PATH_OVERRUN_M = 10.0

# "Turn around" is a rotation of about 180 degrees in either direction; the
# stage is measured like a left/right turn but against this larger target.
TURN_AROUND_MIN_PROGRESS_DEG = 150.0
# One shared doorway-stage classifier for the captioner contract, the judge's
# measured-crossing rule and the agent's doorway lock. Audit on 400 sampled
# val_unseen stages (docs 2026-08-31): the old per-site regexes disagreed and
# the captioner one missed 29% of true doorway stages ("Go through the door",
# "entryway", "entrance"). A turn instruction wins over a doorway mention:
# "Turn left at the doorway" is a rotation whose location happens to be a door.
DOORWAY_STAGE_PATTERN = (
    r"\b(?:doorways?|doors?|exit|leave|enter|entrance|entryway|arch(?:way)?|"
    r"threshold|cross(?:ed|ing)?|out\s+of|walk\s+out|(?:go|walk|pass)\s+through)\b"
)
TURN_STAGE_PATTERN = r"\bturn\s+(?:left|right)\b"


def stage_is_doorway(text: str) -> bool:
    """Doorway-type stage: its completion is passing a structural opening."""
    import re

    if re.search(TURN_STAGE_PATTERN, text, flags=re.IGNORECASE):
        return False
    if re.search(TURN_AROUND_PATTERN, text, flags=re.IGNORECASE):
        return False
    return bool(re.search(DOORWAY_STAGE_PATTERN, text, flags=re.IGNORECASE))


TURN_AROUND_PATTERN = (
    r"\b(?:turn\s+(?:all\s+the\s+way\s+)?(?:around|back|about)|about[- ]face|"
    r"turn\s+180|(?:do|make)\s+a\s+180|reverse\s+direction)\b"
)
TURN_ABANDON_DEG = 180.0
# Bound every online visual request. Temporal-memory requests are also resized
# by TemporalCaptioner before reaching the engine.
VLM_IMAGE_MIN_PIXELS = 64**2
# 224 px left a 640x480 frame as ~35 visual tokens, too few to place a
# doorway; 448 px is ~140 tokens, still cheap now that inference is bf16.
VLM_IMAGE_MAX_PIXELS = _env_int("VLN_IMAGE_MAX_PIXELS", 448**2)
# 16 frames x 320 px is ~1200 visual tokens per scene call. At 160 px the
# landmark position the Captioner reports was unusable.
TEMPORAL_MAX_IMAGE_EDGE = _env_int("VLN_TEMPORAL_MAX_IMAGE_EDGE", 320)
# Below this Captioner confidence a localized landmark is not trusted to
# override the waypoint model's own TURN/PREVIEW request.
# A landmark the Captioner marks visible with a pixel is a structured claim
# that the depth veto (LANDMARK_RANGE_VETO_M) and committed-point odometry
# already check; its confidence field is advisory (see FINAL_STOP note).
LANDMARK_STEER_MIN_CONFIDENCE = 0.0
# A turn stage is satisfied the moment its landmark sits within this band
# of the image centre (normalized 0..1000; 150 is about 13 degrees at a
# 90-degree field of view). Turning further only turns away from it.
TURN_TARGET_CENTRED_U = 150
# ...provided the camera has actually turned at least one primitive; a
# landmark already centred before any rotation is not a completed turn.
TURN_TARGET_MIN_PROGRESS_DEG = 15.0
# The actor schema is now a nested discriminated union, which costs roughly
# twenty-five more tokens per reply than the flat waypoint shape.  At the
# previous 64-token budget a typical reply would truncate before its closing
# braces, and a truncated reply fails validation and silently degrades into the
# safe fallback waypoint.  A PREVIEW reply is far shorter and ends early.
# 96 tokens truncated ~3 % of replies once the evidence clause grew; 192
# leaves room for the nested shape plus a 20-word evidence.
STRUCTURED_VLM_MAX_TOKENS = _env_int("VLN_STRUCTURED_VLM_MAX_TOKENS", 192)
BEHAVIOR_HISTORY_SIZE = 8
LANDMARK_HISTORY_SIZE = 6
TURN_ALIGNMENT_DEG = 60.0

NavigationIntent = Literal[
    "FOLLOW_CORRIDOR",
    "APPROACH_LANDMARK",
    "TURN_LEFT",
    "TURN_RIGHT",
    "FINAL_APPROACH",
    "STOP",
]


SUBGOAL_PROMPT = """You are an indoor navigation task planner. Decompose the
instruction into a short ordered list of stages. Write one stage per line in
exactly this form, and output nothing else 鈥?no JSON, no brackets, no bullets,
no commentary, no blank lines:
id|description|completion criterion

Example of a complete answer:
1|Go up the stairs|The camera has passed the stairs: they are below and behind it and the upper hallway fills the view
2|Walk down the hall to the marked doorway|The marked doorway is directly ahead of the camera, within a step
3|Turn left at the marked doorway|After turning left, the kitchen entrance is centred in the view
4|Enter the kitchen|The camera has crossed the kitchen threshold and the kitchen interior is central in the view

Rules:
1. One stage per line, in instruction order, with IDs counting 1, 2, 3 with no
   gaps. Each line holds exactly three fields separated by two "|" characters;
   never write "|" inside a description or a criterion.
2. Every stage is ONE atomic action: a walk to a landmark, a doorway
   crossing, a single turn, or a corridor follow. Never join two actions with
   "and", "then" or a comma. "Enter the bedroom and turn left" is two stages.
3. Every completion criterion must describe a visually verifiable endpoint,
   never an action merely being attempted or currently in progress.
4. A stage that walks TO a landmark completes when that landmark is beside
   or directly ahead of the camera, within a step. A stage that goes UP,
   DOWN, THROUGH or ALONG something (stairs, a hallway, a room) completes
   when that thing has been passed: it is behind or below the camera and the
   space beyond it fills the view. Neither may claim the agent has already
   turned; that belongs to the turn stage.
5. A turn stage's criterion names what must be centred in the view after the
   turn: the landmark or opening of the following stage.
6. A doorway-crossing stage completes when the camera crosses the threshold
   and the destination room becomes the central/dominant view. Brief
   peripheral visibility of the starting room does not invalidate a crossing.
7. Final arrival requires reaching and stopping beside the destination;
   merely seeing it at a distance is insufficient.
8. Never add route geometry absent from the instruction. In particular,
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
