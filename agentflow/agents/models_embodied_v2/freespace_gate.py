"""The single place that decides whether FORWARD is legal this step.

A wall does not stop the policy model. The target is still visible somewhere
beyond the obstacle, so ModelB keeps emitting FORWARD and the episode burns its
entire step budget pressed against the same surface. The wall-stuck detector in
temporal memory only rules after six such steps, and its recovery macro turns a
blind 180 degrees -- neither the detection nor the escape looks at where the
floor actually is.

This gate closes that gap with the same mechanism the stop gate uses: a failed
FORWARD makes FORWARD *unrepresentable* on the next step rather than merely
discouraged. It also remembers which relative headings have already proven to be
walls, so the escape direction it asks for is one the robot has not already
disproved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from .Actor import FORWARD, STOP, TURN_LEFT, TURN_RIGHT

# The camera-motion label temporal memory emits when nothing moved.
STATIONARY_MOTION = "STATIONARY"


@dataclass(frozen=True)
class FreeSpaceDecision:
    """Whether the robot may drive straight ahead right now, and why."""

    forward_allowed: bool
    reason: str
    blocked_streak: int = 0

    def __bool__(self) -> bool:
        return self.forward_allowed


@dataclass(frozen=True)
class EscapeDirection:
    """A traversable heading, expressed as primitives ModelB could have chosen.

    Recovery stays inside the ordinary action space on purpose: an escape is a
    sequence of turns followed by a short forward probe, so nothing downstream
    needs a privileged action to execute it.
    """

    turn_action: str
    turn_steps: int
    opening: str = "unknown"
    reason: str = ""

    def __post_init__(self) -> None:
        if self.turn_action not in (TURN_LEFT, TURN_RIGHT):
            raise ValueError(
                f"turn_action must be {TURN_LEFT} or {TURN_RIGHT}, "
                f"got {self.turn_action!r}"
            )
        if self.turn_steps < 1:
            raise ValueError("turn_steps must be at least 1")

    def actions(self, *, probe_forward_steps: int = 2) -> tuple[str, ...]:
        """Turn onto the heading, then probe it with a few forward steps."""
        return (self.turn_action,) * self.turn_steps + (FORWARD,) * max(
            0, probe_forward_steps
        )

    def describe(self, *, turn_degrees: float = 15.0) -> str:
        side = "left" if self.turn_action == TURN_LEFT else "right"
        degrees = self.turn_steps * turn_degrees
        return (
            f"turn {side} {degrees:.0f} degrees toward {self.opening}"
            + (f" ({self.reason})" if self.reason else "")
        )


class FreeSpaceGate:
    """Track whether forward motion is physically possible, and where to go."""

    def __init__(
        self,
        *,
        turn_degrees: float = 15.0,
        stall_similarity: float = 0.97,
        stall_confidence: float = 0.85,
        block_after: int = 1,
        escape_after: int = 2,
        probe_forward_steps: int = 2,
        max_turn_steps: int = 12,
        sweep_turn_steps: int = 2,
    ):
        if turn_degrees <= 0:
            raise ValueError("turn_degrees must be positive")
        if block_after < 1:
            raise ValueError("block_after must be at least 1")
        if escape_after < block_after:
            raise ValueError("escape_after must not be smaller than block_after")
        if max_turn_steps < 1:
            raise ValueError("max_turn_steps must be at least 1")
        for name, value in (
            ("stall_similarity", stall_similarity),
            ("stall_confidence", stall_confidence),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        self.turn_degrees = turn_degrees
        # Headings are a circle, not a line: the shortest way onto a bearing 200
        # degrees to the right is 160 degrees to the left. Everything below works
        # in buckets modulo one revolution, so a half turn is the longest turn
        # worth emitting and a scan can never drift off in one direction leaving
        # half the room unprobed.
        self.steps_per_revolution = max(2, round(360.0 / turn_degrees))
        max_turn_steps = min(max_turn_steps, self.steps_per_revolution // 2)
        if not 1 <= sweep_turn_steps <= max_turn_steps:
            raise ValueError(
                f"sweep_turn_steps must be in [1, {max_turn_steps}] for a "
                f"{turn_degrees:.0f}-degree turn"
            )
        self.stall_similarity = stall_similarity
        self.stall_confidence = stall_confidence
        self.block_after = block_after
        self.escape_after = escape_after
        self.probe_forward_steps = probe_forward_steps
        self.max_turn_steps = max_turn_steps
        self.sweep_turn_steps = sweep_turn_steps
        self._blocked_streak = 0
        self._heading_bucket = 0
        self._blocked_buckets: set[int] = set()
        self._reason = ""
        self._last_step_id: Optional[int] = None
        self._scan_sign = 0
        self._scan_pass = 0

    def reset(self) -> None:
        """Clear every episode-scoped free-space belief."""
        self._blocked_streak = 0
        self._heading_bucket = 0
        self._blocked_buckets = set()
        self._reason = ""
        self._last_step_id = None
        self._scan_sign = 0
        self._scan_pass = 0

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def observe_step(
        self,
        *,
        commanded_action: str,
        observed_motion: Optional[str] = None,
        motion_confidence: Optional[float] = None,
        frame_similarity: Optional[float] = None,
        collision: Optional[bool] = None,
        step_id: Optional[int] = None,
    ) -> None:
        """Fold one completed transition into the free-space belief.

        Only FORWARD carries information about traversability; a turn merely
        moves the heading the next FORWARD would test.
        """
        if step_id is not None and step_id == self._last_step_id:
            return
        self._last_step_id = step_id
        action = str(commanded_action or "").strip().upper()
        if action == TURN_LEFT:
            self._heading_bucket = (
                self._heading_bucket - 1
            ) % self.steps_per_revolution
            return
        if action == TURN_RIGHT:
            self._heading_bucket = (
                self._heading_bucket + 1
            ) % self.steps_per_revolution
            return
        if action != FORWARD:
            return
        blocked, reason = self._forward_failed(
            observed_motion=observed_motion,
            motion_confidence=motion_confidence,
            frame_similarity=frame_similarity,
            collision=collision,
        )
        if blocked:
            self._blocked_streak += 1
            self._blocked_buckets.add(self._heading_bucket)
            self._reason = reason
            return
        # Driving through means the obstacle map we built is about a place the
        # robot has already left, so none of it is evidence any more.
        self._blocked_streak = 0
        self._heading_bucket = 0
        self._blocked_buckets = set()
        self._reason = ""
        self._scan_sign = 0
        self._scan_pass = 0

    def _forward_failed(
        self,
        *,
        observed_motion: Optional[str],
        motion_confidence: Optional[float],
        frame_similarity: Optional[float],
        collision: Optional[bool],
    ) -> tuple[bool, str]:
        # An explicit collision sensor outranks every visual inference, in both
        # directions: a False reading means the robot really did move.
        if collision is True:
            return True, "the collision sensor fired on the last FORWARD"
        if collision is False:
            return False, ""
        motion = str(observed_motion or "").strip().upper()
        confidence = motion_confidence if motion_confidence is not None else 0.0
        if motion == STATIONARY_MOTION and confidence >= self.stall_confidence:
            return True, (
                f"FORWARD produced no camera motion (confidence {confidence:.2f})"
            )
        if (
            frame_similarity is not None
            and frame_similarity >= self.stall_similarity
        ):
            return True, (
                "the view is unchanged after FORWARD "
                f"(similarity {frame_similarity:.3f})"
            )
        return False, ""

    # ------------------------------------------------------------------
    # Rulings
    # ------------------------------------------------------------------
    def evaluate(self) -> FreeSpaceDecision:
        """Decide whether FORWARD may be offered to the policy this step.

        The mask is scoped to the heading that failed, never to FORWARD as
        such. A gate that stayed shut until the next successful FORWARD could
        never reopen -- the only action that clears it is the one it forbids --
        and the robot would trade banging into the wall for spinning in front
        of it. Turning away from a proven wall is therefore enough to earn
        another attempt, and each fresh failure only costs one more heading.
        """
        if (
            self._blocked_streak < self.block_after
            or self._heading_bucket not in self._blocked_buckets
        ):
            return FreeSpaceDecision(
                True,
                "the way ahead appears drivable",
                self._blocked_streak,
            )
        plural = "" if self._blocked_streak == 1 else "s"
        return FreeSpaceDecision(
            False,
            f"this heading already failed {self._blocked_streak} FORWARD "
            f"step{plural}: {self._reason}",
            self._blocked_streak,
        )

    @property
    def blocked_streak(self) -> int:
        return self._blocked_streak

    @property
    def blocked_heading_count(self) -> int:
        """How many distinct headings have been disproved since getting stuck."""
        return len(self._blocked_buckets)

    def needs_escape(self) -> bool:
        """Whether ModelB has had enough chances to turn away on its own."""
        return self._blocked_streak >= self.escape_after

    def filter_actions(self, allowed: Sequence[str]) -> tuple[str, ...]:
        """Mask FORWARD out of the action space while the path is blocked.

        Turns always survive the mask, so the policy is never handed an empty
        action space -- the worst case is that it must turn.
        """
        if self.evaluate().forward_allowed:
            return tuple(allowed)
        remaining = tuple(action for action in allowed if action != FORWARD)
        movement = [action for action in remaining if action != STOP]
        return remaining if movement else tuple(allowed)

    # ------------------------------------------------------------------
    # Escape planning
    # ------------------------------------------------------------------
    def _signed_offset(self, bucket: int) -> int:
        """Shortest signed rotation from the current heading onto `bucket`."""
        half = self.steps_per_revolution // 2
        return (
            bucket - self._heading_bucket + half
        ) % self.steps_per_revolution - half

    def blocked_note(self) -> str:
        """Describe already-disproved headings relative to the current view."""
        if not self._blocked_buckets:
            return ""
        offsets = sorted(
            self._signed_offset(bucket) * self.turn_degrees
            for bucket in self._blocked_buckets
        )
        parts = []
        for offset in offsets:
            if abs(offset) < self.turn_degrees / 2:
                parts.append("straight ahead")
            elif offset < 0:
                parts.append(f"{abs(offset):.0f} degrees to the left")
            else:
                parts.append(f"{offset:.0f} degrees to the right")
        return (
            "Already tried and blocked, relative to the current view: "
            + ", ".join(parts)
            + ". Do not send the robot back into any of them."
        )

    def fallback_escape(self) -> EscapeDirection:
        """A deterministic scan for when the escape query gives no answer.

        The scan circles in one direction rather than fanning out to either
        side: an alternating fan re-crosses the room on every probe and costs a
        multiple of the turns a single revolution needs. The first pass stops
        every `sweep_turn_steps` -- close enough together that a doorway is
        unlikely to fall between two probes -- and a second pass fills in the
        bearings the first one skipped, so no opening stays invisible forever.
        """
        if self._scan_sign == 0:
            # Circle away from the side where the failures cluster; ties open to
            # the left, which only has to be deterministic, not correct.
            bias = sum(
                self._signed_offset(bucket) for bucket in self._blocked_buckets
            )
            self._scan_sign = -1 if bias > 0 else 1
        for coarse in (self._scan_pass == 0, False):
            for offset in range(1, self.steps_per_revolution + 1):
                target = (
                    self._heading_bucket + self._scan_sign * offset
                ) % self.steps_per_revolution
                if target in self._blocked_buckets:
                    continue
                if coarse and target % self.sweep_turn_steps != 0:
                    continue
                return EscapeDirection(
                    TURN_LEFT if self._scan_sign < 0 else TURN_RIGHT,
                    min(offset, self.max_turn_steps),
                    opening="unscouted",
                    reason=(
                        "no scouted opening; scanning for one"
                        if coarse
                        else "no scouted opening; scanning between the first pass"
                    ),
                )
            # The coarse ring is exhausted; the fine pass owns the rest of this
            # stuck episode, so it never has to be rediscovered.
            self._scan_pass = 1
        return EscapeDirection(
            TURN_LEFT if self._scan_sign < 0 else TURN_RIGHT,
            self.max_turn_steps,
            opening="unscouted",
            reason="every scanned heading is blocked; turning around",
        )

    def adjust(self, direction: EscapeDirection) -> EscapeDirection:
        """Push a proposed heading off any bucket already known to be a wall."""
        candidates = [
            (direction.turn_action, steps)
            for steps in range(direction.turn_steps, self.max_turn_steps + 1)
        ]
        opposite = (
            TURN_RIGHT if direction.turn_action == TURN_LEFT else TURN_LEFT
        )
        candidates += [
            (opposite, steps) for steps in range(1, self.max_turn_steps + 1)
        ]
        for turn_action, steps in candidates:
            sign = -1 if turn_action == TURN_LEFT else 1
            target = (
                self._heading_bucket + sign * steps
            ) % self.steps_per_revolution
            if target not in self._blocked_buckets:
                if (turn_action, steps) == (
                    direction.turn_action,
                    direction.turn_steps,
                ):
                    return direction
                return EscapeDirection(
                    turn_action,
                    steps,
                    opening=direction.opening,
                    reason=(
                        direction.reason
                        + " (shifted off an already-blocked heading)"
                    ).strip(),
                )
        # Every reachable heading has failed: turn as far as the gate allows and
        # let the next probe rebuild the map from wherever that lands.
        return EscapeDirection(
            direction.turn_action,
            self.max_turn_steps,
            opening=direction.opening,
            reason="every scanned heading is blocked; turning around",
        )
