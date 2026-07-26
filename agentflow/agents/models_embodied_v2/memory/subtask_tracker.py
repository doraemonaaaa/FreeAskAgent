"""Structured subtask tracker whose progress invariants are enforced in code.

The planner LLM proposes status changes, but it never gets the final say: every
proposal is filtered through the rules below before it can touch the tracker.
This is what keeps a single hallucinated `COMPLETE` from cascading into an early
STOP.

Invariants (see `SubtaskTracker.apply`):

1. The subtask list is frozen at decomposition time -- text and ordering can
   never be rewritten, only statuses change.
2. Only the *active* subtask (the first non-COMPLETE one) may change status, so
   the model cannot reach ahead and complete a later subtask.
3. Progress is monotonic and advances at most one rank per update, which makes
   `NOT STARTED -> COMPLETE` in a single frame impossible.
4. An arrival subtask may only reach COMPLETE when a *measured* distance is
   present and within the arrival radius. The model reports a number; this code
   applies the threshold.
5. The final subtask needs the same COMPLETE verdict on several consecutive
   updates before it sticks.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple


class Status(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETE = "COMPLETE"
    BLOCKED = "BLOCKED"


# Progress ranks used for the monotonicity checks. BLOCKED is a side state that
# means "started but stalled", so it shares IN_PROGRESS's rank.
_RANK: Dict[Status, int] = {
    Status.NOT_STARTED: 0,
    Status.IN_PROGRESS: 1,
    Status.BLOCKED: 1,
    Status.COMPLETE: 2,
}

_STATUS_ALIASES = {
    "NOT STARTED": Status.NOT_STARTED,
    "NOTSTARTED": Status.NOT_STARTED,
    "PENDING": Status.NOT_STARTED,
    "IN PROGRESS": Status.IN_PROGRESS,
    "INPROGRESS": Status.IN_PROGRESS,
    "ONGOING": Status.IN_PROGRESS,
    "DONE": Status.COMPLETE,
    "COMPLETED": Status.COMPLETE,
}


def parse_status(raw: Any) -> Optional[Status]:
    """Best-effort coercion of a model-written status string into a `Status`."""
    if isinstance(raw, Status):
        return raw
    if not isinstance(raw, str):
        return None
    key = raw.strip().upper().replace("-", " ").replace("_", " ")
    key = re.sub(r"\s+", " ", key)
    if key in _STATUS_ALIASES:
        return _STATUS_ALIASES[key]
    try:
        return Status(key.replace(" ", "_"))
    except ValueError:
        return None


def extract_json(text: str) -> Any:
    """Pull a JSON value out of a model reply that may be fenced or padded."""
    candidate = text.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", candidate, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            candidate = fenced.group(1)
    start, end = candidate.find("{"), candidate.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            pass
    start, end = candidate.find("["), candidate.rfind("]")
    if start != -1 and end > start:
        try:
            return json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not extract JSON from model reply: {text!r}")


def coerce_distance(raw: Any) -> Optional[float]:
    """Read a distance in meters from a number or a string like `~6.5 m`."""
    if isinstance(raw, bool) or raw is None:
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
    elif isinstance(raw, str):
        match = re.search(r"-?\d+(?:\.\d+)?", raw)
        if not match:
            return None
        value = float(match.group())
    else:
        return None
    return value if value >= 0 else None


@dataclass(frozen=True)
class Subtask:
    """One immutable navigation subtask plus its mutable status fields."""

    index: int
    text: str
    is_arrival: bool = False
    status: Status = Status.NOT_STARTED
    evidence: str = ""
    distance_m: Optional[float] = None
    blocked_reason: str = ""

    def render(self) -> str:
        parts = [f"{self.index}. Subtask: {self.text} | Completion status: {self.status.value}"]
        if self.status is Status.BLOCKED and self.blocked_reason:
            parts.append(f"(blocked: {self.blocked_reason})")
        if self.is_arrival and self.distance_m is not None:
            parts.append(f"(measured distance to target: {self.distance_m:.1f} m)")
        return " ".join(parts)


@dataclass(frozen=True)
class ApplyResult:
    """What `SubtaskTracker.apply` actually did, for logging and debugging."""

    changed: bool
    rejections: Tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.changed


class SubtaskTracker:
    """A fixed subtask list whose statuses only move under code-checked rules."""

    def __init__(
        self,
        goal: str,
        subtasks: Sequence[Subtask],
        *,
        target_phrase: str = "",
        arrival_radius_m: float = 2.0,
        confirmations_required: int = 3,
    ):
        if not subtasks:
            raise ValueError("A tracker needs at least one subtask.")
        if arrival_radius_m <= 0:
            raise ValueError("arrival_radius_m must be positive.")
        if confirmations_required < 1:
            raise ValueError("confirmations_required must be at least 1.")
        self.goal = goal
        self.target_phrase = target_phrase or subtasks[-1].text
        self.arrival_radius_m = arrival_radius_m
        self.confirmations_required = confirmations_required
        self._subtasks: List[Subtask] = list(subtasks)
        self._final_confirmations = 0

    # ---- construction ----

    @classmethod
    def from_plan(cls, goal: str, plan: Any, **kwargs: Any) -> "SubtaskTracker":
        """Build a tracker from the decomposition model's JSON reply."""
        if isinstance(plan, str):
            plan = extract_json(plan)
        if isinstance(plan, list):
            plan = {"subtasks": plan}
        if not isinstance(plan, dict):
            raise ValueError(f"Subtask plan must be a JSON object or list, got {type(plan).__name__}.")
        raw_subtasks = plan.get("subtasks")
        if not isinstance(raw_subtasks, list) or not raw_subtasks:
            raise ValueError("Subtask plan contains no `subtasks` list.")

        subtasks: List[Subtask] = []
        for position, entry in enumerate(raw_subtasks, start=1):
            if isinstance(entry, str):
                entry = {"text": entry}
            if not isinstance(entry, dict):
                raise ValueError(f"Subtask #{position} is not an object: {entry!r}")
            text = str(entry.get("text") or entry.get("subtask") or "").strip()
            if not text:
                raise ValueError(f"Subtask #{position} has empty text.")
            subtasks.append(Subtask(index=position, text=text, is_arrival=bool(entry.get("is_arrival"))))

        # The last subtask is the arrival step by construction, whatever the
        # model labelled: stopping is only ever legal at the end of the plan.
        subtasks = [replace(task, is_arrival=(task.index == len(subtasks))) for task in subtasks]
        target_phrase = str(plan.get("target_phrase") or "").strip()
        return cls(goal, subtasks, target_phrase=target_phrase, **kwargs)

    # ---- queries ----

    @property
    def subtasks(self) -> Tuple[Subtask, ...]:
        return tuple(self._subtasks)

    @property
    def active_index(self) -> Optional[int]:
        """1-based index of the first non-COMPLETE subtask, or None if all done."""
        for task in self._subtasks:
            if task.status is not Status.COMPLETE:
                return task.index
        return None

    @property
    def active_subtask(self) -> Optional[Subtask]:
        index = self.active_index
        return None if index is None else self._subtasks[index - 1]

    @property
    def final_subtask(self) -> Subtask:
        return self._subtasks[-1]

    @property
    def all_complete(self) -> bool:
        return self.active_index is None

    @property
    def awaiting_arrival(self) -> bool:
        """True once the arrival subtask is the one being worked on."""
        active = self.active_subtask
        return active is not None and active.is_arrival

    @property
    def final_confirmations(self) -> int:
        return self._final_confirmations

    # ---- mutation ----

    def apply(self, updates: Any, *, measured_distance_m: Optional[float] = None) -> ApplyResult:
        """Apply model-proposed status updates, enforcing every invariant.

        `measured_distance_m` is an authoritative distance to the final target
        from a dedicated arrival check; when supplied it overrides whatever the
        planner claimed in its own update.
        """
        if isinstance(updates, str):
            try:
                updates = extract_json(updates)
            except ValueError as exc:
                # An unreadable reply must not abort the episode. Skipping the
                # update just leaves progress where it was, which is the safe
                # direction: no progress can never cause an early stop.
                return ApplyResult(False, (str(exc),))
        if isinstance(updates, dict):
            updates = updates.get("updates", updates.get("subtasks", []))
        if not isinstance(updates, list):
            return ApplyResult(False, ("Update payload was not a list of subtask updates.",))

        active_index = self.active_index
        if active_index is None:
            return ApplyResult(False, ("All subtasks are already COMPLETE; ignoring further updates.",))

        rejections: List[str] = []
        proposal: Optional[Dict[str, Any]] = None
        for entry in updates:
            if not isinstance(entry, dict):
                rejections.append(f"Ignored non-object update {entry!r}.")
                continue
            index = entry.get("index")
            index = int(index) if isinstance(index, (int, float, str)) and str(index).strip().isdigit() else None
            if index is None or not 1 <= index <= len(self._subtasks):
                rejections.append(f"Ignored update with out-of-range index {entry.get('index')!r}.")
                continue
            if index != active_index:
                # Invariant 2: reaching ahead (or re-editing finished work) is
                # exactly how a spurious COMPLETE used to slip in.
                rejections.append(
                    f"Ignored update to subtask {index}; only active subtask {active_index} may change."
                )
                continue
            proposal = entry

        if proposal is None:
            return ApplyResult(False, tuple(rejections))

        task = self._subtasks[active_index - 1]
        proposed = parse_status(proposal.get("status"))
        if proposed is None:
            rejections.append(f"Ignored unrecognised status {proposal.get('status')!r}.")
            return ApplyResult(False, tuple(rejections))

        distance = measured_distance_m
        if distance is None:
            distance = coerce_distance(proposal.get("distance_estimate_m", proposal.get("distance_m")))

        granted, reason = self._clamp(task, proposed, distance)
        if reason:
            rejections.append(reason)

        updated = replace(
            task,
            status=granted,
            evidence=str(proposal.get("evidence") or task.evidence)[:400],
            distance_m=distance if distance is not None else task.distance_m,
            blocked_reason=str(proposal.get("blocked_reason") or "")[:200] if granted is Status.BLOCKED else "",
        )
        changed = updated != task
        self._subtasks[active_index - 1] = updated
        return ApplyResult(changed, tuple(rejections))

    def _clamp(self, task: Subtask, proposed: Status, distance: Optional[float]) -> Tuple[Status, str]:
        """Reduce a proposed status to the strongest one the rules permit."""
        current = task.status

        # Invariant 3a: progress never regresses.
        if _RANK[proposed] < _RANK[current]:
            return current, f"Rejected regression on subtask {task.index}: {current.value} -> {proposed.value}."

        # Invariant 3b: at most one rank of progress per update.
        if _RANK[proposed] - _RANK[current] > 1:
            return (
                Status.IN_PROGRESS,
                f"Clamped subtask {task.index} to IN_PROGRESS: cannot jump {current.value} -> {proposed.value} in one step.",
            )

        if proposed is not Status.COMPLETE:
            if task.is_arrival:
                self._final_confirmations = 0
            return proposed, ""

        # Invariant 4: arrival needs a number inside the radius. The model
        # supplies the measurement; the threshold is applied here.
        if task.is_arrival:
            if distance is None:
                self._final_confirmations = 0
                return (
                    Status.IN_PROGRESS,
                    f"Held subtask {task.index} at IN_PROGRESS: COMPLETE claimed with no distance estimate.",
                )
            if distance > self.arrival_radius_m:
                self._final_confirmations = 0
                return (
                    Status.IN_PROGRESS,
                    f"Held subtask {task.index} at IN_PROGRESS: distance {distance:.1f} m exceeds "
                    f"the {self.arrival_radius_m:.1f} m arrival radius.",
                )

        # Invariant 5: the final subtask must be confirmed repeatedly.
        if task.index == len(self._subtasks):
            self._final_confirmations += 1
            if self._final_confirmations < self.confirmations_required:
                return (
                    Status.IN_PROGRESS,
                    f"Held final subtask at IN_PROGRESS: confirmation "
                    f"{self._final_confirmations}/{self.confirmations_required}.",
                )

        return Status.COMPLETE, ""

    # ---- rendering ----

    def render(self) -> str:
        """The tracker as text, for planner prompts and task memory."""
        lines = [f"Original task: {self.goal}"]
        lines.extend(task.render() for task in self._subtasks)
        active = self.active_subtask
        lines.append(
            f"Active subtask: {active.index} ({active.text})" if active else "Active subtask: none (all COMPLETE)"
        )
        return "\n".join(lines)

    def render_for_actor(self) -> str:
        """A compact progress summary handed to the action policy each step."""
        lines = [f"Original goal: {self.goal}", "Subtask progress:"]
        for task in self._subtasks:
            marker = ">" if task.index == self.active_index else " "
            lines.append(f" {marker} {task.index}. [{task.status.value}] {task.text}")
        active = self.active_subtask
        if active is not None:
            lines.append(f"You are currently working on subtask {active.index}: {active.text}")
        final = self.final_subtask
        if final.distance_m is not None:
            lines.append(
                f"Measured distance to the final target ({self.target_phrase}): {final.distance_m:.1f} m "
                f"(must be <= {self.arrival_radius_m:.1f} m before stopping)"
            )
        return "\n".join(lines)
