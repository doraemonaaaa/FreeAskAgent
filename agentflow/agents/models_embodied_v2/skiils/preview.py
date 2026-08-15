"""Preview view selection, owned by the Captioner side.

When a PREVIEW decision's surrounding views reach Temporal Memory, the memory
asks a ``PreviewSelector`` which one of them the actor should act on.  The actor
never makes this choice: it deposits the views, and then picks a floor point
inside whichever view comes back, using its ordinary waypoint policy.

This module fixes the contract — the strict reply schema, its parser, and the
selector interface.  The judgement itself is not implemented here: filling in a
real ``PreviewSelector`` is the Captioner side's work, and
``UnimplementedPreviewSelector`` is what stands in until then.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, model_validator

from agentflow.agents.models_embodied_v2.data_models import (
    PreviewSelection,
    Subgoal,
)


# The views are labelled with their index and heading offset in the same order
# they are held. The reply binds one navigable floor point to one exact view.
PREVIEW_SELECTION_PROMPT = """You are judging which way an indoor navigation
agent should go. The images are simultaneous views from one standing position,
given in order, each labelled with its view_index and its heading offset in
degrees from the agent's current facing: negative is to the left, positive is
to the right, and 0 is straight ahead.

Choose the single view that best advances the active navigation subgoal, then
choose one reachable FLOOR pixel in that view. Report pixel coordinates on a
normalized 0..1000 grid: u=0 is the left edge, u=1000 the right edge, v=0 the
top, and v=1000 the bottom. For a doorway, put the point on visible walkable
floor just inside its structural threshold; do not default to image centre
when the opening is off-centre. Prefer floor that visibly leads through the
required opening over open floor that merely stays in the current room.

Reply only with one exact JSON object:
{"view_index":integer,"u":integer,"v":integer,"confidence":0.0,"evidence":"brief visual reason"}"""


class PreviewSelectionOutput(BaseModel):
    """Strict reply schema for the Captioner's view judgement."""

    model_config = ConfigDict(extra="forbid", strict=True)

    view_index: int
    u: int
    v: int
    confidence: float
    evidence: str

    @model_validator(mode="after")
    def valid_selection(self) -> "PreviewSelectionOutput":
        if self.view_index < 0:
            raise ValueError("view_index must not be negative")
        if not 0 <= self.u <= 1000 or not 0 <= self.v <= 1000:
            raise ValueError("u and v must be normalized integers in [0, 1000]")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        self.evidence = self.evidence.strip()
        if not self.evidence:
            raise ValueError("evidence must not be empty")
        return self


def parse_preview_selection(
    payload: dict,
    *,
    view_count: int,
) -> PreviewSelection:
    """Validate a reply and bound it to the views actually held.

    The index is checked against ``view_count`` here rather than downstream: a
    waypoint back-projected through the wrong view's depth and camera transform
    produces a plausible-looking world coordinate in the wrong direction, which
    fails silently.
    """
    if view_count < 1:
        raise ValueError("selecting a preview view requires at least one view")
    output = PreviewSelectionOutput.model_validate(payload)
    if output.view_index >= view_count:
        raise ValueError(f"view_index must be in [0, {view_count - 1}]")
    return PreviewSelection(
        view_index=output.view_index,
        u=output.u,
        v=output.v,
        confidence=output.confidence,
        evidence=output.evidence,
    )


class PreviewSelector(Protocol):
    """What Temporal Memory calls the moment preview views arrive."""

    def select(
        self,
        *,
        subgoal: Optional[Subgoal],
        views: Sequence[Any],
    ) -> Optional[PreviewSelection]:
        """Return the chosen view, or None to decline."""
        ...


class UnimplementedPreviewSelector:
    """Stand-in selector that declines to choose.

    Returning None is deliberate rather than guessing a heading: the actor
    falls back to the most forward view and records that no judgement was made,
    so a placeholder can never be mistaken for a real one in the logs.
    """

    def select(
        self,
        *,
        subgoal: Optional[Subgoal] = None,
        views: Sequence[Any] = (),
    ) -> Optional[PreviewSelection]:
        return None


__all__ = (
    "PREVIEW_SELECTION_PROMPT",
    "PreviewSelection",
    "PreviewSelectionOutput",
    "PreviewSelector",
    "UnimplementedPreviewSelector",
    "parse_preview_selection",
)
