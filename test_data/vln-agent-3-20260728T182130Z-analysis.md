# VLN Agent 3: 8-Episode Failure Analysis

## Technical summary

The run evaluated the first eight `val_unseen` episodes, not a random sample.
Episodes 632 and 505 are the two previously tuned examples. Aggregate success
was 2/8 (25%), mean SPL was 0.187, and mean final distance to goal was 5.03 m.

The dominant verified defect is premature STOP acceptance. All five episodes
that issued STOP did so without completing the final subgoal. Episodes 632 and
505 happened to be within the Habitat success radius at that moment; episodes
1631, 1192, and 992 stopped 14.97 m, 4.75 m, and 9.47 m from the goal.
Therefore the two successes do not demonstrate reliable final-subgoal
completion.

The other three episodes timed out at 100 steps without leaving subgoal 1.
Their failure modes combine overly strict doorway completion, a semantically
incorrect generated subgoal for walking around a bed, and frequent error-mode
recoveries that perturb navigation.

## Episode evidence

| Episode | Result | Steps | Final DTG | Last subgoal transition | Diagnosis |
|---:|---:|---:|---:|---|---|
| 632 | Success | 83 | 0.55 m | `1 -> 2`, then STOP | Premature final-stage STOP happened inside success radius |
| 505 | Success | 38 | 1.74 m | `3 -> 4`, then STOP | Premature final-stage STOP happened inside success radius |
| 1631 | Failure | 51 | 14.97 m | Stayed on `1`, caption `wip`, then STOP | Final-subgoal waypoint STOP accepted without completion |
| 1192 | Failure | 23 | 4.75 m | `1 -> 2`, then STOP | Newly active final subgoal stopped immediately |
| 1628 | Failure | 100 | 3.83 m | Stayed on `1` | Bathroom-exit completion never verified |
| 1521 | Failure | 100 | 0.91 m | Stayed on `1` | Reached near the goal but exit-stage completion never advanced |
| 992 | Failure | 33 | 9.47 m | `2 -> 3`, then STOP | Newly active final subgoal stopped immediately |
| 1769 | Failure | 100 | 4.01 m | Stayed on `1` | Planner converted “around the bed” into an incorrect full circuit |

## Premature STOP is the primary code defect

`WaypointPolicyMixin._select_pixel` accepts any model STOP whenever the current
subgoal is the final list element. It does not require Temporal Memory to mark
that final subgoal complete. Because Temporal Memory advances the subgoal
before waypoint selection in the same step, a transition into the final stage
immediately exposes this permissive branch.

This pattern is visible in four multi-stage episodes:

- 632: caption completes subgoal 1, state becomes subgoal 2, then action STOP.
- 505: caption completes subgoal 3, state becomes subgoal 4, then action STOP.
- 1192: caption completes subgoal 1, state becomes subgoal 2, then action STOP.
- 992: caption completes subgoal 2, state becomes subgoal 3, then action STOP.

Episode 1631 has only one subgoal. Its caption still reported `wip`, but the
waypoint model's STOP was accepted because that one subgoal was also final.

The correct invariant is: Habitat STOP should be accepted only after Task
Memory reports the whole task complete, or after an independently verified,
multi-frame final-target guard. Merely being on the final subgoal is
insufficient.

## The timeout failures expose completion and planning gaps

### Doorway completion is too brittle

Episodes 1628 and 1521 remained on their initial “exit the room” subgoal for
all 100 steps. Episode 1521 ended only 0.91 m from the goal, which is strong
evidence that physical progress and task-memory progress diverged. The current
doorway guard requires a particular ordered combination of landmark
visibility, proximity, measured threshold translation, and a
destination-dominant view. If any tracker state is missed, the growing window
does not advance the task.

### The planner changed the meaning of episode 1769

The instruction says to walk clockwise around the bed and then exit. The
generated criterion instead requires returning to the starting point after a
full clockwise circuit. That is not implied by the instruction and prevents
progress to the exit subgoal. This is a subgoal-generation semantic error, not
just a visual completion error.

### Error-mode output is noisy and recovery is heavily exercised

The three timeout episodes reported non-`NONE` error modes on 58% to 69% of
steps. Deterministic recovery bypassed waypoint inference on 28 steps for
episode 1628, 9 for 1521, and 27 for 1769. Error modes are also noisy in a
successful episode: 632 reported a non-`NONE` mode on 52 of 83 steps and used
18 deterministic recovery steps.

This does not prove every recovery action was harmful, but it proves error mode
is not a reliable discriminator of failure in this run. It should not dominate
navigation until its precision is measured on more episodes.

## Recommended correction order

1. Reject waypoint-model STOP unless Task Memory is complete. Keep a narrowly
   defined multi-frame final-target exception only if it is separately
   validated.
2. Prevent a subgoal transition from immediately triggering STOP in the same
   observation.
3. Add completion fallbacks for doorway stages using measured path progress
   and stable room-transition evidence, without requiring one exact tracker
   sequence.
4. Reject planner criteria that introduce route semantics absent from the
   instruction, such as “full circuit” or “return to the starting point.”
5. Temporarily disable action-level error recovery, or require much stronger
   motion-grounded confirmation, while measuring its effect in an ablation.
6. Re-run a broader deterministic sample that excludes 505 and 632 before
   starting the full split.

## Scope and limitations

The evidence comes from the eight rank logs and rank summary JSON files in the
specified run directory. No videos were recorded, and per-step world position
or distance-to-goal was not logged. Consequently the STOP defect and planner
error are verified directly, while the exact visual event missed by the
doorway tracker remains unresolved without replay/video or richer pose logs.
