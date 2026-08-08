# VLN Agent 3 Rank 3 and Rank 7 Diagnosis

## Technical summary

Rank 3 failed because the repeated final-target guard treated a nearby,
still-ahead doorway as proof that the agent was already beyond that doorway.
The immediate same-frame STOP defect is fixed, but the replacement guard is
still semantically too weak for relational goals.

Rank 7 did not navigate at all. Plan-fidelity validation correctly rejected an
invented requirement to complete a full clockwise circuit and return to the
starting point, but both model attempts returned the same invalid plan. The
runner then aborted during `prepare_task`.

## Rank 3: nearby landmark was confused with completed spatial relation

- Episode: 1192
- Instruction: turn left, then stop just past the doorway straight ahead.
- At step 22, subgoal 1 completed and state advanced to subgoal 2.
- Steps 22 and 23 continued moving; the old same-frame final STOP no longer
  occurred.
- At step 24, Captioner still reported `wip`, but the agent issued STOP.
- Final distance to goal was 5.25 m.

The only non-memory path capable of producing this STOP is the repeated
final-target evidence guard. Its predicate accepts:

1. a word overlap with the final subgoal, such as `doorway`;
2. a generic near/visible relation;
3. waypoint depth at most 2.75 m;
4. at least 0.25 m progress;
5. two positive votes.

That proves the doorway is visible and nearby, but not that the camera has
crossed it and is positioned past it. Relational final goals therefore need
relation-specific completion evidence. A `past`, `through`, `inside`, `on`,
or `beside` goal cannot be reduced to generic target visibility.

## Rank 7: strict plan rejection has no repair path

- Episode: 1769
- Instruction: walk clockwise around the bed toward the exit, then stop on the
  hallway rug.
- Both planning attempts generated a criterion requiring a full clockwise
  circuit and returning to the starting point.
- The new fidelity validator rejected this invented route requirement.
- After two identical invalid responses, `prepare_task` raised `ValueError`;
  no episode steps or metrics were produced.

The validator is working as intended, but retry feedback is only generic. The
planner is not told which semantic requirement was invented, and there is no
deterministic normalization fallback. A recoverable planning error therefore
becomes a rank-level crash.

## Required fixes

1. Disable the generic final-target vote for goals involving `past`, `through`,
   `enter`, `inside`, `on`, or directional route relations.
2. For those goals, require Temporal Memory completion or a matching
   motion/landmark relation guard.
3. Include the exact fidelity-validation error in the planner retry prompt.
4. If the second response repeats an invented full circuit/return-to-start,
   deterministically rewrite only that completion criterion instead of
   aborting the episode.
5. Catch episode preparation errors in the runner so one invalid plan does not
   terminate an entire GPU shard.

## Evidence limitations

Rank 3 was not run with navigation-debug lines, so the exact waypoint evidence
strings are absent from the log. The STOP path is still identifiable from the
state: Captioner remained `wip`, Task Memory was incomplete, and the STOP
occurred after the required repeated-vote delay. Rank 7's exception and
repeated invalid response are recorded directly in the log.
