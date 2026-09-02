# VLN Task and Temporal Memory

`TaskMemory` owns the episode instruction, ordered subgoals, latest RGB
observation, and events used by the planner. `TemporalMemory` retains copied
observations for the current subgoal and selects bounded evidence windows.
Neither its internal `MemoryFrame` nor its public API contains actions.

## Per-frame update

```python
task_memory.record_input(rgb)
result = temporal_memory.update_from_task_memory()
```

`update_from_task_memory()` reads `TaskMemory.get_latest_observation()`,
ignores an already-consumed observation, and analyzes every distinct frame.
Completion uses at most nine selected frames; error diagnosis uses its recent
suffix of at most eight frames. The captioner judges:

- whether the current subgoal is visually complete;
- whether the recent sequence shows a cumulative visual error.

Temporal Memory publishes compact `ERROR` and `SUBGOAL_COMPLETED` events back
through `TaskMemory.publish_temporal_event()`.

## Episode reset

Reset Task Memory first, then Temporal Memory (this is what
`VLNAgent.reset_memory()` does):

```python
task_memory.reset(goal=instruction, subgoals=subgoals)
temporal_memory.reset()
```

Every Task Memory reset increments `get_reset_generation()`. Temporal Memory
checks that generation before every update and public read, so even a direct
Task Memory reset cannot expose stale frames from the previous episode.
