# VLN Memory Interfaces

`TemporalMemory` is one typed memory module. Agent and ablation code should
normally call it through `TemporalMemoryInterface`, which owns image/action
alignment and generates internal step IDs and timestamps.

## Standalone Temporal ablation

```python
from agentflow.agents.models_embodied_v2.memory import (
    TemporalMemory,
    TemporalMemoryInterface,
)

temporal = TemporalMemoryInterface(
    TemporalMemory(goal=instruction, captioner=captioner)
)
temporal.reset(episode_id="episode-1", goal=instruction)

temporal.observe(frame_0)
temporal.stage_action("FORWARD")
temporal.observe(frame_1)  # closes FORWARD with its post-action frame

# Repeat until three completed transitions exist.
record = temporal.latest_record
metrics = temporal.diagnostics()["timing"]
```

The first three-step analysis requires four observations. Only the three
post-action observations are sent to the Video Understanding model. The
interface snapshots each small RGB observation, so a simulator may safely
reuse its mutable frame buffer without corrupting earlier steps.

By default the deterministic rules and optical flow update after every action,
while the foundation model runs once per three newly completed actions
(`window_size=3`, `analysis_stride=3`). This avoids turning a shorter window
into more model calls.

## Memory ablations

`CompositeMemory` and `AsyncThinkActVLN(memory_mode=...)` support:

- `task`: Task Memory only.
- `temporal`: Temporal Memory only.
- `task+temporal`: both modules, with separately labelled planner context.
- `none`: coordinator/action-history baseline (main Python API only).

Habitat still sends only the instruction at reset and one RGB image per agent
call. It does not construct memory metadata.

## Timing definitions

Diagnostics expose additive counts and totals so results from episodes or GPU
ranks can be weighted correctly:

- `task_memory`: one inference sample is one `record_input` update.
- `temporal_memory`: one inference sample is one eligible three-step
  `analyze_if_ready` call, including request construction, foundation-model
  call, validation, rule fusion, and commit.
- `video_understanding`: the synchronized foundation-model call only.
- `temporal_memory_interface`: image-only `observe` updates, including warm-up
  observations before the first three-step window. Terminal finalization is
  reported separately under `operations.finish_episode`.

Every component reports `inference_count`, `total_inference_ms`, and
`average_inference_ms`; model failures and the five-second latency budget are
reported separately. With `include_raw_response=True`, successful analyses
include `latest_analysis.raw_response`; failed validation includes
`last_failed_raw_response` so malformed or truncated model output is auditable.
