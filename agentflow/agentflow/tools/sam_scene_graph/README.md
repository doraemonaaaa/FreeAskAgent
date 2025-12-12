# SAM2 Scene Graph Tool

Generate a task-friendly scene graph (objects + spatial relations) from the raw output of `SAM2_Perception_Tool`. The tool filters masks, computes normalized positions, and emits relations such as `left_of`, `ahead_of`, and optional `agent` interactions. Results can fuel reasoning prompts, navigation policies, or visual analytics (via `agentflow/util/scene_graph_viz.py`).

## Quick Start

```python
from agentflow.agentflow.tools.sam_perception.tool import SAM2_Perception_Tool
from agentflow.agentflow.tools.sam_scene_graph.tool import SAM2_SceneGraph_Tool

sam_tool = SAM2_Perception_Tool(model_cfg="sam2.1_hiera_l", device="cuda")
scene_graph_tool = SAM2_SceneGraph_Tool()

sam_result = sam_tool.execute(
    image_path="/path/to/image.jpg",
    mode="automatic",
    top_k=5,
    output_dir="tmp",
)

graph = scene_graph_tool.execute(
    sam_result=sam_result,
    min_area=1500,
    relation_threshold=0.04,
    save_path="tmp/scene_graph.json",
)

print(f"objects: {len(graph['objects'])}, relations: {len(graph['relations'])}")
```

## Inputs

| Parameter | Description |
|-----------|-------------|
| `sam_result` | In-memory dict returned by `SAM2_Perception_Tool.execute`. |
| `sam_result_path` | Path to a JSON dump compatible with the structure above (useful when running offline). |
| `min_area` | Drop segments with an area (in pixels) below this value. Defaults to `2000`. |
| `relation_threshold` | Minimum normalized delta (0–1) required before emitting a spatial relation. |
| `agent_position` | Reference origin for agent-object relations (default `(0.0, 0.0)`). |
| `include_agent_relations` | Adds `agent -> object` edges that tag in-path obstacles. |
| `save_path` | Optional target path for writing the resulting JSON graph. |

## Outputs

The tool returns a dictionary with:

- `objects`: filtered SAM segments with normalized bounding boxes, centered coordinates, and obstacle metadata.
- `relations`: spatial predicates among objects and (optionally) the agent.
- `metadata`: bookkeeping fields (image info, filter thresholds, counts).
- `saved_to`: absolute JSON path when `save_path` is provided.

The resulting JSON is compatible with `agentflow/util/scene_graph_viz.py`:

```python
from agentflow.util.scene_graph_viz import plot_scene_graph_matplotlib
plot_scene_graph_matplotlib(memory_like_object, outfile="tmp/scene_graph.png")
```

## Enabling the Tool in the Agent

Add `"SAM2_SceneGraph_Tool"` to the `enabled_tools` list when constructing the solver (see `quick_start.py`). This allows the planner/executor stack to call the scene graph tool after running SAM2 segmentation.
