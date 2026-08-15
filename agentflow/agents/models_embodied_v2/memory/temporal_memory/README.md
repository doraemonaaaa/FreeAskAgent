# Temporal Memory

`temporal_memory` 是 `vln_agent_4.py` 使用的统一时序视觉模块。它保存当前
subgoal 的连续观测，通过一个 Captioner VLM 调用判断当前场景，并把完成或错误
事件发布给 `TaskMemory`。Waypoint 规划和 Habitat 动作执行不属于本模块。

统一从包入口导入：

```python
from agentflow.agents.models_embodied_v2.memory.temporal_memory import (
    TemporalCaptioner,
    TemporalMemory,
)
```

## 文件职责

| 文件 | 职责 |
| --- | --- |
| `temporal_memory.py` | `TemporalMemory` 总入口；同步 TaskMemory、处理 reset/subgoal 切换、消费观测并协调分析与事件发布。 |
| `temporal_captioner.py` | 唯一 Scene VLM；编码图像、调用视频视觉模型、校验 JSON，并输出 landmark、完成、错误和最终目标判断。门口 subgoal 会在同一次调用中使用专用穿门提示。 |
| `completion_judge.py` | 保存当前 subgoal 的有界证据窗口，组织 `SceneAnalysisRequest`，调用 Captioner，并把模型判断转换为 `CaptionResult`。它不再使用累计路径、转角或投票替模型判定完成。 |
| `frame_history.py` | 在 RGB 进入历史前做一次防御性复制，避免 observation buffer 被复用后污染旧帧。 |
| `event_publisher.py` | 校验结果，并按固定顺序发布 `ERROR`、`SUBGOAL_COMPLETED`。 |
| `preview_store.py` | 保存同一位置的 PREVIEW 多视角、Captioner 选择的视角与归一化落脚点，以及异常；这些视角不会混入时序帧。 |
| `interfaces.py` | 定义 TaskMemory、Captioner 和 Preview Selector 的调用协议，便于 mock 或替换实现。 |
| `__init__.py` | 唯一公开 API，集中导出总类、Captioner、配置、数据类型、接口和错误类。 |

共享数据结构仍位于父模块：

- `data_models.py`：`Subgoal`、`MemoryFrame`、`SceneAnalysisRequest`、
  `SceneAnalysisResult`、`CaptionResult`、配置和事件类型。
- `task_memory.py`：任务计划、当前 subgoal、最新 RGB，以及 temporal 事件接收。

## 每一步的调用流程

```text
vln_agent_4.act(rgb, depth, pose)
        │
        ├─ 计算本步 translation_m / yaw_delta_deg
        ├─ TaskMemory.record_input(rgb)
        └─ TemporalMemory.update_from_task_memory()
                  │
                  ├─ 同步 reset generation 与当前 subgoal
                  ├─ 复制一次 RGB，写入当前 subgoal 的历史窗口
                  ├─ 组织 1–16 帧 SceneAnalysisRequest
                  ├─ TemporalCaptioner.analyze_scene()  ← 一次 VLM
                  │      ├─ 当前 landmark 与位置
                  │      ├─ subgoal 是否完成
                  │      ├─ 累积执行错误
                  │      └─ 最终目标是否到达
                  ├─ Completion Judge 转换、校验结果
                  └─ Event Publisher 更新 TaskMemory
        │
        └─ vln_agent_4 根据新 TaskMemory 状态继续 Waypoint 或校验 STOP
```

Temporal Memory 每个被分析的步骤最多一次 Scene VLM 请求。朝已锁定、距离仍较远的
门口移动时只记录帧而延迟分析，不发起必然为未完成的 Scene 请求。它不会再分别调用 Landmark VLM、
Completion VLM 和 Error VLM，也不会把下一个 subgoal 的描述交给 Captioner，因此
不会用后续目标（例如“pool”）解释当前门口 subgoal。Waypoint 自己的动作规划调用
仍是独立的，因为它回答的是“下一步怎么走”，不是“当前 subgoal 是否完成”。

## Captioner 输出

`TemporalCaptioner.analyze_scene()` 接收当前 subgoal、是否为最终 subgoal，以及按
时间排序的帧。一次结构化结果包含：

- `landmark`：当前目标是否可见、方向、远近、归一化坐标和置信度；
- `completed` / `completion_confidence`：当前 subgoal 的视觉完成判断；
- `error_mode` / `error_confidence`：`WALL_STUCK`、
  `TURN_OSCILLATION`、`IN_PLACE_SPIN`、`GET_NOWHERE` 或 `NONE`；
- `final_target`：仅最终 subgoal 使用的目标可见性和 `FAR/NEAR/AT`；
- `evidence`：用于日志和问题定位的简短逐帧依据。

Captioner 只返回结构化结果，不直接修改 TaskMemory，也不决定 Habitat 动作。

## 门口完成如何判断

门口仍由视频理解模型判断，但使用 Captioner 内部的专用单次提示，不增加 VLM
次数。模型必须分别回答：

- `door_state`：未看到、接近、门槛处、穿越中或已穿越；
- `door_camera_side`：门前、门口或门后；
- `door_transition`：最近连续帧是无变化、转头离开、接近，还是确实看到门框
  和门槛越过摄像机；
- `current_room_side`：当前画面仍是原房间、已经是远侧房间，或无法确定。

只有模型同时给出 `CROSSED`、`AFTER_DOOR`、`PASSED_THROUGH`、`FAR_SIDE`，
确认远侧空间主导当前画面，并且相机到 Waypoint 模型定位的结构门点足够近时，
Completion Judge 才接受完成。看到门后的房间、门因转向
离开画面、累计走了一段距离，都不能单独证明穿门。这里没有恢复旧的“累计路径
超过阈值即穿门”规则；所有门口证据仍由视频模型从有序帧中识别。

最终目标也采用语义判断。提示中明确区分 swimming pool 与浴缸、水槽、蓝色
瓷砖或透过门看到的蓝色区域，避免新 subgoal 首帧把浴室误判成泳池目标。最终
subgoal 还会校验 Captioner 自己的结构化结论：`NEAR` 表示仍需继续，只有
`final_target.visible=true`、`proximity=AT` 且模型同时返回完成时才由 Temporal
Memory 推进完成。为避免历史窗口模型漏判后永远无法停止，Waypoint 还有独立的
最终 STOP 仲裁：仅最终 subgoal、置信度至少 0.90、证据明确点名目标并说明已在
紧邻/近场位置、且不含“远处、门后、尚未到达”等矛盾语句时才计一票；最近 5 次
Waypoint 视觉判断至少 2 票才真正 STOP。该仲裁不伪造 TaskMemory 完成事件。

## PREVIEW 与 Waypoint

Waypoint VLM 的明确 `TURN_LEFT/RIGHT` 会按 15° Habitat 原语执行，不再被强制
改写成 PREVIEW。只有模型主动请求 PREVIEW 时，runner 才在同一位置渲染多方向
视图，并交给 `TemporalCaptioner.select()`。选择结果同时包含：

- `view_index`：应该使用哪一个相机朝向；
- `u/v`：该视图中 0–1000 归一化的可行走落脚点；
- `confidence/evidence`：置信度和视觉依据。

因此偏离画面中心的门洞不会再被固定中心点替换。Captioner 选好视角和落脚点后
直接使用对应视图的 depth、intrinsics 和 camera-to-world 反投影，不再对同一
PREVIEW 画面额外调用一次 Waypoint VLM。预览 yaw 与 Habitat runner 统一为负值
向左、正值向右；深度只校验所选点是否可用，不会用“最空旷方向”覆盖语义门洞。

## 状态与性能边界

- RGB 历史只属于当前 subgoal，最多 16 帧，不随 episode 无限增长。
- 新 subgoal、task reset 或新 episode 会清空旧帧、Scene 结果和 preview 状态。
- PREVIEW 是同一位置的多方向观察，不表示时间推进，不进入 temporal 历史。
- RGB 写入历史时只复制一次。
- Captioner 使用有界的对象身份 PNG 缓存；保留帧不会每步重复 resize/编码。
- 门口专用判断只是切换 system prompt 和 JSON 校验，仍是一次模型请求，没有新增
  RGB 拷贝、图片编码或 VLM 调用。

## 事件与 subgoal 切换

每次成功分析产生一个 `CaptionResult`。`event_publisher.py` 先发布 `ERROR`，再
发布 `SUBGOAL_COMPLETED`。只有完成事件的 `subgoal_id` 与当前活动 subgoal
一致时，TaskMemory 才推进计划。进入新 subgoal 后，Temporal Memory 立即建立
新的帧窗口。`vln_agent_4` 可在最后一个 subgoal 的 Temporal 完成事件后停止，也可
在上述重复、严格校验的当前帧 Waypoint STOP 证据成立后停止。

## 测试与 632 回归结论

单元测试覆盖单次 Scene 调用、JSON 校验、PNG 缓存、有界历史、reset/subgoal
隔离、事件顺序，以及门口结构化证据一致性。

R2R episode 632 已用完整带视频在线回归验证。v14 在 frame 41–42 穿过门槛，
frame 43 完全进入泳池房，frame 44–45 门外原房间不再可见且泳池近距离占据前景；
第二次严格校验的 Waypoint STOP 随后终止 episode。结果为 45 steps、
`success=1.0`、`SPL=1.0`，最终 geodesic distance 为 0.58m。验证视频在 R2R
工作区的 `videos_waypoint_fix_v14/632.mp4`。
