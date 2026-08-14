# Temporal Memory

`temporal_memory` 是 VLN agent 的时序状态与视觉判断模块。它围绕当前
`Subgoal` 持续保存观测证据，判断子目标是否完成、识别近期执行错误，并把
结构化事件发布给 `TaskMemory`。模块入口是：

```python
from agentflow.agents.models_embodied_v2.memory.temporal_memory import (
    TemporalMemory,
    TemporalCaptioner,
)
```

## 模块组成

| 文件 | 职责 |
| --- | --- |
| `temporal_memory.py` | `TemporalMemory` 总入口。负责生命周期同步、每步更新、协调判断、保存结果和发布事件。 |
| `temporal_captioner.py` | VLM 时序视觉判断器。把有序图像和子目标输入 Qwen-VL，严格解析完成与错误 JSON。 |
| `frame_history.py` | 观测帧保存辅助逻辑；在观测写入 memory 前复制 RGB，防止外部缓冲区后续修改历史证据。 |
| `completion_judge.py` | 增长式完成判断。选择 completion/error 双窗口，调用 Captioner，并用运动与地标证据校验门口、决策点和转向完成。 |
| `event_publisher.py` | 校验 `CaptionResult` 并向 `TaskMemory` 按固定顺序发布时序事件。 |
| `preview_store.py` | 保存同一站位的 PREVIEW 多视角、选择结果和 selector 错误；这些视角不会混入真实时间序列。 |
| `interfaces.py` | `TaskMemory`、Captioner、Preview selector 的 Protocol 接口，支持依赖替换和单元测试 mock。 |
| `__init__.py` | 对外导出唯一的 temporal 模块 API。 |

共享的数据模型不在本目录：`data_models.py` 定义 `Subgoal`、`MemoryFrame`、
`TemporalFrameInput`、`CaptionResult`、事件和配置；`task_memory.py` 保存任务
计划、活动子目标和最新观测。

## 每一步的数据流

```text
RGB / depth / pose
      │
      ├─ LandmarkTracker 生成 landmark 证据
      ├─ VLNAgent 计算 translation_m、yaw_delta_deg
      │
      ▼
TemporalMemory.set_motion_evidence / set_landmark_evidence
      │
      ▼
TaskMemory.record_input(RGB)
      │
      ▼
TemporalMemory.update_from_task_memory()
      │
      ├─ frame_history：复制并记录带运动/地标元数据的帧
      ├─ completion_judge：选择证据窗口
      ├─ temporal_captioner：VLM 完成判断与错误判断
      ├─ completion_judge：运动完成护栏修正完成结果
      └─ event_publisher：发布 ERROR、SUBGOAL_COMPLETED
      │
      ▼
TaskMemory：更新错误状态；完成时推进活动 Subgoal
```

`VLNAgent` 随后读取本步 `CaptionResult`：完成事件会使下一步使用新的活动
subgoal；错误结果只有在置信度和近期实测运动都满足恢复条件时，才会驱动
waypoint recovery。

## 证据窗口与性能边界

完整历史只在当前 subgoal 内保留，不会跨 subgoal 或 episode 延续。为了保持
在线推理成本恒定，不会把全部历史交给视觉模型：

| 判断 | 输入帧 | 目的 |
| --- | --- | --- |
| completion | 最多 9 帧：首帧锚点和最近进展等代表证据 | 判断当前子目标是否被视觉证明完成。 |
| error | completion 窗口的最近后缀，最多 8 帧 | 判断 `WALL_STUCK`、`TURN_OSCILLATION`、`IN_PLACE_SPIN`、`GET_NOWHERE`。 |

因此历史长度增长不会增加单次 VLM 请求的图像数量。`TemporalCaptioner` 的图片
缩放、prompt、模型实例和调用次数均由其自身配置控制，Temporal Memory 不会
额外编码图像或额外发起模型请求。

## 结果、事件与状态切换

每次成功判断会得到一个 `CaptionResult`：

- `completed`：当前 subgoal 是否完成；
- `error`、`error_mode`：近期是否有明确执行错误；
- `error_confidence`、`error_evidence`：错误判断的可信度和依据。

`event_publisher` 保持如下发布顺序：先发布 `ERROR`，再发布
`SUBGOAL_COMPLETED`。`TaskMemory` 仅在 `SUBGOAL_COMPLETED=True` 的事件 ID 与
当前活动 subgoal 相同时推进计划；错误事件不会跳过任何 subgoal。

## Reset 与 Preview

`TemporalMemory` 在每次公开读取和更新前检查 `TaskMemory` 的 reset generation。
任务被重置或活动 subgoal 改变时，旧帧、最新判断和 preview 状态会被清除，避免
上一个阶段的视觉证据影响下一个阶段。

PREVIEW 视图来自同一站位的不同朝向，不代表时间推进。它们仅由
`preview_store.py` 保留给 waypoint 决策，绝不计入路径长度、completion evidence
或 error evidence。
