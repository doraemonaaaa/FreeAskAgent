# VLN Agent v3 架构说明及相对 v2 的功能增量

本文以当前代码实现为准，说明 `vln_agent_3` 的在线执行流程，以及它相对 `vln_agent_2` 新增的模块和约束。v3 并非重写 RGB-D waypoint 导航器：它保留 v2 的深度有效性检查、可行走像素选择和相机坐标到 Habitat 世界坐标的反投影；其主要增量是将“子目标计划—时序证据—地标与运动校验—受约束动作”组成闭环。

## 1. 设计目标

v2 已能将第一人称 RGB-D 观测转化为一个世界坐标 waypoint，但它的动作主要由当前帧和当前子目标驱动。对于“穿过门口”“在路口左转”“到达后停下”这类必须依赖先后顺序的指令，仅凭单帧难以确认进展，也难以区分正常视角变化和原地受阻。

v3 因而新增以下能力：

1. 在任务准备阶段生成并严格校验有序子目标；
2. 在每一步用当前子目标内的时序证据判断是否完成，并将结果以事件同步给任务记忆；
3. 将真实位移、偏航角和地标状态与视觉证据结合，限制不符合物理过程的完成判断；
4. 显式预测导航意图、限制过早转向或停止，并在可信的受阻模式下执行确定性恢复；
5. 输出完整的中间状态和延迟诊断，便于复现实验与分析失败轨迹。

## 2. 总体执行流程

每个 episode 先调用 `prepare_task(instruction)`。随后环境每提供一帧 RGB-D 观测，`Actor.act(...)` 执行一次下列闭环：

```text
instruction
    │
    ├─ prepare_task ──> validated subgoal plan ──> Task Memory
    │
RGB-D observation + camera pose
    │
    ├─ measure translation / yaw
    ├─ Landmark Tracker (active subgoal only)
    ├─ Growing Temporal Memory
    │      ├─ completion judgement on selected temporal evidence
    │      └─ error judgement on the recent evidence suffix
    ├─ typed events: SUBGOAL_COMPLETED / ERROR ──> Task Memory
    ├─ intent-aware waypoint policy + guards / recovery
    └─ depth validation + RGB-D back-projection ──> Habitat waypoint or STOP
```

关键的时序顺序是：Temporal Memory 先发布本步事件，Task Memory 再更新当前子目标，最后 waypoint 策略只读取更新后的活动子目标。因此，已经完成的子目标不会在下一步继续牵引 waypoint。

## 3. v2 与 v3 的对比

| 维度 | `vln_agent_2` | `vln_agent_3` |
| --- | --- | --- |
| 子目标生成 | 一次生成并进行基本 JSON 解析。 | 生成失败时最多重试一次；以 Pydantic 严格校验字段、非空文本和从 `1` 开始的连续唯一 ID，并修正无依据的环路要求和非最终 `stop` 条件。 |
| 子目标语义 | 计划可描述视觉完成条件，但没有专门的转向决策点规范化。 | 对“先直行、再在某处转向”的指令，将直行阶段对齐到下一转向的决策点，避免把“正在靠近”误认为完成。 |
| 时序记忆 | 标准 `TemporalMemory` 最多保留 8 帧，默认仅输出完成判断。 | `GrowingCompletionMemory` 为当前子目标积累观测；每次调用至多选择 9 帧代表证据完成判断，另取其中最近 8 帧进行错误诊断。 |
| 时序证据 | 图像帧为主。 | 每帧额外记录实测平移、偏航变化、子目标内累计路径长度以及地标状态，作为视觉判断的辅助证据。 |
| 完成判定 | VLM 判断直接驱动状态。 | 视觉完成判断还受门槛穿越、决策点到达、净转角等运动条件校验；只有匹配活动子目标的正完成事件才会推进计划。 |
| 地标理解 | 没有独立地标状态。 | 新增地标跟踪器，输出可见性、相对方向、距离层级、是否通过、目标区域是否主导画面、置信度和依据。 |
| waypoint 输出 | 仅返回像素坐标或 `stop`。 | 输出 `intent`、归一化坐标、置信度和视觉依据；意图包括走廊跟随、接近地标、左右转、最终接近和停止。 |
| 行为约束 | 主要依赖单步 VLM 选择和深度可行性。 | 维护行为历史、走廊朝向锁定和导航阶段；未达到转向决策点时抑制侧向转入。 |
| 异常恢复 | 无面向时序错误的恢复策略。 | 支持 `WALL_STUCK`、`TURN_OSCILLATION`、`IN_PLACE_SPIN`、`GET_NOWHERE`；错误必须满足 VLM 高置信度、运动一致性和多步投票才触发确定性恢复。 |
| 停止控制 | waypoint 模型的 `stop` 可直接结束动作。 | 非最终子目标的停止被忽略；最终停止需计划完成或重复的近目标证据支持，避免远处看见目标即停止。 |
| 运行诊断 | 提供基础耗时与模型回复。 | Worker 和 Habitat 日志额外记录时序窗口、所选帧 ID、双窗口结果、事件、地标、恢复模式、导航意图、停止原因及分阶段延迟。 |

## 4. v3 的核心模块

### 4.1 严格的子目标计划与 Task Memory

`prepare_task` 要求视觉语言模型产生如下语义结构：

```json
{
  "subgoals": [
    {
      "subgoal_id": "1",
      "description": "...",
      "completion_criteria": "..."
    }
  ]
}
```

每个 `Subgoal` 包含三项数据：唯一字符串标识符、可执行描述和可视觉验证的完成条件。v3 使用 `SubgoalPlanOutput` 验证输出，并在计划规范化阶段完成三类处理：

- 删除指令中未出现的“完整绕行、回到起点”等额外路径要求；
- 删除中间阶段不应出现的停止条件；
- 为后续需要左/右转的直行阶段补充对应的决策点语义。

`TaskMemory` 是计划推进的唯一状态源。它保存原始指令、子目标列表、当前活动子目标、最近观测以及时序事件。`SUBGOAL_COMPLETED=True` 仅在事件 ID 与活动子目标 ID 相同的情况下推进索引；错误事件只更新错误状态，不会越过任何子目标。控制器向 VLM 提供的也是活动子目标而不是完整计划，从而减少已完成阶段对当前动作的干扰。

### 4.2 带双窗口的 Temporal Memory

v3 使用 `GrowingCompletionMemory` 替代 v2 的固定 8 帧 `TemporalMemory`。在一个活动子目标内，它持续记录帧；每个帧附有：

- `translation_m`：相邻观测间从相机位姿得到的实测平移；
- `yaw_delta_deg`：相邻观测间的实测偏航变化；
- `subgoal_path_length_m`：当前子目标开始后的累计路径长度；
- 地标的可见性、方向、接近程度、是否通过、置信度及文本依据。

但 v3 不会把不断增长的全部历史一次性输入模型。若当前子目标帧数不超过 9，则所有帧用于完成判断；在当前 `MAX_COMPLETION_EVIDENCE_FRAMES=9` 和 `RECENT_COMPLETION_EVIDENCE_FRAMES=8` 的配置下，历史超过 9 帧后，完成窗口实际为“首帧锚点 + 最近 8 帧”。代码保留了对地标状态变化帧和均匀历史帧的候选选择接口，但在该上限配置下没有额外名额将其加入窗口。这样，完成判断可同时保留“从何处开始”和“最近进展”，且视觉 token 开销有明确上界。

完成判断和错误判断使用不同窗口和不同严格 JSON 模式：

```text
completion window: selected representative frames, at most 9
error window:      recent suffix of the completion window, at most 8
```

完成判断只回答 `{"completed": bool}`，其目标是确认当前子目标的视觉完成条件是否被证明。错误判断在最近 8 帧足够时才启用，输出 `error`、`error_mode`、`confidence` 和 `evidence`；其候选模式为 `WALL_STUCK`、`TURN_OSCILLATION`、`IN_PLACE_SPIN`、`GET_NOWHERE` 或 `NONE`。因此，长期进展确认与局部异常诊断不会相互混淆。

在默认 v3 配置下，时序图像会缩放至边长不超过 160，完成和错误的结构化输出分别受到 token 上限控制；错误判断从第 8 个证据帧起才运行。这是在增加时序推理的同时限制在线显存与延迟增长的实际实现。

### 4.3 视觉完成判断的运动校验

Temporal Captioner 的视觉输出不是唯一决定因素。`GrowingCompletionMemory` 对三类容易被单帧误判的阶段增加运动约束：

- **门口/阈值穿越**：需要地标跟踪器给出可信的通过证据，或者先出现近距离阈值、随后有充分前向位移且偏航变化受限；仅看到门口或目标房间不足以完成。
- **转向前的决策点**：需要在子目标内走过最低路径长度，并测得与下一阶段方向一致的偏航变化。
- **纯转向阶段**：需要累计净偏航达到规定阈值（当前为 $60^\circ$）。

这些规则只为视觉时序判断提供物理可验证的护栏：它们可以在明确的运动证据下确认完成，也可以否决缺乏进展的过早完成，而不以启发式图像相似度覆盖 Captioner 的错误判断。

### 4.4 Landmark Tracker

每一步时序分析前，`LandmarkTrackerMixin` 使用当前 RGB、活动子目标、下一路线阶段、近期地标历史和实测运动生成一个结构化地标状态。地标输出包括：

```text
visible, direction, proximity, passed,
destination_dominant, confidence, evidence
```

其中 `direction` 为 `LEFT/CENTER/RIGHT/UNKNOWN`，`proximity` 为 `FAR/NEAR/AT/UNKNOWN`。对门口，`passed=True` 还必须通过顺序性校验：例如先居中接近阈值，再以较小偏航向前推进，最后目的区域成为画面主导；若无法满足该过程，模型的“已通过”声明会被撤销。该状态既进入 Temporal Memory，也作为 waypoint VLM 的辅助上下文。

### 4.5 意图化 waypoint 策略与行为约束

v3 仍利用 v2 的深度图对请求像素进行可行走性验证并反投影到世界坐标，但其 VLM waypoint 输出已扩展为：

```json
{
  "stop": false,
  "intent": "FOLLOW_CORRIDOR",
  "u": 500,
  "v": 750,
  "confidence": 0.0,
  "evidence": "..."
}
```

坐标统一在 `[0, 1000]` 归一化平面中表示，输出通过 Pydantic 严格验证；无效回复会重试，并保留安全的确定性退路。动作策略结合活动子目标、导航阶段、地标状态和最近行为历史进行选择，并额外施加：

- **走廊朝向锁定**：进入走廊跟随阶段后，连续前进建立朝向锁定；在真正到达下一转向决策点前，侧向 waypoint 或无依据转弯会被抑制。
- **转向放行**：只有活动子目标及其可测运动证据支持时，才放行所需的 `TURN_LEFT` 或 `TURN_RIGHT`。
- **深度失败恢复**：当深度图没有合法可行走点时，不中止 episode，而是给出合成的侧向转向目标，由低层执行器尝试脱离近场障碍。

### 4.6 错误确认、恢复与停止护栏

错误恢复刻意比 Captioner 输出更保守。系统先要求错误置信度不低于 `0.9`，再检查最近运动是否支持该模式，并在最近 5 次判断中累积至少 4 次同一候选，才启动恢复。已确认的恢复保持 2 步：

- `WALL_STUCK` 或 `GET_NOWHERE`：采用稳定的左侧转向 waypoint；
- `TURN_OSCILLATION` 或 `IN_PLACE_SPIN`：采用稳定的下方中央前向 waypoint；
- `NO_VALID_DEPTH`：由动作层触发侧向转向恢复。

停止同样是受保护的动作。中间子目标的 `STOP` 被转换为继续导航；最终子目标的 `STOP` 也首先被延后，除非所有子目标均完成，或最近窗口中至少 2/3 次近目标视觉与路径条件共同支持最终到达。这样可以避免模型仅因为远处出现目标、或在中间路段看到相似物体而提前结束。

## 5. 运行时可观测性与代价

v3 的 R2R worker 会将每步的内部状态通过 JSON 返回；Habitat runner 可将其写入日志。除常规动作外，日志包括：

- 当前子目标、完成状态、`ERROR` 和 `SUBGOAL_COMPLETED` 事件；
- 当前时序历史长度、完成窗口大小、实际选中的帧 ID、错误窗口结果、模型原始输出和 Captioner 延迟；
- 地标状态、行为历史、模型请求的 waypoint、实际应用的意图、保护规则原因、恢复模式和停止原因；
- RGB 处理、时序推理、深度、像素选择、waypoint 反投影及环境步进的分段耗时。

相较 v2，v3 的主要额外在线开销来自地标模型调用和时序完成判断；从第 8 帧起还会增加一次错误窗口调用。实现通过至多 9/8 帧的双窗口、160 边长时序图像及受限结构化输出控制这一开销。因而在比较吞吐或延迟时，应将 v3 视为“带在线状态验证和恢复的控制器”，而不是仅增加一个静态记忆缓存。

## 6. 代码入口与模块对应

| 作用 | 代码位置 |
| --- | --- |
| v2 基线 actor 与 RGB-D waypoint 反投影 | `agentflow/agents/vln_agent_2.py` |
| v3 actor 和 episode 级控制状态 | `agentflow/agents/vln_agent_3.py` |
| v3 协议、提示词、常量及严格输出模型 | `agentflow/agents/vln_agent_3_protocol.py` |
| 子目标计划解析与忠实性修正 | `agentflow/agents/vln_agent_3_planning.py` |
| 地标跟踪与门口穿越校验 | `agentflow/agents/vln_agent_3_landmark.py` |
| waypoint 验证、意图护栏和恢复 | `agentflow/agents/vln_agent_3_waypoint.py` |
| v3 增长式完成记忆与运动完成护栏 | `agentflow/agents/models_embodied_v2/memory/growing_completion_memory.py` |
| 双窗口 Captioner | `agentflow/agents/models_embodied_v2/TemporalCaptioner.py` |
| 任务状态及类型化事件接收端 | `agentflow/agents/models_embodied_v2/memory/task_memory.py` |
| R2R actor worker 与调试字段 | `FreeAskAgent_R2R/integrations/vln_waypoint_worker.py` |
| R2R/Habitat 日志与启动入口 | `FreeAskAgent_R2R/integrations/run_habitat_2.py`、`run_vln_agent_3_r2r_ce_8gpu.sh` |

## 7. 简要结论

`vln_agent_2` 的核心是“当前观测驱动的 RGB-D waypoint”。`vln_agent_3` 在其上增加了一个可验证的状态闭环：严格的子目标计划定义要完成什么，地标与位姿测量描述已经发生什么，双窗口 Temporal Memory 判断当前阶段是否完成或出现异常，受约束的 waypoint 策略决定下一步如何继续。v3 的新增模块并不替代 RGB-D 几何执行，而是为其提供时序状态、行为约束和恢复能力。
