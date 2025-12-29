---
# models_embodied 模块技术文档

路径：`agentflow/agentflow/models_embodied`

说明：本文件聚焦于该目录下的实现细节、组件职责与接口契约，面向具身智能（Embodied AI）的工程师与架构师。

## 1. 模块架构总览

- 主要目标  
  - `models_embodied` 包实现了 AgentFlow 中“具身”变体的核心推理与执行模块。它把高级规划（Planner）、工具执行（Executor）、以及记忆（Memory）等作为可组合的组件，以支持多步骤、多模态（图像+文本）任务，例如 VLN（视觉语言导航）类任务。

- 面向具身体（embodied agent）的抽象  
  - 基于职责分离的模块设计：  
    - Planner：将用户输入与感知结果（图像 bytes）转成结构化的“下一步/子目标/工具调用”决策；负责 prompt 构造、LLM 调用与解析。  
    - Executor：将 Planner 产生的工具命令（通常为生成的 Python 调用片段）安全地执行并返回结果，管理工具实例缓存与执行超时。  
    - Memory（短期/长期）：提供会话内/跨会话的上下文存储，记录文件、动作与每步执行的元数据。  
    - Initializer：启动时扫描 `tools/` 并并行加载工具，构建工具元数据表与工具实例缓存；提供工具发现与映射能力。  
    - Formatters/Prompts：定义 LLM 期望的结构化输出类型（Pydantic 模型）与任务/ToM/VLN 特定的 prompt 模板。

- 接口与数据流（简化）  
  - 输入：用户 query (str)、可选的 image paths 或 image bytes 列表、工具元数据（由 Initializer 提供）；LLM 引擎通过 `create_llm_engine(...)` 被注入到 Planner/Executor。  
  - Planner.analyze_query(question, image_paths) -> 返回 QueryAnalysis（或字符串），该方法把 prompt + image bytes 作为 LLM 输入（input_data 列表）并请求结构化输出。  
  - Planner.generate_direct_output/ generate_final_output -> 构造最终 prompt（包含 memory.get_actions() 等）并返回 LLM 生成结果（字符串或结构化）。  
  - Executor.generate_tool_command(question, image, context, sub_goal, tool_name, tool_metadata, ...) -> 返回 ToolCommand（analysis, explanation, command）。  
  - Executor.extract_explanation_and_command(response) -> 分析并提取实际可执行的 Python 片段（以 `execution = tool.execute(...)` 形式）。  
  - Executor.execute_tool_command(tool_name, command) -> 安全执行命令片段（带超时保护），返回执行结果列表或错误信息。  
  - Memory 提供 add_file / add_action / get_actions / get_files / set_query 等 API，供 Planner 与 Executor 读写。

## 2. 文件详细分析

（按文件名逐条说明；列出核心类/函数、职责和继承/依赖）

### `initializer.py`
- 核心类/函数：`Initializer`，辅助函数 `_get_optimal_workers()`。  
- 职责：  
  - 启动阶段扫描 `tools/` 目录，解析 `tool.py` 中的 `TOOL_NAME` 并建立短名->长名与长名->内部类/目录映射。  
  - 并行/串行加载工具模块、实例化工具类、缓存工具实例与收集工具 metadata（`tool_instances_cache`, `toolbox_metadata`）。  
  - 运行 demo 命令（`run_demo_commands`）以确认可用工具并更新 `available_tools`。  
  - 提供并行加载的策略与智能 worker 数量计算（支持 GNU parallel/SLURM 环境检测）。  
- 继承与依赖：无继承体系，依赖 Python 的 importlib/inspect、并发模块、以及项目中工具约定（`tools/*.tool`）。

### `planner.py`
- 核心类/函数：`Planner`（方法包括：`summarize_input_data()`, `analyze_query()`, `extract_context_subgoal_and_tool()`, `generate_direct_output()`）。  
- 职责：  
  - 对用户 query 与图像进行预处理，构造 QueryAnalysis prompt（使用 `QuerynalysisPrompt`），将 image bytes 附加到 LLM 的输入数组中。  
  - 封装 LLM 引擎（通过 `create_llm_engine` 创建 `llm_engine` 与 `llm_engine_fixed`），并按 is_multimodal 配置确定是否把 bytes 传入。  
  - 解析 LLM 返回（既支持 JSON -> Pydantic `NextStep`，也支持半结构化文本解析），以提取 context、sub_goal 与 tool_name（`extract_context_subgoal_and_tool` 中实现了正则与容错逻辑）。  
  - 负责构造最终的多模态提示（`generate_direct_output`），把 memory.get_actions() 与工具元数据注入 prompt。  
- 继承与依赖：无继承；依赖 `engine.factory.create_llm_engine`、`models_embodied.formatters`（`NextStep`, `QueryAnalysis`）、prompt 模块（`vln`、`query_analysis`）与 utils（`get_image_info`, `normalize_image_paths`）。  

### `memory.py`
- 核心类/函数：`Memory`。  
- 职责：  
  - 提供一个轻量的会话记忆抽象：保存 query、文件列表（带类型说明）、以及按步骤记录的 actions（由 Executor/Planner 写入）。  
  - 包含文件类型映射与默认描述生成逻辑（`_get_default_description`）。  
  - API：`set_query`, `add_file`, `add_action`, `get_query`, `get_files`, `get_actions`。  
- 继承与依赖：无继承；依赖标准库 `os`。  

### `long_memory.py`
- 核心类/函数：实现长期记忆逻辑（文件开头已读但内容可能更多）。  
- 职责：长期持久化与检索接口（通常与磁盘/检索器/向量库衔接），用于跨会话上下文维护。  
- 继承与依赖：通常会扩展或与 `Memory` 协同（本仓库中做法是提供一个独立模块用于长期存储策略）。（注：当前实现细节可在文件中查看具体 API）

### `short_memory.py`
- 核心类/函数：实现短期记忆（session-local）策略。  
- 职责：保存本次会话的最近行动/结果，快速读写以供 Planner 在同一解题流程内复用。  
- 继承与依赖：与 `Memory` 概念一致，侧重于非持久化的短期数据结构。

### `executor.py`
- 核心类/函数：`Executor`（方法包括 `set_query_cache_dir`, `generate_tool_command`, `extract_explanation_and_command`, `execute_tool_command`）。  
- 职责：  
  - 接收 Planner 生成的工具调用 intent，调用 LLM 生成具体可执行命令（`generate_tool_command` -> `ToolCommand`）。  
  - 解析 LLM 产出的 `ToolCommand`（尝试 JSON -> Pydantic，失败时使用正则与回退策略），并提取最终的 Python 代码片段（`execution = tool.execute(...)` 风格）。  
  - 安全执行这些代码片段：使用 `signal.alarm` 超时保护、在局部 context 中运行 `exec`，并把 `tool` 注入本地上下文（`execute_with_timeout`）。  
  - 管理工具实例缓存（`tool_instances_cache`）；如缓存中没有则尝试按映射动态导入工具模块并实例化（支持短名/长名映射回退）。  
  - 返回执行结果的列表或错误信息，且在执行前为工具设置 `set_custom_output_dir` 以组织输出。  
- 继承与依赖：无继承；依赖 `create_llm_engine`、`formatters.ToolCommand`、动态 import（importlib），并使用 signal 进行超时控制。  

### `formatters.py`
- 核心类/函数：Pydantic 模型 `QueryAnalysis`, `NextStep`, `MemoryVerification`, `ToolCommand`。  
- 职责：定义结构化的数据契约（LLM response schemas），便于 Planner/Executor 将 LLM 输出反序列化为强类型对象并安全解析。  
  - `QueryAnalysis` 带 `__str__` 方法以易读形式输出分析结果。  
  - `NextStep` 定义 planner 希望从 LLM 获取的字段（justification, context, sub_goal, tool_name）。  
  - `ToolCommand` 定义 Executor 期望的工具命令输出字段（analysis, explanation, command）。  
- 继承与依赖：继承自 `pydantic.BaseModel`，无内部继承关系；被 Planner 与 Executor 导入使用。

### `__init__.py`
- 内容：当前为空（没有导出符号或易用快捷接口）。  
- 作用：占位模块，当前没有向包外暴露额外快捷 API；外部需要从具体模块导入所需类（例如 `from agentflow.agentflow.models_embodied.planner import Planner`）。

### `prompts/` 子目录
（提示模板与 task-specific prompt builders）

- `prompts/vln.py`  
  - 核心函数：`vln_prompt()`（返回基于 ToM 的 VLN 任务模板）。  
  - 职责：封装导航任务的 action-space、状态/策略逻辑示例与多段示例输出，供 Planner 在多模态导航任务上构造强约束 prompt（输出格式严格）。  
  - 依赖：`prompts.tom.build_tom_specified_task_prompt`。

- `prompts/tom.py`  
  - 核心函数：`build_tom_specified_task_prompt(specified_task, specified_examples)`，并定义 `TOM_CORE_PROMPT`（Theory of Mind 框架的核心提示）。  
  - 职责：统一 ToM 基础 prompt 的内容，以便在不同任务（VLN、QueryAnalysis 等）中复用 ToM 风格指导。  

- `prompts/query_analysis.py`  
  - 核心函数：`QuerynalysisPrompt(available_tools, toolbox_metadata, question, image_info=None)`。  
  - 职责：构造 Query Analysis 的提示模板，包含可用工具、工具元数据、图像信息（可选）与明确的输出项（summary, required skills, relevant tools, additional considerations）。  
  - 依赖：引用 `TOM_CORE_PROMPT` 保证分析借助 ToM 思路完成。

## 3. 关键接口汇总（快速参考）

- Planner  
  - analyze_query(question: str, image: str) -> str (QueryAnalysis 或文本)  
  - generate_direct_output(question, image, memory) -> str  
  - extract_context_subgoal_and_tool(response) -> (context, sub_goal, tool_name)

- Executor  
  - set_query_cache_dir(dir)  
  - generate_tool_command(question, image, context, sub_goal, tool_name, tool_metadata, step_count=0, json_data=None) -> ToolCommand  
  - extract_explanation_and_command(response) -> (analysis, explanation, command)  
  - execute_tool_command(tool_name, command) -> list | error

- Memory  
  - set_query(query), add_file(file_name, description=None), add_action(step_count, tool_name, sub_goal, command, result), get_actions(), get_files()

## 4. 注意事项与改进建议（面向具身场景）

- 安全执行：当前 Executor 使用 exec() 执行生成代码片段，虽有超时保护但仍存在安全/沙箱风险。建议采用更受控的执行环境（容器化、沙箱解释器或受限 API 调用层）。  
- Schema 强化：增加 Pydantic 校验与更严格的失败回退策略（例如当 LLM 未返回合规 ToolCommand 时，触发回退 prompt 或人工规则）。  
- Memory 后端：短/长期 memory 提供不同后端（内存/文件/向量数据库）插拔，并在接口层统一抽象检索方法（例如 retrieve(context_embedding, k)）。  
- Prompt 层可参数化：把 ToM、VLN 的严格输出格式封装为可验证的 JSON schema 并在 LLM 调用时使用以减少解析歧义。

---

（基于对 `agentflow/agentflow/models_embodied` 目录中源码的静态分析。若需要我可以把每个文件的逐行重要片段（例如类定义与关键方法）以代码引用形式追加到文档中。） 


