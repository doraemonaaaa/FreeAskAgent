# 需求文档

## 介绍
A-MEM (Agentic Memory for LLM Agents) 是 NeurIPS 2025 论文中提出的代理记忆系统，旨在为 LLM 代理提供长期记忆能力，支持智能的记忆存储、检索和演化。本次迁移的目标是在不破坏 AgentFlow 现有架构的前提下，将 A-MEM 的核心记忆机制集成到 AgentFlow 项目中，为其多轮对话和复杂推理任务提供更强大的记忆支持。

AgentFlow 目前使用简单的 Memory 类来存储查询、文件和动作信息，而 A-MEM 能够提供：
- LLM 驱动的智能记忆分析和元数据生成
- 混合检索系统（BM25 + 语义搜索）
- 记忆演化机制，支持记忆间的连接和强化
- 持久化存储和管理

## 需求

### 需求 1 - 基于情境的记忆检索

**用户故事：** As a AgentFlow开发者, I want Agent具备基于情境的记忆检索能力, So that 在多轮对话中能够根据当前查询语境智能检索最相关的历史记忆片段.

#### 验收标准

1. While 系统初始化, when 加载历史对话, the 记忆模块 shall 根据当前Query检索最相关的Top-K记忆片段.
2. While 对话进行中, when 产生新的交互, the 记忆模块 shall 更新并维护记忆库的结构.
3. While 用户查询时, when 提供查询内容, the 记忆模块 shall 使用混合检索（BM25+语义）返回相关度排序的记忆结果.
4. While 记忆存储时, when 添加新记忆, the 记忆模块 shall 自动分析记忆内容并生成关键词、标签和上下文信息.

### 需求 2 - LLM 驱动的记忆分析

**用户故事：** As a AgentFlow开发者, I want Agent具备LLM驱动的记忆分析能力, So that 能够自动理解和分类记忆内容，提高检索准确性.

#### 验收标准

1. While 添加记忆时, when 提供记忆内容, the 记忆模块 shall 使用LLM自动提取关键词、上下文和标签信息.
2. While 记忆分析时, when 处理复杂内容, the 记忆模块 shall 支持多种LLM后端（OpenAI、LiteLLM等）.
3. While 元数据生成时, when 分析记忆, the 记忆模块 shall 确保生成的元数据格式统一且可解析.

### 需求 3 - 记忆演化机制

**用户故事：** As a AgentFlow开发者, I want Agent具备记忆演化能力, So that 能够动态更新记忆间的关系和重要性，实现记忆的智能化管理.

#### 验收标准

1. While 记忆积累时, when 达到演化阈值, the 记忆模块 shall 自动触发记忆巩固过程.
2. While 相似记忆出现时, when 检测到相关记忆, the 记忆模块 shall 强化记忆间的连接关系.
3. While 记忆更新时, when 执行演化动作, the 记忆模块 shall 更新邻域记忆的上下文和标签信息.

### 需求 4 - 混合检索系统

**用户故事：** As a AgentFlow开发者, I want Agent具备混合检索能力, So that 能够结合关键词匹配和语义相似度提供准确的记忆检索.

#### 验收标准

1. While 执行检索时, when 提供查询, the 记忆模块 shall 同时使用BM25和语义搜索并加权组合结果.
2. While 检索配置时, when 设置参数, the 记忆模块 shall 支持调整BM25和语义搜索的权重比例.
3. While 检索优化时, when 处理大规模记忆库, the 记忆模块 shall 保持检索效率和准确性平衡.

### 需求 5 - 记忆持久化存储

**用户故事：** As a AgentFlow开发者, I want Agent具备记忆持久化能力, So that 能够保存和恢复记忆状态，支持跨会话的连续性.

#### 验收标准

1. While 系统重启时, when 加载记忆, the 记忆模块 shall 从持久化存储中恢复完整的记忆状态.
2. While 记忆更新时, when 修改记忆内容, the 记忆模块 shall 自动同步到持久化存储.
3. While 存储管理时, when 管理存储文件, the 记忆模块 shall 支持配置存储路径和格式.

### 需求 6 - 配置管理

**用户故事：** As a AgentFlow开发者, I want Agent具备灵活的配置管理能力, So that 能够方便地配置API密钥、模型参数和存储路径等设置.

#### 验收标准

1. While 系统初始化时, when 读取配置, the 记忆模块 shall 从 `/root/autodl-tmp/FreeAskAgent/agentflow/agentflow/models/` 目录加载配置文件.
2. While 配置API密钥时, when 设置LLM后端, the 记忆模块 shall 支持环境变量和配置文件两种方式.
3. While 参数调整时, when 修改模型设置, the 记忆模块 shall 支持运行时动态配置embedding模型、检索参数等.
4. While 存储路径配置时, when 指定目录, the 记忆模块 shall 支持自定义记忆存储位置.

### 需求 7 - 命令行测试接口

**用户故事：** As a AgentFlow开发者, I want Agent具备命令行测试接口, So that 能够在 `/root/autodl-tmp/FreeAskAgent/agentflow/agentflow/models` 目录下直接测试记忆功能.

#### 验收标准

1. While 命令行启动时, when 运行测试脚本, the 记忆模块 shall 提供简单的命令行界面进行记忆添加和查询.
2. While 添加记忆时, when 输入"时代广场内有盒马和永辉两家超市", the 记忆模块 shall 成功存储并分析该记忆内容.
3. While 查询记忆时, when 输入"时代广场有什么超市", the 记忆模块 shall 根据历史记忆返回相关结果.
4. While 跨会话测试时, when 重启命令行, the 记忆模块 shall 保持之前的记忆状态并支持连续查询.
5. While 交互测试时, when 进行多轮对话, the 记忆模块 shall 展示记忆的积累和演化效果.