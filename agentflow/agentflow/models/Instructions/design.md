# A-MEM Memory System 集成需求文档

## 介绍

本阶段的目标是将新迁移的A-MEM (Agentic Memory for LLM Agents) 记忆模块集成到FreeAskAgent的核心组件中，使Agent具备长期记忆、混合检索和记忆演化能力。

## 系统架构概述

### 现有Agent架构
```
FreeAskAgent Core Architecture
├── Agent Controller (主控制器)
│   ├── Planner (规划器)
│   │   ├── LLM Engine (qwen2.5-72b-instruct)
│   │   ├── Tool Metadata (工具信息)
│   │   └── Memory Interface (现有Memory类)
│   │       ├── Query Storage (set_query/get_query)
│   │       ├── File Management (add_file/get_files)
│   │       └── Action Tracking (add_action/get_actions)
│   ├── Verifier (验证器)
│   │   ├── LLM Engine (qwen2.5-72b-instruct)
│   │   ├── Tool Metadata (工具信息)
│   │   └── Memory Interface (现有Memory类)
│   └── Executor (执行器)
│       ├── Tool Instances Cache
│       └── Command Execution Engine
```

### 集成后目标架构
```
Enhanced FreeAskAgent with A-MEM
├── Agent Controller (增强的主控制器)
│   ├── Planner (智能规划器)
│   │   ├── LLM Engine (qwen2.5-72b-instruct)
│   │   ├── Tool Metadata
│   │   ├── Memory Interface (兼容现有)
│   │   ├── HybridRetriever (新增) - 历史记忆检索
│   │   └── Historical Memory Store - 规划经验库
│   ├── Verifier (智能验证器)
│   │   ├── LLM Engine (qwen2.5-72b-instruct)
│   │   ├── Tool Metadata
│   │   ├── Memory Interface (兼容现有)
│   │   ├── HybridRetriever (新增) - 验证记忆辅助
│   │   └── Verification Cases Store - 验证案例库
│   ├── Executor (执行器)
│   └── AgenticMemorySystem (集成层)
│       ├── Basic Memory (原有功能)
│       ├── A-MEM Layer (增强功能)
│       │   ├── HybridRetriever (BM25+语义)
│       │   ├── ContentAnalyzer (LLM分析)
│       │   ├── LLM Controllers (多后端支持)
│       │   └── Memory Persistence (状态保存)
│       └── Configuration Manager (参数配置)
```

### A-MEM模块核心组件

- **HybridRetriever**: 混合检索引擎
  - BM25算法：基于关键词匹配的传统检索
  - 语义搜索：基于向量相似度的深度检索
  - API嵌入：支持GPT-5等现代嵌入模型

- **ContentAnalyzer**: 内容分析器
  - LLM驱动的内容理解和关键词提取
  - 上下文分析和标签生成
  - 记忆重要性评估

- **LLM Controllers**: 多后端LLM支持
  - OpenAI Controller (GPT-5)
  - LiteLLM Controller (通用接口)
  - Ollama Controller (本地模型)
  - SGLang Controller (高性能推理)

- **Memory Note**: 结构化记忆单元
  - 标准化记忆格式
  - 元数据管理和关联
  - 记忆演化支持

## 需求规格

### 需求 1 - Planner记忆集成

**用户故事：** As a Agent System, I want Planner能够访问长期记忆, So that 生成的计划能参考过去的成功/失败经验，避免重复错误，提高规划质量.

**功能描述：**
- Planner在初始化时实例化HybridRetriever用于历史记忆检索
- 在生成规划时，基于当前查询检索相关历史记忆
- 将检索到的记忆内容注入到规划prompt中，指导LLM生成更优计划
- 支持记忆的动态添加和管理

**技术实现：**
```python
class Planner:
    def __init__(self, ..., use_amem: bool = True, retriever_config: dict = None):
        # 初始化HybridRetriever
        if use_amem:
            self.retriever = HybridRetriever(
                use_api_embedding=config.get('use_api_embedding', True),
                alpha=config.get('alpha', 0.5)  # BM25 vs 语义搜索权重
            )

    def generate_next_step(self, question: str, ...):
        # 检索相关历史记忆
        relevant_memories = self._retrieve_relevant_memories(question, k=3)

        # 注入到规划prompt
        prompt = f"""
        Context: {question}
        Historical Experiences:
        {self._format_memories_for_prompt(relevant_memories)}

        Based on the above context and historical experiences,
        determine the optimal next step...
        """
```

#### 验收标准

1. **初始化验收**: Agent初始化时，Planner正确实例化A-MEM HybridRetriever并加载历史记忆数据
2. **检索验收**: 规划时能够调用retriever.retrieve()获取相关上下文并注入到prompt中
3. **优化验收**: 基于历史相似任务的记忆来优化sub-goal选择和工具使用
4. **学习验收**: 遇到类似问题时，从记忆中学习避免重复错误，提高成功率

### 需求 2 - Verifier记忆集成

**用户故事：** As a Agent System, I want Verifier能够利用记忆辅助验证, So that 验证过程更准确且考虑历史经验，避免遗漏和错误.

**功能描述：**
- Verifier在验证过程中查询相关历史验证案例
- 基于历史经验辅助判断结果的完整性和准确性
- 当发现当前结果与历史模式不一致时进行标记
- 验证完成后将结果存储到记忆中用于未来参考

**技术实现：**
```python
class Verifier:
    def __init__(self, ..., use_amem: bool = True, retriever_config: dict = None):
        # 初始化验证记忆检索器
        if use_amem:
            self.retriever = HybridRetriever(...)
            self.verification_memories = []

    def verificate_context(self, question: str, memory: Memory, ...):
        # 构建验证上下文
        context = f"Query: {question}, Actions: {memory.get_actions()}"

        # 检索历史验证案例
        similar_cases = self._get_similar_historical_verifications(context, k=2)

        # 注入到验证prompt
        prompt = f"""
        Current Verification Context:
        {context}

        Historical Verification Cases:
        {self._format_verification_memories_for_prompt(similar_cases)}

        Based on the above context and historical cases,
        evaluate if additional verification is needed...
        """
```

#### 验收标准

1. **查询验收**: 结果验证时，Verifier查询相关历史记忆来辅助完整性判断
2. **不一致检测**: 当当前结果与历史矛盾时，标记需要进一步验证
3. **记忆存储**: 验证完成后，将验证结果和经验存储到记忆中

### 需求 3 - 记忆演化机制

**用户故事：** As a Agent System, I want Agent具备记忆演化能力, So that 能够动态更新和强化记忆关系，提高记忆质量和检索效率.

**功能描述：**
- 随着记忆积累，触发记忆演化过程优化记忆结构
- 发现相似任务时强化相关记忆连接
- 检测记忆冲突时通过LLM分析解决矛盾
- 动态调整记忆重要性和关联强度

**技术实现：**
```python
class AgenticMemorySystem:
    def evolve_memories(self):
        """记忆演化过程"""
        if len(self.long_term_memories) >= self.evolution_threshold:
            # 触发演化
            self._consolidate_similar_memories()
            self._resolve_conflicts_via_llm()
            self._update_memory_importance()

    def _consolidate_similar_memories(self):
        """合并相似记忆"""
        # 使用向量相似度检测相似记忆
        # 合并重复内容，强化重要模式

    def _resolve_conflicts_via_llm(self):
        """通过LLM解决记忆冲突"""
        # 检测矛盾信息
        # 调用LLM分析和解决冲突
```

#### 验收标准

1. **演化触发**: 积累足够记忆时自动触发记忆演化过程
2. **连接强化**: 发现相似任务时强化相关记忆连接
3. **冲突解决**: 发现矛盾信息时通过LLM分析解决冲突

### 需求 4 - 向后兼容性保证

**用户故事：** As a Developer, I want 集成后的系统保持向后兼容, So that 现有功能不受影响，迁移成本最小化.

**功能描述：**
- 保持所有现有Memory类接口完全不变
- 通过配置开关控制A-MEM功能启用/禁用
- 当A-MEM依赖不可用时自动降级到原有功能
- 支持渐进式升级，无需修改现有代码

**技术实现：**
```python
# 完全兼容现有接口
class AgenticMemorySystem(Memory):
    def __init__(self, use_amem: bool = True, ...):
        super().__init__()  # 初始化基础Memory
        self.use_amem = use_amem
        if use_amem:
            try:
                self._init_amem_components()
            except ImportError:
                self.use_amem = False  # 优雅降级

    # 所有原有方法保持不变
    def get_actions(self):
        return self.basic_memory.get_actions()  # 直接委托

    def add_action(self, ...):
        self.basic_memory.add_action(...)  # 基础存储
        if self.use_amem:
            self._add_to_long_term_memory(...)  # 增强功能
```

#### 验收标准

1. **接口兼容**: 现有Memory类方法正常工作，无任何修改需求
2. **降级处理**: A-MEM依赖缺失或API失败时自动降级到原有功能
3. **配置切换**: 禁用A-MEM时完全使用原有逻辑，无性能损失

### 需求 5 - 配置管理和监控

**用户故事：** As a Developer, I want 能够配置和监控记忆功能, So that 可以调整参数，观察效果，快速诊断问题.

**功能描述：**
- 支持环境变量、配置文件等多种配置方式
- 运行时记录详细的记忆检索统计和性能指标
- 提供调试信息和错误诊断支持
- 实时监控记忆系统健康状态

**技术实现：**
```python
class MemoryConfig:
    def __init__(self):
        self._config = {
            'use_amem': True,
            'retriever_config': {
                'use_api_embedding': True,
                'alpha': 0.5,
                'model': 'gpt-5'
            },
            'storage_config': {
                'max_memories': 1000,
                'enable_persistence': True
            }
        }

class AgenticMemorySystem:
    def get_stats(self) -> Dict[str, Any]:
        return {
            'total_memories': len(self.long_term_memories),
            'retrieval_count': self.stats['retrieval_count'],
            'avg_retrieval_time': self.stats['avg_retrieval_time'],
            'success_rate': self.stats['performance_metrics']['success_rate'],
            'amem_enabled': self.use_amem,
            'memory_utilization': len(self.long_term_memories) / self.max_memories
        }
```

#### 验收标准

1. **配置支持**: 支持embedding模型、检索阈值、存储参数等完整配置
2. **监控记录**: 运行时记录记忆检索统计、性能指标和使用模式
3. **调试支持**: 提供详细的记忆状态信息和错误诊断功能

## 技术约束和规范

### 性能要求
- **检索延迟**: 平均检索时间 < 500ms，最大 < 5秒
- **内存使用**: 1000个记忆占用 < 500MB
- **成功率**: 检索成功率 > 95%
- **并发支持**: 支持至少10个并发检索操作

### 兼容性要求
- **API兼容**: 100%兼容现有Memory类接口
- **数据兼容**: 支持现有数据格式无缝迁移
- **版本兼容**: 支持Python 3.8+
- **依赖兼容**: A-MEM功能为可选依赖，不影响核心功能

### 安全性要求
- **数据隔离**: 用户记忆数据完全隔离
- **API安全**: 支持安全的API密钥管理
- **错误处理**: 优雅的错误处理，不暴露敏感信息
- **资源限制**: 防止内存泄漏和无限增长

### 可扩展性要求
- **模块化设计**: 支持新增检索算法和内容分析器
- **配置驱动**: 所有功能通过配置开关控制
- **插件架构**: 支持自定义记忆处理器和存储后端
- **监控接口**: 提供标准监控和指标接口

## 验收测试标准

### 自动化测试覆盖
- **单元测试**: 核心功能覆盖率 > 90%
- **集成测试**: 端到端流程完整覆盖
- **性能测试**: 压力测试和基准测试
- **兼容性测试**: 回归测试确保无破坏性变更

### 生产就绪检查
- [ ] 所有需求验收标准100%通过
- [ ] 性能指标满足生产要求
- [ ] 监控和日志系统完整
- [ ] 文档和部署指南完备
- [ ] 故障恢复机制验证通过
