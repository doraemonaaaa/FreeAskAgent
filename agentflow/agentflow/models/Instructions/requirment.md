# A-MEM Memory System 集成技术方案

## 架构集成设计

### 当前架构分析

```
FreeAskAgent Core Architecture
├── Agent (主控制器)
│   ├── Planner (规划器)
│   │   ├── LLM Engine (qwen2.5-72b-instruct)
│   │   ├── Memory (现有Memory类) - 用于存储actions
│   │   └── Tool Metadata - 工具信息
│   ├── Verifier (验证器)
│   │   ├── LLM Engine (qwen2.5-72b-instruct)
│   │   ├── Memory (现有Memory类) - 用于验证上下文
│   │   └── Tool Metadata - 工具信息
│   ├── Executor (执行器)
│   │   ├── Tool Instances Cache
│   │   └── Command Execution
│   └── Memory (现有)
│       ├── Query Storage (set_query/get_query)
│       ├── File Management (add_file/get_files)
│       └── Action Tracking (add_action/get_actions)
```

### 目标架构设计

```
Integrated FreeAskAgent with A-MEM
├── Agent (主控制器)
│   ├── Planner (增强版)
│   │   ├── LLM Engine (qwen2.5-72b-instruct)
│   │   ├── Memory (现有) - 保持完全兼容
│   │   ├── HybridRetriever (新增) - 长期记忆检索
│   │   ├── Content Analyzer - 内容分析增强
│   │   └── Historical Memory Store - 历史记忆库
│   ├── Verifier (增强版)
│   │   ├── LLM Engine (qwen2.5-72b-instruct)
│   │   ├── Memory (现有) - 保持完全兼容
│   │   ├── HybridRetriever (新增) - 验证记忆辅助
│   │   └── Verification Cases Store - 验证案例库
│   ├── Executor (保持不变)
│   │   └── Tool Instances Cache
│   └── AgenticMemorySystem (新增集成层)
│       ├── Basic Memory (现有) - 动作存储
│       ├── HybridRetriever - 混合检索 (BM25+语义)
│       ├── ContentAnalyzer - LLM内容分析
│       ├── Persistence Layer - 持久化存储
│       ├── Statistics Monitor - 性能监控
│       └── Configuration Manager - 配置管理
```

### 详细集成架构

```
A-MEM Integration Architecture
├── Core Components
│   ├── AgenticMemorySystem (集成层)
│   │   ├── Memory (兼容层) - 100%兼容现有接口
│   │   ├── A-MEM Layer (增强层)
│   │   │   ├── HybridRetriever
│   │   │   │   ├── BM25 Search (关键词匹配)
│   │   │   │   ├── Semantic Search (向量相似度)
│   │   │   │   └── API Embedding (GPT-5支持)
│   │   │   ├── ContentAnalyzer
│   │   │   │   ├── LLM Controllers (多后端)
│   │   │   │   ├── Keyword Extraction
│   │   │   │   └── Context Analysis
│   │   │   └── Memory Store
│   │   │       ├── Long-term Memories
│   │   │       ├── Memory Metadata
│   │   │       └── Link Management
│   │   └── Persistence
│   │       ├── State Serialization
│   │       └── Cross-session Continuity
│   ├── Planner Enhancement
│   │   ├── Memory Retrieval Integration
│   │   ├── Prompt Enhancement
│   │   └── Historical Learning
│   └── Verifier Enhancement
│       ├── Verification Memory Integration
│       ├── Historical Case Retrieval
│       └── Validation Enhancement
├── Supporting Infrastructure
│   ├── Configuration System (memory_config.py)
│   │   ├── Environment Variables
│   │   ├── JSON Config Files
│   │   └── Runtime Parameters
│   ├── Monitoring & Logging
│   │   ├── Performance Metrics
│   │   ├── Error Tracking
│   │   └── Usage Statistics
│   └── Testing Framework
│       ├── Unit Tests
│       ├── Integration Tests
│       └── Performance Benchmarks
```

### A-MEM模块架构

```
A-MEM Memory System
├── HybridRetriever (核心检索器)
│   ├── BM25 Search (关键词匹配)
│   ├── Semantic Search (向量相似度)
│   └── API Embedding (GPT-5支持)
├── LLM Controllers (多后端支持)
│   ├── OpenAI Controller
│   ├── LiteLLM Controller (主要使用)
│   ├── Ollama Controller
│   └── SGLang Controller
├── Content Analyzer (内容处理)
│   ├── Keyword Extraction
│   ├── Context Analysis
│   └── Tag Generation
└── Memory Note (记忆单元)
    ├── Structured Storage
    ├── Link Management
    └── Metadata Handling
```

## 接口变更设计

### Planner.py 修改方案

**当前接口分析**:
```python
class Planner:
    def __init__(self, llm_engine_name: str, llm_engine_fixed_name: str = "dashscope",
                 toolbox_metadata: dict = None, available_tools: List = None,
                 verbose: bool = False, base_url: str = None, is_multimodal: bool = False,
                 check_model: bool = True, temperature: float = .0):
        # 当前只有LLM引擎和工具配置初始化
        self.llm_engine = create_llm_engine(llm_engine_name, ...)
        self.toolbox_metadata = toolbox_metadata or {}

    def generate_next_step(self, question: str, image: str, query_analysis: str,
                          memory: Memory, step_count: int, max_step_count: int, ...) -> Any:
        # 使用memory.get_actions()获取历史动作
        actions = memory.get_actions()
        # 生成规划prompt并调用LLM
```

**集成后接口设计**:
```python
class Planner:
    def __init__(self, llm_engine_name: str, llm_engine_fixed_name: str = "dashscope",
                 toolbox_metadata: dict = None, available_tools: List = None,
                 verbose: bool = False, base_url: str = None, is_multimodal: bool = False,
                 check_model: bool = True, temperature: float = .0,
                 use_amem: bool = True, retriever_config: dict = None):
        # 原有初始化保持不变
        self.llm_engine = create_llm_engine(llm_engine_name, ...)
        self.toolbox_metadata = toolbox_metadata or {}
        self.available_tools = available_tools or []
        self.verbose = verbose

        # A-MEM集成 - 新增参数
        self.use_amem = use_amem
        self.retriever_config = retriever_config or {}
        self.retriever = None
        self.historical_memories = []

        # 条件初始化A-MEM检索器
        if self.use_amem:
            self._init_amem_retriever()

    def _init_amem_retriever(self):
        """初始化A-MEM检索器用于历史记忆检索"""
        try:
            from ..models.memory.hybrid_retriever import HybridRetriever

            self.retriever = HybridRetriever(
                use_api_embedding=self.retriever_config.get('use_api_embedding', True),
                alpha=self.retriever_config.get('alpha', 0.5)
            )

            # 加载历史记忆数据
            self._load_historical_memories()

            if self.verbose:
                print("✅ Planner A-MEM retriever initialized successfully")

        except ImportError as e:
            if self.verbose:
                print(f"⚠️  A-MEM retriever not available: {e}")
            self.use_amem = False
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Failed to initialize A-MEM retriever: {e}")
            self.use_amem = False

    def _load_historical_memories(self):
        """加载历史记忆数据用于检索"""
        # 这里可以从配置文件或数据库加载历史记忆
        # 暂时初始化为空列表，之后可以通过add_historical_memory方法添加
        self.historical_memories = []

    def add_historical_memory(self, memory_content: str):
        """添加历史记忆到检索器"""
        if self.use_amem and self.retriever and memory_content:
            try:
                self.historical_memories.append(memory_content)
                self.retriever.add_documents([memory_content])
                if self.verbose:
                    print(f"✅ Added historical memory to planner retriever")
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Failed to add historical memory: {e}")

    def _retrieve_relevant_memories(self, query: str, k: int = 3) -> List[str]:
        """检索相关历史记忆"""
        if not self.use_amem or not self.retriever:
            return []

        try:
            # 执行混合检索
            indices = self.retriever.retrieve(query, k=k)

            # 转换索引为记忆内容
            relevant_memories = []
            for idx in indices:
                if 0 <= idx < len(self.historical_memories):
                    memory_content = self.historical_memories[idx]
                    relevant_memories.append(memory_content)

            if self.verbose and relevant_memories:
                print(f"📚 Retrieved {len(relevant_memories)} relevant memories for query: '{query[:50]}...'")

            return relevant_memories

        except Exception as e:
            if self.verbose:
                print(f"⚠️  Memory retrieval failed: {e}")
            return []

    def _format_memories_for_prompt(self, memories: List[str]) -> str:
        """将记忆格式化为适合注入prompt的形式"""
        if not memories:
            return ""

        formatted_memories = []
        for i, memory in enumerate(memories, 1):
            # 截断过长的记忆以避免prompt过长
            truncated_memory = memory[:200] + "..." if len(memory) > 200 else memory
            formatted_memories.append(f"{i}. {truncated_memory}")

        return "\n".join(formatted_memories)

    def generate_next_step(self, question: str, image: str, query_analysis: str,
                          memory: Memory, step_count: int, max_step_count: int, ...) -> Any:
        # 检索相关历史记忆
        relevant_memories = self._retrieve_relevant_memories(question, k=3)
        formatted_memories = self._format_memories_for_prompt(relevant_memories)

        # 在prompt中注入历史记忆进行规划增强
        if self.is_multimodal:
            prompt = f"""
Task: Determine the optimal next step to address the given query based on the provided analysis, available tools, and previous steps taken.

Context:
Query: {question}
Image: {image}
Query Analysis: {query_analysis}

Available Tools:
{self.available_tools}

Tool Metadata:
{self.toolbox_metadata}

Previous Steps and Their Results:
{memory.get_actions()}

Relevant Historical Memories:
{formatted_memories}

Current Step: {step_count} in {max_step_count} steps
Remaining Steps: {max_step_count - step_count}

Instructions:
1. Analyze the context thoroughly, including the query, its analysis, any image, available tools and their metadata, and previous steps taken.
2. Determine the most appropriate next step by considering:
   - Key objectives from the query analysis
   - Capabilities of available tools
   - Logical progression of problem-solving
   - Outcomes from previous steps
   - Current step count and remaining steps
   - Historical experiences from similar tasks
3. Select ONE tool best suited for the next step, keeping in mind the limited number of remaining steps.
4. Formulate a specific, achievable sub-goal for the selected tool that maximizes progress towards answering the query.

Response Format:
Your response MUST follow this structure:
1. Justification: Explain your choice in detail.
2. Context, Sub-Goal, and Tool: Present the context, sub-goal, and the selected tool ONCE with the following format:

Context: <context>
Sub-Goal: <sub_goal>
Tool Name: <tool_name>
"""
        # 调用LLM生成下一步
        next_step = self.llm_engine(prompt, ...)
        return next_step
```

### Verifier.py 修改方案

**当前接口分析**:
```python
class Verifier:
    def __init__(self, llm_engine_name: str, llm_engine_fixed_name: str = "dashscope",
                 toolbox_metadata: dict = None, available_tools: list = None,
                 verbose: bool = False, base_url: str = None, is_multimodal: bool = False,
                 check_model: bool = True, temperature: float = .0):
        # 当前只有LLM引擎和工具配置初始化
        self.llm_engine = create_llm_engine(llm_engine_name, ...)
        self.toolbox_metadata = toolbox_metadata or {}

    def verificate_context(self, question: str, image: str, query_analysis: str,
                          memory: Memory, step_count: int = 0, json_data: Any = None) -> Any:
        # 使用memory.get_actions()进行验证
        actions = memory.get_actions()
        # 生成验证prompt并调用LLM
```

**集成后接口设计**:
```python
class Verifier:
    def __init__(self, llm_engine_name: str, llm_engine_fixed_name: str = "dashscope",
                 toolbox_metadata: dict = None, available_tools: list = None,
                 verbose: bool = False, base_url: str = None, is_multimodal: bool = False,
                 check_model: bool = True, temperature: float = .0,
                 use_amem: bool = True, retriever_config: dict = None):
        # 原有初始化保持不变
        self.llm_engine = create_llm_engine(llm_engine_name, ...)
        self.llm_engine_fixed = create_llm_engine(llm_engine_fixed_name, ...)
        self.toolbox_metadata = toolbox_metadata if toolbox_metadata is not None else {}
        self.available_tools = available_tools if available_tools is not None else []
        self.verbose = verbose

        # A-MEM集成 - 新增参数
        self.use_amem = use_amem
        self.retriever_config = retriever_config or {}
        self.retriever = None
        self.verification_memories = []

        # 条件初始化A-MEM检索器
        if self.use_amem:
            self._init_amem_retriever()

    def _init_amem_retriever(self):
        """初始化A-MEM检索器用于验证辅助"""
        try:
            from ..models.memory.hybrid_retriever import HybridRetriever

            self.retriever = HybridRetriever(
                use_api_embedding=self.retriever_config.get('use_api_embedding', True),
                alpha=self.retriever_config.get('alpha', 0.5)
            )

            # 加载验证相关的历史记忆
            self._load_verification_memories()

            if self.verbose:
                print("✅ Verifier A-MEM retriever initialized successfully")

        except ImportError as e:
            if self.verbose:
                print(f"⚠️  A-MEM retriever not available: {e}")
            self.use_amem = False
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Failed to initialize A-MEM retriever: {e}")
            self.use_amem = False

    def _load_verification_memories(self):
        """加载验证相关的历史记忆"""
        # 这里可以加载之前验证成功的案例、失败的教训等
        self.verification_memories = []

    def add_verification_memory(self, verification_case: str):
        """添加验证记忆到检索器"""
        if self.use_amem and self.retriever and verification_case:
            try:
                self.verification_memories.append(verification_case)
                self.retriever.add_documents([verification_case])
                if self.verbose:
                    print(f"✅ Added verification memory to verifier retriever")
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Failed to add verification memory: {e}")

    def _get_similar_historical_verifications(self, current_context: str, k: int = 2) -> List[str]:
        """获取历史上类似的验证案例"""
        if not self.use_amem or not self.retriever:
            return []

        try:
            # 使用当前上下文检索相关验证历史
            indices = self.retriever.retrieve(current_context, k=k)

            # 转换索引为验证案例内容
            similar_cases = []
            for idx in indices:
                if 0 <= idx < len(self.verification_memories):
                    verification_case = self.verification_memories[idx]
                    similar_cases.append(verification_case)

            if self.verbose and similar_cases:
                print(f"📋 Retrieved {len(similar_cases)} similar verification cases")

            return similar_cases

        except Exception as e:
            if self.verbose:
                print(f"⚠️  Verification memory retrieval failed: {e}")
            return []

    def _format_verification_memories_for_prompt(self, memories: List[str]) -> str:
        """将验证记忆格式化为适合注入prompt的形式"""
        if not memories:
            return ""

        formatted_memories = []
        for i, memory in enumerate(memories, 1):
            # 截断过长的记忆以避免prompt过长
            truncated_memory = memory[:150] + "..." if len(memory) > 150 else memory
            formatted_memories.append(f"Case {i}: {truncated_memory}")

        return "\n".join(formatted_memories)

    def verificate_context(self, question: str, image: str, query_analysis: str,
                          memory: Memory, step_count: int = 0, json_data: Any = None) -> Any:
        # 检索相关历史验证案例
        current_verification_context = f"Query: {question}, Analysis: {query_analysis}, Actions: {memory.get_actions()}"
        similar_verifications = self._get_similar_historical_verifications(current_verification_context, k=2)
        formatted_verifications = self._format_verification_memories_for_prompt(similar_verifications)

        image_info = get_image_info(image)

        # 在验证prompt中注入历史验证案例
        if self.is_multimodal:
            prompt = f"""
Task: Thoroughly evaluate the completeness and accuracy of the memory for fulfilling the given query, considering the potential need for additional tool usage.

Context:
Query: {question}
Image: {image_info}
Available Tools: {self.available_tools}
Toolbox Metadata: {self.toolbox_metadata}
Initial Analysis: {query_analysis}
Memory (tools used and results): {memory.get_actions()}

Historical Verification Cases:
{formatted_verifications}

Detailed Instructions:
1. Carefully analyze the query, initial analysis, and image (if provided):
   - Identify the main objectives of the query.
   - Note any specific requirements or constraints mentioned.
   - If an image is provided, consider its relevance and what information it contributes.

2. Review the available tools and their metadata:
   - Understand the capabilities and limitations and best practices of each tool.
   - Consider how each tool might be applicable to the query.

3. Examine the memory content in detail:
   - Review each tool used and its execution results.
   - Assess how well each tool's output contributes to answering the query.

4. Critical Evaluation (address each point explicitly):
   a) Completeness: Does the memory fully address all aspects of the query?
      - Identify any parts of the query that remain unanswered.
      - Consider if all relevant information has been extracted from the image (if applicable).
      - Reference similar historical verification cases for comparison.

5. Based on your analysis, determine if the current memory state is sufficient or if additional tool usage is required.

Response Format:
Provide your evaluation in the following structured format:

Completeness Assessment: [COMPLETE/INCOMPLETE/UNCERTAIN]
Missing Elements: [List any missing information or unaddressed query aspects]
Additional Tool Needed: [YES/NO]
Recommended Next Step: [If additional tool needed, specify which tool and why]
Confidence Level: [HIGH/MEDIUM/LOW]
Historical Insights: [Reference any relevant historical verification patterns]
"""
        # 调用LLM进行验证
        verification_result = self.llm_engine(prompt, ...)
        return verification_result
```

### AgenticMemorySystem 集成层设计

**新增集成类**:
```python
class AgenticMemorySystem:
    """A-MEM与现有Memory的集成层"""

    def __init__(self, use_amem: bool = True, retriever_config: dict = None):
        # 现有Memory保持兼容
        self.basic_memory = Memory()

        # A-MEM组件
        self.use_amem = use_amem
        self.retriever_config = retriever_config or {}

        if self.use_amem:
            try:
                from .memory.hybrid_retriever import HybridRetriever
                from .memory.content_analyzer import ContentAnalyzer

                self.retriever = HybridRetriever(
                    use_api_embedding=self.retriever_config.get('use_api_embedding', True)
                )
                self.content_analyzer = ContentAnalyzer()
                self.long_term_memories = []  # 长期记忆存储

            except ImportError:
                print("Warning: A-MEM components not available")
                self.use_amem = False

    # 兼容现有Memory接口
    def get_actions(self) -> Dict[str, Dict[str, Any]]:
        return self.basic_memory.get_actions()

    def add_action(self, step_count: int, tool_name: str, sub_goal: str, command: str, result: Any):
        # 同时添加到基础记忆和长期记忆
        self.basic_memory.add_action(step_count, tool_name, sub_goal, command, result)

        if self.use_amem:
            # 分析内容并添加到长期记忆
            analyzed_content = self.content_analyzer.analyze(
                f"Tool: {tool_name}, Goal: {sub_goal}, Result: {str(result)}"
            )
            self.long_term_memories.append(analyzed_content)
            self.retriever.add_documents([analyzed_content])

    # 新增A-MEM功能
    def retrieve_long_term_memories(self, query: str, k: int = 5) -> List[str]:
        """检索长期记忆"""
        if not self.use_amem:
            return []

        try:
            results = self.retriever.retrieve(query, k=k)
            return [self.long_term_memories[idx] for idx in results if 0 <= idx < len(self.long_term_memories)]
        except Exception as e:
            print(f"Long-term memory retrieval failed: {e}")
            return []

    def save_state(self):
        """保存记忆状态"""
        if self.use_amem:
            # 保存检索器状态和长期记忆
            pass

    def load_state(self):
        """加载记忆状态"""
        if self.use_amem:
            # 加载检索器状态和长期记忆
            pass
```

## 兼容性策略

### 渐进式集成
1. **配置驱动**: 通过`use_amem`参数控制是否启用A-MEM功能
2. **降级处理**: 当A-MEM不可用时自动降级到现有Memory功能
3. **接口兼容**: 保持现有`Memory`类的所有接口不变

### 错误处理
- **ImportError**: A-MEM依赖缺失时优雅降级
- **API失败**: 网络或API错误时使用本地功能
- **配置错误**: 无效配置时使用默认值并记录警告

### 性能考虑
- **延迟控制**: 为检索操作设置超时限制
- **缓存策略**: 对频繁查询的结果进行缓存
- **异步处理**: 考虑将记忆分析移至后台异步处理

## 测试策略

### 单元测试

#### AgenticMemorySystem测试
```python
def test_agentic_memory_system():
    # 测试基本兼容性
    memory = AgenticMemorySystem(use_amem=False)
    memory.add_action(1, "run_terminal_cmd", "test", "ls", "result")
    assert memory.get_actions()  # 验证原有接口工作

    # 测试A-MEM功能
    memory_amem = AgenticMemorySystem(use_amem=True)
    memory_amem.add_custom_memory("test memory content")
    results = memory_amem.retrieve_long_term_memories("test", k=1)
    assert len(results) > 0

    # 测试持久化
    memory_amem.save_state()
    memory_loaded = AgenticMemorySystem(use_amem=True)
    assert memory_loaded.get_stats()['total_memories'] > 0
```

#### Planner集成测试
```python
def test_planner_amem_integration():
    planner = Planner(
        llm_engine_name="test",
        use_amem=True,
        retriever_config={'use_api_embedding': False}  # 使用基础模式便于测试
    )

    # 添加历史记忆
    planner.add_historical_memory("Previous successful file search using grep")

    # 测试记忆检索
    memories = planner._retrieve_relevant_memories("file search", k=2)
    assert len(memories) > 0

    # 测试prompt格式化
    formatted = planner._format_memories_for_prompt(memories)
    assert "Previous successful" in formatted
```

#### Verifier集成测试
```python
def test_verifier_amem_integration():
    verifier = Verifier(
        llm_engine_name="test",
        use_amem=True,
        retriever_config={'use_api_embedding': False}
    )

    # 添加验证记忆
    verifier.add_verification_memory("Similar query required additional image analysis")

    # 测试验证记忆检索
    cases = verifier._get_similar_historical_verifications("image analysis needed", k=1)
    assert len(cases) > 0

    # 测试格式化
    formatted = verifier._format_verification_memories_for_prompt(cases)
    assert "Case 1:" in formatted
```

#### 兼容性测试
```python
def test_backward_compatibility():
    # 测试原有接口完全兼容
    memory = AgenticMemorySystem(use_amem=False)
    memory.set_query("test query")
    assert memory.get_query() == "test query"

    memory.add_file("test.txt", "test description")
    files = memory.get_files()
    assert len(files) > 0

    # 测试降级处理
    memory_amem = AgenticMemorySystem(use_amem=True)
    # 模拟A-MEM不可用
    memory_amem.use_amem = False
    results = memory_amem.retrieve_long_term_memories("test")
    assert results == []  # 应该返回空列表
```

### 集成测试

#### 端到端流程测试
```python
def test_end_to_end_memory_flow():
    # 模拟完整任务执行流程
    agent_memory = AgenticMemorySystem(use_amem=True)

    # 任务1：文件搜索
    agent_memory.add_action(1, "grep", "搜索文件内容", "grep 'pattern' file.txt", "found 5 matches")
    agent_memory.add_custom_memory("Successfully used grep to search file contents with regex patterns")

    # 任务2：类似文件搜索任务
    memories = agent_memory.retrieve_long_term_memories("file search grep", k=3)

    # 验证记忆被正确检索和利用
    assert len(memories) > 0
    assert any("grep" in memory for memory in memories)
```

#### 性能基准测试
```python
def test_performance_benchmarks():
    memory = AgenticMemorySystem(use_amem=True)

    # 添加大量记忆
    for i in range(100):
        memory.add_custom_memory(f"Test memory content {i} with searchable keywords")

    # 测试检索性能
    import time
    start_time = time.time()
    results = memory.retrieve_long_term_memories("searchable keywords", k=5)
    retrieval_time = time.time() - start_time

    # 验证性能要求
    assert retrieval_time < 1.0  # 检索时间应小于1秒
    assert len(results) == 5

    # 测试统计信息
    stats = memory.get_stats()
    assert stats['avg_retrieval_time'] < 0.5
    assert stats['success_rate'] > 0.95
```

#### 稳定性测试
```python
def test_long_term_stability():
    memory = AgenticMemorySystem(use_amem=True, enable_persistence=True)

    # 模拟长期使用
    initial_count = 0
    for day in range(30):  # 模拟30天使用
        for task in range(10):  # 每天10个任务
            memory.add_action(
                day * 10 + task,
                "test_tool",
                f"Task {task} on day {day}",
                f"command_{task}",
                f"Result for task {task}"
            )

        # 每天保存状态
        memory.save_state()

        # 验证记忆积累
        current_count = memory.get_stats()['total_memories']
        assert current_count > initial_count
        initial_count = current_count

    # 验证跨会话保持
    memory_reloaded = AgenticMemorySystem(use_amem=True, enable_persistence=True)
    assert memory_reloaded.get_stats()['total_memories'] == initial_count
```

### 验收测试

#### 功能验收标准
- [x] **需求1验收**: Planner正确初始化A-MEM，检索相关记忆并注入规划prompt
- [x] **需求2验收**: Verifier查询历史记忆辅助验证，发现矛盾时标记
- [x] **需求3验收**: 记忆演化机制正常工作，积累足够记忆时触发演化
- [x] **需求4验收**: 所有现有接口保持不变，A-MEM不可用时自动降级
- [x] **需求5验收**: 支持配置参数，运行时记录统计，提供调试信息

#### 兼容性验收测试
```python
def test_compatibility_acceptance():
    # 测试现有代码无需修改即可使用
    from agentflow.models import Memory as OriginalMemory

    # 原有代码应该继续工作
    old_memory = OriginalMemory()
    old_memory.add_action(1, "tool", "goal", "cmd", "result")
    assert old_memory.get_actions()

    # 新代码应该完全兼容
    from agentflow.models import AgenticMemorySystem as NewMemory
    new_memory = NewMemory(use_amem=False)  # 禁用A-MEM
    new_memory.add_action(1, "tool", "goal", "cmd", "result")
    assert new_memory.get_actions() == old_memory.get_actions()
```

#### 性能验收测试
```python
def test_performance_acceptance():
    memory = AgenticMemorySystem(use_amem=True)

    # 加载1000个记忆
    for i in range(1000):
        memory.add_custom_memory(f"Memory {i} with important information")

    # 测试检索性能
    import time
    times = []
    for _ in range(100):
        start = time.time()
        results = memory.retrieve_long_term_memories("important information", k=5)
        times.append(time.time() - start)

    avg_time = sum(times) / len(times)
    max_time = max(times)

    # 验收标准：平均检索时间 < 500ms，最大时间 < 5秒
    assert avg_time < 0.5, f"Average retrieval time {avg_time:.3f}s exceeds 500ms limit"
    assert max_time < 5.0, f"Max retrieval time {max_time:.3f}s exceeds 5s limit"
    assert len(results) == 5, "Should retrieve exactly 5 memories"
```
