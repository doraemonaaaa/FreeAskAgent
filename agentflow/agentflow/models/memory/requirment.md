# 技术方案设计

## 架构设计

A-MEM 将作为 AgentFlow 记忆模块的增强版本进行集成。现有 AgentFlow 的 Memory 类主要用于存储简单的查询、文件和动作信息，而 A-MEM 将提供更丰富的记忆管理能力。

### 集成架构

```
AgentFlow Memory System
├── Base Memory (现有)
│   ├── Query Storage
│   ├── File Management
│   └── Action Tracking
│
└── A-MEM Enhancement (新增)
    ├── MemoryNote (记忆单元)
    │   ├── Content Analysis (LLM驱动)
    │   ├── Metadata Generation (关键词/标签/上下文)
    │   └── Link Management (记忆连接)
    ├── HybridRetriever (混合检索)
    │   ├── BM25 Search (关键词匹配)
    │   └── Semantic Search (向量相似度)
    ├── AgenticMemorySystem (核心控制器)
    │   ├── Memory Evolution (记忆演化)
    │   ├── Persistence (持久化存储)
    │   └── API Interface (统一接口)
    └── LLM Controllers (多后端支持)
        ├── OpenAI Controller
        ├── LiteLLM Controller
        ├── Ollama Controller
        └── SGLang Controller
```

### 兼容性设计

- **继承现有接口**: 新增的 A-MEM 类将继承或扩展现有的 Memory 类
- **渐进式迁移**: 支持在现有系统中逐步启用 A-MEM 功能
- **配置驱动**: 通过配置参数控制是否启用 A-MEM 增强功能

## 技术选型

### Embedding 模型
- **选择**: `sentence-transformers/all-MiniLM-L6-v2`
- **原因**: 轻量级、高效、支持中文，在 A-MEM 原论文中验证有效
- **备选**: 支持自定义 embedding 模型配置

### 存储介质
- **主要存储**: JSON + Pickle 混合存储
  - JSON: 结构化记忆数据和元数据
  - Pickle: 检索器状态（BM25模型、embeddings）
- **检索缓存**: NumPy 文件存储 embeddings 向量
- **配置**: 支持自定义存储路径和格式

### 算法逻辑

#### 混合检索算法
```
检索流程:
1. 查询预处理 -> 分词 + 向量化
2. 并行检索:
   ├── BM25 检索 -> 关键词匹配得分
   └── 语义检索 -> 余弦相似度得分
3. 得分融合: hybrid_score = α * bm25_score + (1-α) * semantic_score
4. 结果排序: 按 hybrid_score 降序返回 Top-K
```

#### 记忆演化算法
```
演化触发条件: 记忆数量达到阈值 (evo_threshold)
演化流程:
1. 识别新记忆的 K 个最近邻
2. LLM 分析决定演化动作:
   ├── strengthen: 强化记忆连接
   ├── update_neighbor: 更新邻域记忆
3. 执行演化动作并更新记忆状态
```

## 接口设计

### 核心类定义

```python
class AgenticMemorySystem(BaseMemory):
    """A-MEM 核心记忆系统"""

    def __init__(self,
                 model_name: str = 'all-MiniLM-L6-v2',
                 llm_backend: str = "litellm",
                 llm_model: str = "gpt-4o-mini",
                 evo_threshold: int = 100,
                 storage_dir: str = "./memory_store",
                 enable_llm_features: bool = True):

    def add_memory(self, content: str, **kwargs) -> str:
        """添加新记忆，返回记忆ID"""

    def retrieve_memories(self, query: str, k: int = 5) -> List[MemoryNote]:
        """检索相关记忆"""

    def get_memory(self, memory_id: str) -> Optional[MemoryNote]:
        """根据ID获取记忆"""

    def update_memory(self, memory_id: str, **updates) -> bool:
        """更新记忆内容"""

    def delete_memory(self, memory_id: str) -> bool:
        """删除记忆"""

    def consolidate_memories(self) -> None:
        """手动触发记忆巩固"""

    def get_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""

    def save_state(self) -> None:
        """保存记忆状态到磁盘"""

    def load_state(self) -> bool:
        """从磁盘加载记忆状态"""

class MemoryNote:
    """记忆单元结构"""

    def __init__(self,
                 content: str,
                 id: Optional[str] = None,
                 keywords: Optional[List[str]] = None,
                 context: Optional[str] = None,
                 tags: Optional[List[str]] = None,
                 importance_score: Optional[float] = None,
                 links: Optional[List[int]] = None,
                 timestamp: Optional[str] = None):

class HybridRetriever:
    """混合检索器"""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', alpha: float = 0.5):

    def add_documents(self, documents: List[str]) -> None:
        """添加文档到检索索引"""

    def retrieve(self, query: str, k: int = 5) -> List[int]:
        """执行混合检索，返回文档索引列表"""
```

### 配置管理设计

```python
class MemoryConfig:
    """记忆系统配置管理类"""

    def __init__(self, config_file: str = None):
        self.config_file = config_file or "/root/autodl-tmp/FreeAskAgent/agentflow/agentflow/models/memory_config.env"

    def load_config(self) -> Dict[str, Any]:
        """从配置文件加载配置"""

    def save_config(self, config: Dict[str, Any]) -> None:
        """保存配置到文件"""

    def get_llm_config(self) -> Dict[str, str]:
        """获取LLM相关配置（API密钥、模型等）"""

    def get_storage_config(self) -> Dict[str, str]:
        """获取存储相关配置（路径、格式等）"""

    def get_retrieval_config(self) -> Dict[str, Any]:
        """获取检索相关配置（模型、参数等）"""
```

### 命令行测试接口设计

```python
class MemoryCLITester:
    """命令行记忆测试接口"""

    def __init__(self, config_path: str = "/root/autodl-tmp/FreeAskAgent/agentflow/agentflow/models/"):
        self.config_path = config_path
        self.memory_system = None

    def initialize_system(self) -> bool:
        """初始化记忆系统，从config_path加载配置"""

    def run_interactive_session(self) -> None:
        """运行交互式命令行会话"""

    def add_memory_interactive(self, content: str) -> str:
        """交互式添加记忆"""

    def query_memory_interactive(self, query: str) -> List[Dict[str, Any]]:
        """交互式查询记忆"""

    def show_stats_interactive(self) -> Dict[str, Any]:
        """显示系统统计信息"""

    def save_and_exit(self) -> None:
        """保存状态并退出"""

# 命令行测试脚本示例 (amem_test_cli.py)
def main():
    """主函数，支持以下使用模式：

    # 交互模式
    python amem_test_cli.py

    # 直接添加记忆
    python amem_test_cli.py add "时代广场内有盒马和永辉两家超市"

    # 直接查询记忆
    python amem_test_cli.py query "时代广场有什么超市"
    """
    pass
```

## 测试策略

### 单元测试
- **MemoryNote**: 测试记忆创建、元数据生成、序列化
- **HybridRetriever**: 测试 BM25 和语义检索的准确性
- **AgenticMemorySystem**: 测试记忆 CRUD 操作

### 集成测试
- **检索准确性**: 使用标准问答数据集评估检索质量
- **演化效果**: 测试记忆演化对检索性能的影响
- **持久化完整性**: 测试保存/加载的完整性和一致性

### 性能测试
- **检索延迟**: 测试不同规模记忆库的检索响应时间
- **内存占用**: 监控系统在大量记忆下的内存使用情况
- **并发性能**: 测试多线程环境下的系统稳定性

### 与 AgentFlow 集成测试
- **兼容性验证**: 确保 A-MEM 与现有 AgentFlow 组件正常协作
- **功能增强验证**: 验证 A-MEM 对 AgentFlow 推理性能的提升
- **端到端测试**: 完整的多轮对话场景测试