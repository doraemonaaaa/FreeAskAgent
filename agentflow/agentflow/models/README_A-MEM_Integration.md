# A-MEM Memory System 集成使用指南

## 🎯 概述

A-MEM (Agentic Memory for LLM Agents) 记忆系统已成功集成到AgentFlow中，为Agent提供长期记忆、混合检索和记忆演化能力。

## 🚀 快速开始

### 1. 运行快速测试

```bash
cd /root/autodl-tmp/FreeAskAgent/agentflow/agentflow/models
python quick_start.py
```

### 2. 测试结果说明

运行后你会看到：
- ✅ **基础兼容性**: AgenticMemorySystem与现有Memory类完全兼容
- ✅ **持久化存储**: 记忆状态自动保存和加载
- ✅ **性能监控**: 详细的检索统计和性能指标
- ✅ **降级处理**: 当A-MEM组件不可用时自动降级到基础功能

## 📋 核心功能

### AgenticMemorySystem

```python
from agentic_memory_system import AgenticMemorySystem

# 初始化（基础模式）
memory = AgenticMemorySystem(use_amem=False)

# 初始化（完整A-MEM模式）
memory = AgenticMemorySystem(
    use_amem=True,
    retriever_config={
        'use_api_embedding': True,
        'alpha': 0.5  # BM25与语义搜索权重
    }
)
```

### Planner集成

```python
from planner import Planner

planner = Planner(
    llm_engine_name="qwen2.5-72b-instruct",
    use_amem=True,  # 启用A-MEM增强
    retriever_config={'use_api_embedding': True}
)

# 规划时会自动检索相关历史记忆并注入到prompt中
```

### Verifier集成

```python
from verifier import Verifier

verifier = Verifier(
    llm_engine_name="qwen2.5-72b-instruct",
    use_amem=True,  # 启用A-MEM增强
    retriever_config={'use_api_embedding': True}
)

# 验证时会自动查询历史验证案例并辅助判断
```

## ⚙️ 配置说明

### 环境变量配置 (.env)

```env
# GPT-5 API Configuration (Primary)
MODEL=gpt-5
BASE_URL=https://yinli.one/v1
API_KEY=your-gpt5-api-key

# Qwen API Configuration (Alternative)
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_API_KEY=sk-b2a7128ecd0547009c2e9e48a6773133
QWEN_MODEL=qwen2.5-72b-instruct

# Test Configuration
TEST_LLM_BACKEND=litellm
TEST_LLM_MODEL=qwen2.5-72b-instruct
TEST_LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
TEST_LLM_API_KEY=sk-b2a7128ecd0547009c2e9e48a6773133
```

### 配置参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `use_amem` | 是否启用A-MEM功能 | `True` |
| `use_api_embedding` | 是否使用API嵌入 | `True` |
| `alpha` | BM25与语义搜索权重 | `0.5` |
| `max_memories` | 最大记忆数量 | `1000` |

## 📊 性能指标

### 测试结果示例

```
🎉 A-MEM核心功能测试完成！
============================================================
✅ 核心功能验证:
   - AgenticMemorySystem: 基础模式/正常
   - 记忆存储: X 条
   - 检索功能: X 次查询
   - 持久化存储: 启用

📊 性能指标:
   - 平均检索时间: X.XXXs
   - 检索成功率: XX.X%
```

### 扩展性

- **文档规模**: 支持数千文档处理
- **并发性能**: 批量API调用优化
- **存储效率**: JSON + Pickle混合存储
- **检索速度**: ~15秒 (包含API调用)

## 🔧 故障排除

### 常见问题

1. **A-MEM组件不可用**
   ```bash
   # 检查依赖安装
   pip install -r requirements_amem.txt
   ```

2. **API调用失败**
   ```bash
   # 检查.env文件配置
   cat memory/.env
   ```

3. **导入错误**
   ```bash
   # 从models目录运行
   cd /root/autodl-tmp/FreeAskAgent/agentflow/agentflow/models
   python quick_start.py
   ```

4. **性能问题**
   ```bash
   # 启用性能监控
   export AMEM_VERBOSE=true
   ```

## 📁 文件结构

```
/models/
├── agentic_memory_system.py      # 🧠 核心集成层
├── planner.py                     # 🎯 增强版规划器
├── verifier.py                    # 🔍 增强版验证器
├── memory/                        # 📚 A-MEM核心模块
│   ├── hybrid_retriever.py       # 🔍 混合检索器
│   ├── llm_controllers.py        # 🤖 LLM控制器
│   ├── content_analyzer.py       # 📝 内容分析器
│   ├── memory_note.py           # 📋 记忆单元
│   ├── .env                      # ⚙️ 环境配置
│   └── requirements_amem.txt     # 📦 依赖列表
├── Instructions/                  # 📋 集成文档
│   ├── design.md                 # 🎯 需求文档
│   ├── requirment.md            # 🛠️ 技术文档
│   └── task.md                   # ✅ 任务文档
├── memory_config.py              # ⚙️ 配置管理
└── quick_start.py               # 🚀 快速测试
```

## 🎉 成功标志

当你看到以下输出时，说明A-MEM集成成功：

```
✅ 核心功能验证:
   - AgenticMemorySystem: 正常
   - 记忆存储: X 条
   - 检索功能: X 次查询
   - 持久化存储: 启用

📝 测试结果:
   - 完全向后兼容 ✅
   - A-MEM功能正常 ✅
   - 性能表现良好 ✅
   - 监控日志完整 ✅
```

## 💡 进阶使用

### 1. 启用API嵌入
```python
memory = AgenticMemorySystem(
    use_amem=True,
    retriever_config={'use_api_embedding': True}
)
```

### 2. 自定义记忆分析
```python
from memory.content_analyzer import ContentAnalyzer
analyzer = ContentAnalyzer()
analyzed_content = analyzer.analyze("你的记忆内容")
```

### 3. 性能监控
```python
stats = memory.get_stats()
memory.log_performance_report()
```

### 4. 记忆持久化
```python
# 自动保存
memory.save_state()

# 自动加载
memory = AgenticMemorySystem()  # 会自动加载之前的状态
```

---

**🚀 A-MEM Memory System 已准备就绪！开始体验智能记忆增强的Agent能力吧！**

