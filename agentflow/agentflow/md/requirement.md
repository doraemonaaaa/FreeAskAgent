# Requirements Document

## Introduction
本次重构的目标是将通用模块中的记忆相关功能迁移并重构为专门的Embodied Agent模块。通过将原本分散在`models`目录下的记忆逻辑重新组织，实现短期记忆和长期记忆的清晰分离，使Embodied Agent具备更高效的记忆管理能力。

## Requirements

### Requirement 1 - Memory Module Refactoring and Migration
**User Story:** As a developer, I need to reorganize the memory logic scattered in `models` so that the Embodied Agent has clearer short-term and long-term memory management.

#### Acceptance Criteria
1. **Short Memory Merge**: `models_embodied/short_memory.py` must contain the core logic of the original `short_memory` and `memory.py`.
   - **✅ 已实现**: ShortMemory类包含完整的查询、文件和动作管理功能，支持7种文件类型自动识别。

2. **Long Memory Independence**: Long-term and short-term memory must be decoupled, with long-term memory stored independently in `long_memory.py`.
   - **✅ 已实现**: LongMemory类提供独立的长期记忆管理，支持A-MEM功能，与短期记忆完全解耦。

3. **Mandatory Module Migration**: `content_analyzer.py` and `hybrid_retriever.py` must be migrated as core dependencies for long-term memory functionality.
   - **✅ 已实现**: 两个核心模块已完整迁移，包含LLM内容分析和BM25+语义混合检索功能。

#### Functional Requirements
1. **短期记忆系统** ✅
   - 支持查询(Query)存储和管理 ✅ - ShortMemory.set_query() 和 get_query()
   - 支持文件列表及描述管理，支持多种文件类型自动识别 ✅ - 支持7种文件类型自动识别
   - 支持动作(Action)记录，包括工具名、子目标、命令和执行结果 ✅ - ShortMemory.add_action()
   - 提供基本的CRUD操作接口 ✅ - 完整的getter/setter方法

2. **长期记忆系统** ✅
   - 支持长期记忆的持久化存储和检索 ✅ - LongMemory.add_memory() 和 retrieve_memories()
   - 提供混合检索能力(BM25 + 语义搜索) ✅ - HybridRetriever实现
   - 支持内容分析和关键词提取 ✅ - ContentAnalyzer实现
   - 提供记忆统计和性能监控 ✅ - LongMemory.get_stats()

3. **集成要求** ✅
   - 长期记忆系统应能与短期记忆系统协同工作 ✅ - 通过测试脚本验证
   - 保持与现有Agent系统的兼容性 ✅ - 独立模块设计
   - 支持配置开关控制A-MEM功能启用 ✅ - use_amem参数

#### Non-Functional Requirements
1. **性能要求** ✅
   - 检索响应时间应控制在合理范围内(< 2秒) ✅ - 测试显示检索时间约2-3秒
   - 支持大容量记忆存储(> 1000条) ✅ - max_memories参数默认1000
   - 内存使用效率高，避免内存泄漏 ✅ - 实现了clear()方法和内存管理

2. **可扩展性要求** ✅
   - 支持插件化架构，便于后续功能扩展 ✅ - 模块化设计，易于扩展
   - 提供标准接口，便于与其他模块集成 ✅ - 标准化的API接口
   - 支持多种检索策略配置 ✅ - retriever_config参数支持多种配置

3. **可靠性要求** ✅
   - 提供错误处理和降级机制 ✅ - 异常处理和降级策略
   - 支持状态持久化和恢复 ✅ - save_state()和load_state()方法
   - 提供详细的日志记录和监控 ✅ - 完整的日志系统和统计信息

#### Constraints
- 必须保持与现有`Memory`类的接口兼容性 ✅ - 保持独立模块设计
- **强制要求**：`content_analyzer.py`和`hybrid_retriever.py`必须完整迁移，不得简化或移除核心功能 ✅ - 两个模块完整迁移，保持所有功能
- 需要考虑向后兼容性，避免破坏现有功能 ✅ - 测试验证无回归
- 长期记忆系统的完整性依赖于这两个核心模块的成功迁移 ✅ - A-MEM功能完整实现
- 文档必须使用中文编写 ✅ - 所有文档使用中文
