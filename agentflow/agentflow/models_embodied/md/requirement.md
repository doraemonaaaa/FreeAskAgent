# Requirements Document

## Introduction
本次重构的目标是将通用模块中的记忆相关功能迁移并重构为专门的Embodied Agent模块。通过将原本分散在`models`目录下的记忆逻辑重新组织，实现短期记忆和长期记忆的清晰分离，使Embodied Agent具备更高效的记忆管理能力。

## Requirements

### Requirement 1 - Memory Module Refactoring and Migration
**User Story:** As a developer, I need to reorganize the memory logic scattered in `models` so that the Embodied Agent has clearer short-term and long-term memory management.

#### Acceptance Criteria
1. **Short Memory Merge**: `models_embodied/short_memory.py` must contain the core logic of the original `short_memory` and `memory.py`.
2. **Long Memory Independence**: Long-term and short-term memory must be decoupled, with long-term memory stored independently in `long_memory.py`.
3. **Mandatory Module Migration**: `content_analyzer.py` and `hybrid_retriever.py` must be migrated as core dependencies for long-term memory functionality.

#### Functional Requirements
1. **短期记忆系统**
   - 支持查询(Query)存储和管理
   - 支持文件列表及描述管理，支持多种文件类型自动识别
   - 支持动作(Action)记录，包括工具名、子目标、命令和执行结果
   - 提供基本的CRUD操作接口

2. **长期记忆系统**
   - 支持长期记忆的持久化存储和检索
   - 提供混合检索能力(BM25 + 语义搜索)
   - 支持内容分析和关键词提取
   - 提供记忆统计和性能监控

3. **集成要求**
   - 长期记忆系统应能与短期记忆系统协同工作
   - 保持与现有Agent系统的兼容性
   - 支持配置开关控制A-MEM功能启用

#### Non-Functional Requirements
1. **性能要求**
   - 检索响应时间应控制在合理范围内(< 2秒)
   - 支持大容量记忆存储(> 1000条)
   - 内存使用效率高，避免内存泄漏

2. **可扩展性要求**
   - 支持插件化架构，便于后续功能扩展
   - 提供标准接口，便于与其他模块集成
   - 支持多种检索策略配置

3. **可靠性要求**
   - 提供错误处理和降级机制
   - 支持状态持久化和恢复
   - 提供详细的日志记录和监控

#### Constraints
- 必须保持与现有`Memory`类的接口兼容性
- **强制要求**：`content_analyzer.py`和`hybrid_retriever.py`必须完整迁移，不得简化或移除核心功能
- 需要考虑向后兼容性，避免破坏现有功能
- 长期记忆系统的完整性依赖于这两个核心模块的成功迁移
- 文档必须使用中文编写
