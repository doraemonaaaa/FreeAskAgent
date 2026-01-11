# Requirements Document

## Introduction
本次重构的目标是对 FreeAskAgent/agentflow 的 embodied 模块进行全面工程化重构，包括记忆机制完善、prompt 目录规范化、代码结构优化。目标是实现一个结构清晰、功能完整、可维护性强的 Embodied Agent 模块。

## Requirements

### Requirement 1 - Memory System Architecture
**User Story:** As a developer, I need a complete memory system that automatically manages conversation history and provides intelligent retrieval for Embodied Agent interactions.

#### Acceptance Criteria
1. **Memory Directory Structure**: `models_embodied/memory/` contains `short_memory.py`, `long_memory.py`, `content_analyzer.py`, `hybrid_retriever.py`, `memory_manager.py`
2. **Automatic Conversation Logging**: Every user/assistant message is automatically recorded in short_memory
3. **Intelligent Summarization**: Every 2-3 conversation turns trigger automatic summarization and storage in long_memory
4. **Context Retrieval**: New conversations automatically retrieve relevant historical summaries as context
5. **Memory Persistence**: Memory state is automatically saved and restored between sessions

#### Functional Requirements
1. **短期记忆系统 (ShortMemory)**
   - 自动记录每轮对话（role、content、timestamp、turn_id、session_id）
   - 提供对话窗口管理（默认3轮对话为一窗口）
   - 提供查询、文件、动作记录的CRUD接口
   - 支持多种文件类型自动识别和描述生成

2. **长期记忆系统 (LongMemory)**
   - 存储对话窗口的智能摘要（包含关键词、上下文、标签）
   - 提供混合检索能力（BM25 + 语义向量搜索）
   - 支持内容分析和A-MEM增强功能
   - 提供记忆统计、性能监控和持久化存储

3. **记忆管理器 (MemoryManager)**
   - 协调短期和长期记忆的交互
   - 自动触发对话窗口总结
   - 提供统一的记忆检索接口
   - 支持配置化的记忆参数

#### Non-Functional Requirements
1. **性能要求**
   - 记忆检索响应时间 < 2秒
   - 支持 > 1000 条记忆存储
   - 内存使用效率高，无内存泄漏

2. **可靠性要求**
   - 完整的错误处理和降级机制
   - 自动状态持久化和恢复
   - 详细的日志记录和监控

### Requirement 2 - Prompt Externalization
**User Story:** As a developer, I need all prompt templates to be externalized for easy maintenance and updates.

#### Acceptance Criteria
1. **Prompt Directory**: `models_embodied/prompts/` contains all prompt template files
2. **File-based Loading**: All prompts loaded from external files, not hardcoded strings
3. **Template Format**: Support for .txt files with variable substitution
4. **No Hardcoded Prompts**: Zero inline prompt strings in Python code

#### Functional Requirements
1. **Prompt文件组织**
   - `final_output_multimodal.txt`: 多模态最终输出prompt
   - `final_output_with_memory.txt`: 带记忆的文本输出prompt
   - `final_output_simple.txt`: 简单文本输出prompt
   - 其他模块的prompt文件按功能分类

2. **加载机制**
   - 相对路径加载（基于__file__解析）
   - 错误处理（文件不存在时的降级）
   - 模板变量替换功能

### Requirement 3 - LLM Integration Architecture
**User Story:** As a developer, I need a clean separation between embodied logic and LLM engine calls.

#### Acceptance Criteria
1. **No Custom Controllers**: No separate `llm_controllers.py` - use unified `engine/` directory
2. **Engine Integration**: All LLM calls go through `engine.factory.create_llm_engine()`
3. **Clean Interfaces**: Embodied modules focus on prompt assembly and result processing
4. **Configuration Support**: LLM engine selection via configuration parameters

#### Functional Requirements
1. **统一的LLM调用**
   - Planner直接使用engine创建的LLM实例
   - ContentAnalyzer使用engine工厂方法
   - 无重复的LLM封装逻辑

2. **配置化支持**
   - 支持多种LLM引擎（OpenAI, Anthropic, etc.）
   - 可配置的模型参数（temperature, max_tokens等）
   - 环境变量和配置文件支持

### Requirement 4 - Code Quality and Engineering Standards
**User Story:** As a developer, I need clean, maintainable, and well-engineered code.

#### Acceptance Criteria
1. **No Debug Code**: Zero print statements or temporary debug code in production
2. **Proper Logging**: All logging through Python logging module with appropriate levels
3. **Type Hints**: Complete type annotations for all public methods
4. **Documentation**: Comprehensive docstrings and inline comments
5. **Error Handling**: Proper exception handling with meaningful error messages

#### Functional Requirements
1. **代码质量**
   - 清晰的类和函数边界
   - 合理的命名规范
   - 完整的类型注解
   - 有意义的docstring

2. **工程实践**
   - 配置项集中管理，避免magic numbers
   - 依赖注入而非硬编码
   - 单元测试覆盖核心功能
   - CI/CD集成支持

### Requirement 5 - Integration and Testing
**User Story:** As a developer, I need confidence that the refactored system works correctly.

#### Acceptance Criteria
1. **Import Success**: All modules import without errors
2. **Memory Workflow**: Conversation → Short Memory → Summarization → Long Memory → Retrieval
3. **Prompt Loading**: All prompts load correctly from external files
4. **Backward Compatibility**: Existing functionality preserved
5. **Performance**: Memory operations meet performance requirements

## Verification Criteria

### Automated Tests
- [ ] Unit tests for ShortMemory core functionality
- [ ] Unit tests for LongMemory retrieval and storage
- [ ] Integration tests for MemoryManager workflow
- [ ] Prompt loading and template substitution tests
- [ ] Import and initialization tests for all modules

### Manual Verification
- [ ] Conversation recording in short_memory verified
- [ ] Automatic summarization after 3 turns verified
- [ ] Memory retrieval in new conversations verified
- [ ] Prompt externalization working correctly
- [ ] No console output or debug prints in production

### Performance Benchmarks
- [ ] Memory retrieval < 2 seconds for 1000 memories
- [ ] Memory storage operations < 1 second
- [ ] Import time < 5 seconds for all modules
- [ ] Memory footprint < 500MB for 1000 memories
