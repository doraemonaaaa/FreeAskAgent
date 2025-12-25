# A-MEM 依赖分析报告

## 依赖兼容性分析

### 现有 AgentFlow 依赖
- openai==1.75.0
- transformers (无版本约束)
- python-dotenv==1.0.1
- 其他工具库...

### A-MEM 新增依赖
- sentence-transformers>=3.4.1 (新增)
- scikit-learn>=1.6.1 (新增)
- nltk>=3.9.1 (新增)
- rank-bm25>=0.2.2 (新增)
- litellm>=1.59.1 (AgentFlow中被注释，可能需要启用)
- ollama>=0.3.3 (AgentFlow中被注释，可能需要启用)
- numpy>=1.24.3 (可能与现有版本兼容)
- torch>=2.4.0 (可能与现有版本兼容)

### 版本兼容性问题
1. **openai**: A-MEM需要1.61.1，AgentFlow使用1.75.0
   - 建议：使用AgentFlow的版本 (1.75.0)，A-MEM代码需要兼容性测试

2. **litellm 和 ollama**: 在AgentFlow中被注释
   - 建议：添加到AgentFlow主requirements.txt中，或作为A-MEM的条件依赖

### 解决策略
1. 创建独立的A-MEM依赖文件 `requirements_amem.txt`
2. 在A-MEM模块中进行条件导入，避免与AgentFlow核心冲突
3. 对于版本冲突的包，使用AgentFlow的主版本
4. 添加错误处理，当依赖不可用时提供降级功能

### 安装建议
```bash
# 安装A-MEM专用依赖
pip install -r agentflow/models/memory/requirements_amem.txt

# 或添加到AgentFlow主依赖中
echo "sentence-transformers>=3.4.1" >> requirements.txt
echo "scikit-learn>=1.6.1" >> requirements.txt
echo "nltk>=3.9.1" >> requirements.txt
echo "rank-bm25>=0.2.2" >> requirements.txt
echo "litellm>=1.59.1" >> requirements.txt
echo "ollama>=0.3.3" >> requirements.txt
```

## 结论
依赖兼容性良好，主要依赖可以与AgentFlow现有环境共存。建议采用渐进式集成策略，先独立测试A-MEM功能，再考虑合并到主依赖中。
