#!/usr/bin/env python3
"""
简化的 AgenticMemory 功能演示

直接展示记忆分析和检索功能，不依赖复杂的导入
"""

import os
import json

def load_config():
    """加载配置"""
    config_file = "agentflow/agentflow/models/memory/config.env"
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
        return True
    return False

def demonstrate_memory_analysis():
    """演示记忆分析功能"""
    print("🧠 记忆分析功能演示")
    print("=" * 40)

    try:
        import litellm
        from litellm import completion

        # 测试记忆内容
        test_memories = [
            "时代广场中有盒马、永辉等超市，提供新鲜蔬果和日用品",
            "时代广场附近有星巴克咖啡店，环境舒适，适合工作和休息",
            "时代广场周边交通便利，有地铁站和多个公交站点"
        ]

        print("📝 正在分析记忆内容...\n")

        for i, content in enumerate(test_memories, 1):
            print(f"记忆 {i}: {content}")

            # 构建分析提示
            analysis_prompt = f"""Generate a structured analysis of the following content by:
1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)
2. Extracting core themes and contextual elements
3. Creating relevant categorical tags

Format the response as a JSON object:
{{
    "keywords": ["keyword1", "keyword2", ...],
    "context": "One sentence summarizing the content",
    "tags": ["tag1", "tag2", ...]
}}

Content for analysis:
{content}"""

            # 调用 LLM 分析
            response = completion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": analysis_prompt}],
                api_key=os.environ.get('LITELLM_API_KEY'),
                api_base=os.environ.get('LITELLM_API_BASE'),
                temperature=0.3,
                max_tokens=200
            )

            # 解析结果
            result_text = response.choices[0].message.content

            # 清理 JSON
            result_text = result_text.strip()
            if result_text.startswith('```json'):
                result_text = result_text[7:]
            if result_text.endswith('```'):
                result_text = result_text[:-3]
            result_text = result_text.strip()

            try:
                analysis = json.loads(result_text)
                print("   🔑 关键词:", analysis.get('keywords', []))
                print("   📝 上下文:", analysis.get('context', ''))
                print("   🏷️ 标签:", analysis.get('tags', []))
            except json.JSONDecodeError:
                print("   ⚠️ 分析结果解析失败")
                print(f"   原始结果: {result_text[:100]}...")

            print()

    except Exception as e:
        print(f"❌ 记忆分析演示失败: {e}")

def demonstrate_semantic_search():
    """演示语义搜索功能"""
    print("🔍 语义搜索功能演示")
    print("=" * 40)

    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        # 初始化模型
        print("🤖 加载语义模型...")
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # 记忆库
        memories = [
            "时代广场中有盒马、永辉等超市，提供新鲜蔬果和日用品",
            "时代广场附近有星巴克咖啡店，环境舒适，适合工作和休息",
            "时代广场周边交通便利，有地铁站和多个公交站点",
            "时代广场是城市中心商业区，有很多餐厅和娱乐场所",
            "机器学习是人工智能的重要分支",
            "深度学习使用神经网络进行特征提取"
        ]

        print(f"📚 记忆库包含 {len(memories)} 条记忆\n")

        # 生成嵌入
        print("🔢 计算语义嵌入...")
        embeddings = model.encode(memories)

        # 测试查询
        queries = [
            "时代广场周边有什么超市",
            "时代广场附近有咖啡店吗",
            "时代广场交通怎么样",
            "什么是机器学习"
        ]

        for query in queries:
            print(f"❓ 查询: {query}")

            # 计算查询嵌入
            query_emb = model.encode([query])

            # 计算相似度
            similarities = cosine_similarity(query_emb, embeddings)[0]

            # 获取最相关的结果
            top_indices = np.argsort(similarities)[-3:][::-1]  # Top 3
            top_scores = similarities[top_indices]

            print("🎯 最相关结果:")
            for i, (idx, score) in enumerate(zip(top_indices, top_scores), 1):
                if score > 0.1:  # 只显示相关度足够高的结果
                    print(".3f")
                    print(f"   记忆: {memories[idx]}")

            print()

    except Exception as e:
        print(f"❌ 语义搜索演示失败: {e}")
        print("请确保安装了 sentence-transformers 和 scikit-learn")

def demonstrate_cli_usage():
    """演示 CLI 使用方法"""
    print("💻 命令行工具使用演示")
    print("=" * 40)

    print("您可以运行以下命令来使用交互式记忆工具:")
    print()
    print("1. 启动交互式工具:")
    print("   cd /root/autodl-tmp/FreeAskAgent")
    print("   python memory_cli.py")
    print()
    print("2. 在工具中可以执行以下操作:")
    print()
    print("   添加记忆:")
    print("   📝 请输入命令: 在时代广场中有盒马、永辉等超市")
    print("   或")
    print("   📝 请输入命令: add 在时代广场中有盒马、永辉等超市")
    print()
    print("   查询记忆:")
    print("   📝 请输入命令: query 时代广场周边有什么超市")
    print()
    print("   列出所有记忆:")
    print("   📝 请输入命令: list")
    print()
    print("   查看统计:")
    print("   📝 请输入命令: stats")
    print()
    print("   获取帮助:")
    print("   📝 请输入命令: help")
    print()
    print("   退出工具:")
    print("   📝 请输入命令: quit")

def show_integration_example():
    """显示集成示例"""
    print("🔗 代码集成示例")
    print("=" * 40)

    print("""
# 在您的 Python 代码中使用 AgenticMemory

import os
# 设置 API 密钥
os.environ['LITELLM_API_KEY'] = 'sk-mQRVq6Mved8vHoJklaJQnLabN0sT9KEnc2Vw45bniUAvBYPL'
os.environ['LITELLM_API_BASE'] = 'https://yinli.one/v1'

# 注意：由于导入问题，建议直接使用组件
from sentence_transformers import SentenceTransformer
import litellm
from litellm import completion

# 1. 创建语义搜索功能
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. 维护记忆库
memories = []

def add_memory(content):
    \"\"\"添加记忆\"\"\"
    # 使用 LLM 分析记忆
    analysis_prompt = f'''分析这段内容，提取关键词和标签:
{content}

返回 JSON 格式: {{"keywords": [], "tags": []}}'''

    response = completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": analysis_prompt}],
        api_key=os.environ['LITELLM_API_KEY'],
        api_base=os.environ['LITELLM_API_BASE']
    )

    # 存储记忆
    memories.append({
        'content': content,
        'embedding': model.encode([content])[0],
        'analysis': response.choices[0].message.content
    })

def search_memories(query, top_k=3):
    \"\"\"搜索相关记忆\"\"\"
    if not memories:
        return []

    query_emb = model.encode([query])[0]
    similarities = []

    for mem in memories:
        sim = np.dot(query_emb, mem['embedding']) / (
            np.linalg.norm(query_emb) * np.linalg.norm(mem['embedding'])
        )
        similarities.append((sim, mem))

    # 排序并返回 top_k
    similarities.sort(reverse=True, key=lambda x: x[0])
    return similarities[:top_k]

# 使用示例
add_memory("时代广场中有盒马、永辉等超市")
results = search_memories("时代广场周边有什么超市")

for score, mem in results:
    print(f"相似度: {score:.3f}")
    print(f"内容: {mem['content']}")
    print(f"分析: {mem['analysis']}")
    """)

def main():
    """主函数"""
    print("🎯 AgenticMemory 功能演示")
    print("让 AI 记住一切，随时查询！")
    print("=" * 50)

    # 加载配置
    if not load_config():
        print("⚠️ 未找到配置文件，使用默认设置")

    # 检查 API Key
    if not os.getenv('LITELLM_API_KEY'):
        print("❌ 未设置 API Key，请检查配置")
        return

    print("✅ 配置检查通过\n")

    # 运行演示
    demonstrate_memory_analysis()
    demonstrate_semantic_search()
    demonstrate_cli_usage()
    show_integration_example()

    print("\n🎉 演示完成！")
    print("💡 现在您可以使用 AgenticMemory 功能了！")
    print("🚀 运行 'python memory_cli.py' 开始交互式体验")

if __name__ == "__main__":
    main()
