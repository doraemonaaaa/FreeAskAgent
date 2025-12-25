#!/usr/bin/env python3
"""
API嵌入功能演示脚本

展示GPT-5 API嵌入功能的效果，避免下载本地模型。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hybrid_retriever import HybridRetriever


def demo_api_embedding():
    """演示API嵌入功能"""
    print("🎯 GPT-5 API嵌入功能演示")
    print("=" * 50)

    # 创建使用API嵌入的检索器
    print("1. 初始化API嵌入检索器...")
    retriever = HybridRetriever(use_api_embedding=True)
    print("   ✅ API嵌入模式已启用")

    # 显示配置信息
    stats = retriever.get_stats()
    print(f"   📊 配置状态:")
    print(f"      - API嵌入: {stats['use_api_embedding']}")
    print(f"      - 语义搜索可用: {stats['semantic_available']}")
    print(f"      - LLM控制器可用: {stats['llm_controller_available']}")

    print("\n2. 添加测试文档...")
    documents = [
        "时代广场内有盒马和永辉两家超市",
        "永辉超市位于时代广场附近",
        "技术编程课程很有趣",
        "学习Python编程语言"
    ]

    success = retriever.add_documents(documents)
    if success:
        print(f"   ✅ 成功添加 {len(documents)} 个文档")
        print(f"   📄 当前文档数量: {len(retriever.corpus)}")
    else:
        print("   ❌ 添加文档失败")
        return

    print("\n3. 执行检索测试...")
    test_queries = [
        "时代广场 超市",
        "编程 课程",
        "Python 学习"
    ]

    for query in test_queries:
        print(f"\n   🔍 查询: '{query}'")
        try:
            results = retriever.retrieve(query, k=2)
            print(f"   📋 返回索引: {results}")

            if results:
                print("   📖 相关文档:")
                for idx in results:
                    if 0 <= idx < len(retriever.corpus):
                        doc = retriever.corpus[idx]
                        print(f"      - {doc}")
        except Exception as e:
            print(f"   ❌ 检索出错: {e}")

    print("\n4. 性能测试...")
    import time

    start_time = time.time()
    for _ in range(5):
        retriever.retrieve("测试查询", k=1)
    end_time = time.time()

    avg_time = (end_time - start_time) / 5
    print(f"   ⏱️ 平均响应时间: {avg_time:.2f}秒"))
    print("🎉 API嵌入功能演示完成！")


def demo_config_info():
    """显示配置信息"""
    print("🔧 当前配置信息")
    print("=" * 30)

    # 读取环境变量
    config_vars = [
        'MODEL',
        'BASE_URL',
        'API_KEY',
        'USE_API_EMBEDDING',
        'EMBEDDING_MODEL',
        'EMBEDDING_API_BASE',
        'RETRIEVER_BACKEND'
    ]

    for var in config_vars:
        value = os.getenv(var, '未设置')
        # 隐藏API密钥
        if 'KEY' in var or 'key' in var:
            if len(value) > 10:
                value = value[:6] + '***' + value[-4:]
        print(f"      {var}: {value}")


if __name__ == "__main__":
    try:
        demo_config_info()
        print()
        demo_api_embedding()
    except Exception as e:
        print(f"❌ 演示脚本出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
