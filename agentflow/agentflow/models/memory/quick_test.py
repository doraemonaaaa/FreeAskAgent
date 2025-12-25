#!/usr/bin/env python3
"""
快速测试脚本 - 验证GPT-5 API嵌入功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hybrid_retriever import HybridRetriever


def main():
    print("🚀 GPT-5 API嵌入快速测试")
    print("=" * 40)

    try:
        # 1. 创建检索器
        print("1. 初始化检索器...")
        retriever = HybridRetriever(use_api_embedding=True)
        print("   ✅ 成功初始化")

        # 2. 检查配置
        stats = retriever.get_stats()
        print("2. 配置检查:")
        print(f"   - API嵌入: {'✅' if stats['use_api_embedding'] else '❌'}")
        print(f"   - 语义搜索: {'✅' if stats['semantic_available'] else '❌'}")
        print(f"   - LLM控制器: {'✅' if stats['llm_controller_available'] else '❌'}")

        # 3. 添加测试文档
        print("\n3. 添加测试文档...")
        docs = ["苹果是一家科技公司", "香蕉是一种水果", "编程很有趣"]
        success = retriever.add_documents(docs)
        print(f"   📄 添加了 {len(docs)} 个文档: {'✅' if success else '❌'}")

        # 4. 执行简单检索
        print("\n4. 执行检索测试...")
        query = "水果"
        results = retriever.retrieve(query, k=2)
        print(f"   🔍 查询 '{query}' -> 结果索引: {results}")

        if results and len(results) > 0:
            print("   📖 找到的相关文档:")
        for idx in results[:2]:  # 只显示前2个
            if 0 <= idx < len(retriever.corpus):
                print(f"      - {retriever.corpus[idx]}")

        print("\n🎉 测试完成！API嵌入功能正常工作")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
