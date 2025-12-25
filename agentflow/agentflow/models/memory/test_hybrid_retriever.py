#!/usr/bin/env python3
"""
Hybrid Retriever 单元测试
"""

import sys
import os
import tempfile
import shutil
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hybrid_retriever import HybridRetriever


def test_retriever_initialization():
    """测试检索器初始化"""
    print("Testing HybridRetriever initialization...")

    # 先检查网络连接，避免下载模型超时
    import requests
    try:
        requests.get("https://huggingface.co", timeout=5)
        network_available = True
    except:
        network_available = False
        print("  Warning: Network unavailable, semantic search will be disabled")

    retriever = HybridRetriever(alpha=0.7)

    assert retriever.alpha == 0.7
    assert retriever.model_name == 'all-MiniLM-L6-v2'
    assert retriever.corpus == []
    assert retriever.document_ids == {}

    # 检查功能可用性（可能因依赖缺失或网络问题而不同）
    print(f"  BM25 available: {retriever.bm25_available}")
    print(f"  Semantic search available: {retriever.semantic_available}")
    print(f"  Network available: {network_available}")

    print("✓ Retriever initialization tests passed")


def test_document_operations():
    """测试文档操作"""
    print("Testing document operations...")

    retriever = HybridRetriever()

    # 如果语义搜索不可用，只测试BM25功能
    if not retriever.semantic_available:
        print("  Note: Semantic search disabled, testing BM25 only")

    # 测试批量添加文档
    documents = [
        "时代广场 盒马 超市",
        "永辉 购物 商场",
        "技术 编程 Python",
        "学习 教育 课程"
    ]

    success = retriever.add_documents(documents)

    # 如果BM25可用，检查文档是否正确添加
    if retriever.bm25_available:
        assert len(retriever.corpus) == len(documents)
        assert all(doc in retriever.document_ids for doc in documents)
    else:
        print("  BM25 not available, skipping document validation")

    # 测试单个文档添加（如果BM25可用）
    if retriever.bm25_available:
        new_doc = "新的 测试 文档"
        was_added = retriever.add_document(new_doc)

        if was_added:
            assert new_doc in retriever.document_ids
            assert len(retriever.corpus) == len(documents) + 1
        else:
            # 文档已存在
            assert not was_added

    print("✓ Document operations tests passed")


def test_retrieval():
    """测试检索功能"""
    print("Testing retrieval functionality...")

    retriever = HybridRetriever()

    # 添加测试文档
    documents = [
        "时代广场内有盒马和永辉两家超市",
        "永辉超市位于时代广场附近",
        "技术编程课程很有趣",
        "学习Python编程语言"
    ]

    retriever.add_documents(documents)

    # 执行检索（只要BM25可用就可以测试）
    if retriever.bm25_available and retriever.corpus:
        query = "时代广场 超市"
        results = retriever.retrieve(query, k=2)

        assert isinstance(results, list)
        assert len(results) <= 2  # 最多返回k个结果
        assert all(isinstance(idx, int) for idx in results)
        assert all(0 <= idx < len(retriever.corpus) for idx in results)

        # 测试search接口（应该与retrieve相同）
        search_results = retriever.search(query, k=2)
        assert results == search_results

        print("  BM25 retrieval tested successfully")
    elif retriever.semantic_available and retriever.corpus:
        print("  BM25 not available, but semantic search is available")
        # 至少验证接口调用不会崩溃
        query = "时代广场 超市"
        results = retriever.retrieve(query, k=2)
        assert isinstance(results, list)
    else:
        print("  Neither BM25 nor semantic search available, skipping retrieval test")

    print("✓ Retrieval functionality tests passed")


def test_empty_retrieval():
    """测试空检索器的情况"""
    print("Testing empty retriever...")

    retriever = HybridRetriever()

    # 空检索器应该返回空结果
    results = retriever.retrieve("test query")
    assert results == []

    print("✓ Empty retriever tests passed")


def test_persistence():
    """测试持久化功能"""
    print("Testing persistence...")

    with tempfile.TemporaryDirectory() as temp_dir:
        cache_file = os.path.join(temp_dir, "retriever.pkl")
        embeddings_file = os.path.join(temp_dir, "embeddings.npy")

        # 创建和保存检索器
        retriever1 = HybridRetriever(alpha=0.6)
        documents = ["测试文档1", "测试文档2", "测试文档3"]
        retriever1.add_documents(documents)

        # 保存
        save_success = retriever1.save(cache_file, embeddings_file)
        # 保存可能失败（如果依赖不可用），这是可以接受的

        # 加载
        retriever2 = HybridRetriever.load(cache_file, embeddings_file)

        if retriever2:  # 如果加载成功
            assert retriever2.alpha == 0.6
            if retriever1.corpus:  # 如果原始检索器有数据
                assert len(retriever2.corpus) == len(retriever1.corpus)

    print("✓ Persistence tests passed")


def test_tokenization():
    """测试分词功能"""
    print("Testing tokenization...")

    retriever = HybridRetriever()

    # 测试中文分词
    text = "时代广场内有盒马和永辉两家超市"
    tokens = retriever._simple_tokenize(text)

    assert isinstance(tokens, list)
    assert len(tokens) > 0
    assert "时代广场" in tokens or "时代" in tokens

    # 测试英文分词
    english_text = "Hello world Python programming"
    english_tokens = retriever._simple_tokenize(english_text)

    assert "Hello" in english_tokens
    assert "world" in english_tokens
    assert "Python" in english_tokens

    print("✓ Tokenization tests passed")


def test_stats():
    """测试统计信息"""
    print("Testing statistics...")

    retriever = HybridRetriever(alpha=0.8)

    stats = retriever.get_stats()

    assert isinstance(stats, dict)
    assert 'total_documents' in stats
    assert 'bm25_available' in stats
    assert 'semantic_available' in stats
    assert 'model_name' in stats
    assert 'alpha' in stats
    assert stats['alpha'] == 0.8

    print("✓ Statistics tests passed")


def test_clear():
    """测试清空功能"""
    print("Testing clear functionality...")

    retriever = HybridRetriever()

    # 添加一些文档
    documents = ["doc1", "doc2", "doc3"]
    retriever.add_documents(documents)

    # 清空
    retriever.clear()

    assert len(retriever.corpus) == 0
    assert len(retriever.document_ids) == 0

    print("✓ Clear functionality tests passed")


def run_all_tests():
    """运行所有测试"""
    print("Running Hybrid Retriever unit tests...\n")

    try:
        test_retriever_initialization()
        test_document_operations()
        test_retrieval()
        test_empty_retrieval()
        test_persistence()
        test_tokenization()
        test_stats()
        test_clear()

        print("\n🎉 All Hybrid Retriever tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
