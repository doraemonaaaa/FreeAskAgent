#!/usr/bin/env python3
"""
MemoryNote 单元测试
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory_note import MemoryNote
import json


def test_memory_note_creation():
    """测试记忆创建"""
    print("Testing MemoryNote creation...")

    # 基本创建
    note = MemoryNote(content="时代广场内有盒马和永辉两家超市")
    assert note.content == "时代广场内有盒马和永辉两家超市"
    assert len(note.id) > 0
    assert note.keywords == []
    assert note.tags == []
    assert note.importance_score == 1.0

    # 带参数创建
    note2 = MemoryNote(
        content="测试记忆内容",
        keywords=["测试", "关键词"],
        tags=["测试标签"],
        importance_score=0.8,
        category="测试分类"
    )
    assert note2.keywords == ["测试", "关键词"]
    assert note2.tags == ["测试标签"]
    assert note2.importance_score == 0.8
    assert note2.category == "测试分类"

    print("✓ MemoryNote creation tests passed")


def test_memory_note_operations():
    """测试记忆操作"""
    print("Testing MemoryNote operations...")

    note = MemoryNote(content="测试内容")

    # 测试标签操作
    note.add_tag("标签1")
    note.add_tag("标签2")
    assert "标签1" in note.tags
    assert "标签2" in note.tags

    note.remove_tag("标签1")
    assert "标签1" not in note.tags
    assert "标签2" in note.tags

    # 测试连接操作
    note.add_link(1)
    note.add_link(2)
    assert 1 in note.links
    assert 2 in note.links

    note.remove_link(1)
    assert 1 not in note.links
    assert 2 in note.links

    # 测试检索计数
    initial_count = note.retrieval_count
    note.increment_retrieval_count()
    assert note.retrieval_count == initial_count + 1

    print("✓ MemoryNote operations tests passed")


def test_memory_note_serialization():
    """测试记忆序列化"""
    print("Testing MemoryNote serialization...")

    # 创建测试记忆
    note = MemoryNote(
        content="时代广场内有盒马和永辉两家超市",
        keywords=["时代广场", "盒马", "永辉", "超市"],
        tags=["地点", "购物"],
        context="购物场所信息",
        category="地点信息"
    )

    # 测试字典序列化
    data = note.to_dict()
    assert data["content"] == note.content
    assert data["keywords"] == note.keywords
    assert data["tags"] == note.tags

    # 测试从字典反序列化
    note2 = MemoryNote.from_dict(data)
    assert note2.content == note.content
    assert note2.keywords == note.keywords
    assert note2.tags == note.tags
    assert note2.id == note.id

    # 测试JSON序列化
    json_str = note.to_json()
    note3 = MemoryNote.from_json(json_str)
    assert note3.content == note.content
    assert note3.keywords == note.keywords

    print("✓ MemoryNote serialization tests passed")


def test_memory_note_evolution():
    """测试记忆演化记录"""
    print("Testing MemoryNote evolution...")

    note = MemoryNote(content="初始内容")

    # 添加演化记录
    note.add_evolution_record("strengthen", {"connected_to": [1, 2]})
    note.add_evolution_record("update_context", {"old_context": "初始", "new_context": "更新后"})

    assert len(note.evolution_history) == 2
    assert note.evolution_history[0]["action"] == "strengthen"
    assert note.evolution_history[1]["action"] == "update_context"

    # 测试序列化包含演化历史
    data = note.to_dict()
    note2 = MemoryNote.from_dict(data)
    assert len(note2.evolution_history) == 2

    print("✓ MemoryNote evolution tests passed")


def run_all_tests():
    """运行所有测试"""
    print("Running MemoryNote unit tests...\n")

    try:
        test_memory_note_creation()
        test_memory_note_operations()
        test_memory_note_serialization()
        test_memory_note_evolution()

        print("\n🎉 All MemoryNote tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
