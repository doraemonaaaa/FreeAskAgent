#!/usr/bin/env python3
"""
Content Analyzer 单元测试
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from content_analyzer import ContentAnalyzer
from llm_controllers import LLMController


class MockLLMController:
    """模拟LLM控制器，用于测试"""

    def get_completion(self, prompt, response_format=None, temperature=0.7):
        # 返回模拟的JSON响应
        mock_response = '''{
            "keywords": ["时代广场", "盒马", "永辉", "超市"],
            "context": "Shopping location information about Times Square area",
            "tags": ["location", "shopping", "commerce"]
        }'''
        return mock_response


def test_fallback_analysis():
    """测试降级分析功能"""
    print("Testing fallback content analysis...")

    analyzer = ContentAnalyzer()  # 不提供LLM控制器，使用降级模式

    # 测试中文内容分析
    content = "时代广场内有盒马和永辉两家超市"
    result = analyzer.analyze_content(content)

    assert "keywords" in result
    assert "context" in result
    assert "tags" in result
    assert isinstance(result["keywords"], list)
    assert isinstance(result["context"], str)
    assert isinstance(result["tags"], list)

    # 检查是否提取了相关关键词
    keywords_str = " ".join(result["keywords"])
    assert any(word in keywords_str for word in ["时代广场", "盒马", "永辉", "超市"]), f"Keywords not extracted properly: {result['keywords']}"

    print("✓ Fallback analysis tests passed")


def test_mock_llm_analysis():
    """测试模拟LLM分析功能"""
    print("Testing mock LLM content analysis...")

    # 使用模拟LLM控制器
    mock_controller = MockLLMController()
    analyzer = ContentAnalyzer(llm_controller=mock_controller)

    content = "时代广场内有盒马和永辉两家超市"
    result = analyzer.analyze_content(content)

    assert "keywords" in result
    assert "context" in result
    assert "tags" in result

    # 检查模拟响应是否正确解析
    assert "时代广场" in result["keywords"]
    assert "盒马" in result["keywords"]
    assert "shopping" in result["context"].lower()
    assert "location" in result["tags"]

    print("✓ Mock LLM analysis tests passed")


def test_empty_content():
    """测试空内容分析"""
    print("Testing empty content analysis...")

    analyzer = ContentAnalyzer()

    # 测试空内容
    result = analyzer.analyze_content("")
    assert result["keywords"] == ["general"]
    assert result["context"] == "General content"
    assert result["tags"] == ["general"]

    # 测试None内容
    result = analyzer.analyze_content(None)
    assert result["keywords"] == ["general"]

    print("✓ Empty content analysis tests passed")


def test_context_inference():
    """测试上下文推断功能"""
    print("Testing context inference...")

    analyzer = ContentAnalyzer()

    # 测试购物相关内容
    shopping_content = "时代广场 盒马 永辉 超市 购物"
    result = analyzer.analyze_content(shopping_content)
    assert "shopping" in result["context"].lower()

    # 测试技术相关内容
    tech_content = "Python 编程 开发 代码 技术"
    result = analyzer.analyze_content(tech_content)
    assert any(word in result["tags"] for word in ["technology", "programming"])

    print("✓ Context inference tests passed")


def test_response_parsing():
    """测试响应解析功能"""
    print("Testing response parsing...")

    analyzer = ContentAnalyzer()

    # 测试正常JSON响应
    normal_response = '{"keywords": ["test"], "context": "test context", "tags": ["test"]}'
    result = analyzer._parse_llm_response(normal_response)
    assert result["keywords"] == ["test"]
    assert result["context"] == "test context"
    assert result["tags"] == ["test"]

    # 测试带额外文本的JSON响应
    extra_text_response = 'Here is the analysis: {"keywords": ["test"], "context": "test context", "tags": ["test"]} And some more text.'
    result = analyzer._parse_llm_response(extra_text_response)
    assert result["keywords"] == ["test"]

    # 测试无效JSON响应（应该降级）
    invalid_response = "This is not JSON at all"
    result = analyzer._parse_llm_response(invalid_response)
    # 应该返回降级结果
    assert isinstance(result["keywords"], list)
    assert isinstance(result["context"], str)

    print("✓ Response parsing tests passed")


def test_analyzer_update():
    """测试分析器更新功能"""
    print("Testing analyzer update...")

    analyzer = ContentAnalyzer()

    # 初始状态：无LLM控制器
    assert not analyzer.llm_available

    # 更新为有LLM控制器的状态
    mock_controller = MockLLMController()
    analyzer.update_llm_controller(mock_controller)
    assert analyzer.llm_available
    assert analyzer.llm_controller is mock_controller

    # 再次更新为无LLM控制器
    analyzer.update_llm_controller(None)
    assert not analyzer.llm_available

    print("✓ Analyzer update tests passed")


def run_all_tests():
    """运行所有测试"""
    print("Running Content Analyzer unit tests...\n")

    try:
        test_fallback_analysis()
        test_mock_llm_analysis()
        test_empty_content()
        test_context_inference()
        test_response_parsing()
        test_analyzer_update()

        print("\n🎉 All Content Analyzer tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
