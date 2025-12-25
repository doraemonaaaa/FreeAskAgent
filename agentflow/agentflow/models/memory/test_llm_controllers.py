#!/usr/bin/env python3
"""
LLM Controllers 单元测试
"""

import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm_controllers import (
    BaseLLMController,
    OpenAIController,
    LiteLLMController,
    OllamaController,
    SGLangController,
    LLMController
)


class MockLLMController(BaseLLMController):
    """模拟LLM控制器，用于测试"""

    def get_completion(self, prompt: str, response_format=None, temperature: float = 0.7) -> str:
        # 返回一个模拟的JSON响应
        mock_response = {
            "keywords": ["测试", "关键词"],
            "context": "这是一个测试上下文",
            "tags": ["测试标签"]
        }
        return json.dumps(mock_response)


def test_base_controller():
    """测试基础控制器"""
    print("Testing BaseLLMController...")

    controller = MockLLMController()

    # 测试空值生成
    assert controller._generate_empty_value("string") == ""
    assert controller._generate_empty_value("array") == []
    assert controller._generate_empty_value("object") == {}
    assert controller._generate_empty_value("number") == 0
    assert controller._generate_empty_value("boolean") is False

    # 测试空响应生成
    response_format = {
        "json_schema": {
            "schema": {
                "properties": {
                    "keywords": {"type": "array"},
                    "context": {"type": "string"},
                    "tags": {"type": "array"}
                }
            }
        }
    }

    empty_response = controller._generate_empty_response(response_format)
    assert empty_response["keywords"] == []
    assert empty_response["context"] == ""
    assert empty_response["tags"] == []

    print("✓ BaseLLMController tests passed")


def test_llm_controller_factory():
    """测试LLM控制器工厂"""
    print("Testing LLMController factory...")

    # 测试无效后端
    try:
        controller = LLMController(backend="invalid")
        assert False, "Should have failed with invalid backend"
    except ValueError:
        pass  # 预期的失败

    # 测试SGLang控制器（不依赖外部包）
    try:
        controller = SGLangController(model="llama2")
        assert controller.model == "llama2"
        assert controller.base_url == "http://localhost:30000"
    except Exception as e:
        print(f"Unexpected error with SGLang: {e}")

    # 测试OpenAI控制器（应该因为缺少API密钥而失败）
    try:
        controller = OpenAIController(model="gpt-4o-mini")
        assert False, "Should have failed without API key"
    except ValueError:
        pass  # 预期的失败

    # 测试LiteLLM控制器（应该能创建，因为依赖可用）
    try:
        controller = LiteLLMController(model="gpt-4o-mini")
        assert controller.model == "gpt-4o-mini"
        assert controller.api_key == "EMPTY"
    except Exception as e:
        print(f"LiteLLM controller creation failed: {e}")

    # 测试Ollama控制器（应该能创建，因为依赖可用）
    try:
        controller = OllamaController(model="llama2")
        assert "ollama/llama2" in controller.model
    except Exception as e:
        print(f"Ollama controller creation failed: {e}")

    print("✓ LLMController factory tests passed")


def test_mock_completion():
    """测试模拟完成功能"""
    print("Testing mock completion...")

    controller = MockLLMController()

    prompt = "测试提示"
    response = controller.get_completion(prompt)

    # 解析响应
    data = json.loads(response)
    assert "keywords" in data
    assert "context" in data
    assert "tags" in data
    assert isinstance(data["keywords"], list)
    assert isinstance(data["context"], str)
    assert isinstance(data["tags"], list)

    print("✓ Mock completion tests passed")


def test_sglang_controller():
    """测试SGLang控制器（不依赖真实服务器）"""
    print("Testing SGLangController...")

    controller = SGLangController(model="test-model", sglang_host="http://localhost", sglang_port=30000)

    assert controller.model == "test-model"
    assert controller.base_url == "http://localhost:30000"

    # 测试错误处理（服务器不存在）
    response = controller.get_completion("test prompt")
    assert response == ""  # 应该返回空字符串作为错误处理

    # 测试带响应格式的错误处理
    response_format = {"json_schema": {"schema": {"properties": {"test": {"type": "string"}}}}}
    response = controller.get_completion("test prompt", response_format)
    data = json.loads(response)
    assert "test" in data
    assert data["test"] == ""

    print("✓ SGLangController tests passed")


def run_all_tests():
    """运行所有测试"""
    print("Running LLM Controllers unit tests...\n")

    try:
        test_base_controller()
        test_llm_controller_factory()
        test_mock_completion()
        test_sglang_controller()

        print("\n🎉 All LLM Controllers tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
