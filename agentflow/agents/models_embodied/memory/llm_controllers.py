"""
LLM Controllers for A-MEM

移植自 A-MEM 的 LLM 控制器，支持多种后端（OpenAI、LiteLLM、Ollama、SGLang）。
适配 AgentFlow 的配置系统。
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Literal
import os
import json
import requests
from pathlib import Path

# 条件导入，增加错误处理
try:
    from litellm import completion
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


class BaseLLMController(ABC):
    """基础LLM控制器抽象类"""

    @abstractmethod
    def get_completion(self, prompt: str, response_format: Optional[Dict] = None, temperature: float = 0.7) -> str:
        """获取LLM完成结果"""
        pass

    def _generate_empty_value(self, schema_type: str, schema_items: Optional[Dict] = None) -> Any:
        """生成空值用于错误处理"""
        if schema_type == "array":
            return []
        elif schema_type == "string":
            return ""
        elif schema_type == "object":
            return {}
        elif schema_type in ["number", "integer"]:
            return 0
        elif schema_type == "boolean":
            return False
        return None

    def _generate_empty_response(self, response_format: Optional[Dict] = None) -> Dict[str, Any]:
        """生成空响应用于错误处理"""
        if not response_format or "json_schema" not in response_format:
            return {}

        schema = response_format["json_schema"]["schema"]
        result = {}

        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                result[prop_name] = self._generate_empty_value(prop_schema["type"],
                                                             prop_schema.get("items"))

        return result


class OpenAIController(BaseLLMController):
    """OpenAI API 控制器"""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not found. Install with: pip install openai")

        self.model = model
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')

        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

        self.client = OpenAI(api_key=self.api_key)

    def get_completion(self, prompt: str, response_format: Optional[Dict] = None, temperature: float = 0.7) -> str:
        try:
            messages = [
                {"role": "system", "content": "You must respond with a JSON object."},
                {"role": "user", "content": prompt}
            ]

            completion_args = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 1000
            }

            if response_format:
                completion_args["response_format"] = response_format

            response = self.client.chat.completions.create(**completion_args)
            return response.choices[0].message.content

        except Exception as e:
            print(f"OpenAI completion error: {e}")
            if response_format:
                return json.dumps(self._generate_empty_response(response_format))
            return ""


class LiteLLMController(BaseLLMController):
    """LiteLLM 统一控制器"""

    def __init__(self, model: str, api_base: Optional[str] = None, api_key: Optional[str] = None):
        if not LITELLM_AVAILABLE:
            raise ImportError("LiteLLM package not found. Install with: pip install litellm")

        self.model = model
        self.api_base = api_base
        self.api_key = api_key or "EMPTY"  # LiteLLM 默认值

    def get_completion(self, prompt: str, response_format: Optional[Dict] = None, temperature: float = 0.7) -> str:
        try:
            completion_args = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You must respond with a JSON object."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature
            }

            if response_format:
                completion_args["response_format"] = response_format
            if self.api_base:
                completion_args["api_base"] = self.api_base
            if self.api_key:
                completion_args["api_key"] = self.api_key

            response = completion(**completion_args)
            return response.choices[0].message.content

        except Exception as e:
            print(f"LiteLLM completion error: {e}")
            if response_format:
                return json.dumps(self._generate_empty_response(response_format))
            return ""


class OllamaController(BaseLLMController):
    """Ollama 本地控制器"""

    def __init__(self, model: str = "llama2"):
        if not LITELLM_AVAILABLE:
            raise ImportError("LiteLLM package required for Ollama. Install with: pip install litellm")

        # 使用 LiteLLM 调用 Ollama
        self.model = f"ollama/{model}" if not model.startswith("ollama/") else model

    def get_completion(self, prompt: str, response_format: Optional[Dict] = None, temperature: float = 0.7) -> str:
        try:
            completion_args = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You must respond with a JSON object."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "api_base": "http://localhost:11434",
                "api_key": "EMPTY"
            }

            if response_format:
                completion_args["response_format"] = response_format

            response = completion(**completion_args)
            return response.choices[0].message.content

        except Exception as e:
            print(f"Ollama completion error: {e}")
            if response_format:
                return json.dumps(self._generate_empty_response(response_format))
            return ""


class SGLangController(BaseLLMController):
    """SGLang 服务器控制器"""

    def __init__(self, model: str = "llama2", sglang_host: str = "http://localhost", sglang_port: int = 30000):
        self.model = model
        self.sglang_host = sglang_host
        self.sglang_port = sglang_port
        self.base_url = f"{sglang_host}:{sglang_port}"

    def get_completion(self, prompt: str, response_format: Optional[Dict] = None, temperature: float = 0.7) -> str:
        try:
            # 准备 SGLang 请求
            payload = {
                "text": prompt,
                "sampling_params": {
                    "temperature": temperature,
                    "max_new_tokens": 1000
                }
            }

            # 添加 JSON schema 支持
            if response_format and "json_schema" in response_format:
                json_schema = response_format["json_schema"]["schema"]
                json_schema_str = json.dumps(json_schema)
                payload["sampling_params"]["json_schema"] = json_schema_str

            # 发送请求到 SGLang 服务器
            response = requests.post(
                f"{self.base_url}/generate",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("text", "")
            else:
                print(f"SGLang server returned status {response.status_code}: {response.text}")
                raise Exception(f"SGLang server error: {response.status_code}")

        except Exception as e:
            print(f"SGLang completion error: {e}")
            if response_format:
                return json.dumps(self._generate_empty_response(response_format))
            return ""


class LLMController:
    """
    LLM 控制器工厂类

    支持多种后端：openai, litellm, ollama, sglang
    """

    def __init__(self,
                 backend: Literal["openai", "litellm", "ollama", "sglang"] = "openai",
                 model: str = "gpt-4o-mini",
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 sglang_host: str = "http://localhost",
                 sglang_port: int = 30000):
        """
        初始化LLM控制器

        Args:
            backend: 后端类型 ("openai", "litellm", "ollama", "sglang")
            model: 模型名称
            api_key: API密钥
            api_base: API基础URL
            sglang_host: SGLang服务器主机
            sglang_port: SGLang服务器端口
        """
        self.backend = backend
        self.model = model

        try:
            if backend == "openai":
                self.llm = OpenAIController(model, api_key)
            elif backend == "litellm":
                self.llm = LiteLLMController(model, api_base, api_key)
            elif backend == "ollama":
                self.llm = OllamaController(model)
            elif backend == "sglang":
                self.llm = SGLangController(model, sglang_host, sglang_port)
            else:
                raise ValueError(f"Unsupported backend: {backend}. Must be 'openai', 'litellm', 'ollama', or 'sglang'")
        except ImportError as e:
            raise ImportError(f"Required package for backend '{backend}' not found: {e}")

    def get_completion(self, prompt: str, response_format: Optional[Dict] = None, temperature: float = 0.7) -> str:
        """获取LLM完成结果"""
        return self.llm.get_completion(prompt, response_format, temperature)