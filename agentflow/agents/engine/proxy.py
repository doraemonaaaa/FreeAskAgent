import os
import json
import base64
import platformdirs
from typing import List, Union, Optional
from pathlib import Path

try:
    from openai import OpenAI
    import openai
except ImportError:
    raise ImportError("请执行 `pip install openai` 安装依赖。")

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

class ProxyAI:
    """
    通用的代理 AI 引擎，适配所有支持 OpenAI 兼容格式的模型
    (包含 Claude, DeepSeek, GPT, Llama, Gemini 等)
    """
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
        self,
        model_string: str = "gpt-4o", # 也可以是 "deepseek-reasoner" 或 "claude-3-5-sonnet"
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        use_cache: bool = True,
        **kwargs
    ):
        self.model_string = model_string
        self.system_prompt = system_prompt
        self.use_cache = use_cache

        # 优先级：构造函数参数 > 环境变量 > 默认值
        final_base_url = base_url or os.getenv("Proxy_API_BASE") or "https://api.gptplus5.com/v1"
        final_api_key = api_key or os.getenv("OPENAI_API_KEY") or "sk-xxx"

        self.client = OpenAI(
            api_key=final_api_key,
            base_url=final_base_url
        )

        # 缓存初始化
        if self.use_cache:
            root = platformdirs.user_cache_dir("proxy_ai_cache")
            os.makedirs(root, exist_ok=True)
            # 注意：此处假设基类 CachedEngine 的逻辑已被正确继承或实现
            # self.cache_path = os.path.join(root, f"{self.model_string}.db")

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(3))
    def generate(self, content: Union[str, List[Union[str, bytes]]], system_prompt=None, **kwargs):
        """
        统一生成入口：自动判断是纯文本还是多模态内容
        """
        sys_prompt = system_prompt if system_prompt else self.system_prompt
        
        # 1. 构建消息体
        messages = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})

        # 2. 处理内容 (多模态或纯文本)
        if isinstance(content, str):
            messages.append({"role": "user", "content": content})
        elif isinstance(content, list):
            formatted_content = self._format_multimodal_content(content)
            messages.append({"role": "user", "content": formatted_content})
        else:
            raise ValueError("Unsupported content type.")

        # 3. 发送请求 (通用接口)
        try:
            # 移除所有特定模型的 if/else 判断，直接透传参数
            # 大多数现代模型（DeepSeek/Claude）都支持标准的 chat.completions.create
            response = self.client.chat.completions.create(
                model=self.model_string,
                messages=messages,
                **kwargs # 透传 temperature, top_p, max_tokens, response_format 等
            )
            
            return response.choices[0].message.content

        except Exception as e:
            print(f"API Error ({self.model_string}): {str(e)}")
            return {"error": type(e).__name__, "message": str(e)}

    def _format_multimodal_content(self, content: List[Union[str, bytes]]) -> List[dict]:
        """
        将列表内容转化为通用的 OpenAI 多模态格式
        """
        formatted = []
        for item in content:
            if isinstance(item, bytes):
                b64_data = base64.b64encode(item).decode('utf-8')
                formatted.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"}
                })
            elif isinstance(item, str):
                formatted.append({"type": "text", "text": item})
        return formatted

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)