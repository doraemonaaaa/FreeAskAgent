"""
Content Analyzer for A-MEM

实现LLM驱动的内容分析功能，自动提取关键词、上下文和标签。
支持降级机制，当LLM不可用时提供默认值。
"""

import json
import re
from typing import Dict, List, Any
import logging

from ...engine.factory import create_llm_engine
from ..prompts.content_analysis import build_content_analysis_prompt
from .interfaces import ContentAnalyzerInterface, BaseMemoryComponent


class ContentAnalyzer(BaseMemoryComponent, ContentAnalyzerInterface):
    """
    内容分析器 - 简洁实现

    使用LLM自动分析记忆内容，提取关键词、上下文和标签信息。
    支持错误处理和降级机制。
    """

    def __init__(self, llm_engine_name: str = "gpt-4o", temperature: float = 0.3):
        super().__init__()
        self.llm_engine_name = llm_engine_name
        self.temperature = temperature
        self.llm_engine = None
        self.llm_available = False

        self._setup_logging()
        self._init_llm_engine()

    def _setup_logging(self) -> None:
        """设置日志"""
        self.logger = logging.getLogger('ContentAnalyzer')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _init_llm_engine(self) -> None:
        """初始化LLM引擎"""
        try:
            self.llm_engine = create_llm_engine(
                model_string=self.llm_engine_name,
                temperature=self.temperature
            )
            self.llm_available = True
        except Exception:
            self.llm_engine = None
            self.llm_available = False

    def analyze_content(self, content: str) -> Dict[str, Any]:
        """分析内容，提取元数据"""
        if self.llm_available and self.llm_engine:
            return self._analyze_with_llm(content)
        else:
            return self._analyze_fallback(content)

    def _analyze_with_llm(self, content: str) -> Dict[str, Any]:
        """使用LLM进行内容分析"""
        prompt = build_content_analysis_prompt(content)

        try:
            response = self.llm_engine.generate(prompt)
            return self._parse_llm_response(response)
        except Exception:
            return self._analyze_fallback(content)

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """解析LLM响应"""
        try:
            if not response:
                return self._analyze_fallback("")

            # 清理和解析JSON
            response_cleaned = self._clean_json_response(response)
            analysis = json.loads(response_cleaned)

            # 规范化结果
            return self._normalize_analysis_result(analysis)

        except (json.JSONDecodeError, Exception):
            return self._analyze_fallback("")

    def _clean_json_response(self, response: str) -> str:
        """清理JSON响应文本"""
        response_cleaned = response.strip()

        if not response_cleaned.startswith('{'):
            start_idx = response_cleaned.find('{')
            if start_idx != -1:
                response_cleaned = response_cleaned[start_idx:]

        if not response_cleaned.endswith('}'):
            end_idx = response_cleaned.rfind('}')
            if end_idx != -1:
                response_cleaned = response_cleaned[:end_idx+1]

        return response_cleaned

    def _normalize_analysis_result(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """规范化分析结果"""
        return {
            "keywords": analysis.get("keywords", []) if isinstance(analysis.get("keywords"), list) else [],
            "context": analysis.get("context", "General") if isinstance(analysis.get("context"), str) else "General",
            "tags": analysis.get("tags", []) if isinstance(analysis.get("tags"), list) else []
        }

    def _analyze_fallback(self, content: str) -> Dict[str, Any]:
        """降级分析方法"""
        return {
            "keywords": self._extract_keywords_fallback(content),
            "context": self._infer_context_fallback(content),
            "tags": self._generate_tags_fallback(content)
        }

    def _extract_keywords_fallback(self, content: str) -> List[str]:
        """降级关键词提取"""
        if not content:
            return ["general"]

        words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', content.lower())
        stop_words = {'的', '了', '和', '是', '就', '都', '而', '及', '与', '着', '或'}

        keywords = [word for word in words if len(word) > 1 and word not in stop_words]
        unique_keywords = list(dict.fromkeys(keywords))

        return unique_keywords[:5] if unique_keywords else ["general"]

    def _infer_context_fallback(self, content: str) -> str:
        """降级上下文推断"""
        if not content:
            return "General content"

        content_lower = content.lower()

        if any(word in content_lower for word in ['时代广场', '超市', '购物']):
            return "Shopping and location information"
        elif any(word in content_lower for word in ['技术', '代码', '编程']):
            return "Technology and programming related content"
        elif any(word in content_lower for word in ['学习', '教育']):
            return "Educational content"
        else:
            return "General information content"

    def _generate_tags_fallback(self, content: str) -> List[str]:
        """降级标签生成"""
        tags = ["general"]

        if not content:
            return tags

        content_lower = content.lower()

        if any(word in content_lower for word in ['时代广场', '地点', '位置']):
            tags.extend(["location", "place"])
        if any(word in content_lower for word in ['超市', '购物', '商场']):
            tags.extend(["shopping", "commerce"])
        if any(word in content_lower for word in ['技术', '代码', '编程']):
            tags.extend(["technology", "programming"])
        if any(word in content_lower for word in ['学习', '教育']):
            tags.extend(["education", "learning"])

        return list(dict.fromkeys(tags))

    def update_llm_engine(self, llm_engine_name: str, temperature: float = 0.3) -> None:
        """更新LLM引擎"""
        self.llm_engine_name = llm_engine_name
        self.temperature = temperature
        self._init_llm_engine()
