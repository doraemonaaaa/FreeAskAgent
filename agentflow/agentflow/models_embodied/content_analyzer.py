"""
Content Analyzer for A-MEM

实现LLM驱动的内容分析功能，自动提取关键词、上下文和标签。
支持降级机制，当LLM不可用时提供默认值。
"""

import json
import re
from typing import Dict, List, Optional, Any
import sys
import os

from .llm_controllers import LLMController


class ContentAnalyzer:
    """
    内容分析器

    使用LLM自动分析记忆内容，提取关键词、上下文和标签信息。
    支持错误处理和降级机制。
    """

    def __init__(self, llm_controller: Optional[LLMController] = None):
        """
        初始化内容分析器

        Args:
            llm_controller: LLM控制器，如果为None则使用降级模式
        """
        self.llm_controller = llm_controller
        self.llm_available = llm_controller is not None

    def analyze_content(self, content: str) -> Dict[str, Any]:
        """
        分析内容，提取元数据

        Args:
            content: 要分析的内容文本

        Returns:
            包含keywords、context、tags的字典
        """
        if self.llm_available and self.llm_controller:
            return self._analyze_with_llm(content)
        else:
            return self._analyze_fallback(content)

    def _analyze_with_llm(self, content: str) -> Dict[str, Any]:
        """
        使用LLM进行内容分析

        Args:
            content: 要分析的内容

        Returns:
            分析结果字典
        """
        prompt = self._build_analysis_prompt(content)

        try:
            response = self.llm_controller.get_completion(
                prompt=prompt,
                response_format=self._get_response_format(),
                temperature=0.3  # 较低温度以获得更一致的结果
            )

            return self._parse_llm_response(response)

        except Exception as e:
            print(f"LLM content analysis failed: {e}")
            return self._analyze_fallback(content)

    def _build_analysis_prompt(self, content: str) -> str:
        """
        构建分析提示

        Args:
            content: 要分析的内容

        Returns:
            完整的分析提示
        """
        return f"""Generate a structured analysis of the following content by:
1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)
2. Extracting core themes and contextual elements
3. Creating relevant categorical tags

Format the response as a JSON object:
{{
    "keywords": [
        // several specific, distinct keywords that capture key concepts and terminology
        // Order from most to least important
        // Don't include keywords that are the name of the speaker or time
        // At least three keywords, but don't be too redundant.
    ],
    "context":
        // one sentence summarizing:
        // - Main topic/domain
        // - Key arguments/points
        // - Intended audience/purpose
    ,
    "tags": [
        // several broad categories/themes for classification
        // Include domain, format, and type tags
        // At least three tags, but don't be too redundant.
    ]
}}

Content for analysis:
{content}"""

    def _get_response_format(self) -> Dict[str, Any]:
        """获取响应格式定义"""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "content_analysis",
                "schema": {
                    "type": "object",
                    "properties": {
                        "keywords": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "关键词列表"
                        },
                        "context": {
                            "type": "string",
                            "description": "上下文摘要"
                        },
                        "tags": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "标签列表"
                        },
                    },
                    "required": ["keywords", "context", "tags"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        解析LLM响应

        Args:
            response: LLM原始响应

        Returns:
            解析后的分析结果
        """
        try:
            # 清理响应文本
            response_cleaned = response.strip()

            # 尝试找到JSON内容
            if not response_cleaned.startswith('{'):
                start_idx = response_cleaned.find('{')
                if start_idx != -1:
                    response_cleaned = response_cleaned[start_idx:]

            if not response_cleaned.endswith('}'):
                end_idx = response_cleaned.rfind('}')
                if end_idx != -1:
                    response_cleaned = response_cleaned[:end_idx+1]

            # 解析JSON
            analysis = json.loads(response_cleaned)

            # 验证必需字段
            required_fields = ["keywords", "context", "tags"]
            for field in required_fields:
                if field not in analysis:
                    analysis[field] = self._get_default_value(field)

            # 确保字段类型正确
            analysis["keywords"] = analysis.get("keywords", [])
            analysis["context"] = analysis.get("context", "General")
            analysis["tags"] = analysis.get("tags", [])

            # 转换为正确类型
            if isinstance(analysis["keywords"], list):
                analysis["keywords"] = [str(k) for k in analysis["keywords"]]
            else:
                analysis["keywords"] = []

            if not isinstance(analysis["context"], str):
                analysis["context"] = str(analysis["context"])

            if isinstance(analysis["tags"], list):
                analysis["tags"] = [str(t) for t in analysis["tags"]]
            else:
                analysis["tags"] = []

            return analysis

        except json.JSONDecodeError as e:
            print(f"JSON parsing error in content analysis: {e}")
            print(f"Raw response: {response}")
            return self._analyze_fallback("")
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return self._analyze_fallback("")

    def _analyze_fallback(self, content: str) -> Dict[str, Any]:
        """
        降级分析方法（当LLM不可用时使用）

        Args:
            content: 要分析的内容

        Returns:
            默认分析结果
        """
        # 简单的基于规则的关键词提取
        keywords = self._extract_keywords_fallback(content)

        # 简单的上下文推断
        context = self._infer_context_fallback(content)

        # 简单的标签生成
        tags = self._generate_tags_fallback(content)

        return {
            "keywords": keywords,
            "context": context,
            "tags": tags
        }

    def _extract_keywords_fallback(self, content: str) -> List[str]:
        """
        降级关键词提取

        Args:
            content: 内容文本

        Returns:
            关键词列表
        """
        if not content:
            return ["general"]

        # 简单的中文分词模拟（实际应用中可以使用jieba等库）
        # 这里使用简单的空格和标点分割
        words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', content.lower())

        # 过滤常见停用词和单字词
        stop_words = {'的', '了', '和', '是', '就', '都', '而', '及', '与', '着', '或', '一个', '没有', '这个', '那个'}
        keywords = [word for word in words if len(word) > 1 and word not in stop_words]

        # 去重并限制数量
        unique_keywords = list(dict.fromkeys(keywords))  # 保持顺序去重
        return unique_keywords[:5] if unique_keywords else ["general"]

    def _infer_context_fallback(self, content: str) -> str:
        """
        降级上下文推断

        Args:
            content: 内容文本

        Returns:
            上下文描述
        """
        if not content:
            return "General content"

        # 根据关键词判断主题
        content_lower = content.lower()

        if any(word in content_lower for word in ['时代广场', '超市', '购物', '商场']):
            return "Shopping and location information"
        elif any(word in content_lower for word in ['技术', '代码', '编程', '开发']):
            return "Technology and programming related content"
        elif any(word in content_lower for word in ['学习', '教育', '课程']):
            return "Educational content"
        else:
            return "General information content"

    def _generate_tags_fallback(self, content: str) -> List[str]:
        """
        降级标签生成

        Args:
            content: 内容文本

        Returns:
            标签列表
        """
        tags = ["general"]

        if not content:
            return tags

        content_lower = content.lower()

        # 基于内容添加标签
        if any(word in content_lower for word in ['时代广场', '地点', '位置', '地址']):
            tags.extend(["location", "place"])
        if any(word in content_lower for word in ['超市', '购物', '商场', '商店']):
            tags.extend(["shopping", "commerce"])
        if any(word in content_lower for word in ['技术', '代码', '编程']):
            tags.extend(["technology", "programming"])
        if any(word in content_lower for word in ['学习', '教育']):
            tags.extend(["education", "learning"])

        # 去重
        return list(dict.fromkeys(tags))

    def _get_default_value(self, field: str) -> Any:
        """
        获取字段的默认值

        Args:
            field: 字段名

        Returns:
            默认值
        """
        defaults = {
            "keywords": [],
            "context": "General",
            "tags": []
        }
        return defaults.get(field, None)

    def update_llm_controller(self, llm_controller: LLMController) -> None:
        """
        更新LLM控制器

        Args:
            llm_controller: 新的LLM控制器
        """
        self.llm_controller = llm_controller
        self.llm_available = llm_controller is not None
