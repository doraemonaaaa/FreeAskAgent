"""
Content Analysis Prompts for Embodied Agent

内容分析相关的prompt模板，用于LLM驱动的内容分析。
"""


def build_content_analysis_prompt(content: str) -> str:
    """
    构建内容分析prompt

    Args:
        content: 要分析的内容文本

    Returns:
        完整的分析prompt字符串
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


def get_content_analysis_response_format() -> dict:
    """
    获取内容分析的响应格式定义

    Returns:
        JSON schema响应格式
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "content_analysis",
            "schema": {
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "关键词列表，按重要性排序"
                    },
                    "context": {
                        "type": "string",
                        "description": "内容摘要，包括主题、要点和受众"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "分类标签列表"
                    }
                },
                "required": ["keywords", "context", "tags"],
                "additionalProperties": False
            },
            "strict": True
        }
    }