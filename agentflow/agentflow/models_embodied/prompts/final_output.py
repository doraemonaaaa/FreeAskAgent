"""
Final Output Prompts for Embodied Agent

最终输出相关的prompt模板，包括多模态和文本模式的输出生成。
"""

import os
from pathlib import Path


def _load_prompt_template(template_name: str) -> str:
    """
    从文件加载prompt模板

    Args:
        template_name: 模板文件名（不含扩展名）

    Returns:
        prompt模板字符串
    """
    template_path = Path(__file__).parent / f"{template_name}.txt"
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def build_multimodal_final_output_prompt(question: str, image_info: str, memory_context: str, actions_taken: str, available_tools: str, toolbox_metadata: str) -> str:
    """
    构建多模态最终输出prompt

    Args:
        question: 用户问题
        image_info: 图像信息
        memory_context: 记忆上下文
        actions_taken: 已执行的动作
        available_tools: 可用工具
        toolbox_metadata: 工具元数据

    Returns:
        完整的prompt字符串
    """
    from .vln import vln_prompt

    template = _load_prompt_template("final_output_multimodal")
    return template.format(
        question=question,
        image_info=image_info,
        memory_context=memory_context,
        actions_taken=actions_taken,
        vln_prompt=vln_prompt(),
        available_tools=available_tools,
        toolbox_metadata=toolbox_metadata
    )


def build_text_final_output_prompt_with_memory(question: str, memory_context: str) -> str:
    """
    构建带记忆的文本最终输出prompt

    Args:
        question: 用户问题
        memory_context: 记忆上下文

    Returns:
        完整的prompt字符串
    """
    template = _load_prompt_template("final_output_with_memory")
    return template.format(
        memory_context=memory_context,
        question=question
    )


def build_text_final_output_prompt_simple(question: str) -> str:
    """
    构建简单文本最终输出prompt

    Args:
        question: 用户问题

    Returns:
        完整的prompt字符串
    """
    template = _load_prompt_template("final_output_simple")
    return template.format(question=question)
