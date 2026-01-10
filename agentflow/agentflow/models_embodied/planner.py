import json
import os
import re
import logging
from collections.abc import Sequence
from typing import Any, Dict, List, Optional, Tuple, Union

from ..engine.factory import create_llm_engine
from ..models_embodied.formatters import NextStep, QueryAnalysis
from .memory.short_memory import ShortMemory
from ..utils.utils import get_image_info, normalize_image_paths
from .prompts.vln import vln_prompt
from .prompts.query_analysis import QuerynalysisPrompt
from .prompts.final_output import (
    build_multimodal_final_output_prompt,
    build_text_final_output_prompt_with_memory,
    build_text_final_output_prompt_simple
)

class Planner:
    """
    Embodied Agent规划器

    负责查询分析、工具规划和最终输出生成。
    集成短期记忆和长期记忆机制，支持对话历史的检索和总结。
    """

    # 默认配置常量
    DEFAULT_TEMPERATURE = 0.0
    DEFAULT_MAX_RETRIES = 3

    def __init__(self,
                 llm_engine_name: str,
                 llm_engine_fixed_name: str = "dashscope",
                 toolbox_metadata: Optional[Dict[str, Any]] = None,
                 available_tools: Optional[List[str]] = None,
                 verbose: bool = False,
                 base_url: Optional[str] = None,
                 is_multimodal: bool = False,
                 check_model: bool = True,
                 temperature: float = DEFAULT_TEMPERATURE,
                 max_retries: int = DEFAULT_MAX_RETRIES):
        """
        初始化规划器

        Args:
            llm_engine_name: 主LLM引擎名称
            llm_engine_fixed_name: 固定LLM引擎名称（用于特定任务）
            toolbox_metadata: 工具箱元数据
            available_tools: 可用工具列表
            verbose: 是否启用详细日志
            base_url: LLM API基础URL
            is_multimodal: 是否支持多模态输入
            check_model: 是否检查模型可用性
            temperature: 生成温度
            max_retries: 最大重试次数
        """
        # 配置参数
        self.llm_engine_name = llm_engine_name
        self.llm_engine_fixed_name = llm_engine_fixed_name
        self.is_multimodal = is_multimodal
        self.temperature = temperature
        self.max_retries = max_retries
        self.toolbox_metadata = toolbox_metadata or {}
        self.available_tools = available_tools or []

        # 初始化日志
        self.logger = logging.getLogger('Planner')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG if verbose else logging.INFO)

        # 初始化LLM引擎
        try:
            self.llm_engine_fixed = create_llm_engine(
                model_string=llm_engine_fixed_name,
                is_multimodal=is_multimodal,
                temperature=temperature
            )
            self.llm_engine = create_llm_engine(
                model_string=llm_engine_name,
                is_multimodal=is_multimodal,
                base_url=base_url,
                temperature=temperature
            )
            self.logger.info(f"Planner initialized with engines: {llm_engine_name}, {llm_engine_fixed_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM engines: {e}")
            raise

    def extract_context_subgoal_and_tool(self, response: Union[str, Dict[str, Any]]) -> Tuple[Optional[str], Optional[str], Optional[str]]:

        def normalize_tool_name(tool_name: str) -> str:
            """
            Normalizes a tool name robustly using regular expressions.
            It handles any combination of spaces and underscores as separators.
            """
            def to_canonical(name: str) -> str:
                # Split the name by any sequence of one or more spaces or underscores
                parts = re.split('[ _]+', name)
                # Join the parts with a single underscore and convert to lowercase
                return "_".join(part.lower() for part in parts)

            normalized_input = to_canonical(tool_name)
            
            for tool in self.available_tools:
                if to_canonical(tool) == normalized_input:
                    return tool
                    
            return f"No matched tool given: {tool_name}"

        try:
            if isinstance(response, str):
                # Attempt to parse the response as JSON
                try:
                    response_dict = json.loads(response)
                    response = NextStep(**response_dict)
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse response as JSON: {e}")
                    return None, None, None

            if isinstance(response, NextStep):
                context = response.context.strip()
                sub_goal = response.sub_goal.strip()
                tool_name = response.tool_name.strip()
            else:
                # Parse text response
                text = str(response).replace("**", "")

                # Pattern to match the exact format
                pattern = r"Context:\s*(.*?)Sub-Goal:\s*(.*?)Tool Name:\s*(.*?)\s*(?:```)?\s*(?=\n\n|\Z)"
                matches = re.findall(pattern, text, re.DOTALL)

                if not matches:
                    self.logger.warning("No matches found in response text")
                    return None, None, None

                # Return the last match (most recent/relevant)
                context, sub_goal, tool_name = matches[-1]
                context = context.strip()
                sub_goal = sub_goal.strip()

            tool_name = normalize_tool_name(tool_name)
            return context, sub_goal, tool_name

        except Exception as e:
            self.logger.error(f"Error extracting context, sub-goal, and tool name: {e}")
            return None, None, None

        return context, sub_goal, tool_name
    
    def analyze_query(self, question: str, image: str, relevant_memories: Optional[List[Dict[str, Any]]] = None) -> str:
        image_info = get_image_info(image)


        query_prompt = QuerynalysisPrompt(self.available_tools, self.toolbox_metadata, question, image_info, "")
        input_data = [query_prompt]
        
        image_paths = normalize_image_paths(image)
        for path in image_paths:
            try:
                with open(path, 'rb') as file:
                    image_bytes = file.read()
                input_data.append(image_bytes)
            except Exception as e:
                self.logger.warning(f"Error reading image file '{path}': {str(e)}")

        self.logger.debug(f"Input data summary: {self.summarize_input_data(input_data)}")

        self.query_analysis = self.llm_engine(input_data, response_format=QueryAnalysis)

        return str(self.query_analysis).strip()

    def summarize_input_data(self, input_data: List[Any]) -> str:
        """
        汇总输入数据用于调试

        Args:
            input_data: 输入数据列表

        Returns:
            汇总字符串
        """
        summary_parts = []
        for i, item in enumerate(input_data):
            if isinstance(item, str):
                summary_parts.append(f"Text[{i}]: {len(item)} chars")
            elif isinstance(item, bytes):
                summary_parts.append(f"Bytes[{i}]: {len(item)} bytes")
            else:
                summary_parts.append(f"Other[{i}]: {type(item).__name__}")
        return " | ".join(summary_parts)

    def generate_direct_output(self, question: str, image: str, memory: ShortMemory, relevant_memories: Optional[List[Dict[str, Any]]] = None) -> str:
        image_info = get_image_info(image)

        # Format relevant memories for context
        memory_context = ""
        if relevant_memories:
            memory_items = []
            for mem in relevant_memories:
                if isinstance(mem, dict):
                    # Priority: original_content > content
                    content = mem.get('original_content') or mem.get('content', '')
                    if isinstance(content, str) and len(content.strip()) > 0:
                        # Clean up content - remove processing artifacts
                        clean_content = content
                        # Remove English parts that are processing artifacts
                        if ' Shopping' in clean_content:
                            clean_content = clean_content.split(' Shopping')[0]
                        if ' general' in clean_content:
                            clean_content = clean_content.split(' general')[0]
                        if ' commerce' in clean_content:
                            clean_content = clean_content.split(' commerce')[0]

                        # Only keep meaningful Chinese content
                        if len(clean_content.strip()) > 3 and any('\u4e00' <= char <= '\u9fff' for char in clean_content):
                            memory_items.append(clean_content.strip())

            if memory_items:
                memory_context = "\n\n重要提示：请基于以下已知信息回答问题：\n" + "\n".join([f"• {item}" for item in memory_items]) + "\n\n这些信息是准确的，请直接使用它们来回答用户的问题。"

        if self.is_multimodal:
            prompt_generate_final_output = build_multimodal_final_output_prompt(
                question=question,
                image_info=image_info,
                memory_context=memory_context,
                actions_taken=str(memory.actions),
                available_tools=str(self.available_tools),
                toolbox_metadata=str(self.toolbox_metadata)
            )
        else:
            # For non-multimodal text-only queries
            if memory_context:
                prompt_generate_final_output = build_text_final_output_prompt_with_memory(
                    question=question,
                    memory_context=memory_context
                )
            else:
                prompt_generate_final_output = build_text_final_output_prompt_simple(question=question)

        input_data = [prompt_generate_final_output]
        image_paths = normalize_image_paths(image)
        if len(image_paths) > 1:
            filenames = ", ".join(os.path.basename(path) for path in image_paths)
            input_data.append(
                f"Consider the following {len(image_paths)} frames in chronological order: {filenames}."
            )
        for path in image_paths:
            if not os.path.isfile(path):
                self.logger.warning(f"Image file not found '{path}' - skipping.")
                continue
            try:
                with open(path, 'rb') as file:
                    image_bytes = file.read()
                input_data.append(image_bytes)
            except Exception as e:
                self.logger.warning(f"Error reading image file '{path}': {str(e)}")

        final_output = self.llm_engine(input_data)

        return final_output

