import json
import os
import re
from collections.abc import Sequence
from typing import Any, Dict, List, Optional, Tuple

from ..engine.factory import create_llm_engine
from ..models_embodied.formatters import NextStep, QueryAnalysis
from ..models_embodied.memory.memory import Memory
from ..utils.utils import get_image_info, normalize_image_paths
from ..models_embodied.prompts.vln import vln_prompt
from ..models_embodied.prompts.query_analysis import QuerynalysisPrompt

class Planner:
    def __init__(self, llm_engine_name: str, llm_engine_fixed_name: str = "dashscope",
                 toolbox_metadata: dict = None, available_tools: List = None,
                 verbose: bool = False, base_url: str = None, is_multimodal: bool = False,
                 check_model: bool = True, temperature : float = .0):
        self.llm_engine_name = llm_engine_name
        self.llm_engine_fixed_name = llm_engine_fixed_name
        self.is_multimodal = is_multimodal
        # Allow downstream engines to ingest image bytes when available.
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
        self.toolbox_metadata = toolbox_metadata if toolbox_metadata is not None else {}
        self.available_tools = available_tools if available_tools is not None else []

        self.verbose = verbose
    
    # 调试输出：只打印安全的元信息，不输出原始字节
    def summarize_input_data(self, items):
        summary = []
        for i, item in enumerate(items):
            if isinstance(item, (bytes, bytearray)):
                summary.append({
                    "index": i,
                    "type": "bytes",
                    "length": len(item)
                })
            else:
                # 对长文本做截断，避免日志过长
                s = str(item)
                summary.append({
                    "index": i,
                    "type": type(item).__name__,
                    "preview": (s[:200] + "...") if len(s) > 200 else s
                })
        return summary

    def extract_context_subgoal_and_tool(self, response: Any) -> Tuple[str, str, str]:

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
                except Exception as e:
                    print(f"Failed to parse response as JSON: {str(e)}")
            if isinstance(response, NextStep):
                print("arielg 1")
                context = response.context.strip()
                sub_goal = response.sub_goal.strip()
                tool_name = response.tool_name.strip()
            else:
                print("arielg 2")
                text = response.replace("**", "")

                # Pattern to match the exact format
                pattern = r"Context:\s*(.*?)Sub-Goal:\s*(.*?)Tool Name:\s*(.*?)\s*(?:```)?\s*(?=\n\n|\Z)"

                # Find all matches
                matches = re.findall(pattern, text, re.DOTALL)

                # Return the last match (most recent/relevant)
                context, sub_goal, tool_name = matches[-1]
                context = context.strip()
                sub_goal = sub_goal.strip()
            tool_name = normalize_tool_name(tool_name)
        except Exception as e:
            print(f"Error extracting context, sub-goal, and tool name: {str(e)}")
            return None, None, None

        return context, sub_goal, tool_name
    
    def analyze_query(self, question: str, image: str) -> str:
        image_info = get_image_info(image)

        query_prompt = QuerynalysisPrompt(self.available_tools, self.toolbox_metadata, question, image_info)
        input_data = [query_prompt]
        
        image_paths = normalize_image_paths(image)
        for path in image_paths:
            try:
                with open(path, 'rb') as file:
                    image_bytes = file.read()
                input_data.append(image_bytes)
            except Exception as e:
                print(f"Error reading image file '{path}': {str(e)}")

        print("Input data of `analyze_query()`: ", self.summarize_input_data(input_data))

        # self.query_analysis = self.llm_engine_mm(input_data, response_format=QueryAnalysis)
        self.query_analysis = self.llm_engine(input_data, response_format=QueryAnalysis)
        # self.query_analysis = self.llm_engine_fixed(input_data, response_format=QueryAnalysis)

        return str(self.query_analysis).strip()

    def generate_direct_output(self, question: str, image: str, memory: Memory) -> str:
        image_info = get_image_info(image)
        if self.is_multimodal:
            prompt_generate_final_output = f"""
# Context:
## Query: {question}
## Image: {image_info}

## Actions Taken:
{memory.get_actions()}

## VLN Task Principle Prompt:
{vln_prompt()}

## Tools:
Available tools: {self.available_tools}
Metadata for the tools: {self.toolbox_metadata}
# End of Context
"""

        input_data = [prompt_generate_final_output]
        image_paths = normalize_image_paths(image)
        if len(image_paths) > 1:
            filenames = ", ".join(os.path.basename(path) for path in image_paths)
            input_data.append(
                f"Consider the following {len(image_paths)} frames in chronological order: {filenames}."
            )
        for path in image_paths:
            if not os.path.isfile(path):
                print(f"Warning: image file not found '{path}' - skipping.")
                continue
            try:
                with open(path, 'rb') as file:
                    image_bytes = file.read()
                input_data.append(image_bytes)
            except Exception as e:
                print(f"Error reading image file '{path}': {str(e)}")

        final_output = self.llm_engine(input_data)
        # final_output = self.llm_engine_fixed(input_data)
        # final_output = self.llm_engine_mm(input_data)

        return final_output
    