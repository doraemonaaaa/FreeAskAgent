import json
import os
import re
from collections.abc import Sequence
from typing import Any, Dict, List, Optional, Tuple

from ..engine.factory import create_llm_engine
from ..models.formatters import NextStep, QueryAnalysis
from ..models.memory import Memory
from ..utils.utils import get_image_info, normalize_image_paths
from ..models_embodied.prompts.vln import vln_prompt

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
    
#     def analyze_query(self, question: str, image: str) -> str:
#         image_info = get_image_info(image)

#         if self.is_multimodal:
#             query_prompt = f"""
# Task: Analyze the given query with accompanying inputs and determine the skills and tools needed to address it effectively.

# Available tools: {self.available_tools}

# Metadata for the tools: {self.toolbox_metadata}

# Image: {image_info}

# Query: {question}

# Instructions:
# 1. Carefully read and understand the query and any accompanying inputs.
# 2. Identify the main objectives or tasks within the query.
# 3. List the specific skills that would be necessary to address the query comprehensively.
# 4. Examine the available tools in the toolbox and determine which ones might relevant and useful for addressing the query. Make sure to consider the user metadata for each tool, including limitations and potential applications (if available).
# 5. Provide a brief explanation for each skill and tool you've identified, describing how it would contribute to answering the query.

# Your response should include:
# 1. A concise summary of the query's main points and objectives, as well as content in any accompanying inputs.
# 2. A list of required skills, with a brief explanation for each.
# 3. A list of relevant tools from the toolbox, with a brief explanation of how each tool would be utilized and its potential limitations.
# 4. Any additional considerations that might be important for addressing the query effectively.

# Please present your analysis in a clear, structured format.
#                         """
#         else: 
#             query_prompt = f"""
# Task: Analyze the given query to determine necessary skills and tools.

# Inputs:
# - Query: {question}
# - Available tools: {self.available_tools}
# - Metadata for tools: {self.toolbox_metadata}

# Instructions:
# 1. Identify the main objectives in the query.
# 2. List the necessary skills and tools.
# 3. For each skill and tool, explain how it helps address the query.
# 4. Note any additional considerations.

# Format your response with a summary of the query, lists of skills and tools with explanations, and a section for additional considerations.

# Be biref and precise with insight. 
# """
#         input_data = [query_prompt]
#         if image_info:
#             try:
#                 with open(image_info["image_path"], 'rb') as file:
#                     image_bytes = file.read()
#                 input_data.append(image_bytes)
#             except Exception as e:
#                 print(f"Error reading image file: {str(e)}")

#         print("Input data of `analyze_query()`: ", input_data)

#         # self.query_analysis = self.llm_engine_mm(input_data, response_format=QueryAnalysis)
#         # self.query_analysis = self.llm_engine(input_data, response_format=QueryAnalysis)
#         self.query_analysis = self.llm_engine_fixed(input_data, response_format=QueryAnalysis)

#         return str(self.query_analysis).strip()

    def generate_direct_output(self, question: str, image: str, memory: Memory) -> str:
        image_info = get_image_info(image)
        if self.is_multimodal:
            prompt_generate_final_output = f"""
Context:
Query: {question}
Image: {image_info}

Actions Taken:
{memory.get_actions()}

Please generate the concise output based on the query, image information, initial analysis, and actions taken. Break down the process into clear, logical, and conherent steps. Conclude with a precise and direct answer to the query.

Task Prompt:
{vln_prompt()}

Tools:
Available tools: {self.available_tools}
Metadata for the tools: {self.toolbox_metadata}
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

        # final_output = self.llm_engine(input_data)
        final_output = self.llm_engine_fixed(input_data)
        # final_output = self.llm_engine_mm(input_data)

        return final_output

