import json
import os
import re
from collections.abc import Sequence
from typing import Any, Dict, List, Optional, Tuple

from ..engine.factory import create_llm_engine
from ..models.formatters import NextStep, QueryAnalysis
from ..models.memory import Memory
from ..utils.utils import get_image_info, normalize_image_paths

class Planner:
    def __init__(self, llm_engine_name: str, llm_engine_fixed_name: str = "dashscope",
                 toolbox_metadata: dict = None, available_tools: List = None,
                 verbose: bool = False, base_url: str = None, is_multimodal: bool = False,
                 check_model: bool = True, temperature : float = .0,
                 use_amem: bool = True, retriever_config: dict = None):
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

        # A-MEM集成
        self.use_amem = use_amem
        self.retriever_config = retriever_config or {}
        self.retriever = None
        self.historical_memories = []

        # 初始化A-MEM检索器
        if self.use_amem:
            self._init_amem_retriever()

    def _init_amem_retriever(self):
        """初始化A-MEM检索器用于历史记忆检索"""
        try:
            from ..models.memory.hybrid_retriever import HybridRetriever

            self.retriever = HybridRetriever(
                use_api_embedding=self.retriever_config.get('use_api_embedding', True),
                alpha=self.retriever_config.get('alpha', 0.5)
            )

            # 加载历史记忆数据（如果有的话）
            self._load_historical_memories()

            if self.verbose:
                print("✅ Planner A-MEM retriever initialized successfully")

        except ImportError as e:
            if self.verbose:
                print(f"⚠️  A-MEM retriever not available: {e}")
            self.use_amem = False
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Failed to initialize A-MEM retriever: {e}")
            self.use_amem = False

    def _load_historical_memories(self):
        """加载历史记忆数据用于检索"""
        # 这里可以从配置文件或数据库加载历史记忆
        # 暂时初始化为空列表，之后可以通过add_historical_memory方法添加
        self.historical_memories = []

        # 如果有持久化的记忆文件，可以在这里加载
        # 例如：从agentic_memory_system加载共享的记忆
        pass

    def add_historical_memory(self, memory_content: str):
        """添加历史记忆到检索器"""
        if self.use_amem and self.retriever and memory_content:
            try:
                self.historical_memories.append(memory_content)
                self.retriever.add_documents([memory_content])
                if self.verbose:
                    print(f"✅ Added historical memory to planner retriever")
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Failed to add historical memory: {e}")

    def _retrieve_relevant_memories(self, query: str, k: int = 3) -> List[str]:
        """
        检索相关历史记忆

        Args:
            query: 查询字符串（通常是当前任务描述）
            k: 返回的记忆数量

        Returns:
            List[str]: 相关记忆内容的列表
        """
        if not self.use_amem or not self.retriever:
            return []

        try:
            # 执行混合检索
            indices = self.retriever.retrieve(query, k=k)

            # 转换索引为记忆内容
            relevant_memories = []
            for idx in indices:
                if 0 <= idx < len(self.historical_memories):
                    memory_content = self.historical_memories[idx]
                    relevant_memories.append(memory_content)

            if self.verbose and relevant_memories:
                print(f"📚 Retrieved {len(relevant_memories)} relevant memories for query: '{query[:50]}...'")

            return relevant_memories

        except Exception as e:
            if self.verbose:
                print(f"⚠️  Memory retrieval failed: {e}")
            return []

    def _format_memories_for_prompt(self, memories: List[str]) -> str:
        """
        将记忆格式化为适合注入prompt的形式

        Args:
            memories: 记忆内容列表

        Returns:
            str: 格式化的记忆字符串
        """
        if not memories:
            return ""

        formatted_memories = []
        for i, memory in enumerate(memories, 1):
            # 截断过长的记忆以避免prompt过长
            truncated_memory = memory[:200] + "..." if len(memory) > 200 else memory
            formatted_memories.append(f"{i}. {truncated_memory}")

        return "\n".join(formatted_memories)
    
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

    def generate_base_response(self, question: str, image: str, max_tokens: int = 2048) -> str:
        image_info = get_image_info(image)

        input_data = [question]
        image_paths = normalize_image_paths(image)
        if len(image_paths) > 1:
            filenames = ", ".join(os.path.basename(path) for path in image_paths)
            input_data.append(
                f"The following {len(image_paths)} frames are provided in chronological order (oldest to newest): {filenames}."
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

        print("Input data of `generate_base_response()` (summary):", self.summarize_input_data(input_data))

        self.base_response = self.llm_engine(input_data, max_tokens=max_tokens)
        # self.base_response = self.llm_engine_fixed(input_data, max_tokens=max_tokens)

        return self.base_response

    def analyze_query(self, question: str, image: str) -> str:
        image_info = get_image_info(image)

        if self.is_multimodal:
            query_prompt = f"""
Task: Analyze the given query with accompanying inputs and determine the skills and tools needed to address it effectively.

Available tools: {self.available_tools}

Metadata for the tools: {self.toolbox_metadata}

Image: {image_info}

Query: {question}

Instructions:
1. Carefully read and understand the query and any accompanying inputs.
2. Identify the main objectives or tasks within the query.
3. List the specific skills that would be necessary to address the query comprehensively.
4. Examine the available tools in the toolbox and determine which ones might relevant and useful for addressing the query. Make sure to consider the user metadata for each tool, including limitations and potential applications (if available).
5. Provide a brief explanation for each skill and tool you've identified, describing how it would contribute to answering the query.

Your response should include:
1. A concise summary of the query's main points and objectives, as well as content in any accompanying inputs.
2. A list of required skills, with a brief explanation for each.
3. A list of relevant tools from the toolbox, with a brief explanation of how each tool would be utilized and its potential limitations.
4. Any additional considerations that might be important for addressing the query effectively.

Please present your analysis in a clear, structured format.
                        """
        else: 
            query_prompt = f"""
Task: Analyze the given query to determine necessary skills and tools.

Inputs:
- Query: {question}
- Available tools: {self.available_tools}
- Metadata for tools: {self.toolbox_metadata}

Instructions:
1. Identify the main objectives in the query.
2. List the necessary skills and tools.
3. For each skill and tool, explain how it helps address the query.
4. Note any additional considerations.

Format your response with a summary of the query, lists of skills and tools with explanations, and a section for additional considerations.

Be biref and precise with insight. 
"""

        input_data = [query_prompt]
        image_paths = normalize_image_paths(image)
        if len(image_paths) > 1:
            filenames = ", ".join(os.path.basename(path) for path in image_paths)
            input_data.append(
                f"The accompanying sequence contains {len(image_paths)} frames in chronological order: {filenames}."
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

        print("Input data of `analyze_query()`: ", self.summarize_input_data(input_data))

        # self.query_analysis = self.llm_engine_mm(input_data, response_format=QueryAnalysis)
        # self.query_analysis = self.llm_engine(input_data, response_format=QueryAnalysis)
        self.query_analysis = self.llm_engine_fixed(input_data, response_format=QueryAnalysis)

        return str(self.query_analysis).strip()

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

    def generate_next_step(self, question: str, image: str, query_analysis: str, memory: Memory, step_count: int, max_step_count: int, json_data: Any = None) -> Any:
        # 检索相关历史记忆
        relevant_memories = self._retrieve_relevant_memories(question, k=3)
        formatted_memories = self._format_memories_for_prompt(relevant_memories)
        if self.is_multimodal:
            prompt_generate_next_step = f"""
Task: Determine the optimal next step to address the given query based on the provided analysis, available tools, and previous steps taken.

Context:
Query: {question}
Image: {image}
Query Analysis: {query_analysis}

Available Tools:
{self.available_tools}

Tool Metadata:
{self.toolbox_metadata}

Previous Steps and Their Results:
{memory.get_actions()}

Relevant Historical Memories:
{formatted_memories}

Current Step: {step_count} in {max_step_count} steps
Remaining Steps: {max_step_count - step_count}

Instructions:
1. Analyze the context thoroughly, including the query, its analysis, any image, available tools and their metadata, and previous steps taken.

2. Determine the most appropriate next step by considering:
- Key objectives from the query analysis
- Capabilities of available tools
- Logical progression of problem-solving
- Outcomes from previous steps
- Current step count and remaining steps

3. Select ONE tool best suited for the next step, keeping in mind the limited number of remaining steps.

4. Formulate a specific, achievable sub-goal for the selected tool that maximizes progress towards answering the query.

Response Format:
Your response MUST follow this structure:
1. Justification: Explain your choice in detail.
2. Context, Sub-Goal, and Tool: Present the context, sub-goal, and the selected tool ONCE with the following format:

Context: <context>
Sub-Goal: <sub_goal>
Tool Name: <tool_name>

Where:
- <context> MUST include ALL necessary information for the tool to function, structured as follows:
* Relevant data from previous steps
* File names or paths created or used in previous steps (list EACH ONE individually)
* Variable names and their values from previous steps' results
* Any other context-specific information required by the tool
- <sub_goal> is a specific, achievable objective for the tool, based on its metadata and previous outcomes.
It MUST contain any involved data, file names, and variables from Previous Steps and Their Results that the tool can act upon.
- <tool_name> MUST be the exact name of a tool from the available tools list.

Rules:
- Select only ONE tool for this step.
- The sub-goal MUST directly address the query and be achievable by the selected tool.
- The Context section MUST include ALL necessary information for the tool to function, including ALL relevant file paths, data, and variables from previous steps.
- The tool name MUST exactly match one from the available tools list: {self.available_tools}.
- Avoid redundancy by considering previous steps and building on prior results.
- Your response MUST conclude with the Context, Sub-Goal, and Tool Name sections IN THIS ORDER, presented ONLY ONCE.
- Include NO content after these three sections.

Example (do not copy, use only as reference):
Justification: [Your detailed explanation here]
Context: Image path: "example/image.jpg", Previous detection results: [list of objects]
Sub-Goal: Detect and count the number of specific objects in the image "example/image.jpg"
Tool Name: Object_Detector_Tool

Remember: Your response MUST end with the Context, Sub-Goal, and Tool Name sections, with NO additional content afterwards.
                        """
        else:
            prompt_generate_next_step = f"""
Task: Determine the optimal next step to address the query using available tools and previous steps.

Context:
- **Query:** {question}
- **Query Analysis:** {query_analysis}
- **Available Tools:** {self.available_tools}
- **Toolbox Metadata:** {self.toolbox_metadata}
- **Previous Steps:** {memory.get_actions()}
- **Relevant Historical Memories:** {formatted_memories}

Instructions:
1. Analyze the query, previous steps, and available tools.
2. Select the **single best tool** for the next step.
3. Formulate a specific, achievable **sub-goal** for that tool.
4. Provide all necessary **context** (data, file names, variables) for the tool to function.

Response Format:
1.  **Justification:** Explain your choice of tool and sub-goal.
2.  **Context:** Provide all necessary information for the tool.
3.  **Sub-Goal:** State the specific objective for the tool.
4.  **Tool Name:** State the exact name of the selected tool.

Rules:
- Select only ONE tool.
- The sub-goal must be directly achievable by the selected tool.
- The Context section must contain all information the tool needs to function.
- The response must end with the Context, Sub-Goal, and Tool Name sections in that order, with no extra content.
                    """
            
        next_step = self.llm_engine(prompt_generate_next_step, response_format=NextStep)
        if json_data is not None:
            json_data[f"action_predictor_{step_count}_prompt"] = prompt_generate_next_step
            json_data[f"action_predictor_{step_count}_response"] = str(next_step)
        return next_step


    def generate_final_output(self, question: str, image: str, memory: Memory) -> str:
        image_info = get_image_info(image)
        if self.is_multimodal:
            prompt_generate_final_output = f"""
Task: Generate the final output based on the query, image, and tools used in the process.

Context:
Query: {question}
Image: {image_info}
Actions Taken:
{memory.get_actions()}

Instructions:
1. Review the query, image, and all actions taken during the process.
2. Consider the results obtained from each tool execution.
3. Incorporate the relevant information from the memory to generate the step-by-step final output.
4. The final output should be consistent and coherent using the results from the tools.

Output Structure:
Your response should be well-organized and include the following sections:

1. Summary:
   - Provide a brief overview of the query and the main findings.

2. Detailed Analysis:
   - Break down the process of answering the query step-by-step.
   - For each step, mention the tool used, its purpose, and the key results obtained.
   - Explain how each step contributed to addressing the query.

3. Key Findings:
   - List the most important discoveries or insights gained from the analysis.
   - Highlight any unexpected or particularly interesting results.

4. Answer to the Query:
   - Directly address the original question with a clear and concise answer.
   - If the query has multiple parts, ensure each part is answered separately.

5. Additional Insights (if applicable):
   - Provide any relevant information or insights that go beyond the direct answer to the query.
   - Discuss any limitations or areas of uncertainty in the analysis.

6. Conclusion:
   - Summarize the main points and reinforce the answer to the query.
   - If appropriate, suggest potential next steps or areas for further investigation.
"""
        else:
                prompt_generate_final_output = f"""
Task: Generate the final output based on the query and the results from all tools used.

Context:
- **Query:** {question}
- **Actions Taken:** {memory.get_actions()}

Instructions:
1. Review the query and the results from all tool executions.
2. Incorporate the relevant information to create a coherent, step-by-step final output.
"""

        input_data = [prompt_generate_final_output]
        image_paths = normalize_image_paths(image)
        if len(image_paths) > 1:
            filenames = ", ".join(os.path.basename(path) for path in image_paths)
            input_data.append(
                f"The final output should consider {len(image_paths)} frames provided in chronological order: {filenames}."
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

        # final_output = self.llm_engine_mm(input_data)
        # final_output = self.llm_engine(input_data)
        final_output = self.llm_engine_fixed(input_data)

        return final_output


    def generate_direct_output(self, question: str, image: str, memory: Memory, relevant_memories: Optional[List[Dict[str, Any]]] = None) -> str:
        image_info = get_image_info(image)
        if self.is_multimodal:
            prompt_generate_final_output = f"""
Context:
Query: {question}
Image: {image_info}
Initial Analysis:
{self.query_analysis}
Actions Taken:
{memory.get_actions()}

Please generate the concise output based on the query, image information, initial analysis, and actions taken. Break down the process into clear, logical, and conherent steps. Conclude with a precise and direct answer to the query.

Answer:
"""
        else:
            # Format relevant memories for context
            memory_context = ""
            if relevant_memories:
                memory_items = []
                for mem in relevant_memories:
                    if isinstance(mem, dict) and 'content' in mem:
                        # Extract original content, removing processing artifacts
                        content = mem['content']
                        if isinstance(content, str):
                            # Clean up processed content by taking the first part before keywords
                            original_content = content.split('。')[0] + '。' if '。' in content else content
                            memory_items.append(f"- {original_content.strip()}")
                if memory_items:
                    memory_context = "\n- **Relevant Memories:**\n" + "\n".join(memory_items)

            prompt_generate_final_output = f"""
Task: Generate a concise final answer to the query based on all provided context.

Context:
- **Query:** {question}
- **Initial Analysis:** {self.query_analysis}{memory_context}
- **Actions Taken:** {memory.get_actions()}

Instructions:
1. Review the query and the results from all actions.
2. Synthesize the key findings into a clear, step-by-step summary of the process.
3. Provide a direct, precise answer to the original query.

Output Structure:
1.  **Process Summary:** A clear, step-by-step breakdown of how the query was addressed, including the purpose and key results of each action.
2.  **Answer:** A direct and concise final answer to the query.
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