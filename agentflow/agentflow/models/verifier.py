import json
import os
import re
from collections.abc import Sequence
from typing import Any, List, Tuple

from ..engine.factory import create_llm_engine
from ..models.formatters import MemoryVerification
from ..models.memory import Memory
from ..utils.utils import get_image_info, normalize_image_paths


class Verifier:
    def __init__(self, llm_engine_name: str, llm_engine_fixed_name: str = "dashscope",
                 toolbox_metadata: dict = None, available_tools: list = None,
                 verbose: bool = False, base_url: str = None, is_multimodal: bool = False,
                 check_model: bool = True, temperature: float = .0,
                 use_amem: bool = True, retriever_config: dict = None):
        self.llm_engine_name = llm_engine_name
        self.llm_engine_fixed_name = llm_engine_fixed_name
        self.is_multimodal = is_multimodal
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
        self.verification_memories = []

        # 初始化A-MEM检索器
        if self.use_amem:
            self._init_amem_retriever()

    def _init_amem_retriever(self):
        """初始化A-MEM检索器用于验证辅助"""
        try:
            from ..models.memory.hybrid_retriever import HybridRetriever

            self.retriever = HybridRetriever(
                use_api_embedding=self.retriever_config.get('use_api_embedding', True),
                alpha=self.retriever_config.get('alpha', 0.5)
            )

            # 加载验证相关的历史记忆
            self._load_verification_memories()

            if self.verbose:
                print("✅ Verifier A-MEM retriever initialized successfully")

        except ImportError as e:
            if self.verbose:
                print(f"⚠️  A-MEM retriever not available: {e}")
            self.use_amem = False
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Failed to initialize A-MEM retriever: {e}")
            self.use_amem = False

    def _load_verification_memories(self):
        """加载验证相关的历史记忆"""
        # 这里可以加载之前验证成功的案例、失败的教训等
        self.verification_memories = []
        pass

    def add_verification_memory(self, verification_case: str):
        """添加验证记忆到检索器"""
        if self.use_amem and self.retriever and verification_case:
            try:
                self.verification_memories.append(verification_case)
                self.retriever.add_documents([verification_case])
                if self.verbose:
                    print(f"✅ Added verification memory to verifier retriever")
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Failed to add verification memory: {e}")

    def _get_similar_historical_verifications(self, current_context: str, k: int = 2) -> List[str]:
        """
        获取历史上类似的验证案例

        Args:
            current_context: 当前验证上下文（问题+分析+当前步骤）
            k: 返回的验证案例数量

        Returns:
            List[str]: 相关的历史验证案例
        """
        if not self.use_amem or not self.retriever:
            return []

        try:
            # 使用当前上下文检索相关验证历史
            indices = self.retriever.retrieve(current_context, k=k)

            # 转换索引为验证案例内容
            similar_cases = []
            for idx in indices:
                if 0 <= idx < len(self.verification_memories):
                    verification_case = self.verification_memories[idx]
                    similar_cases.append(verification_case)

            if self.verbose and similar_cases:
                print(f"📋 Retrieved {len(similar_cases)} similar verification cases")

            return similar_cases

        except Exception as e:
            if self.verbose:
                print(f"⚠️  Historical verification retrieval failed: {e}")
            return []

    def _format_verification_memories_for_prompt(self, memories: List[str]) -> str:
        """
        将验证记忆格式化为适合注入prompt的形式

        Args:
            memories: 验证记忆内容列表

        Returns:
            str: 格式化的验证记忆字符串
        """
        if not memories:
            return "No relevant historical verification cases found."

        formatted_memories = []
        for i, memory in enumerate(memories, 1):
            # 截断过长的记忆
            truncated_memory = memory[:300] + "..." if len(memory) > 300 else memory
            formatted_memories.append(f"Case {i}: {truncated_memory}")

        return "Relevant historical verification cases:\n" + "\n".join(formatted_memories)

    def verificate_context(self, question: str, image: str, query_analysis: str, memory: Memory, step_count: int = 0, json_data: Any = None) -> Any:
        image_info = get_image_info(image)

        # 检索相关历史验证案例
        current_verification_context = f"Query: {question}, Analysis: {query_analysis}, Actions: {memory.get_actions()}"
        similar_verifications = self._get_similar_historical_verifications(current_verification_context, k=2)
        formatted_verifications = self._format_verification_memories_for_prompt(similar_verifications)
        if self.is_multimodal:
            prompt_memory_verification = f"""
Task: Thoroughly evaluate the completeness and accuracy of the memory for fulfilling the given query, considering the potential need for additional tool usage.

Context:
Query: {question}
Image: {image_info}
Available Tools: {self.available_tools}
Toolbox Metadata: {self.toolbox_metadata}
Initial Analysis: {query_analysis}
Memory (tools used and results): {memory.get_actions()}

Historical Verification Cases:
{formatted_verifications}

Detailed Instructions:
1. Carefully analyze the query, initial analysis, and image (if provided):
   - Identify the main objectives of the query.
   - Note any specific requirements or constraints mentioned.
   - If an image is provided, consider its relevance and what information it contributes.

2. Review the available tools and their metadata:
   - Understand the capabilities and limitations and best practices of each tool.
   - Consider how each tool might be applicable to the query.

3. Examine the memory content in detail:
   - Review each tool used and its execution results.
   - Assess how well each tool's output contributes to answering the query.

4. Critical Evaluation (address each point explicitly):
   a) Completeness: Does the memory fully address all aspects of the query?
      - Identify any parts of the query that remain unanswered.
      - Consider if all relevant information has been extracted from the image (if applicable).

   b) Unused Tools: Are there any unused tools that could provide additional relevant information?
      - Specify which unused tools might be helpful and why.

   c) Inconsistencies: Are there any contradictions or conflicts in the information provided?
      - If yes, explain the inconsistencies and suggest how they might be resolved.

   d) Verification Needs: Is there any information that requires further verification due to tool limitations?
      - Identify specific pieces of information that need verification and explain why.

   e) Ambiguities: Are there any unclear or ambiguous results that could be clarified by using another tool?
      - Point out specific ambiguities and suggest which tools could help clarify them.

5. Final Determination:
   Based on your thorough analysis, decide if the memory is complete and accurate enough to generate the final output, or if additional tool usage is necessary.

Response Format:

If the memory is complete, accurate, AND verified:
Explanation:
<Provide a detailed explanation of why the memory is sufficient. Reference specific information from the memory and explain its relevance to each aspect of the task. Address how each main point of the query has been satisfied.>

Conclusion: STOP

If the memory is incomplete, insufficient, or requires further verification:
Explanation:
<Explain in detail why the memory is incomplete. Identify specific information gaps or unaddressed aspects of the query. Suggest which additional tools could be used, how they might contribute, and why their input is necessary for a comprehensive response.>

Conclusion: CONTINUE

IMPORTANT: Your response MUST end with either 'Conclusion: STOP' or 'Conclusion: CONTINUE' and nothing else. Ensure your explanation thoroughly justifies this conclusion.
"""
        else:
            prompt_memory_verification = f"""
Task: Evaluate if the current memory is complete and accurate enough to answer the query, or if more tools are needed.

Context:
- **Query:** {question}
- **Available Tools:** {self.available_tools}
- **Toolbox Metadata:** {self.toolbox_metadata}
- **Initial Analysis:** {query_analysis}
- **Memory (Tools Used & Results):** {memory.get_actions()}
- **Historical Verification Cases:** {formatted_verifications}

Instructions:
1.  Review the query, initial analysis, and memory.
2.  Assess the completeness of the memory: Does it fully address all parts of the query?
3.  Check for potential issues:
    -   Are there any inconsistencies or contradictions?
    -   Is any information ambiguous or in need of verification?
4.  Determine if any unused tools could provide missing information.

Final Determination:
-   If the memory is sufficient, explain why and conclude with "STOP".
-   If more information is needed, explain what's missing, which tools could help, and conclude with "CONTINUE".

IMPORTANT: The response must end with either "Conclusion: STOP" or "Conclusion: CONTINUE".
"""

        input_data = [prompt_memory_verification]
        image_paths = normalize_image_paths(image)
        if len(image_paths) > 1:
            filenames = ", ".join(os.path.basename(path) for path in image_paths)
            input_data.append(
                f"The verification should consider {len(image_paths)} frames in chronological order: {filenames}."
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

        stop_verification = self.llm_engine_fixed(input_data, response_format=MemoryVerification)
        if json_data is not None:
            json_data[f"verifier_{step_count}_prompt"] = input_data
            json_data[f"verifier_{step_count}_response"] = str(stop_verification)
        return stop_verification

    def extract_conclusion(self, response: Any) -> Tuple[str, str]:
        if isinstance(response, str):
            # Attempt to parse the response as JSON
            try:
                response_dict = json.loads(response)
                response = MemoryVerification(**response_dict)
            except Exception as e:
                print(f"Failed to parse response as JSON: {str(e)}")
        if isinstance(response, MemoryVerification):
            analysis = response.analysis
            stop_signal = response.stop_signal
            if stop_signal:
                return analysis, 'STOP'
            else:
                return analysis, 'CONTINUE'
        else:
            analysis = response
            pattern = r'conclusion\**:?\s*\**\s*(\w+)'
            matches = list(re.finditer(pattern, response, re.IGNORECASE | re.DOTALL))
            if matches:
                conclusion = matches[-1].group(1).upper()
                if conclusion in ['STOP', 'CONTINUE']:
                    return analysis, conclusion

            # If no valid conclusion found, search for STOP or CONTINUE anywhere in the text
            if 'stop' in response.lower():
                return analysis, 'STOP'
            elif 'continue' in response.lower():
                return analysis, 'CONTINUE'
            else:
                print("No valid conclusion (STOP or CONTINUE) found in the response. Continuing...")
                return analysis, 'CONTINUE'