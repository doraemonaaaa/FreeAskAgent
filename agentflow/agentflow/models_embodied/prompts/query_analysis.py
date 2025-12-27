from typing import Any, List, Dict, Optional
from .tom import TOM_CORE_PROMPT

def QuerynalysisPrompt(available_tools: List[str], toolbox_metadata: Dict[str, Dict[str, Any]], question: str, image_info: Optional[str] = None) -> str:
    if image_info:
        return f"""
Analysis principles:{TOM_CORE_PROMPT}

Task: Analyze the given query with accompanying inputs and determine the skills and tools needed to address it effectively.

Available tools: {available_tools}

Metadata for the tools: {toolbox_metadata}

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
        return f"""
Task: Analyze the given query to determine necessary skills and tools.

Inputs:
- Query: {question}
- Available tools: {available_tools}
- Metadata for tools: {toolbox_metadata}

Instructions:
1. Identify the main objectives in the query.
2. List the necessary skills and tools.
3. For each skill and tool, explain how it helps address the query.
4. Note any additional considerations.

Format your response with a summary of the query, lists of skills and tools with explanations, and a section for additional considerations.

Be biref and precise with insight. 

Embodied Task Perception and Query Analysis:
If the query involves embodied tasks (e.g., navigation, object manipulation), consider the following:
- Perceptual Requirements: Identify what sensory inputs (visual, spatial, etc.) are needed to understand and execute the task.
- Environmental Context: Consider the setting in which the task takes place and how it affects tool and skill selection.
"""