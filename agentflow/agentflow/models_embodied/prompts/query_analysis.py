from typing import Any, List, Dict, Optional

def QuerynalysisPrompt(available_tools: List[str], toolbox_metadata: Dict[str, Dict[str, Any]], question: str, image_info: Optional[str] = None, memory_context: Optional[str] = None) -> str:
    """
    Produce a compact analysis prompt that asks the model to:
      - Summarize the query and inputs
      - List required skills and relevant tools (brief)
      - Provide short recommendations for next steps
    Keep language neutral and avoid sensitive instructions.
    """
    base_inputs = f"- Query: {question}\n- Available tools: {available_tools}\n- Tool metadata: {toolbox_metadata}"
    if image_info:
        base_inputs = f"Image info: {image_info}\n" + base_inputs

    if memory_context:
        base_inputs += f"\nMemory context: {memory_context}"

    return f"""
Task: Analyze the query and inputs and produce a concise, structured analysis.

Inputs:
{base_inputs}

Instructions:
1. Provide a one-line summary of the query.
2. List 2-4 required skills or capabilities (one line each).
3. List 1-3 relevant tools and a one-line note on use/limitations.
4. Give one short recommended next step.

Keep responses neutral, factual, and concise.
"""