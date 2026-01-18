import json
import os
import re
from collections.abc import Sequence
from typing import Any, Dict, List, Optional, Tuple

from ..engine.factory import create_llm_engine
from ..models_embodied.formatters import NextStep, QueryAnalysis
from ..models_embodied.memory.memory import Memory
from ..utils.utils import get_image_info, append_image_bytes
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
    
    def analyze_query(self, question: str, image_paths: Any) -> str:
        image_info = get_image_info(image_paths)
        query_prompt = QuerynalysisPrompt(self.available_tools, self.toolbox_metadata, question, image_info)
        input_data = [query_prompt]
        append_image_bytes(input_data, image_paths)
        print("Input data of `analyze_query()`: ", self.summarize_input_data(input_data))

        # self.query_analysis = self.llm_engine_mm(input_data, response_format=QueryAnalysis)
        self.query_analysis = self.llm_engine(input_data, response_format=QueryAnalysis)
        # self.query_analysis = self.llm_engine_fixed(input_data, response_format=QueryAnalysis)

        return str(self.query_analysis).strip()

    def generate_direct_output(self, question: str, image_paths: Any, memory: Memory, latest_verifier_sug: str = "") -> str:  # image_paths 改为 Any
        image_info = get_image_info(image_paths) if self.is_multimodal else "Null"

        prompt = f"""
# Context:
You are an embodied agent. Follow the specific guidance provided by your navigation supervisor.
You are a Planner Agent to plan the trajectory which robot to go.

## Image: {image_info}

## Verification Feedback Integration
The Verifier Agent provides course correction based on global map data. 
Please prioritize the Verifier's feedback to update your path planning. 
If the Verifier indicates a subgoal is complete, proceed to the next step.
{latest_verifier_sug}

## PLANNING OBJECTIVE
Translate the Verifier’s high-level guidance into:
- short-horizon,
- collision-free,
- atomic actions
that make continuous progress toward the current subgoal.

## OBSTACLE-AWARE EXECUTION POLICY
- The Verifier provides approximate direction or intent, NOT exact motion.
- You MUST use perception to avoid collisions, walls, or getting stuck.
- Temporary local deviations are ALLOWED if needed to bypass obstacles.

### Allowed Local Adjustment Example:
- Verifier: “Move forward”
- You detect an obstacle ahead-left
- Valid plan:
  <MoveRight()>
  <Forward(1)>
  <TurnLeft(15)>
  <Forward(1)>

### Forbidden Behavior:
- Turning around to re-check a completed subgoal
- Oscillating left/right without net progress
- Repeating the same action when no movement occurs
- Blindly moving forward into obstacles

## EXECUTION HISTORY (for reference only)
(length = {memory.max_memory_length})
{memory.get_actions()}

## TASK CONTEXT
Original Task ( this have processed by verifier):
{question}

## Tools:
Available tools: {self.available_tools}
Metadata for the tools: {self.toolbox_metadata}
# End of Context

# VLN Task Principles

This describe the detail of vln task principles
---

## Action Space

### <Forward(n)>  # Move forward n meter
### <TurnLeft(n)>  # Turn left n degree
### <TurnRight(n)>  # Turn Right n degree
### <TurnAround()>  # Turn around for 180 degree

### <Ask(text)>
- **Pre-condition**: MUST ONLY be used if a Human is visible AND distance < 2.0 meters.
- **Usage**: Request social or goal-related info (e.g., "Where is the kitchen?", "Are you the person I'm looking for?").
- **Example**: `<Ask("Excuse me, could you tell me where the apple is?")>`

### <Wait(t)>
- **Description**: Pause execution for a specific duration.
- **Parameters**: `t` [Float/Int] in seconds.
- **Example**: `<Wait(5)>`

### <Stop()>
- **Description**: Final action to terminate the task. 
- **Usage**: Execute ONLY when the goal is fully achieved.

---

## States
- Navigating: Target is visible or location is known.
- Exploration: Target location unknown.
  - Self-Exploration: Scan environment.
  - Ask-Strategy: If a pedestrian is encountered naturally, you may ask for directions.

---

## Priority Decision Hierarchy (Strict Order)
- Verifier Override: If the Supervisor says a subgoal is <Completed>, you MUST NOT repeat actions for it. Move to the next subgoal.
- Instruction Alignment: Follow the specific landmarks in the instruction (e.g., "Flowers from Mainland").
- Directional Knowledge(8 Directions): 
All directions are defined relative to the agent’s current rotation.
-- Forward → 0°
-- Turn Around(转身，往回走，往后走) / Backward → 180°
-- Turn Left → −90°
-- Turn Right → +90°
-- Left Forward(左前) → −45°
-- Right Forward(右前) → +45° 
-- Left Backward(左后) → −135°
-- Right Backward(右后) → +135°

## Theory of Mind (concise)

Description must ONLY describe what is visible in the image.

Verifier Instruction:
The instruction about what verifier want me do something.

Robot Belief:
- What do verifier want me to do ?
- Do I know where the target landmark is?
- Is a human visible?

User Belief:
- What does the verifier think about the target location?
- What is the target?

Intention:
- What should I do next to reach the goal?

---

## Output Format (STRICT, Must Use this Format)

Description:
...

Verifier Instruction:
...

Belief:
- Verifier_Goal:
- Target_Visibility:
- Human_Visibility:
- Localization_Confidence:

Intention:
...

State:
<...>

Action:
Six step from ##Action Space
(Example)
1. <TurnLeft(a)>
2. <Forward(b)>
3. <Forward(c)>
4. <TurnRight(d)>
5. <Forward(e)>
6. <Ask(f)>

# End of VLN Task Principles
"""
        # print("Debug Planner Prompt: " + prompt)

        input_data = [prompt]
        append_image_bytes(input_data, image_paths)

        final_output = self.llm_engine(input_data)
        # final_output = self.llm_engine_fixed(input_data)
        # final_output = self.llm_engine_mm(input_data)

        return final_output
    