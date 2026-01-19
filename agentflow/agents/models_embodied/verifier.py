import json
import os
import re
from collections.abc import Sequence
from typing import Any, List, Tuple

from ..engine.factory import create_llm_engine
from ..models.formatters import MemoryVerification
from ..models.memory import Memory
from ..utils.utils import get_image_info, append_image_bytes, normalize_image_inputs


class Verifier:
    def __init__(self, llm_engine_name: str, llm_engine_fixed_name: str = "dashscope",
                 toolbox_metadata: dict = None, available_tools: list = None,
                 verbose: bool = False, base_url: str = None, is_multimodal: bool = False,
                 check_model: bool = True, temperature: float = .0):
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
# delete stop in action space
# ### <Stop()>
# - **Description**: Final action to terminate the task. 
# - **Usage**: Execute ONLY when the goal is fully achieved.

    def verificate_context(self, question: str, image_paths: Any, latest_planner_output: str, memory: Memory, previous_verification_log: str = None) -> Any:
        image_info = get_image_info(image_paths) if self.is_multimodal else "Null"
        prev_log_text = previous_verification_log if previous_verification_log else "None (First Step)"
        prompt_memory_verification = f"""
# Role: VLN System Verifier

You are a VERIFIER agent for a Vision-Language Navigation (VLN) system to monitor PLANNER.
Your job is to evaluate whether the CURRENT SUBGOAL has been completed, and CURRENT COMMAND is correct,
based on the planner's latest output and the accumulated navigation memory.

You do NOT generate actions.
You judge progress, detect errors, and provide structured feedback to improve the planner's next decision.
Carefully evaluate all directional commands (e.g., turn, face a direction) and determine if the current subgoal is completed. Explicitly note any subgoals not yet achieved. Provide structured feedback to the planner to prevent repeating unnecessary or meaningless actions, improving navigation efficiency.
---

## Context

### Task Query:
This is task you should finish.
{question}

### ⏱️ PREVIOUS VERIFICATION STATE (CRITICAL)
This is your judgment from the LAST step. Use this to maintain continuity. 
Do NOT mark a subgoal as 'Not Completed' if it was already 'Completed' previously, unless the robot actively undid it.
--------------------------------------------------
{prev_log_text}
--------------------------------------------------

### Evidence
- History: You can have history information from planner's intention, state, belief, and commands, verification is history verification from you, memory length {memory.max_memory_length}
{memory.get_actions()}
- Visual: [From Images provided] 
Analyze the provided images and map data to verify the navigation progress so far. 
Compare the visual evidence with the previous commands to ensure the actual trajectory aligns with the plan and there is no significant deviation.
{image_info}
- Latest Plan: 
{latest_planner_output}

### VLN Rules (Planner uses these)
- Subgoal completion depends on spatial progress, visibility, ego position and rotation, and action consistency.
- Planner beliefs may become outdated after movement.
- Exploration vs Navigating state must be consistent with belief.

---

## Action Space for Planner
By this you can judge whether planner output good command to complete the subgoal.
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
---

## Internal Task Decomposition (Hierarchy)

### 1. High-level Goal
Final target (object, place, or person).

### 2. Subgoals
Ordered spatial or informational steps (e.g., "Pass the red door", "Reach the hallway").

### 3. Progress Check
Determine if current subgoal is reached via visual perception and command history.

## Subgoal Type Rules (CRITICAL)

Each subgoal MUST be categorized into one of the following types:
- Orientation
- Traversal
- Perceptual
- Interaction

### Orientation Subgoal
- Completed immediately after the corresponding turn action is executed once.
- Do NOT wait for visual landmarks.
- Do NOT keep as <Current> for more than one step.

### Traversal Subgoal
- Requires forward/backward movement.
- May stay <Current> across multiple steps until spatial progress is evident.

### Perceptual Subgoal
- Completed only when the target landmark or object is visible.

### Interaction Subgoal
- Completed only after the interaction action is executed successfully.

---

## Your Tasks

### 1. Identify the SUBGOALs

### 2. Decompose the Task into Sequential Subgoals (Independent of Planner)
- Based ONLY on the original Task Instruction (not the planner's output), decompose the navigation into clear, sequential subgoals.
- Use natural, concise Chinese descriptions (matching the task language).
- Typical decomposition for this kind of instruction:
1. Turn around to face the opposite direction (Orientation)
2. Move backward until passing the sign "Working at Sea" (Traversal / Perceptual)
3. Continue forward to reach the first "Me and George European Goods" (Traversal)
4. At the first "Me and George European Goods", turn right toward "Best Price Best Package" (Orientation)
5. Continue forward to reach the second "Me and George European Goods" (Traversal)

### 3. Evaluate Completion INDEPENDENTLY
- For each subgoal, judge Completed / Not Completed / Current (in progress) based ONLY on:
  - Action history (memory)
  - Current visual evidence (signs, landmarks, ego orientation)
  - Common VLN heuristics (visibility, centering, passing, orientation alignment)
  - Top-Down Cost Map and ego-centric image
- The "Current" subgoal should be the earliest Not Completed one in the sequence.
- If the planner is pursuing a wrong/misordered subgoal, explicitly note it as an error.

### 4. Evidence-Based Reasoning
Explicitly cite:
- Which actions support or contradict the subgoal
- What visual cues indicate success or failure
- Whether ego motion aligns with planner intention

---

### Output Format (STRICT) Example:

Subgoal:
- [Subgoal Planning]:
(This is Example)
1. <Turn around to face the opposite direction> <Completed>  (Orientation)
2. <Move forward until passing the area "Working at Sea"> <Current> (Traversal / Perceptual) ← Next Recommended Subgoal 
3. <Continue forward to reach the first "Me and George European Goods"> <Not Started> (Traversal)
4. <At the first "Me and George European Goods", turn right toward "Best Price Best Package"> <Not Started> (Orientation)
5. <Continue forward to reach the second "Me and George European Goods" (destination)> <Not Started> (Traversal)

Last Action Feedback: 
(This is Example)Planner finished task 1, turn around(memory action command is TurnAround)

Planner Feedback:
(This is Example)
- The orientation subgoal of turning around has been successfully completed, as the robot’s heading has been reversed.
- The current active subgoal is to continue moving forward to reach the area "Working at Sea".
- No visual evidence of the target landmark is visible yet, which is expected at this stage.

"""

        input_data = [prompt_memory_verification]

        print(f"[Verifier]: Reveived {len(image_paths)} images")
        append_image_bytes(input_data, image_paths)

        verification = self.llm_engine(input_data)
        return verification

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
