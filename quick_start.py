# Import the solver
import os
import time
from pathlib import Path

from agentflow.agentflow.solver_fast import construct_fast_solver
from agentflow.agentflow.solver import construct_solver

from dotenv import load_dotenv
load_dotenv(dotenv_path="agentflow/.env")
print("Proxy_API_BASE:" + os.environ.get("Proxy_API_BASE", "Not Set"))
print("OPENAI_API_KEY:" + os.environ.get("OPENAI_API_KEY", "Not Set"))
print("DASHSCOPE_API_KEY:" + os.environ.get("DASHSCOPE_API_KEY", "Not Set"))

# Set the LLM engine name
# llm_engine_name = "dashscope" # you can use "gpt-4o" as well
llm_engine_name = "gpt-4o"

# Construct the solver
# Fast MODE Only go to planner
FAST_MODE = False

if FAST_MODE:
    solver = construct_fast_solver(
        llm_engine_name=llm_engine_name,
        enabled_tools=["Base_Generator_Tool", "SAM2_Perception_Tool"],
        tool_engine=["gpt-4o"],
        output_types="direct",
        max_steps=1,
        max_time=10,
        max_tokens=1024,
        fast_max_tokens=256,
        enable_multimodal=True,
        verbose=True
    )
else:
    solver = construct_solver(
        llm_engine_name=llm_engine_name,
        enabled_tools=["Base_Generator_Tool", "SAM2_Perception_Tool"],
        tool_engine=["gpt-4o"],
        model_engine=["gpt-4o", "gpt-4o", "gpt-4o", "gpt-4o"],
        output_types="direct",
        max_time=300,
        max_steps=1,
        enable_multimodal=True
    )

# Prepare an ordered image sequence so the agent can perceive motion
frame_dir = Path("/home/pengyh/workspace/FreeAskAgent")
# image_sequence = sorted(str(path) for path in frame_dir.glob("frame_*.jpeg"))
image_sequence = None
if not image_sequence:
    image_sequence = ["/home/pengyh/workspace/FreeAskAgent/input_img1.jpg"]

navigation_task_prompt = """"
[Task]
请描述视野中的可行动作并选出后续一连串的导航轨迹指令
你要去面包店
[Rules]
要躲避物体不要撞上
当你离人2m内的时候就可以触发问路
[Policy]
使用最快获取信息的策略，你可选择自己不断探索地点，也可以问人来快速获取信息，尽管可能不精准
[Action Space]
动作空间是[前进，左转，右转，后转, 后退, 停止, 问路][1m, 2m, 3m]
每次动作只能选择一个动作和一个距离, 比如'前进2m'
[Output Format]
请给出后续5步的导航指令序列。
"""

output = solver.solve(
    navigation_task_prompt,
    image_paths=image_sequence[:5],  # take up to 5 chronological frames
)

print(output.get("direct_output"))


