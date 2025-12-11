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
FAST_MODE = True

if FAST_MODE:
    solver = construct_fast_solver(
        llm_engine_name=llm_engine_name,
        enabled_tools=["Base_Generator_Tool"],
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
        enabled_tools=["Base_Generator_Tool"],
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

# Solve the user query with the frame sequence (oldest -> newest)
# output = solver.solve("What is the capital of France?")
navigation_agent_prompt = """
【任务描述】
根据场景和任务目标生成5步导航指令序列。

【目标】
你当前位于户外场景中，需要前往“面包店”。
如果你不知道面包店的位置，你可以在“距离人类小于 2 米”时执行动作“问路”。

【可执行动作空间】
动作 = [前进, 左转, 右转, 后转, 后退, 停止, 问路]
距离 = [1m, 2m, 3m]
每一步指令必须严格是一个“动作 + 距离”的组合，例如：
- 前进2m
- 右转1m
- 问路（若问路不需要距离则写：问路）

【导航规划要求】
- 必须避开所有可见障碍物。
- 输出连续5步导航序列。
- 若在图像中发现“人类”并且导航距离会进入2m范围，可选择动作“问路”。
- 如果场景中没有人，则仅根据路径规划输出动作。

【最终输出内容】
导航指令序列（共5步，每步一个“动作 + 距离”）
"""

output = solver.solve(
    navigation_agent_prompt,
    image_paths=image_sequence[:5],  # take up to 5 chronological frames
)

print(output.get("direct_output"))


