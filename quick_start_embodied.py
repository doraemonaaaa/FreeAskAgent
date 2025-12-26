# Import the solver
import os
from pathlib import Path

from agentflow.agentflow.solver_embodied import construct_solver_embodied

from dotenv import load_dotenv
load_dotenv(dotenv_path="agentflow/.env")
print("Proxy_API_BASE:" + os.environ.get("Proxy_API_BASE", "Not Set"))
print("OPENAI_API_KEY:" + os.environ.get("OPENAI_API_KEY", "Not Set"))
print("DASHSCOPE_API_KEY:" + os.environ.get("DASHSCOPE_API_KEY", "Not Set"))

# Set the LLM engine name
# llm_engine_name = "dashscope" # you can use "gpt-4o" as well
llm_engine_name = "gpt-4o"

solver = construct_solver_embodied(
    llm_engine_name=llm_engine_name,
    enabled_tools=["Base_Generator_Tool", "GroundedSAM2_Tool"],
    tool_engine=["gpt-4o"],
    model_engine=["gpt-4o", "gpt-4o", "gpt-4o"],
    output_types="direct",
    max_time=300,
    max_steps=1,
    enable_multimodal=True
)

# Prepare an ordered image sequence so the agent can perceive motion
frame_dir = Path("test/vln")
# image_sequence = sorted(str(path) for path in frame_dir.glob("frame_*.jpeg"))
image_sequence = None
if not image_sequence:
    image_sequence = ["test/vln/input_img1.jpg"]

navigation_task_prompt = """"
Go to the store, called micheal's store.
"""

output = solver.solve(
    navigation_task_prompt,
    image_paths=image_sequence[:5],  # take up to 5 chronological frames
)

print(output.get("direct_output"))


