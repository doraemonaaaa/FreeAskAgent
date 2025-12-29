#!/usr/bin/env python3
"""
Light-weight test script to compare solver behavior with and without A‑MEM (memory).

Usage:
  - Put all model names / base URLs and API keys into:
      /root/autodl-tmp/FreeAskAgent/.env
  - This script reads that .env and runs two short solves:
      1) without memory (use_amem=False)
      2) with memory    (use_amem=True)
  - It prints both outputs and a simple comparison.

Environment (.env) variable suggestions (fill API keys yourself):
  # Text model (LLM) - used for planning / reasoning
  TEXT_MODEL_NAME=gpt-4o
  TEXT_MODEL_BASE_URL=https://your-openai-compatible-endpoint.example/v1
  TEXT_MODEL_API_KEY=your-text-model-api-key

  # Image / multimodal model (if different from text model)
  IMAGE_MODEL_NAME=gpt-4o
  IMAGE_MODEL_BASE_URL=https://your-multimodal-endpoint.example/v1
  IMAGE_MODEL_API_KEY=your-image-model-api-key

Notes:
  - Quick test; uses the existing AgentFlow `construct_solver` factory.
  - All network/auth parameters are read from /root/autodl-tmp/FreeAskAgent/.env.
  - The script keeps text and image model distinction; set IMAGE_MODEL_* if you want a different endpoint.
"""

import os
import time
from dotenv import load_dotenv
from pathlib import Path

# Load configuration from the project's .env (absolute path required)
ENV_PATH = "/root/autodl-tmp/FreeAskAgent/.env"
load_dotenv(dotenv_path=ENV_PATH)

# Read model configuration (user should supply API keys in .env)
TEXT_MODEL_NAME = os.environ.get("TEXT_MODEL_NAME") or os.environ.get("MODEL") or "gpt-4o"
TEXT_MODEL_BASE_URL = os.environ.get("TEXT_MODEL_BASE_URL") or os.environ.get("OPENAI_API_BASE")
TEXT_MODEL_API_KEY = os.environ.get("TEXT_MODEL_API_KEY") or os.environ.get("OPENAI_API_KEY")

IMAGE_MODEL_NAME = os.environ.get("IMAGE_MODEL_NAME") or TEXT_MODEL_NAME
IMAGE_MODEL_BASE_URL = os.environ.get("IMAGE_MODEL_BASE_URL") or TEXT_MODEL_BASE_URL
IMAGE_MODEL_API_KEY = os.environ.get("IMAGE_MODEL_API_KEY") or TEXT_MODEL_API_KEY

print("Loaded model configuration:")
print(" TEXT_MODEL_NAME:", TEXT_MODEL_NAME)
print(" TEXT_MODEL_BASE_URL:", TEXT_MODEL_BASE_URL)
print(" IMAGE_MODEL_NAME:", IMAGE_MODEL_NAME)
print(" IMAGE_MODEL_BASE_URL:", IMAGE_MODEL_BASE_URL)
print(" (API keys must be filled in the .env file)")

# Import the solver factory from AgentFlow
from agentflow.agentflow.solver import construct_solver

# Simple navigation prompt (copied & shortened from quick_start)
NAV_PROMPT = """
[Task]
请描述视野中的可行动作并选出后续一连串的导航轨迹指令
你要去面包店
[Rules]
要躲避物体不要撞上
当你离人2m内的时候就可以触发问路
[Output Format]
请给出后续5步的导航指令序列。
"""

# Single test runner that constructs a solver and runs one solve
def run_single_test(use_memory: bool, test_name: str, image_paths=None):
    print(f"\n--- Running: {test_name} (use_memory={use_memory}) ---")
    # Construct solver with text model; pass base_url so factory can use the endpoint
    solver = construct_solver(
        llm_engine_name=TEXT_MODEL_NAME,
        enabled_tools=["Base_Generator_Tool", "GroundedSAM2_Tool"],
        tool_engine=[TEXT_MODEL_NAME],
        model_engine=[TEXT_MODEL_NAME, TEXT_MODEL_NAME, TEXT_MODEL_NAME, TEXT_MODEL_NAME],
        output_types="direct",
        max_time=120,
        max_steps=1,
        enable_multimodal=True,
        use_amem=use_memory,
        retriever_config=None,
        base_url=TEXT_MODEL_BASE_URL,
        temperature=0.0,
    )

    # Run the solver; capture wall time
    start = time.time()
    try:
        result = solver.solve(NAV_PROMPT, image_paths=image_paths)
    except Exception as e:
        print("Solver raised exception:", type(e).__name__, e)
        return {"error": str(e)}
    duration = time.time() - start
    print(f"Finished {test_name} in {duration:.2f}s")
    # result is a dict; extract the direct output if present
    direct_output = result.get("direct_output") if isinstance(result, dict) else None
    print("Direct output (truncated):", (direct_output or "")[:400])
    return {"result": result, "duration": duration}

def compare_runs(no_mem_res, mem_res):
    print("\n=== Comparison ===")
    no_text = (no_mem_res.get("result") or {}).get("direct_output", "")
    mem_text = (mem_res.get("result") or {}).get("direct_output", "")
    print("No-memory direct length:", len(no_text))
    print("With-memory direct length:", len(mem_text))
    if no_text == mem_text:
        print("No visible difference in direct output.")
    else:
        print("Outputs differ. Showing head of each:")
        print("\n-- No memory --\n", (no_text or "")[:500])
        print("\n-- With memory --\n", (mem_text or "")[:500])

def main():
    # Prepare an example image path (optional). If no multimodal model is available you can leave None.
    example_image = "/root/autodl-tmp/FreeAskAgent/input_img1.jpg"
    image_paths = [example_image] if Path(example_image).exists() else None

    no_memory = run_single_test(use_memory=False, test_name="No Memory", image_paths=image_paths)
    with_memory = run_single_test(use_memory=True, test_name="With Memory", image_paths=image_paths)

    compare_runs(no_memory, with_memory)

if __name__ == "__main__":
    main()


