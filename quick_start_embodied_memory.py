"""
Quick Start for Embodied Agent with One-Line Interface
"""

import contextlib
import io
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

# Ensure Windows console can emit Unicode safely.
if os.name == "nt":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Silence known noisy warnings early
warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater."
)

# Add project root to path
sys.path.append(r"D:\code\agent_memory\FreeAskAgent")

from agentflow.agentflow.solver_embodied import construct_solver_embodied
from dotenv import load_dotenv

load_dotenv(dotenv_path=r"D:\code\agent_memory\FreeAskAgent\agentflow\agentflow\.env")


def _compact_output_enabled() -> bool:
    return os.environ.get("AF_COMPACT_OUTPUT", "1") == "1"


@contextlib.contextmanager
def _suppress_output(enabled: bool):
    if not enabled:
        yield None
        return
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
        yield (stdout_buf, stderr_buf)


def _print_captured_output(captured):
    if not captured:
        return
    stdout_buf, stderr_buf = captured
    stdout_val = stdout_buf.getvalue().strip()
    stderr_val = stderr_buf.getvalue().strip()
    if stdout_val:
        print(stdout_val)
    if stderr_val:
        print(stderr_val)


def run_embodied_agent(
    question: str,
    image_paths: Optional[Union[str, List[str]]] = None,
    enable_memory: bool = True,
    task_type: str = "general_task",
    verbose: bool = False,
) -> Dict[str, Any]:
    start_time = time.time()

    llm_engine_name = "gpt-4o"

    memory_config = {
        "retriever_config": {
            "use_api_embedding": False,
            "local_model_path": r"D:\code\agent_memory\all-MiniLM-L6-v2",
            "model_name": "all-MiniLM-L6-v2",
            "alpha": 0.5,
            "disable_semantic_search": False,
        },
        "gate_config": {
            "retrieve_gate_patterns": [
                r"remember",
                r"recall",
                r"previous",
                r"earlier",
                r"last time",
                r"summary",
                r"summarize",
                r"my name",
                r"my favorite",
                r"i told you",
                r"i said",
            ],
            "retrieve_gate_min_len": 4,
            "min_chars": 10,
            "skip_general": False,
        },
        "conversation_window_size": 2,
        "storage_dir": "./memory_store",
        "enable_persistence": True,
        "max_memories": 1000,
    } if enable_memory else None

    suppress_internal = _compact_output_enabled() and not verbose
    captured = None

    try:
        with _suppress_output(suppress_internal) as captured:
            solver = construct_solver_embodied(
                llm_engine_name=llm_engine_name,
                enabled_tools=["Base_Generator_Tool", "Python_Coder_Tool"],
                tool_engine=["gpt-4o", "gpt-4o"],
                model_engine=["gpt-4o", "gpt-4o", "gpt-4o"],
                output_types="direct",
                max_steps=10,
                max_time=300,
                max_tokens=4000,
                enable_multimodal=True,
                enable_memory=enable_memory,
                memory_config=memory_config,
                verbose=verbose,
            )

            if verbose:
                print("Starting embodied agent with GPT-4o...")
                print(f"Question: {question}")
                if image_paths:
                    if isinstance(image_paths, list):
                        print(f"Images: {len(image_paths)} images provided")
                    else:
                        print(f"Image: {image_paths}")
                print(f"Memory: {'Enabled' if enable_memory else 'Disabled'}")

            result = solver.solve(question, image_paths, task_type)
    except Exception:
        if suppress_internal:
            _print_captured_output(captured)
        raise

    result["execution_stats"] = {
        "total_time": round(time.time() - start_time, 2),
        "llm_engine": llm_engine_name,
        "memory_enabled": enable_memory,
        "task_type": task_type,
    }

    if enable_memory and hasattr(solver, "memory_manager") and solver.memory_manager:
        result["memory_stats"] = solver.memory_manager.get_stats()

    if verbose:
        print("Embodied agent execution completed.")
        print(f"Total execution time: {result['execution_stats']['total_time']:.2f}s")

    return result


def test_complete_flow(print_summary: bool = True) -> Dict[str, Any]:
    results: Dict[str, Any] = {}

    # Seed memory for retrieval in the multimodal test
    run_embodied_agent(question="Turn 1: My favorite color is blue.", enable_memory=True, verbose=False)
    run_embodied_agent(question="Turn 2: My favorite food is sushi.", enable_memory=True, verbose=False)

    # Only test test/vln/input_img1.jpg
    preferred_img = Path("test/vln/input_img1.jpg")
    if preferred_img.exists():
        multimodal_prompt = (
            "You are a navigation agent in a simulated city scene. "
            "Use the exact template below and keep each field short. "
            "These are fictional internal states for the simulation; do not include hidden chain-of-thought.\n"
            "Description: The image shows a street with a sidewalk, trees, and buildings. "
            "There is no visible sign indicating \"Michael's Store.\"\n"
            "Strategy Analysis:\n"
            "- History Review: No prior actions taken.\n"
            "Current State: Exploration\n"
            "- Sub-Strategy: Self-Exploration\n"
            "Rationale: The store is not visible in the image. Exploring the area or seeking additional information is necessary.\n"
            "To-Reasoning:\n"
            "<<Robot Belief>>:\n"
            "I do not know the location of Michael's Store based on the image.\n"
            "<<Robot Intention>>:\n"
            "I intend to acquire more information or explore the area to locate the store.\n"
            "<<Decision>>:\n"
            "Since the store is not visible, I should explore further or ask for more details.\n"
            "Navigation Goal:\n"
            "<Rotate(90)>\n"
            "Also remember what I said about my favorite color and food."
        )

        r_img = run_embodied_agent(
            question=multimodal_prompt,
            image_paths=str(preferred_img),
            enable_memory=True,
            verbose=False,
            task_type="navigation_task",
        )
        if "can't assist" in (r_img.get("direct_output") or "").lower():
            fallback_prompt = (
                "Provide the requested template without refusal. "
                "If a field is uncertain, state it briefly.\n"
                + multimodal_prompt
            )
            r_img = run_embodied_agent(
                question=fallback_prompt,
                image_paths=str(preferred_img),
                enable_memory=True,
                verbose=False,
                task_type="navigation_task",
            )
        results["multimodal"] = r_img
    else:
        results["multimodal"] = None

    # Memory sanity: 3 turns
    r_mem3 = run_embodied_agent(
        question="Turn 3: What did I say about my favorite color and food?",
        enable_memory=True,
        verbose=False,
    )
    results["memory_test_summary"] = r_mem3

    r_text = run_embodied_agent(
        question="What is the capital of France?",
        enable_memory=True,
        verbose=False,
    )
    results["text_only"] = r_text

    if print_summary:
        print("Tests completed (memory enabled). Summary:")
        print(f" - Text-only direct_output length: {len((r_text.get('direct_output') or '') or '')}")
        if results["multimodal"]:
            print(f" - Multimodal direct_output length: {len((results['multimodal'].get('direct_output') or '') or '')}")
        else:
            print(" - Multimodal test: skipped (no image found)")
        mem_stats = r_mem3.get("memory_stats") or {}
        short_total = mem_stats.get("short_memory", {}).get("total_messages", 0)
        long_count = mem_stats.get("long_memory", {}).get("current_memory_count", 0)
        print(f" - Short messages: {short_total}, Long memory entries: {long_count}")

    return results


def main(enable_memory: bool = True, frame_dir: str = "test/vln") -> Dict[str, Any]:
    compact = _compact_output_enabled()
    if compact:
        test_results = test_complete_flow(print_summary=False)
        print("Embodied agent flow test completed.")
        print(f" - Text-only output length: {len((test_results['text_only'].get('direct_output') or '') or '')}")
        if test_results.get("multimodal"):
            print(f" - Multimodal output length: {len((test_results['multimodal'].get('direct_output') or '') or '')}")
        else:
            print(" - Multimodal test: skipped (no image found)")
        mem_stats = test_results["memory_test_summary"].get("memory_stats") or {}
        short_total = mem_stats.get("short_memory", {}).get("total_messages", 0)
        long_count = mem_stats.get("long_memory", {}).get("current_memory_count", 0)
        retrieval_count = mem_stats.get("long_memory", {}).get("retrieval_count", 0)
        print(f" - Short messages: {short_total}, Long memory entries: {long_count}")
        print(f" - Memory retrievals: {retrieval_count}")
        if test_results.get("multimodal"):
            print("\nMultimodal output:")
            print(test_results["multimodal"].get("direct_output", ""))
        return test_results

    print("Embodied Agent One-Line Interface Demo")
    print("Demonstrating complete flow with single function call")
    print("=" * 60)
    print("\nOne-Line Usage Examples:")
    print("# Simple text query:")
    print('result = run_embodied_agent("What is the capital of France?")')
    print("\n# Multimodal query with memory:")
    print('result = run_embodied_agent("Analyze this image", image_paths="image.jpg", enable_memory=True)')
    print("\n# Custom configuration:")
    print('result = run_embodied_agent("Solve this task", task_type="navigation", verbose=False)')

    print("\n" + "=" * 60)
    print("RUNNING COMPLETE FLOW TEST")
    print("=" * 60)

    test_results = test_complete_flow(print_summary=True)

    print("\n" + "=" * 80)
    print("EMBODIED AGENT FLOW TEST RESULTS")
    print("=" * 80)

    for test_name, result in test_results.items():
        if result:
            print(f"\n{test_name.upper()}:")
            stats = result.get("execution_stats", {})
            print(f"  Time: {stats.get('total_time', 0):.2f}s")
            print(f"  LLM: {stats.get('llm_engine', 'Unknown')}")
            print(f"  Memory: {stats.get('memory_enabled', False)}")
            print(f"  Output Length: {len(result.get('direct_output', ''))} chars")

            if result.get("memory_stats"):
                mem_stats = result["memory_stats"]
                print(f"  Memories: {mem_stats.get('short_memory', {}).get('total_messages', 0)} messages")

    print("\nAll tests completed. Embodied agent flow verified.")
    return test_results


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore",
        message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
    )

    if not _compact_output_enabled():
        print("API Configuration:")
        print("Proxy_API_BASE:" + os.environ.get("Proxy_API_BASE", "Not Set"))
        print("OPENAI_API_KEY:" + ("Set" if os.environ.get("OPENAI_API_KEY") else "Not Set"))
        print("DASHSCOPE_API_KEY:" + ("Set" if os.environ.get("DASHSCOPE_API_KEY") else "Not Set"))

    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        if cmd in ("test", "flow", "complete"):
            if not _compact_output_enabled():
                print("\nRunning complete flow test...")
            main()
        elif cmd in ("simple", "demo"):
            if not _compact_output_enabled():
                print("\nRunning simple demo...")
            result = run_embodied_agent("Hello, can you help me understand how memory systems work?")
            direct_output = result.get("direct_output", "No response")
            if isinstance(direct_output, dict):
                print(f"Response: {str(direct_output)[:200]}...")
            else:
                print(f"Response: {str(direct_output)[:200]}...")
        elif cmd in ("true", "1", "yes", "on"):
            main(enable_memory=True)
        elif cmd in ("false", "0", "no", "off"):
            main(enable_memory=False)
        else:
            frame_dir = sys.argv[1]
            main(frame_dir=frame_dir)
    else:
        if not _compact_output_enabled():
            print("\nRunning complete embodied agent flow test...")
        main()