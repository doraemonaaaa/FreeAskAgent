import argparse
import time
import json
from typing import Any, Optional, Sequence, Tuple, List
from pathlib import Path
import re
import threading

from ..agents.models_embodied.initializer import Initializer
from ..agents.models_embodied.planner import Planner
from ..agents.models_embodied.memory.memory import Memory
from ..agents.models_embodied.executor import Executor
from ..agents.models_embodied.verifier import Verifier

# TODO: No Tool Use
class SolverEmbodied:
    def __init__(
        self,
        planner,
        verifier,
        memory,
        executor,
        output_types: str = "base,final,direct",
        max_steps: int = 10,
        max_time: int = 300,
        max_tokens: int = 4000,
        root_cache_dir: str = "cache",
        verbose: bool = True,
        temperature: float = .0,
        is_enable_memory: bool = True,
        is_use_verifier: bool = True
    ):
        self.planner = planner
        self.memory = memory
        self.executor = executor
        self.max_steps = max_steps
        self.max_time = max_time
        self.max_tokens = max_tokens
        self.root_cache_dir = root_cache_dir

        self.output_types = output_types.lower().split(',')
        self.temperature  = temperature
        assert all(output_type in ["base", "final", "direct"] for output_type in self.output_types), "Invalid output type. Supported types are 'base', 'final', 'direct'."
        self.verbose = verbose

        # paralel running
        self.verifier = verifier
        self.latest_verification_result = None
        self.verifier_image_paths = None
        self.planner_latest_output = None

        self.is_use_verifier = is_use_verifier
        self.is_enable_memory = is_enable_memory
        
    def solve(self, question: str, image_paths: Any = None):
        """
        Solve a single problem from the benchmark dataset.
        
        Args:
            index (int): Index of the problem to solve
        """
        # Update cache directory for the executor
        self.executor.set_query_cache_dir(self.root_cache_dir)

        # Initialize json_data with basic problem information
        json_data = {
            "query": question,
            "images": image_paths
        }
        if self.verbose:
            print(f"\n==> 🔍 Received Query: {question}")
            if image_paths:
                if isinstance(image_paths, Sequence) and not isinstance(image_paths, (str, bytes, bytearray)):
                    for idx, path in enumerate(image_paths):
                        print(f"==> 🖼️ Frame {idx+1}: {path}")
                else:
                    print(f"\n==> 🖼️ Received Image: {image_paths}")

        if self.verbose:
                print(f"\n==> 🐙Verifier is thinking (Deep Thinking...)")

        if self.verifier_image_paths is None:
            self.verifier_image_paths = image_paths

        prev_result_text = self.latest_verification_result
        self.latest_verification_result = self.verifier.verificate_context(
            question, self.verifier_image_paths, self.planner_latest_output, self.memory, previous_verification_log=prev_result_text
        )
        if self.verbose:
            print(f"[Verifier] Finished. Result: {self.latest_verification_result}")
    
        # Continue with query analysis and tool execution if final or direct responses are needed
        if {'final', 'direct'} & set(self.output_types):
            if self.verbose:
                print(f"\n==> 🐙 Reasoning Steps from AgentFlow (Deep Thinking...)")

            palnning_start_time = time.time()

            # Generate direct output if requested
            if 'direct' in self.output_types:
                if self.is_use_verifier == False:
                    self.latest_verification_result = ""
                direct_output = self.planner.generate_direct_output(question, image_paths, self.memory, self.latest_verification_result)
                json_data["direct_output"] = direct_output
                print(f"\n==> 🐙 Final Answer:\n\n{direct_output}")

            # print("Memory: " + json.dumps(memory_data, ensure_ascii=False, indent=2))
            self.planner_latest_output = json_data.get("direct_output", "")

            print(f"\n[Total Time]: {round(time.time() - palnning_start_time, 2)}s")
            print(f"\n==> ✅ Query Solved!")

        return json_data
    
    def write_verify_data(self, image_paths: Any = None, interaction_memory: Optional[str] = None):
        self.verifier_image_paths = image_paths
        # memory
        commands = self.parse_commands(self.planner_latest_output)
        parsed_data = self.memory.parse_vln_output(self.planner_latest_output)
        if commands:
            kwargs = {
                'belief': parsed_data.get('belief'),
                'intention': parsed_data.get('intention'),
                "commands": commands,
                'state': parsed_data.get('state'),
                "verification": self.latest_verification_result
            }
            if interaction_memory is not None:
                kwargs["interaction_memory"] = interaction_memory

            self.memory.add_embodied_action(**kwargs)
    
    @staticmethod
    def parse_commands(output_text: str) -> List[Tuple[str, str]]:
        """
        Parses multiple <Action(param)> commands from LLM output text.

        Returns:
            List of tuples: [(method1, params1), (method2, params2), ...]
        """
        log_path = Path("tmp/llm_raw_text.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(output_text + "\n" + "-"*80 + "\n")

        commands = []

        # 尝试匹配 Action 或 Navigation Goal 标签后的文本
        action_match = re.search(
            r"(?:\*\*Action\*\*|Action|Navigation Goal)\s*:\s*(.*)",
            output_text,
            re.IGNORECASE | re.DOTALL
        )

        if action_match:
            action_text = action_match.group(1).strip()
            print(f"Extracted Action Text:\n{action_text}")

            # 匹配所有 <Method(params)>，多步匹配
            method_matches = re.findall(r"<(\w+)\((.*?)\)>", action_text, re.IGNORECASE | re.DOTALL)
            
            if method_matches:
                for method, params in method_matches:
                    method = method.strip()
                    params = params.strip()
                    commands.append((method, params))
                    print(f"Parsed Method: {method}, Params: {params}")
            else:
                print("No <Method(...)> patterns found in Action text.")
        else:
            print("Label 'Action:' not found in LLM output. No commands extracted.")

        return commands
    
def construct_solver_embodied(llm_engine_name : str = "gpt-4o",
                     enabled_tools : list[str] = ["all"],
                     tool_engine: list[str] = ["Default"],
                     model_engine: list[str] = ["trainable", "dashscope", "dashscope", "dashscope"],  # [planner_main, planner_fixed, executor]
                     output_types : str = "final,direct",
                     max_steps : int = 10,
                     max_time : int = 300,
                     max_tokens : int = 4000,
                     root_cache_dir : str = "solver_cache",
                     verbose : bool = True,
                     vllm_config_path : str = None,
                     base_url : str = None,
                     temperature: float = 0.0,
                     enable_multimodal: Optional[bool] = None,
                     is_enable_memory: bool = True,
                     is_use_verifier: bool = True
                     ):

    # Parse model_engine configuration
    # Format: [planner_main, planner_fixed, executor]
    # "trainable" means use llm_engine_name (the trainable model)
    planner_main_engine = llm_engine_name if model_engine[0] == "trainable" else model_engine[0]
    planner_fixed_engine = llm_engine_name if model_engine[1] == "trainable" else model_engine[1]
    verifier_engine = llm_engine_name if model_engine[2] == "trainable" else model_engine[2]
    executor_engine = llm_engine_name if model_engine[3] == "trainable" else model_engine[3]

    # Instantiate Initializer
    initializer = Initializer(
        enabled_tools=enabled_tools,
        tool_engine=tool_engine,
        model_string=llm_engine_name,
        verbose=verbose,
        vllm_config_path=vllm_config_path,
    )

    # Instantiate Planner
    planner = Planner(
        llm_engine_name=planner_main_engine,
        llm_engine_fixed_name=planner_fixed_engine,
        toolbox_metadata=initializer.toolbox_metadata,
        available_tools=initializer.available_tools,
        verbose=verbose,
        base_url=base_url,
        temperature=temperature,
        is_multimodal=enable_multimodal
    )

    verifier = Verifier(
        llm_engine_name=verifier_engine,
        llm_engine_fixed_name=planner_fixed_engine,
        toolbox_metadata=initializer.toolbox_metadata,
        available_tools=initializer.available_tools,
        verbose=verbose,
        base_url=base_url if verifier_engine == llm_engine_name else None,
        temperature=temperature,
        is_multimodal=enable_multimodal
    )

    # Instantiate Memory
    memory = Memory.get_instance(is_enable=is_enable_memory)

    # Instantiate Executor with tool instances cache
    executor = Executor(
        llm_engine_name=executor_engine,
        root_cache_dir=root_cache_dir,
        verbose=verbose,
        base_url=base_url if executor_engine == llm_engine_name else None,  # Only use base_url for trainable model
        temperature=temperature,
        tool_instances_cache=initializer.tool_instances_cache  # Pass the cached tool instances
    )

    # Instantiate Solver
    solver = SolverEmbodied(
        planner=planner,
        verifier=verifier,
        memory=memory,
        executor=executor,
        output_types=output_types,
        max_steps=max_steps,
        max_time=max_time,
        max_tokens=max_tokens,
        root_cache_dir=root_cache_dir,
        verbose=verbose,
        temperature=temperature,
        is_enable_memory=is_enable_memory,
        is_use_verifier=is_use_verifier
    )
    return solver

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the agentflow demo with specified parameters.")
    parser.add_argument("--llm_engine_name", default="gpt-4o", help="LLM engine name.")
    parser.add_argument(
        "--output_types",
        default="base,final,direct",
        help="Comma-separated list of required outputs (base,final,direct)"
    )
    parser.add_argument("--enabled_tools", default="Base_Generator_Tool", help="List of enabled tools.")
    parser.add_argument("--root_cache_dir", default="solver_cache", help="Path to solver cache directory.")
    parser.add_argument("--max_tokens", type=int, default=4000, help="Maximum tokens for LLM generation.")
    parser.add_argument("--max_steps", type=int, default=10, help="Maximum number of steps to execute.")
    parser.add_argument("--max_time", type=int, default=300, help="Maximum time allowed in seconds.")
    parser.add_argument("--verbose", type=bool, default=True, help="Enable verbose output.")
    return parser.parse_args()
    
def main(args):
    tool_engine=["dashscope-qwen2.5-3b-instruct","dashscope-qwen2.5-3b-instruct","Default","Default"]
    solver = construct_solver_embodied(
        llm_engine_name=args.llm_engine_name,
        enabled_tools=["Base_Generator_Tool","Python_Coder_Tool","Google_Search_Tool","Wikipedia_Search_Tool"],
        tool_engine=tool_engine,
        output_types=args.output_types,
        max_steps=args.max_steps,
        max_time=args.max_time,
        max_tokens=args.max_tokens,
        # base_url="http://localhost:8080/v1",
        verbose=args.verbose,
        temperature=0.7
    )

    # Solve the task or problem
    solver.solve("What is the capital of France?")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)