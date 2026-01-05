import argparse
import time
import json
from typing import Optional, Sequence, Union, Dict, Any

from ..agents.models_embodied.initializer import Initializer
from ..agents.models_embodied.planner import Planner
from ..agents.models_embodied.memory.short_memory import ShortMemory
from ..agents.models_embodied.memory.long_memory import LongMemory
from ..agents.models_embodied.executor import Executor
from ..agents.utils.utils import make_json_serializable_truncated

# TODO: No Tool Use
class SolverEmbodied:
    def __init__(
        self,
        planner,
        memory,
        executor,
        output_types: str = "base,final,direct",
        max_steps: int = 10,
        max_time: int = 300,
        max_tokens: int = 4000,
        root_cache_dir: str = "cache",
        verbose: bool = True,
        temperature: float = .0,
        enable_memory: bool = False,
        memory_config: Optional[Dict[str, Any]] = None
    ):
        self.planner = planner
        self.memory = memory  # Keep backward compatibility with basic memory
        self.executor = executor
        self.max_steps = max_steps
        self.max_time = max_time
        self.max_tokens = max_tokens
        self.root_cache_dir = root_cache_dir

        self.output_types = output_types.lower().split(',')
        self.temperature  = temperature
        assert all(output_type in ["base", "final", "direct"] for output_type in self.output_types), "Invalid output type. Supported types are 'base', 'final', 'direct'."
        self.verbose = verbose

        # Memory system configuration
        self.enable_memory = enable_memory
        self.memory_config = memory_config or {}

        # Initialize advanced memory system if enabled
        self.short_memory = None
        self.long_memory = None
        if self.enable_memory:
            self._init_memory_system()

    def _init_memory_system(self):
        """Initialize the advanced memory system (ShortMemory + LongMemory)"""
        if self.verbose:
            print("🧠 Initializing advanced memory systems...")

        # Initialize ShortMemory for current session management
        self.short_memory = ShortMemory()

        # Initialize LongMemory for persistent memory with A-MEM capabilities
        retriever_config = self.memory_config.get('retriever_config', {'use_api_embedding': True})
        self.long_memory = LongMemory(
            use_amem=True,
            retriever_config=retriever_config,
            storage_dir=self.memory_config.get('storage_dir', "./memory_store"),
            enable_persistence=self.memory_config.get('enable_persistence', True),
            max_memories=self.memory_config.get('max_memories', 1000)
        )

        if self.verbose:
            print("✅ Advanced memory systems initialized and ready")

    def populate_memory_for_task(self, task_description: str, task_type: str = "general_task"):
        """
        Populate memory systems with task-related information

        Args:
            task_description: Description of the current task
            task_type: Type of task for memory categorization
        """
        if not self.enable_memory or not self.short_memory or not self.long_memory:
            return

        # Add task information to short-term memory
        self.short_memory.set_query(task_description)

        # Add contextual memory to long-term memory for better retrieval
        memory_content = f"Task: {task_description}"
        if task_type == "navigation_task":
            memory_content += ". Analyzing environment and planning navigation steps."
        elif task_type == "analysis_task":
            memory_content += ". Performing detailed analysis and reasoning."

        self.long_memory.add_memory(memory_content, task_type)

    def retrieve_relevant_memories(self, query: str, k: int = 5) -> list:
        """
        Retrieve relevant memories for the given query

        Args:
            query: Query to search for relevant memories
            k: Number of memories to retrieve

        Returns:
            List of relevant memory objects
        """
        if not self.enable_memory or not self.long_memory:
            return []

        return self.long_memory.retrieve_memories(query, k)

    def solve(self, question: str, image_paths: Optional[Union[str, Sequence[str]]] = None, task_type: str = "general_task"):
        """
        Solve a single problem from the benchmark dataset.

        Args:
            question: The query/question to solve
            image_paths: Optional paths to images for multimodal understanding
            task_type: Type of task for memory categorization
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

        # Populate memory systems with task information if memory is enabled
        if self.enable_memory:
            self.populate_memory_for_task(question, task_type)

            # Retrieve relevant memories for context enhancement
            relevant_memories = self.retrieve_relevant_memories(question, k=3)
            print(f"DEBUG: Retrieved {len(relevant_memories)} memories at solver level")
            if relevant_memories:
                for i, mem in enumerate(relevant_memories[:2]):
                    print(f"DEBUG: Memory {i}: {mem.get('original_content', mem.get('content', ''))[:50]}...")
            if relevant_memories and self.verbose:
                print(f"📚 Retrieved {len(relevant_memories)} relevant memories for context")
            json_data["relevant_memories"] = relevant_memories

        # Generate base response if requested
        # if 'base' in self.output_types:
        #     base_response = self.planner.generate_base_response(question, image_paths, self.max_tokens)
        #     json_data["base_response"] = base_response
        #     if self.verbose:
        #         print(f"\n==> 📝 Base Response from LLM:\n\n{base_response}")

        # If only base response is needed, save and return
        if set(self.output_types) == {'base'}:
            return json_data
    
        # Continue with query analysis and tool execution if final or direct responses are needed
        if {'final', 'direct'} & set(self.output_types):
            if self.verbose:
                print(f"\n==> 🐙 Reasoning Steps from AgentFlow (Deep Thinking...)")

            # [1] Analyze query
            query_start_time = time.time()
            relevant_memories = json_data.get("relevant_memories", [])
            print(f"DEBUG: Found {len(relevant_memories)} relevant memories for query analysis")
            if relevant_memories:
                for i, mem in enumerate(relevant_memories[:2]):
                    print(f"DEBUG: Memory {i}: {mem.get('original_content', mem.get('content', ''))[:50]}...")
            query_analysis = self.planner.analyze_query(question, image_paths, relevant_memories)
            json_data["query_analysis"] = query_analysis
            if self.verbose:
                print(f"\n==> 🔍 Step 0: Query Analysis\n")
                print(f"{query_analysis}")
                print(f"[Time]: {round(time.time() - query_start_time, 2)}s")

            # Generate final output if requested
            if 'final' in self.output_types:
                final_output = self.planner.generate_final_output(question, image_paths, self.memory)
                json_data["final_output"] = final_output
                print(f"\n==> 🐙 Detailed Solution:\n\n{final_output}")

            # Generate direct output if requested
            if 'direct' in self.output_types:
                relevant_memories = json_data.get("relevant_memories", [])
                direct_output = self.planner.generate_direct_output(question, image_paths, self.memory, relevant_memories)
                json_data["direct_output"] = direct_output
                print(f"\n==> 🐙 Final Answer:\n\n{direct_output}")

            print(f"\n[Total Time]: {round(time.time() - query_start_time, 2)}s")
            print(f"\n==> ✅ Query Solved!")

        return json_data

def construct_solver_embodied(llm_engine_name : str = "gpt-4o",
                     enabled_tools : list[str] = ["all"],
                     tool_engine: list[str] = ["Default"],
                     model_engine: list[str] = ["trainable", "dashscope", "dashscope"],  # [planner_main, planner_fixed, executor]
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
                     enable_memory: bool = False,
                     memory_config: Optional[Dict[str, Any]] = None,
                     ):

    # Parse model_engine configuration
    # Format: [planner_main, planner_fixed, executor]
    # "trainable" means use llm_engine_name (the trainable model)
    planner_main_engine = llm_engine_name if model_engine[0] == "trainable" else model_engine[0]
    planner_fixed_engine = llm_engine_name if model_engine[1] == "trainable" else model_engine[1]
    executor_engine = llm_engine_name if model_engine[2] == "trainable" else model_engine[2]

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

    # Instantiate Memory
    memory = ShortMemory(max_files=50, max_actions=500)

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
        memory=memory,
        executor=executor,
        output_types=output_types,
        max_steps=max_steps,
        max_time=max_time,
        max_tokens=max_tokens,
        root_cache_dir=root_cache_dir,
        verbose=verbose,
        temperature=temperature,
        enable_memory=enable_memory,
        memory_config=memory_config
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