import argparse
import time
import json
from typing import Optional, Sequence, Union, Dict, Any
import re

from .models_embodied.initializer import Initializer
from .models_embodied.planner import Planner
from .models_embodied.memory.short_memory import ShortMemory
from .models_embodied.memory.long_memory import LongMemory
from .models_embodied.memory.memory_manager import MemoryManager
from .models_embodied.executor import Executor
from .utils.utils import make_json_serializable_truncated

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

        # Initialize memory manager (unified memory system)
        self.memory_manager = None
        if self.enable_memory:
            self._init_memory_system()

    def _init_memory_system(self):
        """Initialize the unified memory system with MemoryManager"""
        if self.verbose:
            print("🧠 Initializing unified memory system...")

        # Initialize MemoryManager for coordinated memory operations
        short_memory_config = {
            'max_files': self.memory_config.get('max_files', 100),
            'max_actions': self.memory_config.get('max_actions', 1000),
            'conversation_window_size': self.memory_config.get('conversation_window_size', 3)
        }

        long_memory_config = {
            'use_amem': True,
            'retriever_config': self.memory_config.get('retriever_config', {'use_api_embedding': True}),
            'gate_config': self.memory_config.get('gate_config', {}),
            'storage_dir': self.memory_config.get('storage_dir', "./memory_store"),
            'enable_persistence': self.memory_config.get('enable_persistence', True),
            'max_memories': self.memory_config.get('max_memories', 1000)
        }

        self.memory_manager = MemoryManager(
            short_memory_config=short_memory_config,
            long_memory_config=long_memory_config,
            conversation_window_size=self.memory_config.get('conversation_window_size', 3)
        )

        # Keep backward compatibility
        self.short_memory = self.memory_manager.get_short_memory()
        self.long_memory = self.memory_manager.get_long_memory()

        if self.verbose:
            print("✅ Unified memory system initialized and ready")

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
    # 不要每轮把 task 写入 long_memory，避免污染；
    # long_memory 只存窗口总结（conversation_summary）
        return    

    def retrieve_relevant_memories(self, query: str, k: int = 5) -> list:
        """
        Retrieve relevant memories for the given query

        Args:
            query: Query to search for relevant memories
            k: Number of memories to retrieve

        Returns:
            List of relevant memory objects
        """
        if not self.enable_memory or not self.memory_manager:
            return []

        return self.memory_manager.retrieve_relevant_memories(query, k)

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
            if self.verbose:
                print(f"DEBUG: Retrieved {len(relevant_memories)} memories at solver level")
            if relevant_memories:
                for i, mem in enumerate(relevant_memories[:2]):
                    if self.verbose:
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
    
        def _format_memory_block(memories: list) -> str:
            if not memories:
                return ""
            lines = []
            for mem in memories[:2]:
                text = mem.get("original_content") or mem.get("content") or ""
                if not isinstance(text, str):
                    text = str(text)
                # Strip analyzer metadata and keep a short preview
                text = re.split(r"\n?KEYWORDS:|\n?CONTEXT:|\n?TAGS:", text, maxsplit=1)[0]
                text = re.sub(r"关键词:.*", "", text)
                text = re.sub(r"主要内容:.*", "", text)
                text = re.sub(r"对话记录:.*", "", text)
                text = text.strip()
                if not text:
                    continue
                if len(text) > 200:
                    text = text[:200].rstrip() + "..."
                lines.append(f"- {text}")
            return "\n".join(lines)

        def _inject_memory_block(text: str, memories: list) -> str:
            if not text or "<<Mmeory>>" in text:
                return text
            memory_body = _format_memory_block(memories)
            if not memory_body:
                return text
            memory_block = f"<<Mmeory>>:\n{memory_body}\n"
            if "<<Decision>>:" in text:
                return text.replace("<<Decision>>:", f"{memory_block}\n<<Decision>>:", 1)
            if "<<Decision>>" in text:
                return text.replace("<<Decision>>", f"{memory_block}\n<<Decision>>", 1)
            return text.rstrip() + "\n\n" + memory_block

        # Continue with query analysis and tool execution if final or direct responses are needed
        if {'final', 'direct'} & set(self.output_types):
            if self.verbose:
                print(f"\n==> 🐙 Reasoning Steps from AgentFlow (Deep Thinking...)")

            # [1] Analyze query
            query_start_time = time.time()

            # Record user message to memory if memory system is enabled
            if self.enable_memory and self.memory_manager:
                self.memory_manager.add_message('user', question)

            relevant_memories = json_data.get("relevant_memories", [])
            if self.verbose:
                print(f"DEBUG: Found {len(relevant_memories)} relevant memories for query analysis")
            if relevant_memories:
                for i, mem in enumerate(relevant_memories[:2]):
                    if self.verbose:
                        print(f"DEBUG: Memory {i}: {mem.get('original_content', mem.get('content', ''))[:50]}...")
            query_analysis = self.planner.analyze_query(question, image_paths, relevant_memories)
            json_data["query_analysis"] = query_analysis
            if self.verbose:
                print(f"\n==> 🔍 Step 0: Query Analysis\n")
                print(f"{query_analysis}")
                print(f"[Time]: {round(time.time() - query_start_time, 2)}s")

            # Generate final output if requested
            if 'final' in self.output_types:
                # Attempt original prompt, then retry with progressively simpler fallbacks if blocked by content filters
                fallback_prompts = [
                    question,
                    "Description: Brief one-line neutral description of the image scene."
                ]
                final_output = None
                last_exception = None
                for idx, p in enumerate(fallback_prompts):
                    try:
                        final_output = self.planner.generate_direct_output(p, image_paths, self.memory, relevant_memories)
                        break
                    except Exception as _e:
                        last_exception = _e
                        if not ("content_filter" in str(_e) or "filtered" in str(_e).lower()):
                            # If it's not a content filter error, re-raise immediately
                            raise
                        if self.verbose:
                            print(f"⚠️ Prompt attempt {idx+1} was filtered; trying next fallback...")
                if final_output is None:
                    # All fallbacks failed due to content filtering; surface a clear message
                    raise last_exception
                final_output = _inject_memory_block(final_output, relevant_memories)
                json_data["final_output"] = final_output
                print(f"\n==> 🐙 Detailed Solution:\n\n{final_output}")

            # Generate direct output if requested
            if 'direct' in self.output_types:
                relevant_memories = json_data.get("relevant_memories", [])
                # Retry sequence for direct output similar to final_output
                fallback_prompts = [
                    question,
                    "Description: Brief one-line neutral description of the image scene."
                ]
                direct_output = None
                last_exception = None
                for idx, p in enumerate(fallback_prompts):
                    try:
                        direct_output = self.planner.generate_direct_output(p, image_paths, self.memory, relevant_memories)
                        break
                    except Exception as _e:
                        last_exception = _e
                        if not ("content_filter" in str(_e) or "filtered" in str(_e).lower()):
                            raise
                        if self.verbose:
                            print(f"⚠️ Direct prompt attempt {idx+1} was filtered; trying next fallback...")
                if direct_output is None:
                    raise last_exception
                direct_output = _inject_memory_block(direct_output, relevant_memories)
                json_data["direct_output"] = direct_output
                print(f"\n==> 🐙 Final Answer:\n\n{direct_output}")

                # Record assistant message to memory if memory system is enabled
                if self.enable_memory and self.memory_manager:
                    self.memory_manager.add_message('assistant', direct_output)

                # Auto-extract a concise image/environment description and add to long-term memory
                try:
                    def _extract_description(text: str) -> str:
                        if not text:
                            return ""
                        # Try to capture a labeled "Description" block
                        m = re.search(r"Description[:：]\\s*(.+?)(?:\\n\\n|$)", text, flags=re.S)
                        if m:
                            return m.group(1).strip()
                        # Fallback to first line or first 200 chars
                        first_line = text.strip().splitlines()[0] if text.strip().splitlines() else text.strip()
                        return first_line.strip()[:400]

                    description_text = _extract_description(direct_output or json_data.get("final_output") or "")
                    if description_text and self.enable_memory and self.memory_manager and getattr(self.memory_manager, 'long_memory', None):
                        added = self.memory_manager.long_memory.add_memory(description_text, memory_type="image_description", metadata={"source":"auto_extracted"})
                        if self.verbose:
                            print(f"🧠 Auto memory save (image_description): {added} - {description_text[:120]}...")
                except Exception as _e:
                    if self.verbose:
                        print(f"Warning: failed to auto-save image description to memory: {_e}")
            if self.verbose:
                print(f"\n[Total Time]: {round(time.time() - query_start_time, 2)}s")
                print("\n==> Query Solved!")

        # Save memory state if memory system is enabled
        if self.enable_memory and self.memory_manager:
            self.memory_manager.save_state()

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
                     planner: Optional[Planner] = None,
                     executor: Optional[Executor] = None,
                     memory: Optional[ShortMemory] = None,
                     initializer: Optional[Initializer] = None,
                     ):

    # Parse model_engine configuration
    # Format: [planner_main, planner_fixed, executor]
    # "trainable" means use llm_engine_name (the trainable model)
    planner_main_engine = llm_engine_name if model_engine[0] == "trainable" else model_engine[0]
    planner_fixed_engine = llm_engine_name if model_engine[1] == "trainable" else model_engine[1]
    executor_engine = llm_engine_name if model_engine[2] == "trainable" else model_engine[2]

    # Instantiate Initializer (unless provided)
    if initializer is None:
        initializer = Initializer(
            enabled_tools=enabled_tools,
            tool_engine=tool_engine,
            model_string=llm_engine_name,
            verbose=verbose,
            vllm_config_path=vllm_config_path,
        )

    # Instantiate Planner (unless provided)
    if planner is None:
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

    # Instantiate Memory (unless provided)
    if memory is None:
        memory = ShortMemory(max_files=50, max_actions=500)

    # Instantiate Executor with tool instances cache (unless provided)
    if executor is None:
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
