import time
from collections.abc import Sequence
from typing import Optional, Union

from ..agentflow.models.initializer import Initializer
from ..agentflow.models.memory import Memory
from ..agentflow.models.planner import Planner

class FastSolver:
    """Lightweight solver that only uses the planner to produce answers."""

    def __init__(
        self,
        planner: Planner,
        memory: Memory,
        output_types: str = "direct",
        max_time: int = 60,
        max_tokens: int = 1024,
        fast_max_tokens: Optional[int] = None,
        verbose: bool = False,
        temperature: float = 0.0,
    ) -> None:
        self.planner = planner
        self.memory = memory
        self.max_time = max_time
        self.max_tokens = max_tokens
        self.fast_max_tokens = fast_max_tokens or max_tokens
        self.verbose = verbose
        self.temperature = temperature
        self.output_types = [item.strip().lower() for item in output_types.split(",") if item.strip()]
        if not self.output_types:
            self.output_types = ["direct"]
        valid_outputs = {"base", "final", "direct"}
        unknown_outputs = set(self.output_types) - valid_outputs
        if unknown_outputs:
            raise ValueError(f"Unsupported output types for FastSolver: {unknown_outputs}")

    def solve(self, question: str, image_paths: Optional[Union[str, Sequence[str]]] = None) -> dict:
        start_time = time.time()
        payload: dict = {
            "query": question,
            "image": image_paths,
        }

        if self.verbose:
            print("\n==> 🔍 Fast Solver Received Query:", question)
            if image_paths:
                if isinstance(image_paths, Sequence) and not isinstance(image_paths, (str, bytes, bytearray)):
                    print("==> 🖼️ Image Sequence:")
                    for idx, path in enumerate(image_paths):
                        print(f"    Frame {idx + 1}: {path}")
                else:
                    print("==> 🖼️ Image Path:", image_paths)


        # Step 1: quick base response
        if "base" in self.output_types:
            if self.verbose:
                print("\n==> ⚡ Generating fast base response...")
            payload["base_response"] = self.planner.generate_base_response(
                question,
                image_paths,
                max_tokens=self.fast_max_tokens,
            )
            if self.verbose:
                print("==> ✅ Base response ready.")

        # Step 2: optional lightweight analysis for downstream outputs
        request_deeper_output = {"final", "direct"} & set(self.output_types)
        if request_deeper_output:
            if self.verbose:
                print("\n==> 📊 Running lightweight query analysis...")
            payload["query_analysis"] = self.planner.analyze_query(question, image_paths)

        # Step 3: produce final/direct answers without tool execution
        if "final" in self.output_types:
            if self.verbose:
                print("\n==> 🧾 Producing structured final output...")
            payload["final_output"] = self.planner.generate_final_output(question, image_paths, self.memory)
        if "direct" in self.output_types:
            if self.verbose:
                print("\n==> 🎯 Producing concise direct output...")
            payload["direct_output"] = self.planner.generate_direct_output(question, image_paths, self.memory)

        payload["execution_time"] = round(time.time() - start_time, 2)

        print(f"\n[Total Time]: {payload['execution_time']}s")
        print(f"\n==> ✅ Query Solved!")

        return payload

def construct_fast_solver(
    llm_engine_name: str = "gpt-4o",
    enabled_tools: Optional[list[str]] = None,
    tool_engine: Optional[list[str]] = None,
    output_types: str = "direct",
    max_steps: int = 1,  # retained for API compatibility
    max_time: int = 60,
    max_tokens: int = 1024,
    fast_max_tokens: Optional[int] = 256,
    root_cache_dir: str = "solver_cache",
    verbose: bool = False,
    vllm_config_path: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.0,
    enable_multimodal: Optional[bool] = None,
) -> FastSolver:
    # Defaults to "all" tools when nothing provided to stay aligned with regular solver.
    enabled_tools = enabled_tools or ["all"]
    tool_engine = tool_engine or ["Default"]

    if enable_multimodal is None:
        enable_multimodal = supports_multimodal(llm_engine_name)

    initializer = Initializer(
        enabled_tools=enabled_tools,
        tool_engine=tool_engine,
        model_string=llm_engine_name,
        verbose=verbose,
        vllm_config_path=vllm_config_path,
    )

    planner = Planner(
        llm_engine_name=llm_engine_name,
        llm_engine_fixed_name=llm_engine_name,
        toolbox_metadata=initializer.toolbox_metadata,
        available_tools=initializer.available_tools,
        verbose=verbose,
        base_url=base_url,
        is_multimodal=enable_multimodal,
        temperature=temperature,
    )

    memory = Memory()

    return FastSolver(
        planner=planner,
        memory=memory,
        output_types=output_types,
        max_time=max_time,
        max_tokens=max_tokens,
        fast_max_tokens=fast_max_tokens,
        verbose=verbose,
        temperature=temperature,
    )
