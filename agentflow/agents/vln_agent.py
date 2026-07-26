"""CLI and integration entry point for the asynchronous visual navigation agents."""

import argparse

from agentflow.agents.models_embodied_v2.Actor import ACTION_TOKENS, Actor
from agentflow.agents.models_embodied_v2.Thinker import Thinker

# Compatibility aliases for code that imported the previous names.
VLNAgent = Actor


class AsyncThinkActVLN:
    """Run ModelA thinking before ModelB selects each action."""

    def __init__(self, goal: str, *, policy_model_path="models/Qwen3-VL-8B-Instruct", planner_model_path="models/Qwen3-VL-8B-Instruct", debug_performance=False, use_cache=False):
        self.goal = goal
        self.actor = Actor(policy_model_path, debug_performance=debug_performance, use_cache=use_cache)
        self.thinker = Thinker(goal, self.actor, planner_model_path=planner_model_path, debug_performance=debug_performance, use_cache=use_cache)
        # The task memory starts with this agent's input: (goal, current observation).
        self.task_memory = self.thinker.task_memory

    def act(self, rgb_image):
        # Use the directive inferred from this observation, never the previous one.
        directive = self.thinker.submit_observation(rgb_image, wait_for_completion=True)
        action = self.actor.act(rgb_image, directive)
        self.thinker.record_action(action)
        return action

    def close(self, timeout=None):
        self.thinker.close(timeout)


def run_terminal(agent):
    is_async = isinstance(agent, AsyncThinkActVLN)
    print("Async Thinker + Actor is ready." if is_async else "Actor is ready.")
    print(f"Actions: {', '.join(ACTION_TOKENS)}. Use: /image path/to/rgb.png [instruction]")
    try:
        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                return
            if user_input.lower() in {"exit", "quit", "q"}:
                print("Bye.")
                return
            if not user_input.startswith("/image "):
                print("Usage: /image path/to/rgb.png [instruction]")
                continue
            parts = user_input.split(maxsplit=2)
            if is_async:
                if len(parts) == 3 and parts[2] != agent.goal:
                    print("[ModelA goal] Set the task with --goal; ModelB follows ModelA's directive.")
                action = agent.act(parts[1])
            else:
                action = agent.act(parts[1], parts[2] if len(parts) == 3 else "")
            print(f"\nAction: {action}")
    finally:
        if is_async:
            agent.close(timeout=1)


def parse_args():
    parser = argparse.ArgumentParser(description="Run asynchronous Qwen3-VL visual navigation.")
    parser.add_argument("--model-path", default="models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--planner-model-path", default="models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--goal", default="Navigate safely to the requested destination.")
    parser.add_argument("--single-model", action="store_true", help="Disable ModelA and run ModelB alone.")
    parser.add_argument("--no-debug-performance", action="store_true")
    parser.add_argument("--use-cache", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.single_model:
        agent = Actor(args.model_path, debug_performance=not args.no_debug_performance, use_cache=args.use_cache)
    else:
        agent = AsyncThinkActVLN(goal=args.goal, policy_model_path=args.model_path, planner_model_path=args.planner_model_path, debug_performance=not args.no_debug_performance, use_cache=args.use_cache)
    run_terminal(agent)


if __name__ == "__main__":
    main()
