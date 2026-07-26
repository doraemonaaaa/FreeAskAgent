"""CLI and integration entry point for the asynchronous visual navigation agents."""

import argparse
import time
from collections.abc import Mapping
from typing import Any, Optional

from agentflow.agents.models_embodied_v2.Actor import ACTION_TOKENS, FORWARD, STOP, Actor
from agentflow.agents.models_embodied_v2.TemporalCaptioner import TemporalCaptioner
from agentflow.agents.models_embodied_v2.Thinker import Thinker
from agentflow.agents.models_embodied_v2.stop_gate import StopDecision
from agentflow.agents.models_embodied_v2.memory import (
    MEMORY_MODES,
    CompositeMemory,
    MemoryMode,
    StepExecution,
    TaskMemory,
    TaskMemoryInterface,
    TemporalMemory,
    TemporalMemoryInterface,
    TemporalObservation,
)

# Compatibility aliases for code that imported the previous names.
VLNAgent = Actor


class AsyncThinkActVLN:
    """Run ModelA thinking before ModelB selects each action."""

    def __init__(
        self,
        goal: str,
        *,
        policy_model_path="models/Qwen3-VL-8B-Instruct",
        planner_model_path="models/Qwen3-VL-8B-Instruct",
        temporal_model_path: Optional[str] = None,
        memory_mode: MemoryMode = "temporal",
        debug_performance=False,
        use_cache=False,
        actor: Optional[Any] = None,
        thinker: Optional[Any] = None,
        memory: Optional[CompositeMemory] = None,
        task_memory: Optional[TaskMemory] = None,
        temporal_memory: Optional[TemporalMemory] = None,
        temporal_captioner: Optional[TemporalCaptioner] = None,
        episode_id: str = "episode-0",
        arrival_radius_m: float = 2.0,
        confirmations_required: int = 3,
        min_steps_since_target_seen: int = 4,
        show_output: bool = True,
    ):
        self.goal = goal
        self.show_output = show_output
        self._episode_started_at = time.monotonic()
        self._episode_id = str(episode_id)
        self.actor = actor or Actor(
            policy_model_path,
            debug_performance=debug_performance,
            use_cache=use_cache,
            show_output=show_output,
        )
        if memory_mode not in MEMORY_MODES:
            raise ValueError(
                f"Unsupported memory_mode {memory_mode!r}; "
                f"expected {MEMORY_MODES}"
            )
        if memory is not None and (
            task_memory is not None or temporal_memory is not None
        ):
            raise ValueError(
                "Pass either memory or individual memory modules, not both"
            )
        if memory is None:
            task_interface = (
                TaskMemoryInterface(task_memory or TaskMemory(goal))
                if "task" in memory_mode
                else None
            )
            temporal_interface = (
                TemporalMemoryInterface(
                    temporal_memory
                    or TemporalMemory(
                        goal=goal,
                        episode_id=self._episode_id,
                        captioner=temporal_captioner,
                    )
                )
                if "temporal" in memory_mode
                else None
            )
            memory = CompositeMemory(
                goal,
                episode_id=self._episode_id,
                mode=memory_mode,
                task=task_interface,
                temporal=temporal_interface,
            )
        self.memory = memory
        self.memory_mode = self.memory.mode
        self.temporal_memory = self.memory.temporal_memory
        self.task_memory = self.memory.task_memory
        self.thinker = thinker or Thinker(
            goal,
            self.actor,
            planner_model_path=planner_model_path,
            debug_performance=debug_performance,
            use_cache=use_cache,
            memory=self.memory,
            show_output=show_output,
            arrival_radius_m=arrival_radius_m,
            confirmations_required=confirmations_required,
            min_steps_since_target_seen=min_steps_since_target_seen,
        )
        # A supplied Thinker must consume the same composite coordinator.
        self.thinker.memory = self.memory
        self.thinker.task_memory = self.memory

        if (
            self.temporal_memory is not None
            and temporal_captioner is None
            and self.temporal_memory.captioner is None
        ):
            if temporal_model_path and temporal_model_path != planner_model_path:
                temporal_captioner = TemporalCaptioner(
                    model_path=temporal_model_path,
                    use_cache=use_cache,
                    debug_performance=debug_performance,
                )
            else:
                # Planner and temporal understanding execute sequentially, so
                # sharing this engine avoids loading a third 8B model copy.
                temporal_captioner = TemporalCaptioner(
                    engine=self.thinker.planner_engine,
                    model_path=planner_model_path,
                    use_cache=use_cache,
                    debug_performance=debug_performance,
                )
            self.temporal_memory.set_captioner(temporal_captioner)
        elif self.temporal_memory is not None and temporal_captioner is not None:
            self.temporal_memory.set_captioner(temporal_captioner)
        elif temporal_captioner is not None:
            raise ValueError(
                "temporal_captioner requires a temporal memory mode"
            )

    def reset(self, *, goal: str, episode_id: Optional[str] = None) -> None:
        """Reset every episode-scoped planner and temporal-memory state."""
        self.goal = str(goal)
        self._episode_id = str(episode_id or self._episode_id)
        self._episode_started_at = time.monotonic()
        self.memory.reset(
            episode_id=self._episode_id,
            goal=self.goal,
        )
        self.thinker.reset(self.goal)

    def act(
        self,
        rgb_image: Any,
        observation_metadata: Optional[Mapping[str, Any]] = None,
        previous_execution: Optional[Mapping[str, Any] | StepExecution] = None,
    ) -> str:
        """Close the previous transition, update memory, then select an action.

        The normal Habitat boundary is ``act(rgb_image)``.  Temporal timestamps,
        step IDs, and the previously selected command are inferred internally;
        optional metadata remains available only for non-Habitat instrumented
        callers.
        """
        self.memory.record_input(rgb_image)

        def update_temporal_memory(subgoal_snapshot: str) -> None:
            self.memory.close_previous_action(
                subgoal_snapshot,
                metadata=observation_metadata,
                previous_execution=previous_execution,
            )

        # The Thinker updates the subgoal tracker first, invokes the memory hook,
        # then generates a directive from the newly updated memory context.
        directive = self.thinker.submit_observation(
            rgb_image,
            wait_for_completion=True,
            tracker_updated_callback=update_temporal_memory,
        )
        decision = self.thinker.stop_decision
        recovery_action = self.memory.prepare_recovery_action()
        if recovery_action is None:
            # STOP is absent from the action space unless the gate cleared it,
            # so an early stop is unrepresentable rather than just discouraged.
            action = self.actor.act(
                rgb_image,
                directive,
                context=self.thinker.actor_context(),
                allowed_actions=self.thinker.allowed_actions(),
            )
            action = self._enforce_stop_gate(action, decision)
        else:
            # Go Back primitives are movement-only by construction, so recovery
            # can never smuggle a STOP past the gate.
            action = recovery_action
            self.memory.record_event(
                "Go Back Action",
                f"executing recovery primitive {action}",
            )
        self.memory.stage_action(
            action,
            self.thinker.subtask_tracker or "",
        )
        if recovery_action is not None:
            self.memory.ack_recovery_action(action)
        return action

    @property
    def stop_decision(self) -> StopDecision:
        """Whether the gate would permit STOP right now, and why."""
        return self.thinker.stop_decision

    @property
    def tracker(self):
        """The structured tracker backing the gate, or None before planning."""
        return self.thinker.tracker

    def _enforce_stop_gate(self, action: str, decision: StopDecision) -> str:
        """Last line of defence if a masked STOP still comes back from ModelB."""
        if action != STOP or decision.allowed:
            return action
        self.memory.record_event("Stop denied", decision.reason)
        if self.show_output:
            print(f"[StopGate denied STOP] {decision.reason}", flush=True)
        return FORWARD

    def finish_episode(
        self,
        rgb_image: Any,
        observation_metadata: Optional[Mapping[str, Any]] = None,
        previous_execution: Optional[Mapping[str, Any] | StepExecution] = None,
    ):
        """Close and analyze the final action without producing another action."""
        self.memory.record_input(rgb_image)
        tracker = self.thinker.update_tracker_only(rgb_image)
        return self.memory.finish_episode(
            rgb_image,
            subgoal_snapshot=tracker,
            metadata=observation_metadata,
            previous_execution=previous_execution,
        )

    def memory_diagnostics(
        self,
        *,
        include_raw_response: bool = False,
    ) -> dict[str, Any]:
        return self.memory.diagnostics(
            include_raw_response=include_raw_response,
        )

    def close(self, timeout=None):
        self.thinker.close(timeout)

    def _temporal_observation(
        self,
        rgb_image: Any,
        metadata: Optional[Mapping[str, Any]],
    ) -> TemporalObservation:
        values = dict(metadata or {})
        position = values.get("position_xyz")
        landmarks = values.get("landmark_ids")
        return TemporalObservation(
            image=rgb_image,
            episode_id=str(values.get("episode_id", self._episode_id)),
            timestamp_seconds=float(
                values.get(
                    "timestamp_seconds",
                    time.monotonic() - self._episode_started_at,
                )
            ),
            position_xyz=(
                tuple(float(value) for value in position)
                if position is not None
                else None
            ),
            yaw_degrees=values.get("yaw_degrees"),
            distance_to_goal_meters=values.get(
                "distance_to_goal_meters"
            ),
            landmark_ids=(
                tuple(str(value) for value in landmarks)
                if landmarks is not None
                else None
            ),
        )

    def _step_execution(
        self,
        execution: Optional[Mapping[str, Any] | StepExecution],
        *,
        terminal: bool = False,
    ) -> StepExecution:
        if execution is None:
            return self.temporal_memory.infer_pending_execution(
                terminal=terminal
            )
        if isinstance(execution, StepExecution):
            if terminal and not execution.terminal:
                return StepExecution(
                    step_id=execution.step_id,
                    commanded_action=execution.commanded_action,
                    collision=execution.collision,
                    terminal=True,
                )
            return execution
        return StepExecution(
            step_id=int(execution["step_id"]),
            commanded_action=str(execution["commanded_action"]),
            collision=execution.get("collision"),
            terminal=(
                terminal or bool(execution.get("terminal", False))
            ),
        )


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
    parser.add_argument("--temporal-model-path")
    parser.add_argument(
        "--memory-mode",
        choices=MEMORY_MODES,
        default="temporal",
        help="Memory ablation: none, task, temporal, or task+temporal.",
    )
    parser.add_argument("--goal", default="Navigate safely to the requested destination.")
    parser.add_argument("--single-model", action="store_true", help="Disable ModelA and run ModelB alone.")
    parser.add_argument(
        "--arrival-radius",
        type=float,
        default=2.0,
        help="Meters within which the stop gate may clear STOP.",
    )
    parser.add_argument(
        "--stop-confirmations",
        type=int,
        default=3,
        help="Consecutive arrival confirmations before the final subtask completes.",
    )
    parser.add_argument(
        "--stop-dwell-steps",
        type=int,
        default=4,
        help="Steps required between first sighting the target and stopping.",
    )
    parser.add_argument("--no-debug-performance", action="store_true")
    parser.add_argument("--use-cache", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.single_model:
        agent = Actor(args.model_path, debug_performance=not args.no_debug_performance, use_cache=args.use_cache)
    else:
        agent = AsyncThinkActVLN(
            goal=args.goal,
            policy_model_path=args.model_path,
            planner_model_path=args.planner_model_path,
            temporal_model_path=args.temporal_model_path,
            memory_mode=args.memory_mode,
            debug_performance=not args.no_debug_performance,
            use_cache=args.use_cache,
            arrival_radius_m=args.arrival_radius,
            confirmations_required=args.stop_confirmations,
            min_steps_since_target_seen=args.stop_dwell_steps,
        )
    run_terminal(agent)


if __name__ == "__main__":
    main()
