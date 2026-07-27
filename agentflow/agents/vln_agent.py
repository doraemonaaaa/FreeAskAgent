"""CLI and integration entry point for the asynchronous visual navigation agents."""

import argparse
import time
from collections import deque
from collections.abc import Mapping
from typing import Any, Optional

from workspace.FreeAskAgent.agentflow.agents.models_embodied_v2.deprecated.Actor import ACTION_TOKENS, FORWARD, STOP, Actor
from agentflow.agents.models_embodied_v2.TemporalCaptioner import TemporalCaptioner
from workspace.FreeAskAgent.agentflow.agents.models_embodied_v2.deprecated.Thinker import Thinker
from workspace.FreeAskAgent.agentflow.agents.models_embodied_v2.deprecated.freespace_gate import (
    EscapeDirection,
    FreeSpaceDecision,
    FreeSpaceGate,
)
from workspace.FreeAskAgent.agentflow.agents.models_embodied_v2.deprecated.stop_gate import StopDecision
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
        forward_block_after: int = 1,
        escape_after: int = 2,
        escape_probe_steps: int = 2,
        scout_escape_direction: bool = True,
        max_scout_attempts: int = 3,
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

        # Wall handling lives beside the stop gate rather than inside memory:
        # temporal memory only rules on a wall after six wasted steps, and its
        # recovery macro turns blind. This gate reacts on the first failed
        # FORWARD and asks the planner where the floor actually is.
        self.freespace_gate = FreeSpaceGate(
            block_after=forward_block_after,
            escape_after=escape_after,
            probe_forward_steps=escape_probe_steps,
        )
        self.scout_escape_direction = scout_escape_direction
        self.max_scout_attempts = max_scout_attempts
        self._escape_queue: deque[str] = deque()
        self._last_freespace = FreeSpaceDecision(True, "no observation yet")

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
        self.freespace_gate.reset()
        self._escape_queue.clear()
        self._last_freespace = FreeSpaceDecision(True, "no observation yet")

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
        free_space = self._update_freespace(previous_execution)
        recovery_action = self.memory.prepare_recovery_action()
        # Go Back owns the step whenever it is running: its macro and the escape
        # macro would otherwise interleave into a sequence neither one planned.
        escape_action = (
            None
            if recovery_action is not None
            else self._next_escape_action(rgb_image, free_space)
        )
        if recovery_action is not None:
            # Go Back primitives are movement-only by construction, so recovery
            # can never smuggle a STOP past the gate.
            action = recovery_action
            self.memory.record_event(
                "Go Back Action",
                f"executing recovery primitive {action}",
            )
        elif escape_action is not None:
            # Same guarantee as Go Back: an escape macro is turns plus a forward
            # probe, so it cannot end the episode either.
            action = escape_action
            self.memory.record_event(
                "Escape Action",
                f"executing escape primitive {action}",
            )
        else:
            # STOP is absent from the action space unless the gate cleared it,
            # and FORWARD is absent while the way ahead is known to be blocked,
            # so neither an early stop nor another wall hit is representable.
            action = self.actor.act(
                rgb_image,
                directive,
                context=self._actor_context(free_space),
                allowed_actions=self.freespace_gate.filter_actions(
                    self.thinker.allowed_actions()
                ),
            )
            action = self._enforce_stop_gate(action, decision)
        self.memory.stage_action(
            action,
            self.thinker.subtask_tracker or "",
        )
        if recovery_action is not None:
            self.memory.ack_recovery_action(action)
        elif escape_action is not None:
            # Consume only after staging succeeded, mirroring the Go Back ack so
            # a transient failure cannot silently drop an escape primitive.
            self._escape_queue.popleft()
        return action

    @property
    def stop_decision(self) -> StopDecision:
        """Whether the gate would permit STOP right now, and why."""
        return self.thinker.stop_decision

    @property
    def freespace_decision(self) -> FreeSpaceDecision:
        """Whether FORWARD is legal right now, and why."""
        return self._last_freespace

    def _update_freespace(
        self,
        previous_execution: Optional[Mapping[str, Any] | StepExecution] = None,
    ) -> FreeSpaceDecision:
        """Feed the just-closed transition to the free-space gate.

        The transition is already closed by the time this runs, so the evidence
        row temporal memory just appended describes exactly the FORWARD whose
        outcome decides whether FORWARD stays legal.
        """
        collision = self._reported_collision(previous_execution)
        evidence = self._latest_evidence()
        if evidence is not None:
            self.freespace_gate.observe_step(
                commanded_action=evidence.commanded_action,
                observed_motion=evidence.observed_motion,
                motion_confidence=evidence.motion_confidence,
                frame_similarity=evidence.frame_similarity,
                collision=collision,
                step_id=evidence.step_id,
            )
        elif self.temporal_memory is None and collision is not None:
            # Without temporal memory the collision sensor is the only signal
            # left, and it is only present when the caller instruments the step.
            actions = self.memory.recent_actions()
            self.freespace_gate.observe_step(
                commanded_action=actions[-1] if actions else "",
                collision=collision,
            )
        decision = self.freespace_gate.evaluate()
        if not decision.forward_allowed:
            # A newly blocked heading invalidates a probe that was queued for
            # the old one, so replanning beats finishing a disproved macro.
            if self._escape_queue and self._escape_queue[0] == FORWARD:
                self._escape_queue.clear()
            if self.show_output and decision.reason != self._last_freespace.reason:
                print(f"[FreeSpaceGate blocked FORWARD] {decision.reason}", flush=True)
        self._last_freespace = decision
        return decision

    def _latest_evidence(self) -> Optional[Any]:
        """The most recent image-free transition record, if temporal memory ran."""
        if self.temporal_memory is None:
            return None
        evidence = self.temporal_memory.recent_evidence()
        return evidence[-1] if evidence else None

    @staticmethod
    def _reported_collision(
        previous_execution: Optional[Mapping[str, Any] | StepExecution],
    ) -> Optional[bool]:
        """Read the simulator's collision flag when the caller supplied one."""
        if previous_execution is None:
            return None
        if isinstance(previous_execution, StepExecution):
            return previous_execution.collision
        value = previous_execution.get("collision")
        return None if value is None else bool(value)

    def _actor_context(self, free_space: FreeSpaceDecision) -> str:
        """Tell ModelB why FORWARD vanished, so the mask reads as a reason."""
        context = self.thinker.actor_context()
        if free_space.forward_allowed:
            return context
        return f"{context}\nFORWARD is NOT allowed this step ({free_space.reason})."

    def _next_escape_action(
        self,
        rgb_image: Any,
        free_space: FreeSpaceDecision,
    ) -> Optional[str]:
        """Return the next primitive of a vision-chosen escape, if one is due.

        A single blocked step is left to ModelB -- masking FORWARD is usually
        enough to make it turn. Only a robot still stuck after `escape_after`
        blocked steps gets the scripted macro, because by then its own turning
        has demonstrably failed to find a way through.
        """
        if not self._escape_queue:
            if free_space.forward_allowed or not self.freespace_gate.needs_escape():
                return None
            escape = self._plan_escape(rgb_image)
            self._escape_queue.extend(
                escape.actions(
                    probe_forward_steps=self.freespace_gate.probe_forward_steps
                )
            )
            self.memory.record_event(
                "Escape Plan",
                escape.describe(turn_degrees=self.freespace_gate.turn_degrees),
            )
            if self.show_output:
                print(
                    "[FreeSpaceGate escape] "
                    f"{escape.describe(turn_degrees=self.freespace_gate.turn_degrees)}",
                    flush=True,
                )
        return self._escape_queue[0]

    def _plan_escape(self, rgb_image: Any) -> EscapeDirection:
        """Scout a traversable heading, falling back to a deterministic scan.

        A scout whose bearings are systematically off would otherwise keep
        pointing confidently at the same wrong place. After it has disproved
        `max_scout_attempts` headings the gate stops asking and scans instead:
        slower, but it provably covers every bearing.
        """
        proposal = None
        scout_exhausted = (
            self.freespace_gate.blocked_heading_count > self.max_scout_attempts
        )
        if scout_exhausted and self.show_output:
            print(
                "[FreeSpaceGate escape] the scout has been wrong "
                f"{self.freespace_gate.blocked_heading_count} times; "
                "scanning for an opening instead",
                flush=True,
            )
        if self.scout_escape_direction and not scout_exhausted:
            try:
                proposal = self.thinker.propose_escape(
                    rgb_image,
                    blocked_note=self.freespace_gate.blocked_note(),
                    turn_degrees=self.freespace_gate.turn_degrees,
                    max_turn_steps=self.freespace_gate.max_turn_steps,
                )
            except Exception as exc:  # a failed scout must not end the episode
                if self.show_output:
                    print(f"[FreeSpaceGate escape scout failed] {exc}", flush=True)
        if proposal is None:
            return self.freespace_gate.fallback_escape()
        return self.freespace_gate.adjust(proposal)

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
    parser.add_argument(
        "--forward-block-after",
        type=int,
        default=1,
        help="Failed FORWARD steps before FORWARD is masked out of the action space.",
    )
    parser.add_argument(
        "--escape-after",
        type=int,
        default=2,
        help="Blocked steps before the scouted escape macro overrides the policy.",
    )
    parser.add_argument(
        "--escape-probe-steps",
        type=int,
        default=2,
        help="FORWARD steps used to probe a newly scouted heading.",
    )
    parser.add_argument(
        "--no-escape-scout",
        action="store_true",
        help="Escape with a deterministic sweep instead of asking ModelA for an opening.",
    )
    parser.add_argument(
        "--max-scout-attempts",
        type=int,
        default=3,
        help="Headings a wrong escape scout may disprove before the scan takes over.",
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
            forward_block_after=args.forward_block_after,
            escape_after=args.escape_after,
            escape_probe_steps=args.escape_probe_steps,
            scout_escape_direction=not args.no_escape_scout,
            max_scout_attempts=args.max_scout_attempts,
        )
    run_terminal(agent)


if __name__ == "__main__":
    main()
