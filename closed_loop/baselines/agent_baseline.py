"""Agent baseline implementation using the solver from quick_start.py."""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from dotenv import load_dotenv

# Adjust path to ensure agentflow can be imported if running from FreeAskAgent root
import sys
if str(Path(__file__).parents[2]) not in sys.path:
    sys.path.append(str(Path(__file__).parents[2]))

try:
    from agentflow.agentflow.solver import construct_solver
    from agentflow.agentflow.solver_fast import construct_fast_solver
except ImportError:
    # Fallback or error handling if agentflow is not found
    print("Warning: agentflow not found. AgentBaseline will fail.")
    construct_solver = None
    construct_fast_solver = None

from ..freeaskworld_connector.framework import BaselineResponse, BaselineSession, ClosedLoopBaseline, MessageEnvelope
from ..freeaskworld_connector.messages import NavigationCommand, Step, TransformData

class AgentBaseline(ClosedLoopBaseline):
    """Baseline that uses the AgentFlow solver to control the agent."""

    def __init__(self) -> None:
        self._initialized = False
        self._solver = None
        self._setup_solver()
        self._temp_dir = tempfile.TemporaryDirectory()

    def _setup_solver(self):
        # Load environment variables
        env_path = Path(__file__).parents[2] / "agentflow" / ".env"
        load_dotenv(dotenv_path=env_path)
        
        print("Proxy_API_BASE:" + os.environ.get("Proxy_API_BASE", "Not Set"))
        print("OPENAI_API_KEY:" + os.environ.get("OPENAI_API_KEY", "Not Set"))
        
        llm_engine_name = "gpt-4o"
        
        # Using the configuration from quick_start.py (FAST_MODE = False)
        if construct_solver:
            self._solver = construct_solver(
                llm_engine_name=llm_engine_name,
                enabled_tools=["Base_Generator_Tool", "GroundedSAM2_Tool"],
                tool_engine=["gpt-4o"],
                model_engine=["gpt-4o", "gpt-4o", "gpt-4o", "gpt-4o"],
                output_types="direct",
                max_time=300,
                max_steps=1,
                enable_multimodal=True
            )
        else:
            print("Error: Could not construct solver.")

    async def on_session_start(self, session: BaselineSession) -> None:
        session.metadata.clear()
        session.metadata["frame_history"] = []

    async def on_session_end(self, session: BaselineSession) -> None:
        # Cleanup temp files for this session
        if "frame_history" in session.metadata:
            for path_str in session.metadata["frame_history"]:
                try:
                    Path(path_str).unlink(missing_ok=True)
                except Exception:
                    pass
        session.state.clear()
        # Cleanup temp dir if needed, but we keep it for the lifetime of the baseline instance

    async def handle_envelope(
        self, session: BaselineSession, envelope: MessageEnvelope
    ) -> BaselineResponse | None:
        if envelope.message_type == "json":
            self._handle_json_packet(session, envelope)
        elif envelope.message_type == "rgbd":
            session.metadata["has_rgbd"] = True
            session.state.update_rgbd(envelope.payload)

        if not self._ready_to_respond(session):
            return None

        # Get the latest RGB image
        rgb_frame = session.state.latest_rgbd.color
        # Save to temp file
        img_path = Path(self._temp_dir.name) / f"frame_{session.session_id}_{int(asyncio.get_event_loop().time())}_{len(session.metadata.get('frame_history', []))}.jpg"
        Image.fromarray(rgb_frame).save(img_path)
        
        # Update history
        history = session.metadata.setdefault("frame_history", [])
        history.append(str(img_path))
        if len(history) > 5:
            # Remove old file
            old_file = history.pop(0)
            try:
                Path(old_file).unlink(missing_ok=True)
            except Exception:
                pass

        # Run solver
        # We run this in a separate thread to avoid blocking the async loop
        # Pass the full history
        output = await asyncio.to_thread(self._run_solver, history)
        
        # Parse output
        direct_output = output.get("direct_output", "")
        print(f"Agent Output: {direct_output}")
        
        navigation, text_response = self._parse_output(direct_output)
        
        step = Step()

        messages = [
            {
                "type": "json",
                "json_type": "NavigationCommand",
                "content": navigation.to_dict(),
            },
            {
                "type": "json",
                "json_type": "Step",
                "content": step.to_dict(),
            }
        ]
        
        if text_response:
             messages.append({
                "type": "json",
                "json_type": "AgentText",
                "content": {"text": text_response}
            })

        return BaselineResponse(
            messages=messages,
            reset_state=True,
        )

    def _run_solver(self, image_paths: List[str]) -> Dict[str, Any]:
        navigation_task_prompt = """
[Task]
请描述视野中的可行动作并选出后续一连串的导航轨迹指令
你要去面包店
[Rules]
要躲避物体不要撞上
当你离人2m内的时候就可以触发问路
[Policy]
使用最快获取信息的策略，你可选择自己不断探索地点，也可以问人来快速获取信息，尽管可能不精准
[Action Space]
动作空间是[前进，左转，右转，后转, 后退, 停止, 问路][1m, 2m, 3m]
每次动作只能选择一个动作和一个距离, 比如'前进2m'
[Output Format]
请给出后续5步的导航指令序列。
[Tools]
你可以使用GroundedSAM2_Tool来识别图像中的物体，自己设置prompt比如obst，获取物体的位置和类别信息，辅助你做出导航决策。你可以获取obstacle,street,building等信息
[Image Sequence]
这里有一系列按时间顺序排列的图像帧，展示了你当前的视野。请根据这些图像帧来理解环境。
"""
        if self._solver:
            return self._solver.solve(
                navigation_task_prompt,
                image_paths=image_paths,
            )
        return {}

    def _parse_output(self, output_text: str) -> Tuple[NavigationCommand, str]:
        # Default to stop
        pos_offset = np.zeros(3, dtype=float)
        rot_offset = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        is_stopped = False
        
        # Simple parsing logic
        # Look for the first valid action in the text
        # Regex for actions
        action_pattern = r"(前进|后退|左转|右转|后转|停止|问路)\s*(\d+m)?"
        
        # We assume the output contains a list, we take the first one
        # Example: "1. 前进2m"
        
        match = re.search(action_pattern, output_text)
        if match:
            action = match.group(1)
            distance_str = match.group(2)
            distance = float(distance_str.replace('m', '')) if distance_str else 0.0
            
            if action == "前进":
                # Forward is +Z in local frame (assuming)
                pos_offset[2] = distance
            elif action == "后退":
                pos_offset[2] = -distance
            elif action == "左转":
                # Rotate left (around Y axis)
                # Assuming 90 degrees for turn? Or maybe distance implies angle?
                # Usually "Turn Left" is a discrete action. Let's assume 90 degrees.
                # Quaternion for 90 degrees around Y (0, 1, 0)
                # q = [x, y, z, w] = [0, sin(45), 0, cos(45)] = [0, 0.707, 0, 0.707]
                # Note: Check coordinate system. If Y is up.
                # Left turn -> -90 degrees?
                # Let's use -90 degrees (Right Hand Rule around Y points up -> CCW is positive. Left turn is usually CCW).
                # Wait, if I face Z, Left is -X. Turning left means rotating towards +X? No, rotating towards -X.
                # That is +90 degrees around Y?
                # Let's assume +90 degrees around Y.
                # sin(45) = 0.7071, cos(45) = 0.7071
                rot_offset = np.array([0.0, 0.70710678, 0.0, 0.70710678]) 
            elif action == "右转":
                # -90 degrees around Y
                # sin(-45) = -0.7071
                rot_offset = np.array([0.0, -0.70710678, 0.0, 0.70710678])
            elif action == "后转":
                # 180 degrees around Y
                # sin(90) = 1, cos(90) = 0
                rot_offset = np.array([0.0, 1.0, 0.0, 0.0])
            elif action == "停止":
                is_stopped = True
            elif action == "问路":
                pass # No movement
                
        return NavigationCommand(
            LocalPositionOffset=pos_offset,
            LocalRotationOffset=rot_offset,
            IsStopped=is_stopped
        ), output_text

    def _handle_json_packet(self, session: BaselineSession, envelope: MessageEnvelope) -> None:
        packet = envelope.payload
        session.metadata.setdefault("json_types", set()).add(packet.json_type)

        if packet.json_type == "Init":
            self._initialized = True
        elif packet.json_type == "TransformData" and isinstance(packet.content, dict):
            try:
                session.metadata["transform"] = TransformData.from_dict(packet.content)
            except (KeyError, TypeError, ValueError):
                session.metadata["transform"] = packet.content

    def _ready_to_respond(self, session: BaselineSession) -> bool:
        if not self._initialized:
            return False

        if session.state.latest_rgbd is None:
            return False

        packets = session.state.json_packets
        # We wait for Instruction and TransformData as in simple_baseline, 
        # but maybe we don't strictly need Instruction if we have a fixed task.
        # However, to be safe and consistent with the protocol:
        return "Instruction" in packets and "TransformData" in packets


def create_baseline() -> ClosedLoopBaseline:
    """Factory required by the server to instantiate the baseline."""
    return AgentBaseline()
