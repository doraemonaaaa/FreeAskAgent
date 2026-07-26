#!/usr/bin/env python3
"""Replay one real VLN step window through a TemporalCaptioner checkpoint.

The recorded episode MP4 has one initial observation followed by one
post-action frame per environment step.  Consequently, video frame ``n`` is
the post-action observation for 1-based step ``n``.  This script reconstructs
the same action -> post-frame transitions consumed by TemporalMemory and
produces a JSON report suitable for checkpoint A/B comparisons.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = Path(
    "/data/pengyh/workspace/FreeAskAgent_R2R/outputs/"
    "temporal_cumulative_eval/"
    "cumulative-5tasks-20260726T2233-tasktemporal"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a real three-step VLN window through TemporalMemory."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RUN_DIR / "results.json",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=DEFAULT_RUN_DIR / "rank_1/videos/505.mp4",
    )
    parser.add_argument("--episode-id", default="505")
    parser.add_argument("--step-start", type=int, default=471)
    parser.add_argument("--step-count", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-image-edge", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_episode(results_path: Path, episode_id: str) -> dict[str, Any]:
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    for episode in payload.get("episodes", []):
        if str(episode.get("episode_id")) == episode_id:
            return episode
    raise ValueError(
        f"Episode {episode_id!r} is absent from {results_path}"
    )


def read_rgb_frames(
    video_path: Path,
    frame_indices: list[int],
) -> tuple[dict[int, Any], dict[str, Any]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    try:
        metadata = {
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        frames: dict[int, Any] = {}
        for index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None:
                raise ValueError(
                    f"Could not read video frame {index} from {video_path}"
                )
            frames[index] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()
    return frames, metadata


def main() -> int:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    results_path = args.results.expanduser().resolve()
    video_path = args.video.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if args.step_count != 3:
        raise ValueError("This comparison uses TemporalMemory's fixed 3-step window")
    if args.step_start < 1:
        raise ValueError("--step-start must be at least 1")
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")

    episode = load_episode(results_path, str(args.episode_id))
    actions = list(episode["actions"])
    step_ids = list(
        range(args.step_start, args.step_start + args.step_count)
    )
    if step_ids[-1] > len(actions):
        raise ValueError(
            f"Requested step {step_ids[-1]}, but episode has {len(actions)} actions"
        )
    selected_actions = [actions[step_id - 1] for step_id in step_ids]
    frame_indices = [args.step_start - 1, *step_ids]
    frames, video_metadata = read_rgb_frames(video_path, frame_indices)

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    import torch

    from agentflow.agents.models_embodied_v2.TemporalCaptioner import (
        TemporalCaptioner,
        TemporalCaptionerConfig,
    )
    from agentflow.agents.models_embodied_v2.memory import (
        StepExecution,
        TemporalMemory,
        TemporalMemoryConfig,
        TemporalObservation,
    )

    device_map: Any = (
        "auto" if args.device == "auto" else {"": args.device}
    )
    captioner = TemporalCaptioner(
        model_path=str(model_path),
        config=TemporalCaptionerConfig(
            step_max_tokens=args.max_tokens,
            max_image_edge=args.max_image_edge,
            inference_latency_budget_ms=5000.0,
        ),
        use_cache=False,
        debug_performance=True,
        engine_kwargs={
            "torch_dtype": torch.bfloat16,
            "device_map": device_map,
        },
    )
    memory = TemporalMemory(
        goal=str(episode["instruction"]),
        episode_id=str(args.episode_id),
        captioner=captioner,
        config=TemporalMemoryConfig(
            window_size=3,
            analysis_stride=3,
            get_nowhere_steps=3,
            inference_latency_budget_ms=5000.0,
        ),
    )

    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "model": {
            "label": args.model_label,
            "path": str(model_path),
        },
        "source": {
            "results": str(results_path),
            "video": str(video_path),
            "video_metadata": video_metadata,
            "episode_id": str(args.episode_id),
            "instruction": episode["instruction"],
            "step_ids": step_ids,
            "temporal_step_ids": list(range(1, args.step_count + 1)),
            "actions": selected_actions,
            "frame_indices": step_ids,
            "pre_frame_index": args.step_start - 1,
        },
        "configuration": {
            "device": args.device,
            "torch_dtype": "bfloat16",
            "window_size": 3,
            "analysis_stride": 3,
            "max_tokens": args.max_tokens,
            "max_image_edge": args.max_image_edge,
            "latency_budget_ms": 5000.0,
        },
    }

    started = time.perf_counter()
    record = None
    try:
        for offset, (source_step_id, action) in enumerate(
            zip(step_ids, selected_actions)
        ):
            temporal_step_id = offset + 1
            pre_index = source_step_id - 1
            post_index = source_step_id
            pre = TemporalObservation(
                image=frames[pre_index],
                episode_id=str(args.episode_id),
                timestamp_seconds=pre_index / video_metadata["fps"],
            )
            post = TemporalObservation(
                image=frames[post_index],
                episode_id=str(args.episode_id),
                timestamp_seconds=post_index / video_metadata["fps"],
            )
            memory.stage_action(pre, action, "")
            memory.complete_pending_step(
                post,
                StepExecution(
                    step_id=temporal_step_id,
                    commanded_action=action,
                ),
                "",
            )
            candidate = memory.analyze_if_ready()
            if offset == args.step_count - 1:
                record = candidate

        if record is None:
            raise RuntimeError(
                "TemporalMemory did not produce a record for the completed window"
            )
        report["status"] = "success"
        report["record"] = record.model_dump(
            mode="json",
            exclude={"raw_response"},
        )
        report["raw_response"] = record.raw_response
        report["memory_context"] = memory.context()
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        report["last_raw_response"] = captioner.last_raw_response
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        report["total_elapsed_ms"] = (
            time.perf_counter() - started
        ) * 1000
        report["diagnostics"] = memory.diagnostics(
            include_raw_response=True
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report: {output_path}")
    return 0 if report["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
