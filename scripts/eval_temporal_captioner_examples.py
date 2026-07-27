#!/usr/bin/env python3
"""Evaluate TemporalCaptioner on the 505/632 first-person example videos."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

import cv2


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_DIR = Path(
    "/data/pengyh/workspace/FreeAskAgent_R2R/videos/example"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--episodes", nargs="+", default=["505", "632"])
    parser.add_argument(
        "--model-path",
        type=Path,
        default=REPO_ROOT / "models/Qwen3-VL-8B-Instruct",
    )
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--max-image-edge", type=int, default=224)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--analysis-stride", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "tmp/temporal_captioner_examples",
    )
    parser.add_argument(
        "--events-only",
        action="store_true",
        help=(
            "Write one compact report containing only emitted events, raw "
            "boolean model responses, and measured latency."
        ),
    )
    return parser.parse_args()


def load_task_spec(path: Path) -> tuple[str, str, list[dict[str, str]]]:
    text = path.read_text(encoding="utf-8")

    def section(start: str, end: str) -> str:
        match = re.search(
            rf"\[{re.escape(start)}\]\s*(.*?)\s*\[{re.escape(end)}\]",
            text,
            flags=re.DOTALL,
        )
        if not match:
            raise ValueError(f"Missing [{start}] section in {path}")
        return match.group(1).strip()

    task = section("Task", "Task Guidance Logic")
    guidance = section("Task Guidance Logic", "Subgoals")
    block = section("Subgoals", "Required Evaluation Output")
    matches = re.finditer(
        r"(?ms)^(\d+)\.\s+([^\n]+)\n"
        r"\s*Completion evidence:\s*(.+?)"
        r"(?=\n\n\d+\.|\Z)",
        block,
    )
    subgoals = [
        {
            "subgoal_id": match.group(1),
            "description": match.group(2).strip(),
            "completion_criteria": " ".join(match.group(3).split()),
        }
        for match in matches
    ]
    if not subgoals:
        raise ValueError(f"No subgoals parsed from {path}")
    return task, guidance, subgoals


def sample_left_view(
    path: Path,
    *,
    sample_fps: float,
) -> tuple[list[Any], dict[str, Any]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frame_count < 8 or fps <= 0 or width < 640 or sample_fps <= 0:
            raise ValueError(f"Unexpected video metadata for {path}")
        frame_stride = max(int(round(fps / sample_fps)), 1)
        indices = list(range(0, frame_count, frame_stride))
        if indices[-1] != frame_count - 1:
            indices.append(frame_count - 1)
        frames = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, bgr = capture.read()
            if not ok or bgr is None:
                raise ValueError(f"Could not read frame {index} from {path}")
            frames.append(cv2.cvtColor(bgr[:, :640], cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    return frames, {
        "source_width": width,
        "source_height": height,
        "crop": {"x_start": 0, "x_end": 640, "width": 640},
        "source_fps": fps,
        "requested_sample_fps": sample_fps,
        "actual_sample_fps": fps / frame_stride,
        "frame_count": frame_count,
        "sampled_observation_count": len(indices),
        "frame_indices": indices,
        "timestamps_seconds": [round(index / fps, 3) for index in indices],
    }


def rolling_window_ranges(
    observation_count: int,
    *,
    window_size: int,
    stride: int,
) -> list[tuple[int, int]]:
    if window_size != 8:
        raise ValueError("This evaluation uses a fixed eight-observation window")
    if stride < 1:
        raise ValueError("analysis_stride must be positive")
    if observation_count < window_size:
        return []
    return [
        (start, start + window_size)
        for start in range(
            0,
            observation_count - window_size + 1,
            stride,
        )
    ]


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def latency_stats(values: list[float]) -> dict[str, Any]:
    """Return compact millisecond statistics for successful calls."""
    if not values:
        return {"count": 0}
    ordered = sorted(float(value) for value in values)
    p95_index = max(math.ceil(0.95 * len(ordered)) - 1, 0)
    return {
        "count": len(ordered),
        "mean_ms": mean(ordered),
        "median_ms": median(ordered),
        "p95_ms": ordered[p95_index],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "under_2000ms_count": sum(value < 2000 for value in ordered),
    }


def main() -> int:
    args = parse_args()
    if args.window_size != 8:
        raise ValueError("--window-size must be 8")
    if args.analysis_stride < 1:
        raise ValueError("--analysis-stride must be positive")
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    import torch

    from agentflow.agents.models_embodied_v2.TemporalCaptioner import (
        Subgoal,
        TemporalAnalysisRequest,
        TemporalCaptioner,
        TemporalCaptionerConfig,
        TemporalStepInput,
    )

    captioner = TemporalCaptioner(
        model_path=str(args.model_path.resolve()),
        config=TemporalCaptionerConfig(
            max_tokens=args.max_tokens,
            max_image_edge=args.max_image_edge,
            latency_budget_ms=2000,
        ),
        use_cache=False,
        debug_performance=False,
        engine_kwargs={
            "torch_dtype": torch.bfloat16,
            "device_map": {"": args.device},
        },
    )
    model_slug = re.sub(
        r"[^a-z0-9]+",
        "-",
        args.model_path.name.lower(),
    ).strip("-")
    run_started = time.perf_counter()
    compact_events: list[dict[str, Any]] = []
    compact_failures: list[dict[str, Any]] = []
    episode_summaries: list[dict[str, Any]] = []
    failed = False
    for episode_id in args.episodes:
        video = args.video_dir / f"{episode_id}.mp4"
        spec = args.video_dir / f"{episode_id}_subgoals.txt"
        output = args.output_dir / f"{episode_id}_{model_slug}_result.json"
        report: dict[str, Any] = {
            "schema_version": 2,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "episode_id": episode_id,
            "model_path": str(args.model_path.resolve()),
            "video_path": str(video.resolve()),
            "status": "running",
        }
        try:
            task, guidance, raw_subgoals = load_task_spec(spec)
            frames, sampling = sample_left_view(
                video,
                sample_fps=args.sample_fps,
            )
            tracked_subgoals = raw_subgoals
            subgoal_states = {
                item["subgoal_id"]: {
                    "completed": False,
                    "completed_at_window": None,
                }
                for item in tracked_subgoals
            }
            windows = []
            events = []
            current_subgoal_index = 0
            episode_failed = False
            ranges = rolling_window_ranges(
                len(frames),
                window_size=args.window_size,
                stride=args.analysis_stride,
            )
            next_start = 0
            window_index = 0
            while (
                current_subgoal_index < len(tracked_subgoals)
                and next_start + args.window_size <= len(frames)
            ):
                window_index += 1
                start = next_start
                end = start + args.window_size
                raw_subgoal = tracked_subgoals[current_subgoal_index]
                subgoal = Subgoal(**raw_subgoal)
                observation_ids = list(range(start + 1, end + 1))
                request = TemporalAnalysisRequest(
                    episode_id=episode_id,
                    task=task,
                    task_guidance=guidance,
                    subgoals=(subgoal,),
                    steps=tuple(
                        TemporalStepInput(
                            step_id=start + index + 1,
                            action=None,
                            image=frame,
                            timestamp_seconds=sampling[
                                "timestamps_seconds"
                            ][start + index],
                        )
                        for index, frame in enumerate(frames[start:end])
                    ),
                )
                window: dict[str, Any] = {
                    "window_id": window_index,
                    "observation_ids": observation_ids,
                    "start_seconds": sampling["timestamps_seconds"][start],
                    "end_seconds": sampling["timestamps_seconds"][end - 1],
                    "subgoal_id": subgoal.subgoal_id,
                    "subgoal_activation_observation_id": start + 1,
                }
                try:
                    window_started = time.perf_counter()
                    result = captioner.evaluate_subgoal(request)
                    window_e2e_ms = (
                        time.perf_counter() - window_started
                    ) * 1000
                    window.update(
                        status="success",
                        result=result.model_dump(),
                        window_e2e_ms=window_e2e_ms,
                    )
                    print(
                        f"[{episode_id} window={window_index} "
                        f"obs={observation_ids[0]}-{observation_ids[-1]} "
                        f"subgoal={subgoal.subgoal_id}] "
                        f"completed={result.completed} "
                        f"latency={result.latency_ms:.1f}ms "
                        f"e2e={window_e2e_ms:.1f}ms",
                        flush=True,
                    )
                    event = {
                        "kind": "SUBGOAL_COMPLETED",
                        "value": bool(result.completed),
                        "subgoal_id": subgoal.subgoal_id,
                        "window_id": window_index,
                        "timestamp_seconds": window["end_seconds"],
                    }
                    events.append(event)
                    window["emitted_event"] = event
                    compact_events.append(
                        {
                            "episode_id": episode_id,
                            "window_id": window_index,
                            "subgoal_id": subgoal.subgoal_id,
                            "window_start_seconds": window["start_seconds"],
                            "window_end_seconds": window["end_seconds"],
                            "event": {
                                "kind": event["kind"],
                                "value": event["value"],
                            },
                            "qwen_raw_response": result.raw_response,
                            "captioner_call_ms": result.latency_ms,
                            "event_e2e_ms": window_e2e_ms,
                        }
                    )
                    if result.completed:
                        subgoal_states[subgoal.subgoal_id].update(
                            completed=True,
                            completed_at_window=window_index,
                        )
                        current_subgoal_index += 1
                        # The next subgoal may only use observations captured
                        # after this completion decision became available.
                        next_start = end
                    else:
                        next_start += args.analysis_stride
                except Exception as exc:
                    failed = True
                    episode_failed = True
                    next_start += args.analysis_stride
                    window.update(
                        {
                            "status": "failed",
                            "error": {
                                "type": type(exc).__name__,
                                "message": str(exc),
                            },
                            "raw_response": captioner.last_raw_response,
                        }
                    )
                    compact_failures.append(
                        {
                            "episode_id": episode_id,
                            "window_id": window_index,
                            "subgoal_id": subgoal.subgoal_id,
                            "window_start_seconds": window["start_seconds"],
                            "window_end_seconds": window["end_seconds"],
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                            "qwen_raw_response": captioner.last_raw_response,
                        }
                    )
                    print(
                        f"[{episode_id} window={window_index} "
                        f"subgoal={subgoal.subgoal_id}] "
                        f"failed: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                windows.append(window)
            current_subgoal_id = (
                tracked_subgoals[current_subgoal_index]["subgoal_id"]
                if current_subgoal_index < len(tracked_subgoals)
                else None
            )
            tracking_end_reason = (
                "all_tracked_subgoals_completed"
                if current_subgoal_id is None
                else "video_ended"
            )
            report.update(
                status="partial" if episode_failed else "success",
                task=task,
                task_guidance=guidance,
                all_subgoals=raw_subgoals,
                tracked_subgoals=tracked_subgoals,
                sampling=sampling,
                tracking={
                    "window_size": args.window_size,
                    "analysis_stride": args.analysis_stride,
                    "available_window_count": len(ranges),
                    "analyzed_window_count": len(windows),
                    "subgoal_window_reset_on_completion": True,
                    "tracking_end_reason": tracking_end_reason,
                    "current_subgoal_id_at_video_end": current_subgoal_id,
                    "subgoal_states": subgoal_states,
                    "events": events,
                    "windows": windows,
                },
            )
            episode_summaries.append(
                {
                    "episode_id": episode_id,
                    "status": report["status"],
                    "event_count": len(events),
                    "tracking_end_reason": tracking_end_reason,
                    "current_subgoal_id_at_video_end": current_subgoal_id,
                }
            )
        except Exception as exc:
            failed = True
            report.update(
                status="failed",
                error={"type": type(exc).__name__, "message": str(exc)},
                raw_response=captioner.last_raw_response,
            )
            episode_summaries.append(
                {
                    "episode_id": episode_id,
                    "status": "failed",
                    "event_count": 0,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            print(f"[{episode_id}] failed: {type(exc).__name__}: {exc}")
        if not args.events_only:
            write_report(output, report)
            print(f"[{episode_id}] report={output}", flush=True)

    if args.events_only:
        call_ms = [
            float(event["captioner_call_ms"])
            for event in compact_events
        ]
        e2e_ms = [
            float(event["event_e2e_ms"])
            for event in compact_events
        ]
        compact_report: dict[str, Any] = {
            "schema_version": 1,
            "mode": "qwen_fast_event_only",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": args.model_path.name,
            "config": {
                "episodes": args.episodes,
                "sample_fps": args.sample_fps,
                "window_size": args.window_size,
                "analysis_stride": args.analysis_stride,
                "max_image_edge": args.max_image_edge,
                "max_tokens": args.max_tokens,
            },
            "episodes": episode_summaries,
            "events": compact_events,
            "latency_ms": {
                "measurement": (
                    "captioner_call excludes lazy model loading; event_e2e "
                    "includes it when loading occurs inside that request"
                ),
                "run_wall_ms": (time.perf_counter() - run_started) * 1000,
                "first_event_e2e_ms": e2e_ms[0] if e2e_ms else None,
                "all_captioner_calls": latency_stats(call_ms),
                "steady_state_captioner_calls": latency_stats(call_ms[1:]),
                "all_event_e2e": latency_stats(e2e_ms),
                "steady_state_event_e2e": latency_stats(e2e_ms[1:]),
            },
        }
        if compact_failures:
            compact_report["failures"] = compact_failures
        compact_output = (
            args.output_dir / f"{model_slug}_fast_events.json"
        )
        write_report(compact_output, compact_report)
        print(f"events_report={compact_output}", flush=True)
        print(
            json.dumps(
                compact_report["latency_ms"],
                ensure_ascii=False,
            ),
            flush=True,
        )
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
