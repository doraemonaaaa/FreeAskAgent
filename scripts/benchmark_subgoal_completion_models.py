#!/usr/bin/env python3
"""Paired latency benchmark for binary VLN subgoal completion."""

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

from eval_temporal_captioner_examples import (
    DEFAULT_VIDEO_DIR,
    load_task_spec,
    rolling_window_ranges,
    sample_left_view,
    write_report,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--episodes", nargs="+", default=["505", "632"])
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--max-image-edge", type=int, default=224)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def latency_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
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
            max_tokens=1,
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

    requests = []
    manifest = []
    for episode_id in args.episodes:
        video = args.video_dir / f"{episode_id}.mp4"
        spec = args.video_dir / f"{episode_id}_subgoals.txt"
        task, guidance, raw_subgoals = load_task_spec(spec)
        frames, sampling = sample_left_view(
            video,
            sample_fps=args.sample_fps,
        )
        ranges = rolling_window_ranges(
            len(frames),
            window_size=8,
            stride=1,
        )
        selected = sorted({0, len(ranges) // 2, len(ranges) - 1})
        for raw_subgoal in raw_subgoals:
            subgoal = Subgoal(**raw_subgoal)
            for range_index in selected:
                start, end = ranges[range_index]
                request_id = (
                    f"{episode_id}:subgoal-{subgoal.subgoal_id}:"
                    f"window-{range_index + 1}"
                )
                request = TemporalAnalysisRequest(
                    episode_id=episode_id,
                    task=task,
                    task_guidance=guidance,
                    subgoals=(subgoal,),
                    steps=tuple(
                        TemporalStepInput(
                            step_id=start + offset + 1,
                            action=None,
                            image=frame,
                            timestamp_seconds=sampling[
                                "timestamps_seconds"
                            ][start + offset],
                        )
                        for offset, frame in enumerate(frames[start:end])
                    ),
                )
                requests.append((request_id, request))
                manifest.append(
                    {
                        "request_id": request_id,
                        "episode_id": episode_id,
                        "subgoal_id": subgoal.subgoal_id,
                        "window_id": range_index + 1,
                        "observation_ids": list(range(start + 1, end + 1)),
                        "source_frame_indices": sampling["frame_indices"][
                            start:end
                        ],
                    }
                )

    if not requests:
        raise RuntimeError("benchmark manifest is empty")

    # Warm model kernels and processor caches once; exclude this call from the
    # paired measurements used to compare checkpoints.
    captioner.evaluate_subgoal(requests[0][1])
    captioner.reset_performance_stats()

    results = []
    failed = False
    for request_id, request in requests:
        started = time.perf_counter()
        try:
            result = captioner.evaluate_subgoal(request)
            wall_ms = (time.perf_counter() - started) * 1000
            results.append(
                {
                    "request_id": request_id,
                    "status": "success",
                    "completed": result.completed,
                    "raw_response": result.raw_response,
                    "model_call_ms": result.latency_ms,
                    "window_e2e_ms": wall_ms,
                }
            )
            print(
                f"{request_id} completed={result.completed} "
                f"model={result.latency_ms:.1f}ms "
                f"e2e={wall_ms:.1f}ms",
                flush=True,
            )
        except Exception as exc:
            failed = True
            results.append(
                {
                    "request_id": request_id,
                    "status": "failed",
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                    "raw_response": captioner.last_raw_response,
                }
            )
            print(
                f"{request_id} failed: {type(exc).__name__}: {exc}",
                flush=True,
            )

    successful = [item for item in results if item["status"] == "success"]
    model_slug = re.sub(
        r"[^a-z0-9]+",
        "-",
        args.model_path.name.lower(),
    ).strip("-")
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "partial" if failed else "success",
        "model": model_slug,
        "model_path": str(args.model_path.resolve()),
        "protocol": {
            "episodes": args.episodes,
            "sample_fps": args.sample_fps,
            "window_size": 8,
            "selected_windows": "first,middle,last",
            "max_image_edge": args.max_image_edge,
            "max_new_tokens": 1,
            "warmup_calls": 1,
            "input": "one subgoal instruction plus eight ordered images",
            "output": "exact true or false",
        },
        "manifest": manifest,
        "summary": {
            "success_count": len(successful),
            "failure_count": len(results) - len(successful),
            "model_call": latency_stats(
                [item["model_call_ms"] for item in successful]
            ),
            "window_e2e": latency_stats(
                [item["window_e2e_ms"] for item in successful]
            ),
        },
        "results": results,
    }
    write_report(args.output, report)
    print(f"report={args.output}", flush=True)
    print(json.dumps(report["summary"], ensure_ascii=False), flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
