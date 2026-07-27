#!/usr/bin/env python3
"""Render a Qwen Temporal Memory report as a browser-compatible MP4."""

from __future__ import annotations

import argparse
import json
import math
import os
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np


PANEL_HEIGHT = 200
FONT = cv2.FONT_HERSHEY_SIMPLEX


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def wrap_text(
    text: str,
    *,
    max_width: int,
    scale: float,
    thickness: int,
    max_lines: int = 2,
) -> list[str]:
    words = str(text).split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        width = cv2.getTextSize(
            candidate,
            FONT,
            scale,
            thickness,
        )[0][0]
        if current and width > max_width:
            lines.append(current)
            current = word
            if len(lines) == max_lines:
                break
        else:
            current = candidate
    if len(lines) < max_lines and current:
        lines.append(current)
    if len(lines) == max_lines:
        rendered = " ".join(lines)
        if len(rendered.split()) < len(words):
            lines[-1] = lines[-1].rstrip(".") + "..."
    return lines


def state_at(report: dict[str, Any], timestamp: float) -> dict[str, Any]:
    subgoals = report["tracked_subgoals"]
    events = [
        event
        for event in report["tracking"]["events"]
        if float(event["timestamp_seconds"]) <= timestamp + 1e-9
    ]
    completed_ids: list[str] = []
    for event in events:
        subgoal_id = str(event["subgoal_id"])
        if bool(event["value"]) and subgoal_id not in completed_ids:
            completed_ids.append(subgoal_id)

    current_index = next(
        (
            index
            for index, subgoal in enumerate(subgoals)
            if str(subgoal["subgoal_id"]) not in completed_ids
        ),
        None,
    )
    latest = events[-1] if events else None
    window_by_id = {
        int(item["window_id"]): item
        for item in report["tracking"]["windows"]
    }
    latest_window = (
        window_by_id.get(int(latest["window_id"]))
        if latest is not None
        else None
    )

    if current_index is None:
        status = "TASK COMPLETE"
        status_color = (80, 220, 100)
        current = None
        collected = 8
    else:
        current = subgoals[current_index]
        activation_time = (
            0.0
            if current_index == 0
            else next(
                float(event["timestamp_seconds"])
                for event in events
                if bool(event["value"])
                and str(event["subgoal_id"])
                == str(subgoals[current_index - 1]["subgoal_id"])
            )
        )
        sampled_times = report["sampling"]["timestamps_seconds"]
        if current_index == 0:
            collected = sum(
                float(sample_time) <= timestamp + 1e-9
                for sample_time in sampled_times
            )
        else:
            collected = sum(
                activation_time < float(sample_time) <= timestamp + 1e-9
                for sample_time in sampled_times
            )
        collected = min(collected, 8)
        latest_is_current = (
            latest is not None
            and str(latest["subgoal_id"]) == str(current["subgoal_id"])
        )
        if not latest_is_current:
            status = f"COLLECTING EVIDENCE {collected}/8"
            status_color = (40, 210, 255)
        elif bool(latest["value"]):
            status = "SUBGOAL COMPLETED"
            status_color = (80, 220, 100)
        else:
            status = "CONTINUING CURRENT SUBGOAL"
            status_color = (40, 165, 255)

    return {
        "completed_ids": completed_ids,
        "current_index": current_index,
        "current": current,
        "latest_event": latest,
        "latest_window": latest_window,
        "status": status,
        "status_color": status_color,
        "collected": collected,
    }


def draw_panel(
    frame: np.ndarray,
    report: dict[str, Any],
    timestamp: float,
) -> np.ndarray:
    height, width = frame.shape[:2]
    canvas = np.zeros((height + PANEL_HEIGHT, width, 3), dtype=np.uint8)
    canvas[:height] = frame
    panel = canvas[height:]
    panel[:] = (20, 24, 31)
    cv2.line(panel, (0, 0), (width, 0), (80, 95, 115), 2)

    state = state_at(report, timestamp)
    total = len(report["tracked_subgoals"])
    completed = len(state["completed_ids"])
    latest = state["latest_event"]
    latest_window = state["latest_window"]

    cv2.putText(
        panel,
        (
            f"Qwen3-VL Temporal Memory | Episode {report['episode_id']} | "
            f"t={timestamp:04.1f}s"
        ),
        (20, 27),
        FONT,
        0.64,
        (225, 230, 238),
        2,
        cv2.LINE_AA,
    )

    if state["current"] is None:
        current_label = f"Current subgoal: NONE | All {total} completed"
        description = "Navigation task is complete; tracking is stopped."
    else:
        current_number = int(state["current_index"]) + 1
        current_label = f"Current subgoal {current_number}/{total}"
        description = state["current"]["description"]
    cv2.putText(
        panel,
        current_label,
        (20, 57),
        FONT,
        0.63,
        (255, 205, 80),
        2,
        cv2.LINE_AA,
    )
    for line_index, line in enumerate(
        wrap_text(
            description,
            max_width=width - 40,
            scale=0.56,
            thickness=1,
        )
    ):
        cv2.putText(
            panel,
            line,
            (20, 82 + line_index * 23),
            FONT,
            0.56,
            (210, 218, 228),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        panel,
        f"Status: {state['status']}",
        (20, 132),
        FONT,
        0.59,
        state["status_color"],
        2,
        cv2.LINE_AA,
    )
    if latest is None:
        latest_text = "Latest VLM result: waiting for the first 8-frame window"
    else:
        verdict = "TRUE / COMPLETED" if latest["value"] else "FALSE / CONTINUE"
        latency = float(latest_window["result"]["latency_ms"])
        latest_text = (
            f"Latest W{latest['window_id']} | subgoal "
            f"{latest['subgoal_id']} -> {verdict} | {latency:.0f}ms"
        )
    cv2.putText(
        panel,
        latest_text,
        (20, 158),
        FONT,
        0.52,
        (170, 184, 200),
        1,
        cv2.LINE_AA,
    )

    bar_x, bar_y = 20, 174
    bar_width, bar_height = width - 190, 14
    cv2.rectangle(
        panel,
        (bar_x, bar_y),
        (bar_x + bar_width, bar_y + bar_height),
        (70, 78, 90),
        -1,
    )
    fill_width = round(bar_width * completed / max(total, 1))
    if fill_width:
        cv2.rectangle(
            panel,
            (bar_x, bar_y),
            (bar_x + fill_width, bar_y + bar_height),
            (80, 205, 110),
            -1,
        )
    cv2.rectangle(
        panel,
        (bar_x, bar_y),
        (bar_x + bar_width, bar_y + bar_height),
        (130, 140, 155),
        1,
    )
    cv2.putText(
        panel,
        f"Task progress {completed}/{total}",
        (bar_x + bar_width + 15, 187),
        FONT,
        0.48,
        (220, 225, 232),
        1,
        cv2.LINE_AA,
    )
    return canvas


def render(report_path: Path, output_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if "qwen3" not in str(report.get("model_path", "")).lower():
        raise ValueError("the report must come from Qwen3-VL")
    source_path = Path(report["video_path"])
    if source_path.resolve() == output_path.resolve():
        raise ValueError("output must not overwrite the source video")

    source_container = av.open(str(source_path))
    source_stream = source_container.streams.video[0]
    fps = Fraction(source_stream.average_rate)
    width = int(source_stream.codec_context.width)
    height = int(source_stream.codec_context.height)
    output_height = height + PANEL_HEIGHT
    if width % 2 or output_height % 2:
        raise ValueError("H.264 yuv420p output dimensions must be even")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = output_path.with_name(
        f".{output_path.stem}.{os.getpid()}.tmp.mp4"
    )
    output_container = av.open(
        str(staging_path),
        mode="w",
        format="mp4",
        options={"movflags": "+faststart"},
    )
    output_stream = output_container.add_stream(
        "libx264",
        rate=fps,
        options={"crf": "20", "preset": "fast"},
    )
    output_stream.width = width
    output_stream.height = output_height
    output_stream.pix_fmt = "yuv420p"
    time_base = Fraction(fps.denominator, fps.numerator)

    written = 0
    try:
        for decoded in source_container.decode(source_stream):
            timestamp = written / float(fps)
            frame = decoded.to_ndarray(format="bgr24")
            annotated = draw_panel(frame, report, timestamp)
            output_frame = av.VideoFrame.from_ndarray(
                annotated,
                format="bgr24",
            )
            output_frame.pts = written
            output_frame.time_base = time_base
            for packet in output_stream.encode(output_frame):
                output_container.mux(packet)
            written += 1
        for packet in output_stream.encode():
            output_container.mux(packet)
    except Exception:
        output_container.close()
        source_container.close()
        staging_path.unlink(missing_ok=True)
        raise
    output_container.close()
    source_container.close()
    os.replace(staging_path, output_path)

    raw = output_path.read_bytes()
    with av.open(str(output_path)) as verify_container:
        stream = verify_container.streams.video[0]
        codec = stream.codec_context
        pts = [
            frame.pts
            for frame in verify_container.decode(stream)
        ]
        metadata = {
            "output": str(output_path.resolve()),
            "codec": codec.name,
            "codec_tag": codec.codec_tag,
            "profile": codec.profile,
            "pixel_format": codec.format.name,
            "width": codec.width,
            "height": codec.height,
            "fps": float(stream.average_rate),
            "frame_count": len(pts),
            "duration_seconds": len(pts) / float(stream.average_rate),
            "faststart": (
                raw.find(b"moov") >= 0
                and raw.find(b"moov") < raw.find(b"mdat")
            ),
        }
    if metadata["codec"] != "h264":
        raise RuntimeError("rendered video is not H.264")
    if metadata["pixel_format"] != "yuv420p":
        raise RuntimeError("rendered video is not yuv420p")
    if len(pts) != written or any(
        right <= left for left, right in zip(pts, pts[1:])
    ):
        raise RuntimeError("rendered video has invalid frame timestamps")
    if not metadata["faststart"]:
        raise RuntimeError("rendered video is missing MP4 faststart metadata")
    return metadata


def main() -> int:
    args = parse_args()
    metadata = render(args.report, args.output)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
