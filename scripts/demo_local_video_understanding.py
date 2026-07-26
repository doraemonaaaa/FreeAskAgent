#!/usr/bin/env python3
"""Run a local Qwen3-VL video-understanding demo on one view of a split video.

The source VLN recording contains an egocentric view on the left and a
topological map on the right.  This script first materializes a view-only clip
from t=0 to the requested query end, then evaluates either explicit timestamped
frames, Qwen's native video processor, or both.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import struct
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = Path("/data/pengyh/workspace/FreeAskAgent_R2R/videos/632.mp4")
DEFAULT_MODEL_PATH = REPO_ROOT / "models/Qwen3-VL-8B-Instruct"
DEFAULT_OUTPUT = REPO_ROOT / "tmp/video_demo/632_qwen3vl.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crop one view from a split VLN video and evaluate local "
            "Qwen3-VL temporal understanding."
        )
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument(
        "--view",
        choices=("left", "right", "full"),
        default="left",
        help="View to send to the model. The VLN egocentric view is 'left'.",
    )
    parser.add_argument("--query-start", type=float, default=5.0)
    parser.add_argument("--query-end", type=float, default=10.0)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=40)
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help="Per-frame pixel budget. Defaults to the prepared frame area.",
    )
    parser.add_argument(
        "--mode",
        choices=("timestamped", "native", "both"),
        default="timestamped",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON report path.",
    )
    parser.add_argument(
        "--prepared-video",
        type=Path,
        default=None,
        help=(
            "Optional view-only clip path. By default this is created under "
            "/tmp using the source stem, view, and query end."
        ),
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Create and validate the cropped clip without loading the model.",
    )
    parser.add_argument(
        "--debug-performance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable LocalQwen3VL's detailed generation timing output.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.query_start < 0:
        raise ValueError("--query-start must be non-negative")
    if args.query_end <= args.query_start:
        raise ValueError("--query-end must be greater than --query-start")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if args.max_pixels is not None and args.max_pixels <= 0:
        raise ValueError("--max-pixels must be positive")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")


def probe_video(video_path: Path) -> dict[str, Any]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
    finally:
        cap.release()

    if width <= 0 or height <= 0 or fps <= 0 or frame_count <= 0:
        raise ValueError(f"Invalid video metadata: {video_path}")

    return {
        "path": str(video_path.resolve()),
        "size_bytes": video_path.stat().st_size,
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": frame_count / fps,
        "codec_fourcc": "".join(
            chr((fourcc >> (8 * index)) & 0xFF) for index in range(4)
        ),
    }


def view_bounds(view: str, width: int) -> tuple[int, int]:
    midpoint = width // 2
    if view == "left":
        return 0, midpoint
    if view == "right":
        return midpoint, width
    return 0, width


def prepare_view_clip(
    source_path: Path,
    output_path: Path,
    view: str,
    clip_end_seconds: float,
) -> dict[str, Any]:
    """Write a browser-compatible H.264 clip for [0, clip_end_seconds)."""
    import av

    if source_path.resolve() == output_path.resolve():
        raise ValueError("--prepared-video must not overwrite the source video")

    source = probe_video(source_path)
    if clip_end_seconds > source["duration_seconds"] + 1e-6:
        raise ValueError(
            f"Query end {clip_end_seconds:.3f}s exceeds source duration "
            f"{source['duration_seconds']:.3f}s"
        )

    x_start, x_end = view_bounds(view, source["width"])
    output_width = x_end - x_start
    output_height = source["height"]
    if output_width <= 0:
        raise ValueError(f"Cannot extract {view!r} view from width {source['width']}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = output_path.with_name(
        f".{output_path.stem}.{os.getpid()}.tmp{output_path.suffix}"
    )
    target_frames = min(
        source["frame_count"],
        int(math.ceil(clip_end_seconds * source["fps"] - 1e-9)),
    )

    input_container = None
    output_container = None
    written_frames = 0
    try:
        input_container = av.open(str(source_path))
        input_stream = input_container.streams.video[0]
        output_container = av.open(
            str(staging_path),
            mode="w",
            format="mp4",
            options={"movflags": "+faststart"},
        )
        output_rate = Fraction(str(source["fps"])).limit_denominator(100_000)
        output_stream = output_container.add_stream(
            "libx264",
            rate=output_rate,
            options={"crf": "18", "preset": "fast"},
        )
        output_stream.width = output_width
        output_stream.height = output_height
        output_stream.pix_fmt = "yuv420p"
        frame_time_base = Fraction(
            output_rate.denominator,
            output_rate.numerator,
        )

        for decoded_frame in input_container.decode(input_stream):
            if written_frames >= target_frames:
                break
            frame = decoded_frame.to_ndarray(format="bgr24")
            cropped = frame[:, x_start:x_end]
            if cropped.shape[1] != output_width or cropped.shape[0] != output_height:
                raise RuntimeError(
                    f"Unexpected crop shape at frame {written_frames}: {cropped.shape}"
                )
            output_frame = av.VideoFrame.from_ndarray(cropped, format="bgr24")
            output_frame.pts = written_frames
            output_frame.time_base = frame_time_base
            for packet in output_stream.encode(output_frame):
                output_container.mux(packet)
            written_frames += 1

        for packet in output_stream.encode():
            output_container.mux(packet)
    except Exception:
        staging_path.unlink(missing_ok=True)
        raise
    finally:
        if output_container is not None:
            output_container.close()
        if input_container is not None:
            input_container.close()

    if written_frames != target_frames:
        staging_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Expected {target_frames} frames but wrote {written_frames}"
        )
    os.replace(staging_path, output_path)

    prepared = probe_video(output_path)
    expected = {
        "width": output_width,
        "height": output_height,
        "frame_count": target_frames,
        "fps": source["fps"],
    }
    for key in ("width", "height", "frame_count"):
        if prepared[key] != expected[key]:
            raise RuntimeError(
                f"Prepared video {key}={prepared[key]!r}; expected {expected[key]!r}"
            )
    if not math.isclose(prepared["fps"], expected["fps"], rel_tol=0, abs_tol=0.01):
        raise RuntimeError(
            f"Prepared video fps={prepared['fps']}; expected {expected['fps']}"
        )
    if prepared["codec_fourcc"].lower() not in {"avc1", "h264", "x264"}:
        raise RuntimeError(
            f"Prepared video codec={prepared['codec_fourcc']!r}; expected H.264"
        )

    prepared.update(
        {
            "view": view,
            "source_x_range": [x_start, x_end],
            "clip_start_seconds": 0.0,
            "clip_end_seconds": clip_end_seconds,
            "contains_right_view": view in {"right", "full"},
            "encoder": "libx264",
            "pixel_format": "yuv420p",
            "faststart": True,
        }
    )
    return prepared


def _tensor_numel(shape: list[int]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def inspect_safetensors_model(model_path: Path) -> dict[str, Any]:
    """Count parameters from safetensors headers without loading model weights."""
    shard_paths = sorted(model_path.glob("*.safetensors"))
    if not shard_paths:
        raise FileNotFoundError(f"No safetensors weights found in {model_path}")

    total_parameters = 0
    vision_parameters = 0
    language_parameters = 0
    dtype_parameters: dict[str, int] = {}
    tensor_count = 0
    weight_data_bytes = 0

    for shard_path in shard_paths:
        with shard_path.open("rb") as handle:
            header_size_raw = handle.read(8)
            if len(header_size_raw) != 8:
                raise ValueError(f"Invalid safetensors header: {shard_path}")
            header_size = struct.unpack("<Q", header_size_raw)[0]
            header = json.loads(handle.read(header_size))

        for name, metadata in header.items():
            if name == "__metadata__":
                continue
            parameter_count = _tensor_numel(metadata["shape"])
            total_parameters += parameter_count
            tensor_count += 1
            dtype = metadata["dtype"]
            dtype_parameters[dtype] = dtype_parameters.get(dtype, 0) + parameter_count
            offsets = metadata["data_offsets"]
            weight_data_bytes += offsets[1] - offsets[0]
            if name.startswith("model.visual."):
                vision_parameters += parameter_count
            else:
                language_parameters += parameter_count

    directory_bytes = sum(
        path.stat().st_size for path in model_path.rglob("*") if path.is_file()
    )
    return {
        "path": str(model_path.resolve()),
        "total_parameters": total_parameters,
        "total_parameters_billions": total_parameters / 1e9,
        "language_parameters": language_parameters,
        "language_parameters_billions": language_parameters / 1e9,
        "vision_parameters": vision_parameters,
        "vision_parameters_billions": vision_parameters / 1e9,
        "tensor_count": tensor_count,
        "parameters_by_dtype": dtype_parameters,
        "weight_data_bytes": weight_data_bytes,
        "weight_data_gib": weight_data_bytes / (1024**3),
        "directory_bytes": directory_bytes,
        "directory_gib": directory_bytes / (1024**3),
        "shard_count": len(shard_paths),
    }


def format_time_for_filename(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}".replace(".", "_")


def default_prepared_path(video: Path, view: str, query_end: float) -> Path:
    end_label = format_time_for_filename(query_end)
    return (
        Path(tempfile.gettempdir())
        / f"{video.stem}_{view}_0_{end_label}.mp4"
    )


def build_prompt(query_start: float, query_end: float) -> str:
    return (
        f"视频第 {query_start:g}~{query_end:g} 秒之间发生了什么？按时间顺序描述。"
        "只描述第一视角中能确认的事实，并说明相机是前进、转向还是基本停滞。"
    )


def error_record(stage: str, exc: BaseException, include_traceback: bool) -> dict[str, Any]:
    record = {
        "stage": stage,
        "type": type(exc).__name__,
        "message": str(exc),
    }
    if include_traceback:
        record["traceback"] = traceback.format_exc()
    return record


def measure_input_tokens(
    engine: Any,
    content: list[Any],
) -> tuple[int, dict[str, int], float]:
    """Use the engine's exact processor path to count input tokens."""
    started = time.perf_counter()
    messages = engine._build_messages(content, engine.system_prompt)
    user_items = messages[-1]["content"]
    media = {
        "image_count": sum(item.get("type") == "image" for item in user_items),
        "video_count": sum(item.get("type") == "video" for item in user_items),
        "timestamp_count": sum(
            item.get("type") == "text"
            and str(item.get("text", "")).startswith("t=")
            for item in user_items
        ),
    }
    inputs = engine.processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_tokens = int(inputs["input_ids"].shape[1])
    elapsed_ms = (time.perf_counter() - started) * 1000
    del inputs
    del messages
    gc.collect()
    return input_tokens, media, elapsed_ms


def gpu_device_for_model(engine: Any, torch_module: Any) -> Any | None:
    if not torch_module.cuda.is_available():
        return None
    try:
        device = next(engine.model.parameters()).device
    except StopIteration:
        return None
    return device if device.type == "cuda" else None


def collect_gpu_memory(torch_module: Any, device: Any) -> dict[str, Any] | None:
    if device is None:
        return None
    return {
        "device": str(device),
        "device_name": torch_module.cuda.get_device_name(device),
        "peak_allocated_bytes": torch_module.cuda.max_memory_allocated(device),
        "peak_allocated_gib": (
            torch_module.cuda.max_memory_allocated(device) / (1024**3)
        ),
        "peak_reserved_bytes": torch_module.cuda.max_memory_reserved(device),
        "peak_reserved_gib": (
            torch_module.cuda.max_memory_reserved(device) / (1024**3)
        ),
    }


def run_inference_mode(
    engine: Any,
    torch_module: Any,
    video_input_class: Any,
    mode: str,
    prompt: str,
    prepared_path: Path,
    fps: float,
    max_frames: int,
    max_pixels: int,
    max_tokens: int,
) -> dict[str, Any]:
    video_input = video_input_class(
        path=str(prepared_path),
        fps=fps,
        max_frames=max_frames,
        max_pixels=max_pixels,
        with_timestamps=mode == "timestamped",
    )
    content = [prompt, video_input]
    run: dict[str, Any] = {
        "mode": mode,
        "video_input": {
            "path": str(prepared_path),
            "fps": fps,
            "max_frames": max_frames,
            "max_pixels": max_pixels,
            "with_timestamps": mode == "timestamped",
        },
        "temperature": 0,
        "max_tokens": max_tokens,
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "hit_max_tokens": None,
        "input_preprocessing_ms": None,
        "media": None,
        "generation_elapsed_ms": None,
        "gpu_memory": None,
        "response": None,
        "error": None,
        "warnings": [],
    }

    try:
        input_tokens, media, preprocessing_ms = measure_input_tokens(engine, content)
        run["input_tokens"] = input_tokens
        run["media"] = media
        run["input_preprocessing_ms"] = preprocessing_ms
    except Exception as exc:  # Token accounting should not prevent inference.
        run["warnings"].append(error_record("input_token_measurement", exc, False))

    device = gpu_device_for_model(engine, torch_module)
    if device is not None:
        torch_module.cuda.synchronize(device)
        torch_module.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    try:
        response = engine.generate(
            content,
            temperature=0,
            max_tokens=max_tokens,
        )
        if device is not None:
            torch_module.cuda.synchronize(device)
        run["generation_elapsed_ms"] = (time.perf_counter() - started) * 1000
        run["response"] = response
        run["output_tokens"] = len(
            engine.processor.tokenizer.encode(
                response,
                add_special_tokens=False,
            )
        )
        run["hit_max_tokens"] = run["output_tokens"] >= max_tokens
        if run["input_tokens"] is not None:
            run["total_tokens"] = run["input_tokens"] + run["output_tokens"]
    except Exception as exc:
        if device is not None:
            torch_module.cuda.synchronize(device)
        run["generation_elapsed_ms"] = (time.perf_counter() - started) * 1000
        run["error"] = error_record("generate", exc, True)
    finally:
        run["gpu_memory"] = collect_gpu_memory(torch_module, device)

    return run


def write_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    output_path = args.output.expanduser().resolve()
    video_path = args.video.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    prepared_path = (
        args.prepared_video.expanduser().resolve()
        if args.prepared_video is not None
        else default_prepared_path(video_path, args.view, args.query_end)
    )

    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "initializing",
        "model": None,
        "source_video": None,
        "prepared_video": None,
        "query": {
            "start_seconds": args.query_start,
            "end_seconds": args.query_end,
            "prompt": build_prompt(args.query_start, args.query_end),
        },
        "sampling": {
            "fps": args.fps,
            "max_frames": args.max_frames,
            "max_pixels": args.max_pixels,
        },
        "configuration": {
            "mode": args.mode,
            "view": args.view,
            "device": args.device,
            "torch_dtype": "bfloat16",
            "use_cache": False,
            "debug_performance": args.debug_performance,
            "prepare_only": args.prepare_only,
        },
        "model_load_elapsed_ms": None,
        "runs": [],
        "review_timestamps_seconds": [
            args.query_start,
            (args.query_start + args.query_end) / 2,
            args.query_end,
        ],
        "errors": [],
    }

    try:
        validate_args(args)
        if not video_path.is_file():
            raise FileNotFoundError(f"Source video does not exist: {video_path}")
        if not model_path.is_dir():
            raise FileNotFoundError(f"Model directory does not exist: {model_path}")

        report["source_video"] = probe_video(video_path)
        report["prepared_video"] = prepare_view_clip(
            source_path=video_path,
            output_path=prepared_path,
            view=args.view,
            clip_end_seconds=args.query_end,
        )
        report["model"] = inspect_safetensors_model(model_path)

        max_pixels = args.max_pixels
        if max_pixels is None:
            max_pixels = (
                report["prepared_video"]["width"]
                * report["prepared_video"]["height"]
            )
        report["sampling"]["max_pixels"] = max_pixels
        timestamped_frame_count = min(
            args.max_frames,
            math.ceil(
                report["prepared_video"]["frame_count"]
                / max(
                    int(
                        round(
                            report["prepared_video"]["fps"]
                            / args.fps
                        )
                    ),
                    1,
                )
            ),
        )
        report["sampling"]["timestamped_frame_count"] = timestamped_frame_count

        if args.view == "left":
            expected_width = report["source_video"]["width"] // 2
            if (
                report["prepared_video"]["width"] != expected_width
                or report["prepared_video"]["contains_right_view"]
            ):
                raise RuntimeError("Left-view isolation validation failed")

        if args.prepare_only:
            report["status"] = "prepared"
            write_report(report, output_path)
            print(
                f"Prepared {prepared_path} "
                f"({report['prepared_video']['width']}x"
                f"{report['prepared_video']['height']}, "
                f"{report['prepared_video']['frame_count']} frames)"
            )
            print(f"Report: {output_path}")
            return 0

        import torch

        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device {args.device!r} was requested, but CUDA is unavailable"
            )

        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from agentflow.agents.engine.local_qwen3vl import LocalQwen3VL, VideoInput

        device_map: Any
        if args.device == "auto":
            device_map = "auto"
        else:
            device_map = {"": args.device}

        load_started = time.perf_counter()
        engine = LocalQwen3VL(
            model_path=str(model_path),
            is_multimodal=True,
            use_cache=False,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            debug_performance=args.debug_performance,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        report["model_load_elapsed_ms"] = (
            time.perf_counter() - load_started
        ) * 1000

        modes = ("timestamped", "native") if args.mode == "both" else (args.mode,)
        for mode in modes:
            run = run_inference_mode(
                engine=engine,
                torch_module=torch,
                video_input_class=VideoInput,
                mode=mode,
                prompt=report["query"]["prompt"],
                prepared_path=prepared_path,
                fps=args.fps,
                max_frames=args.max_frames,
                max_pixels=max_pixels,
                max_tokens=args.max_tokens,
            )
            report["runs"].append(run)
            print()
            print(f"[{mode}] response")
            print(run["response"] if run["response"] is not None else run["error"])

        successful_runs = sum(run["error"] is None for run in report["runs"])
        if successful_runs == len(report["runs"]):
            report["status"] = "success"
        elif successful_runs:
            report["status"] = "partial"
        else:
            report["status"] = "failed"

        del engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(error_record("main", exc, True))
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)

    write_report(report, output_path)
    print(f"Report: {output_path}")
    return 0 if report["status"] in {"success", "prepared"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
