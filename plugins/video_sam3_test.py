# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
用文本 prompt 对视频中的指定物体做分割与跟踪，并把结果可视化叠加输出为一个新视频。

用法:
    python sam3_video_segment.py --video input.mp4 --prompt "person" --output output.mp4

video 可以是 .mp4 文件，也可以是一个存放 JPEG 帧（命名为 0.jpg, 1.jpg, ...）的文件夹。
"""
import argparse
import glob
import os

import cv2
import matplotlib

matplotlib.use("Agg")  # 无显示环境下渲染，不弹窗
import matplotlib.pyplot as plt
import numpy as np
import torch

import sam3  # noqa: F401  (仅用于定位包路径，非必需)
from sam3.model_builder import build_sam3_video_predictor
from sam3.visualization_utils import (
    prepare_masks_for_visualization,
    visualize_formatted_frame_output,
)


def load_video_frames(video_path):
    """返回 (RGB 帧列表, fps)，与官方 notebook 的加载逻辑一致。"""
    if video_path.endswith(".mp4"):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames, fps
    else:
        paths = glob.glob(os.path.join(video_path, "*.jpg"))
        try:
            paths.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
        except ValueError:
            paths.sort()
        # 只返回路径，官方可视化函数支持传路径列表（内部按需读取）
        return paths, 25.0


def propagate_in_video(predictor, session_id):
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(type="propagate_in_video", session_id=session_id)
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]
    return outputs_per_frame


def fig_to_bgr_array(fig):
    """把 matplotlib figure 渲染成 OpenCV 可写入视频的 BGR numpy 数组。"""
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = rgba[:, :, :3]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser(description="SAM 3 视频文本 prompt 分割 + 可视化导出")
    parser.add_argument("--video", required=True, help=".mp4 文件路径，或 JPEG 帧文件夹路径")
    parser.add_argument("--prompt", required=True, help='文本 prompt，例如 "person" 或 "red car"')
    parser.add_argument("--output", default="output.mp4", help="输出视频路径")
    parser.add_argument("--frame-idx", type=int, default=0, help="添加 prompt 的帧号（默认第 0 帧）")
    parser.add_argument("--fps", type=float, default=None, help="输出视频帧率，默认沿用输入视频帧率")
    args = parser.parse_args()

    # 1. 构建 predictor（自动用上所有可见 GPU；没有 GPU 会报错，SAM3 目前依赖 CUDA）
    #  gpus_to_use = list(range(torch.cuda.device_count())) or [0]
    gpus_to_use = [1]
    predictor = build_sam3_video_predictor(
        gpus_to_use=gpus_to_use,
        checkpoint_path="models/sam3/sam3.pt",   # path
    )

    # 2. 加载帧（仅用于可视化，模型内部会自己重新读取 video_path）
    video_frames_for_vis, in_fps = load_video_frames(args.video)
    out_fps = args.fps or in_fps

    # 3. 打开推理会话
    response = predictor.handle_request(
        request=dict(type="start_session", resource_path=args.video)
    )
    session_id = response["session_id"]

    # 4. 用文本 prompt 在指定帧上标记要分割的物体
    predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=args.frame_idx,
            text=args.prompt,
        )
    )

    # 5. 从该帧开始向整段视频传播（跟踪）
    outputs_per_frame = propagate_in_video(predictor, session_id)
    outputs_per_frame = prepare_masks_for_visualization(outputs_per_frame)

    # 6. 逐帧渲染 mask 叠加效果，写成输出视频
    output_path = os.path.abspath(args.output)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    writer = None
    written_frames = 0

    for frame_idx in sorted(outputs_per_frame.keys()):
        plt.close("all")
        visualize_formatted_frame_output(
            frame_idx,
            video_frames_for_vis,
            outputs_list=[outputs_per_frame],
            titles=[f'SAM 3: "{args.prompt}"'],
            figsize=(6, 4),
        )
        fig = plt.gcf()
        frame_bgr = fig_to_bgr_array(fig)

        if writer is None:
            h, w = frame_bgr.shape[:2]
            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                out_fps,
                (w, h),
            )
            if not writer.isOpened():
                raise RuntimeError(f"无法打开视频写入器，输出路径: {output_path}")

        writer.write(frame_bgr)
        written_frames += 1

    if writer is None:
        raise RuntimeError("没有生成任何可写入的视频帧，outputs_per_frame 为空")

    writer.release()
    plt.close("all")

    # 7. 关闭会话、释放资源
    predictor.handle_request(request=dict(type="close_session", session_id=session_id))
    predictor.shutdown()

    print(f"已保存分割可视化视频到: {args.output}")


if __name__ == "__main__":
    main()