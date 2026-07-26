import hashlib
import io
import json
import os
import tempfile
import time
from dataclasses import dataclass
from threading import Thread
from typing import List, Optional, Union

import platformdirs
import torch
from PIL import Image

try:
    from transformers import AutoModelForImageTextToText, AutoProcessor, TextIteratorStreamer
except ImportError as exc:
    raise ImportError(
        "If you'd like to use local Qwen3-VL models, please install transformers "
        "and the Qwen3-VL runtime dependencies."
    ) from exc

from .base import CachedEngine, EngineLM


@dataclass
class VideoInput:
    """Represents a video to be understood by the model.

    Exactly one of `path` or `data` must be provided.

    Attributes:
        path: Filesystem path (or URL, if the processor/backend supports it)
            pointing to the video file. Preferred over `data` for large files
            since it avoids buffering the whole video in memory and enables
            stable caching based on file identity rather than content hash.
        data: Raw video bytes (e.g. mp4/webm/mov). Will be written to a
            temporary file before being handed to the processor, since the
            underlying Qwen3-VL video pipeline (decord/torchvision/opencv,
            depending on what's installed) expects a path-like input.
        fps: Frames per second to sample from the video. Lower values reduce
            token usage / memory at the cost of temporal resolution. Qwen3-VL
            processors accept this directly and perform the sampling
            internally.
        max_frames: Optional hard cap on the number of sampled frames,
            regardless of `fps` and video duration. Useful to bound
            memory/token usage for long videos.
        max_pixels: Optional per-frame pixel budget forwarded to the
            processor's image/video resizing logic (same semantics as
            Qwen-VL's `max_pixels`), to control token cost per frame.
        with_timestamps: If True, frames are extracted manually (via OpenCV)
            and interleaved with explicit "t=X.Xs" text markers before each
            frame, instead of handing the raw video to the processor's
            built-in video pipeline. This trades a bit of setup cost for
            reliable, version-independent temporal grounding: the model sees
            exactly which second each frame came from, which noticeably
            improves answers to questions like "what happens around the
            10s mark" or "what happened first, X or Y". Requires
            `opencv-python` to be installed.
    """

    path: Optional[str] = None
    data: Optional[bytes] = None
    fps: float = 2.0
    max_frames: Optional[int] = None
    max_pixels: Optional[int] = None
    with_timestamps: bool = False

    def __post_init__(self):
        if (self.path is None) == (self.data is None):
            raise ValueError("VideoInput requires exactly one of `path` or `data`.")

    def resolve_path(self, tmp_dir: str) -> str:
        """Returns a filesystem path for this video, materializing `data` to disk if needed."""
        if self.path is not None:
            return self.path

        # Write raw bytes out to a temp file so the video backend (decord/
        # torchvision/opencv/qwen-vl-utils) can open it like any other file.
        digest = hashlib.sha256(self.data).hexdigest()[:16]
        tmp_path = os.path.join(tmp_dir, f"video_{digest}.mp4")
        if not os.path.exists(tmp_path):
            with open(tmp_path, "wb") as f:
                f.write(self.data)
        return tmp_path

    def cache_fingerprint(self) -> str:
        """A stable fingerprint used for cache-key hashing (cheap for path inputs)."""
        if self.path is not None:
            try:
                stat = os.stat(self.path)
                identity = f"{os.path.abspath(self.path)}:{stat.st_size}:{stat.st_mtime_ns}"
            except OSError:
                identity = os.path.abspath(self.path)
        else:
            identity = hashlib.sha256(self.data).hexdigest()
        return (
            f"{identity}:{self.fps}:{self.max_frames}:{self.max_pixels}:"
            f"{self.with_timestamps}"
        )


class LocalQwen3VL(EngineLM, CachedEngine):
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."
    DEFAULT_MODEL_PATH = "models/Qwen3-VL-8B-Instruct"
    supports_image_pixel_budget = True

    def __init__(
        self,
        model_string=None,
        model_path=None,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        is_multimodal: bool = False,
        use_cache: bool = True,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=False,
        debug_performance: bool = False,
        generation_use_cache: bool = True,
        **kwargs,
    ):
        self.model_string = model_string or model_path or os.getenv(
            "QWEN3VL_MODEL_PATH", self.DEFAULT_MODEL_PATH
        )
        self.model_path = model_path or self.model_string
        self.use_cache = use_cache
        self.system_prompt = system_prompt
        self.is_multimodal = is_multimodal
        self.debug_performance = debug_performance
        # Some fine-tuned Qwen3-VL checkpoints retain their training-time
        # ``use_cache=false`` config.  KV caching is required for practical
        # autoregressive inference latency and is independent of AgentFlow's
        # persistent response cache above.
        self.generation_use_cache = bool(generation_use_cache)

        # Temp dir used to materialize raw video bytes (VideoInput.data) to disk,
        # since the video decoding backends expect a path rather than bytes.
        self._video_tmp_dir = os.path.join(
            tempfile.gettempdir(), "agentflow_qwen3vl_video_cache"
        )
        os.makedirs(self._video_tmp_dir, exist_ok=True)

        if self.use_cache:
            root = platformdirs.user_cache_dir("agentflow")
            cache_name = hashlib.sha256(self.model_path.encode()).hexdigest()[:16]
            cache_path = os.path.join(root, f"cache_local_qwen3vl_{cache_name}.db")
            super().__init__(cache_path=cache_path)

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=trust_remote_code,
        )

    def generate(self, content: Union[str, List[Union[str, bytes, VideoInput]]], system_prompt=None, **kwargs):
        if isinstance(content, str):
            return self._generate(content, system_prompt=system_prompt, **kwargs)

        if isinstance(content, list):
            if all(isinstance(item, str) for item in content):
                return self._generate("\n".join(content), system_prompt=system_prompt, **kwargs)

            has_media = any(isinstance(item, (bytes, VideoInput)) for item in content)
            if has_media:
                if not self.is_multimodal:
                    raise NotImplementedError(
                        f"Multimodal generation is disabled for {self.model_string}. "
                        "Pass is_multimodal=True when creating the engine."
                    )
                return self._generate(content, system_prompt=system_prompt, **kwargs)

        raise ValueError(
            "Unsupported content: expected str or list containing str/bytes/VideoInput."
        )

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)

    def _generate(
        self,
        content: Union[str, List[Union[str, bytes, VideoInput]]],
        system_prompt=None,
        temperature=0,
        max_tokens=2048,
        top_p=0.99,
        response_format=None,
        image_min_pixels=None,
        image_max_pixels=None,
        **kwargs,
    ):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        cache_key = self._cache_key(sys_prompt_arg, content, temperature, max_tokens, top_p)

        if self.use_cache:
            cache_or_none = self._check_cache(cache_key)
            if cache_or_none is not None:
                return cache_or_none

        messages = self._build_messages(content, sys_prompt_arg, response_format)
        processor_kwargs = self._image_processor_kwargs(
            image_min_pixels=image_min_pixels,
            image_max_pixels=image_max_pixels,
        )
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs=processor_kwargs,
        )
        inputs = inputs.to(self.model.device)

        do_sample = temperature is not None and temperature > 0
        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "use_cache": self.generation_use_cache,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p

        if self.debug_performance or kwargs.get("debug_performance", False):
            response_text = self._generate_with_performance_stats(
                inputs=inputs,
                generation_kwargs=generation_kwargs,
                start_index=inputs.input_ids.shape[1],
            )
        else:
            with torch.inference_mode():
                generated_ids = self.model.generate(**generation_kwargs)

            input_len = inputs.input_ids.shape[1]
            generated_ids = generated_ids[:, input_len:]
            response_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

        if self.use_cache:
            self._save_cache(cache_key, response_text)
        return response_text

    def _generate_with_performance_stats(self, inputs, generation_kwargs, start_index):
        streamer = TextIteratorStreamer(
            tokenizer=self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        generation_kwargs = {
            **generation_kwargs,
            "streamer": streamer,
        }

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        response_text = ""
        first_token = True
        ttft = None

        for new_text in streamer:
            if first_token:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                ttft = time.perf_counter() - start_time
                first_token = False
            response_text += new_text

        thread.join()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        input_tokens = inputs.input_ids.shape[1]
        output_tokens = len(
            self.processor.tokenizer.encode(
                response_text,
                add_special_tokens=False,
            )
        )
        total_tokens = input_tokens + output_tokens
        decode_time = elapsed - (ttft if ttft else 0)
        if decode_time <= 0:
            decode_time = elapsed

        decode_throughput = output_tokens / decode_time if decode_time > 0 else 0
        total_throughput = total_tokens / elapsed if elapsed > 0 else 0
        latency_per_token = decode_time / max(output_tokens, 1) * 1000

        print()
        print("=" * 60)
        print("Performance")
        print("=" * 60)
        print(f"Input Tokens          : {input_tokens}")
        print(f"Output Tokens         : {output_tokens}")
        print(f"Total Tokens          : {total_tokens}")
        print(f"TTFT                  : {(ttft or elapsed):.3f} s")
        print(f"Generation Time       : {elapsed:.3f} s")
        print(f"Decode Time           : {decode_time:.3f} s")
        print(f"Decode Throughput     : {decode_throughput:.2f} tokens/s")
        print(f"End-to-End Throughput : {total_throughput:.2f} tokens/s")
        print(f"Latency per Token     : {latency_per_token:.2f} ms/token")
        print("=" * 60)

        return response_text

    def _extract_frames_with_timestamps(self, video_path, fps, max_frames=None, max_pixels=None):
        """Samples frames from `video_path` at roughly `fps`, returning a list of
        (timestamp_seconds, PIL.Image) tuples in chronological order.

        This bypasses the processor's built-in video pipeline so that we can
        attach an explicit, human-readable timestamp to every frame -- this
        is what lets the model reason precisely about "what happened when"
        rather than only "what came before/after what".
        """
        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "VideoInput(with_timestamps=True) requires opencv-python. "
                "Install it with `pip install opencv-python`."
            ) from exc

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        try:
            native_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if native_fps <= 0 or total_frames <= 0:
                raise ValueError(f"Could not read video metadata for: {video_path}")

            duration = total_frames / native_fps
            sample_stride = max(int(round(native_fps / fps)), 1) if fps > 0 else 1

            frames = []
            frame_index = 0
            while True:
                ok = cap.grab()
                if not ok:
                    break
                if frame_index % sample_stride == 0:
                    ok, frame_bgr = cap.retrieve()
                    if not ok:
                        break
                    timestamp = frame_index / native_fps
                    image = Image.fromarray(frame_bgr[:, :, ::-1])  # BGR -> RGB
                    if max_pixels is not None and image.width * image.height > max_pixels:
                        scale = (max_pixels / (image.width * image.height)) ** 0.5
                        new_size = (max(int(image.width * scale), 1), max(int(image.height * scale), 1))
                        image = image.resize(new_size)
                    frames.append((timestamp, image))
                    if max_frames is not None and len(frames) >= max_frames:
                        break
                frame_index += 1
        finally:
            cap.release()

        if not frames:
            raise ValueError(f"No frames could be sampled from video: {video_path}")

        return frames, duration

    def _build_messages(self, content, system_prompt, response_format=None):
        user_content = []
        if isinstance(content, str):
            user_content.append({"type": "text", "text": content})
        else:
            for item in content:
                if isinstance(item, str):
                    user_content.append({"type": "text", "text": item})
                elif isinstance(item, VideoInput):
                    video_path = item.resolve_path(self._video_tmp_dir)
                    if item.with_timestamps:
                        frames, duration = self._extract_frames_with_timestamps(
                            video_path,
                            fps=item.fps,
                            max_frames=item.max_frames,
                            max_pixels=item.max_pixels,
                        )
                        user_content.append(
                            {
                                "type": "text",
                                "text": (
                                    f"[The following {len(frames)} frames are sampled from a "
                                    f"{duration:.1f}s video, in chronological order. Each frame "
                                    "is preceded by its timestamp in seconds.]"
                                ),
                            }
                        )
                        for timestamp, frame in frames:
                            user_content.append({"type": "text", "text": f"t={timestamp:.2f}s"})
                            user_content.append({"type": "image", "image": frame})
                    else:
                        video_block = {"type": "video", "video": video_path, "fps": item.fps}
                        if item.max_frames is not None:
                            video_block["max_frames"] = item.max_frames
                        if item.max_pixels is not None:
                            video_block["max_pixels"] = item.max_pixels
                        user_content.append(video_block)
                elif isinstance(item, bytes):
                    image = Image.open(io.BytesIO(item)).convert("RGB")
                    user_content.append({"type": "image", "image": image})
                else:
                    raise ValueError(f"Unsupported input type: {type(item)}")

        if response_format is not None:
            schema = response_format.model_json_schema()
            user_content.append(
                {
                    "type": "text",
                    "text": (
                        "Return a JSON object matching this schema and do not include "
                        f"extra commentary:\n{json.dumps(schema, ensure_ascii=False)}"
                    ),
                }
            )

        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

    @staticmethod
    def _image_processor_kwargs(
        *,
        image_min_pixels=None,
        image_max_pixels=None,
    ):
        """Build an opt-in per-call image-token budget.

        Without this option Qwen keeps its native processor defaults, which is
        important for Actor/Thinker image quality. Temporal Memory opts in so
        its deliberately small storyboard frames are not upscaled to 256x256.
        """
        if image_min_pixels is None and image_max_pixels is None:
            return {}
        if image_min_pixels is None or image_max_pixels is None:
            raise ValueError(
                "image_min_pixels and image_max_pixels must be provided together"
            )
        minimum = int(image_min_pixels)
        maximum = int(image_max_pixels)
        if minimum <= 0 or maximum <= 0 or minimum > maximum:
            raise ValueError(
                "image pixel budget must satisfy 0 < min <= max"
            )
        return {
            "images_kwargs": {
                "min_pixels": minimum,
                "max_pixels": maximum,
            }
        }

    def _cache_key(self, system_prompt, content, temperature, max_tokens, top_p):
        hasher = hashlib.sha256()
        hasher.update(system_prompt.encode("utf-8"))
        hasher.update(str(temperature).encode("utf-8"))
        hasher.update(str(max_tokens).encode("utf-8"))
        hasher.update(str(top_p).encode("utf-8"))

        if isinstance(content, str):
            hasher.update(content.encode("utf-8"))
        else:
            for item in content:
                if isinstance(item, str):
                    hasher.update(item.encode("utf-8"))
                elif isinstance(item, VideoInput):
                    hasher.update(item.cache_fingerprint().encode("utf-8"))
                elif isinstance(item, bytes):
                    hasher.update(hashlib.sha256(item).digest())
                else:
                    hasher.update(str(type(item)).encode("utf-8"))
        return hasher.hexdigest()
