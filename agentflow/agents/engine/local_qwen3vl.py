import hashlib
import io
import json
import os
import time
from threading import Thread
from typing import List, Union

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


class LocalQwen3VL(EngineLM, CachedEngine):
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."
    DEFAULT_MODEL_PATH = "models/Qwen3-VL-8B-Instruct"

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

    def generate(self, content: Union[str, List[Union[str, bytes]]], system_prompt=None, **kwargs):
        if isinstance(content, str):
            return self._generate(content, system_prompt=system_prompt, **kwargs)

        if isinstance(content, list):
            if all(isinstance(item, str) for item in content):
                return self._generate("\n".join(content), system_prompt=system_prompt, **kwargs)

            if any(isinstance(item, bytes) for item in content):
                if not self.is_multimodal:
                    raise NotImplementedError(
                        f"Multimodal generation is disabled for {self.model_string}. "
                        "Pass is_multimodal=True when creating the engine."
                    )
                return self._generate(content, system_prompt=system_prompt, **kwargs)

        raise ValueError("Unsupported content: expected str or list containing str/bytes.")

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)

    def _generate(
        self,
        content: Union[str, List[Union[str, bytes]]],
        system_prompt=None,
        temperature=0,
        max_tokens=2048,
        top_p=0.99,
        response_format=None,
        **kwargs,
    ):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        cache_key = self._cache_key(sys_prompt_arg, content, temperature, max_tokens, top_p)

        if self.use_cache:
            cache_or_none = self._check_cache(cache_key)
            if cache_or_none is not None:
                return cache_or_none

        messages = self._build_messages(content, sys_prompt_arg, response_format)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        do_sample = temperature is not None and temperature > 0
        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
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

    def _build_messages(self, content, system_prompt, response_format=None):
        user_content = []
        if isinstance(content, str):
            user_content.append({"type": "text", "text": content})
        else:
            for item in content:
                if isinstance(item, str):
                    user_content.append({"type": "text", "text": item})
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
                elif isinstance(item, bytes):
                    hasher.update(hashlib.sha256(item).digest())
                else:
                    hasher.update(str(type(item)).encode("utf-8"))
        return hasher.hexdigest()
