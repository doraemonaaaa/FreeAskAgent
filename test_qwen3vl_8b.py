import time
from threading import Thread

import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    TextIteratorStreamer,
)

MODEL_PATH = "/data/pengyh/workspace/FreeVLN/models/Qwen3-VL-8B-Instruct"

print("Loading model...")

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(MODEL_PATH)

print("Model loaded.")
print("=" * 60)

image_path = "/data/pengyh/workspace/FreeVLN/test/OIP.webp"

history = []

while True:

    question = input("\nYou: ").strip()

    if question.lower() in ["exit", "quit", "q"]:
        break

    history.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {
                    "type": "text",
                    "text": question,
                },
            ],
        }
    )

    inputs = processor.apply_chat_template(
        history,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer=processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=256,
        do_sample=False,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    thread = Thread(
        target=model.generate,
        kwargs=generation_kwargs,
    )
    thread.start()

    print("\nAssistant: ", end="", flush=True)

    answer = ""
    first_token = True
    ttft = None

    for new_text in streamer:

        if first_token:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            ttft = time.perf_counter() - start_time
            first_token = False

        print(new_text, end="", flush=True)
        answer += new_text

    thread.join()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.perf_counter()

    elapsed = end_time - start_time

    print()

    history.append(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": answer,
                }
            ],
        }
    )

    # ===========================
    # Performance Statistics
    # ===========================

    input_tokens = inputs.input_ids.shape[1]

    output_tokens = len(
        processor.tokenizer.encode(
            answer,
            add_special_tokens=False,
        )
    )

    total_tokens = input_tokens + output_tokens

    decode_time = elapsed - (ttft if ttft else 0)

    if decode_time <= 0:
        decode_time = elapsed

    decode_throughput = output_tokens / decode_time
    total_throughput = total_tokens / elapsed
    latency_per_token = decode_time / max(output_tokens, 1) * 1000

    print()
    print("=" * 60)
    print("Performance")
    print("=" * 60)
    print(f"Input Tokens          : {input_tokens}")
    print(f"Output Tokens         : {output_tokens}")
    print(f"Total Tokens          : {total_tokens}")
    print(f"TTFT                  : {ttft:.3f} s")
    print(f"Generation Time       : {elapsed:.3f} s")
    print(f"Decode Time           : {decode_time:.3f} s")
    print(f"Decode Throughput     : {decode_throughput:.2f} tokens/s")
    print(f"End-to-End Throughput : {total_throughput:.2f} tokens/s")
    print(f"Latency per Token     : {latency_per_token:.2f} ms/token")
    print("=" * 60)

print("Bye.")