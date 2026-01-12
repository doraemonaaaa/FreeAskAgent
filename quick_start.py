#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quick_start.py

A-Mem Unit Test (Core Memory System Only).
Tests adding notes, automatic evolution, and retrieval.
"""

import argparse
import os
import sys
from pathlib import Path as _Path
from typing import List

from dotenv import load_dotenv

# --- 1. 路径设置 ---
REPO_ROOT = _Path(__file__).resolve().parent
# 将项目根目录加入 sys.path，确保能找到 agentflow 包
for p in [str(REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)


def _print_env():
    print("Proxy_API_BASE:" + os.environ.get("Proxy_API_BASE", "Not Set"))
    print("OPENAI_API_KEY:" + ("Set" if os.environ.get("OPENAI_API_KEY") else "Not Set"))


def unit_test_amem(k: int = 5) -> bool:
    """
    Directly test A-Mem retrieval (Unit Test).
    Returns True if the retrieved memory context contains the planted fact.
    """
    print("\n====================")
    print("UNIT TEST: A-Mem core (AgenticMemorySystem)")
    print("====================\n")

    # --- 2. 修正导入路径 ---
    # 根据之前的报错，我们去掉中间重复的 .agentflow
    try:
        from agentflow.models_embodied.memory import AgenticMemorySystem
    except ImportError:
        # 双重保险：如果单层导入失败，尝试双层（视具体目录结构而定）
        try:
            from agentflow.agentflow.models_embodied.memory import AgenticMemorySystem
        except Exception as e:
            print("[ERROR] Failed to import AgenticMemorySystem.")
            print("Reason:", repr(e))
            print("Please check if 'agentflow' is in your python path.")
            return False

    # 自动决定后端 (OpenAI vs SGLang)
    api_key = os.environ.get("OPENAI_API_KEY")
    # 兼容各种 Base URL 环境变量写法
    api_base = os.environ.get("Proxy_API_BASE") or os.environ.get("OPENAI_API_BASE") or os.environ.get(
        "OPENAI_BASE_URL")

    llm_backend = "openai" if api_key else "sglang"
    llm_model = os.environ.get("AMEM_LLM_MODEL", "gpt-4o")

    print(f"[A-Mem] llm_backend={llm_backend}")
    print(f"[A-Mem] llm_model={llm_model}")
    if llm_backend == "openai":
        print(f"[A-Mem] api_base={'(default)' if not api_base else api_base}")

    # --- 3. 初始化记忆系统 ---
    print("\n[Init] Initializing AgenticMemorySystem...")
    try:
        amem = AgenticMemorySystem(
            llm_backend=llm_backend,
            llm_model=llm_model,
            api_key=api_key,
            api_base=api_base,
            # 嵌入模型：如果本地跑不动可以改用更小的模型，或者保持默认
            model_name=os.environ.get("AMEM_EMBED_MODEL", "all-MiniLM-L6-v2"),
            evo_threshold=int(os.environ.get("AMEM_EVO_THRESHOLD", "100")),
        )
    except Exception as e:
        print(f"[ERROR] Failed to initialize AgenticMemorySystem: {e}")
        return False

    # --- 4. 写入记忆 (Add Notes) ---
    # 这一步会触发记忆演化 (Evolution)，调用 LLM 分析
    planted_fact = "micheal's store 在红色门旁边"
    notes = [
        "今天我要去找 micheal's store 买画材。",
        planted_fact,
        "老板叫 Mike，他喜欢把促销海报贴在玻璃门上。",
        "从地铁站出来直走 200 米，右转就能看到红色门。",
    ]

    print("\n[1] Adding notes to A-Mem (Testing Evolution)...")
    for i, n in enumerate(notes, 1):
        try:
            # add_note 返回 note_id
            note_id = amem.add_note(n)
            print(f"  - added note {i}: id={note_id}")
        except Exception as e:
            print(f"[ERROR] add_note failed on note {i}: {repr(e)}")
            print("This likely means the configured LLM backend is not reachable or API Key is invalid.")
            return False

    # --- 5. 检索测试 (Retrieval) ---
    query = "micheal's store 在哪里？请给出关键线索。"
    print(f"\n[2] Retrieving related memories for query: {query}")

    try:
        # find_related_memories 返回 (拼接好的字符串, 索引列表)
        memory_str, indices = amem.find_related_memories(query, k=k)
    except Exception as e:
        print("[ERROR] Retrieval failed:", repr(e))
        return False

    print("\n[Retrieved Memories Content]\n")
    print(memory_str)

    # --- 6. 验证结果 ---
    # 检查我们埋入的关键事实 "红色门" 是否被检索到了
    ok = "红色门" in memory_str or "red" in memory_str.lower() or "Red Door" in memory_str

    print("\n====================")
    print("UNIT RESULT:", "PASS ✅" if ok else "FAIL ❌")
    print("====================\n")

    if not ok:
        print(f"Expected retrieved context to contain: '{planted_fact}'")

    return ok


def main():
    parser = argparse.ArgumentParser(description="A-Mem Unit Test Script")
    parser.add_argument("--k", type=int, default=5, help="Number of memories to retrieve")
    args = parser.parse_args()

    # 加载环境变量
    load_dotenv(dotenv_path="agentflow/.env")
    _print_env()

    # 执行单元测试
    success = unit_test_amem(k=args.k)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()