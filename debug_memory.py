#!/usr/bin/env python3
"""
Debug script to investigate memory retrieval issues
"""

import sys
import os
sys.path.append('/root/autodl-tmp/FreeAskAgent')

from agentflow.agentflow.models_embodied.long_memory import LongMemory

def debug_memory_sync():
    """Debug memory synchronization issues"""

    print("=== Memory Debug Script ===")

    # Initialize memory system
    memory = LongMemory(
        use_amem=True,
        retriever_config={'use_api_embedding': True},
        storage_dir="./memory_store",
        enable_persistence=True
    )

    print(f"Initial long_term_memories count: {len(memory.long_term_memories)}")
    print(f"Initial retriever corpus count: {len(memory.retriever.corpus) if memory.retriever and hasattr(memory.retriever, 'corpus') else 'No corpus'}")

    # Add some memories
    print("\nAdding memories...")
    memory.add_memory("广场内有盒马、永辉等大型超市", memory_type="custom")
    print(f"After adding custom memory - long_term_memories: {len(memory.long_term_memories)}, corpus: {len(memory.retriever.corpus) if memory.retriever else 'No retriever'}")

    memory.add_memory("Task: 广场内有什么超市", memory_type="qa_task")
    print(f"After adding qa_task memory - long_term_memories: {len(memory.long_term_memories)}, corpus: {len(memory.retriever.corpus) if memory.retriever else 'No retriever'}")

    # Try retrieval
    print("\nTesting retrieval...")
    query = "广场内有什么超市"
    try:
        results = memory.retrieve_memories(query, k=5)
        print(f"Retrieved {len(results)} memories")
        for i, result in enumerate(results):
            print(f"  {i}: {result.get('content', '')[:50]}...")
    except Exception as e:
        print(f"Retrieval failed: {e}")
        import traceback
        traceback.print_exc()

    # Check indices manually
    print("\nManual index check:")
    if memory.retriever:
        indices = memory.retriever.retrieve(query, k=5)
        print(f"Retriever returned indices: {indices}")
        print(f"long_term_memories length: {len(memory.long_term_memories)}")

        for idx in indices:
            if 0 <= idx < len(memory.long_term_memories):
                print(f"  Index {idx}: VALID - {memory.long_term_memories[idx][:50]}...")
            else:
                print(f"  Index {idx}: INVALID (range: 0-{len(memory.long_term_memories)-1})")

if __name__ == "__main__":
    debug_memory_sync()
