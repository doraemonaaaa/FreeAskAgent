#!/usr/bin/env python3
"""
Detailed debug script to investigate memory retrieval scoring
"""

import sys
import os
sys.path.append('/root/autodl-tmp/FreeAskAgent')

from agentflow.agentflow.models_embodied.long_memory import LongMemory

def debug_memory_scoring():
    """Debug memory scoring and retrieval"""

    print("=== Detailed Memory Debug Script ===")

    # Initialize memory system
    memory = LongMemory(
        use_amem=True,
        retriever_config={'use_api_embedding': True},
        storage_dir="./memory_store",
        enable_persistence=True
    )

    print(f"Total memories: {len(memory.long_term_memories)}")

    # Print all memories
    print("\nAll memories in long_term_memories:")
    for i, mem in enumerate(memory.long_term_memories):
        print(f"  {i}: {mem[:100]}...")

    # Add our test memories
    print("\nAdding test memories...")
    memory.add_memory("广场内有盒马、永辉等大型超市", memory_type="custom")
    memory.add_memory("Task: 广场内有什么超市", memory_type="qa_task")

    print(f"\nAfter adding, total memories: {len(memory.long_term_memories)}")

    # Print all memories again
    print("\nAll memories after adding:")
    for i, mem in enumerate(memory.long_term_memories):
        print(f"  {i}: {mem[:100]}...")

    # Test retrieval with detailed scoring
    query = "广场内有什么超市"
    print(f"\nTesting retrieval for query: '{query}'")

    # Get retriever scores directly
    if memory.retriever:
        indices = memory.retriever.retrieve(query, k=10)
        print(f"Retriever returned indices: {indices}")

        # Get detailed scores if possible
        try:
            bm25_scores = memory.retriever._get_bm25_scores(query)
            semantic_scores = memory.retriever._get_semantic_scores(query)
            hybrid_scores = memory.retriever._combine_scores(bm25_scores, semantic_scores)

            print(f"BM25 scores: {bm25_scores}")
            print(f"Semantic scores: {semantic_scores}")
            print(f"Hybrid scores: {hybrid_scores}")

            # Show top scoring memories
            print("\nTop scoring memories:")
            sorted_indices = hybrid_scores.argsort()[::-1]  # Sort in descending order
            for rank, idx in enumerate(sorted_indices[:5]):
                score = hybrid_scores[idx]
                content = memory.long_term_memories[idx] if idx < len(memory.long_term_memories) else "INDEX_OUT_OF_RANGE"
                print(".4f")

        except Exception as e:
            print(f"Error getting detailed scores: {e}")

    # Test full retrieval
    results = memory.retrieve_memories(query, k=5)
    print(f"\nFull retrieval returned {len(results)} memories:")
    for i, result in enumerate(results):
        print(f"  {i}: {result.get('content', '')[:100]}...")

if __name__ == "__main__":
    debug_memory_scoring()
