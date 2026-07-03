from typing import Dict, Any, List, Tuple

import numpy as np


def retrieve_from_graph(graph, query_embeddings, topk: int = 5, threshold: float = 0.0):
    nodes = graph.search_text_nodes(query_embeddings)
    clip_scores = {}
    for node_id, score in nodes:
        clip_id = graph.nodes[node_id].metadata.get("timestamp")
        if clip_id is None:
            continue
        clip_scores.setdefault(clip_id, []).append(score)

    clip_ranked = []
    for clip_id, scores in clip_scores.items():
        clip_ranked.append((clip_id, max(scores)))
    clip_ranked.sort(key=lambda x: x[1], reverse=True)
    top_clips = [clip_id for clip_id, score in clip_ranked if score >= threshold][:topk]
    return top_clips, clip_ranked, nodes


def collect_memories(graph, top_clips: List[int]) -> Dict[int, Dict[str, List[str]]]:
    memories = {}
    for clip_id in top_clips:
        if clip_id not in graph.text_nodes_by_clip:
            memories[clip_id] = {"episodic": [], "semantic": []}
            continue
        for node_id in graph.text_nodes_by_clip[clip_id]:
            node = graph.nodes[node_id]
            if node.type == "episodic":
                memories.setdefault(clip_id, {"episodic": [], "semantic": []})["episodic"].extend(node.metadata["contents"])
            elif node.type == "semantic":
                memories.setdefault(clip_id, {"episodic": [], "semantic": []})["semantic"].extend(node.metadata["contents"])
    return memories
