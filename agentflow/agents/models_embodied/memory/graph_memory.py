import math
import random
from typing import Dict, List, Tuple, Any, Optional

import numpy as np


class GraphMemory:
    """
    M3-style memory graph for VLN.
    Node types: img, episodic, semantic.
    Edges connect image nodes to text nodes.
    """

    class Node:
        def __init__(self, node_id: int, node_type: str):
            self.id = node_id
            self.type = node_type
            self.embeddings: List[List[float]] = []
            self.metadata: Dict[str, Any] = {}

    def __init__(
        self,
        max_img_embeddings: int = 5,
        img_matching_threshold: float = 0.3
    ):
        self.nodes: Dict[int, GraphMemory.Node] = {}
        self.edges: Dict[Tuple[int, int], float] = {}
        self.text_nodes: List[int] = []
        self.text_nodes_by_clip: Dict[int, List[int]] = {}
        self.max_img_embeddings = max_img_embeddings
        self.img_matching_threshold = img_matching_threshold
        self.next_node_id = 0

    def add_img_node(self, imgs: Dict[str, Any]) -> int:
        node = self.Node(self.next_node_id, "img")
        embeddings = imgs.get("embeddings", [])
        node.embeddings.extend(embeddings[: self.max_img_embeddings])
        node.metadata["contents"] = imgs.get("contents", [])
        self.nodes[self.next_node_id] = node
        self.next_node_id += 1
        return node.id

    def add_text_node(self, text: Dict[str, Any], clip_id: int, text_type: str = "episodic") -> int:
        if text_type not in ["episodic", "semantic"]:
            raise ValueError("text_type must be either 'episodic' or 'semantic'")
        node = self.Node(self.next_node_id, text_type)
        node.embeddings = text.get("embeddings", [])
        node.metadata["contents"] = text.get("contents", [])
        node.metadata["timestamp"] = clip_id
        self.nodes[self.next_node_id] = node
        self.text_nodes.append(node.id)
        if clip_id not in self.text_nodes_by_clip:
            self.text_nodes_by_clip[clip_id] = []
        self.text_nodes_by_clip[clip_id].append(node.id)
        self.next_node_id += 1
        return node.id

    def add_edge(self, node_id1: int, node_id2: int, weight: float = 1.0) -> bool:
        if node_id1 in self.nodes and node_id2 in self.nodes:
            self.edges[(node_id1, node_id2)] = weight
            self.edges[(node_id2, node_id1)] = weight
            return True
        return False

    def get_connected_nodes(self, node_id: int, types: List[str]) -> List[int]:
        connected = set()
        for (n1, n2), _ in self.edges.items():
            if n1 == node_id and self.nodes[n2].type in types:
                connected.add(n2)
            elif n2 == node_id and self.nodes[n1].type in types:
                connected.add(n1)
        return list(connected)

    def search_text_nodes(self, query_embeddings: List[List[float]], range_nodes: List[int] = None) -> List[Tuple[int, float]]:
        if range_nodes:
            text_nodes = []
            for node_id in range_nodes:
                text_nodes.extend(self.get_connected_nodes(node_id, ["episodic", "semantic"]))
            text_nodes = list(set(text_nodes))
        else:
            text_nodes = self.text_nodes

        if not text_nodes:
            return []

        node_embeddings = [self.nodes[node_id].embeddings for node_id in text_nodes]
        node_ids = text_nodes

        query_embeddings_np = np.array(query_embeddings)
        node_embeddings_np = np.array(node_embeddings)

        n_queries = query_embeddings_np.shape[0]
        n_nodes = node_embeddings_np.shape[0]
        n_embeddings = node_embeddings_np.shape[1]
        embedding_dim = node_embeddings_np.shape[-1]

        sims = self._cosine_similarity_matrix(
            query_embeddings_np.reshape(-1, embedding_dim),
            node_embeddings_np.reshape(-1, embedding_dim),
        )
        sims = sims.reshape(n_queries, n_nodes, n_embeddings)
        sims = np.max(sims, axis=(0, 2))

        results = [(node_id, sim) for node_id, sim in zip(node_ids, sims)]
        return sorted(results, key=lambda x: x[1], reverse=True)

    def search_img_nodes(self, img_embeddings: List[List[float]]) -> List[Tuple[int, float]]:
        target_nodes = [(node_id, node.embeddings) for node_id, node in self.nodes.items() if node.type == "img"]
        if not target_nodes:
            return []
        node_ids, node_embeddings = zip(*target_nodes)
        query_embeddings = np.array(img_embeddings)
        embedding_dim = query_embeddings.shape[-1]

        node_sims = []
        for node_emb in node_embeddings:
            node_emb = np.array(node_emb)
            sims = self._cosine_similarity_matrix(query_embeddings.reshape(-1, embedding_dim), node_emb.reshape(-1, embedding_dim))
            node_sims.append(np.mean(sims))
        results = [(node_id, sim) for node_id, sim in zip(node_ids, node_sims) if sim >= self.img_matching_threshold]
        return sorted(results, key=lambda x: x[1], reverse=True)

    def truncate_memory_by_clip(self, clip_id: int) -> None:
        last_node_id = None
        for node_id, node in self.nodes.items():
            if node.type in ["episodic", "semantic"] and node.metadata.get("timestamp") == clip_id:
                last_node_id = node_id
        if last_node_id is None:
            return

        to_del = [node_id for node_id in list(self.nodes.keys()) if node_id > last_node_id]
        for node_id in to_del:
            del self.nodes[node_id]

        to_del_edges = [edge for edge in list(self.edges.keys()) if edge[0] > last_node_id or edge[1] > last_node_id]
        for edge in to_del_edges:
            del self.edges[edge]

        self.text_nodes = [node_id for node_id in self.text_nodes if node_id <= last_node_id]
        self.text_nodes_by_clip = {
            clip: nodes for clip, nodes in self.text_nodes_by_clip.items() if clip <= clip_id
        }

    @staticmethod
    def _cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if a.size == 0 or b.size == 0:
            return np.zeros((a.shape[0], b.shape[0]))
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        return np.dot(a_norm, b_norm.T)
