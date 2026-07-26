import math
from collections import Counter
from typing import Any, Dict, Optional, Sequence

from .memory.memory import Memory


class UncertaintyEstimator:
    """
    U(o_t, M_t) = w_vis * Uvis + w_policy * Upolicy + w_memory * Umemory

    - Uvis: dispersion of embeddings from repeated/augmented perception of o_t.
    - Upolicy: H(pi(.|I, o_t, M_t)) = -sum_a pi(a) log pi(a), entropy of the
      action distribution (estimated empirically from sampled actions).
    - Umemory: ||f(o_t) - f(M_t)||, inconsistency between current observation
      and stored memory.
    """

    def __init__(self, w_vis: float = 1.0, w_policy: float = 1.0, w_memory: float = 1.0):
        self.w_vis = w_vis
        self.w_policy = w_policy
        self.w_memory = w_memory

    # ---- Umemory ----
    def compute_u_memory(self, memory: Memory, image_paths: Any) -> float:
        return memory.compute_observation_uncertainty(image_paths)

    # ---- Upolicy ----
    @staticmethod
    def action_distribution_from_samples(action_labels: Sequence[str]) -> Dict[str, float]:
        if not action_labels:
            return {}
        counts = Counter(action_labels)
        total = sum(counts.values())
        return {action: count / total for action, count in counts.items()}

    @staticmethod
    def entropy(distribution: Dict[str, float]) -> float:
        if not distribution:
            return 0.0
        return -sum(p * math.log(p) for p in distribution.values() if p > 0)

    def compute_u_policy(self, action_samples: Optional[Sequence[str]]) -> float:
        if not action_samples:
            return 0.0
        distribution = self.action_distribution_from_samples(action_samples)
        return self.entropy(distribution)

    # ---- Uvis ----
    @staticmethod
    def compute_u_vis(embeddings: Optional[Sequence[Sequence[float]]]) -> float:
        if not embeddings or len(embeddings) < 2:
            return 0.0
        dim = len(embeddings[0])
        mean = [0.0] * dim
        for emb in embeddings:
            for i, x in enumerate(emb):
                mean[i] += x
        mean = [x / len(embeddings) for x in mean]
        variance = sum(
            sum((x - m) ** 2 for x, m in zip(emb, mean))
            for emb in embeddings
        ) / len(embeddings)
        return math.sqrt(variance)

    # ---- Combined ----
    def compute(
        self,
        memory: Memory,
        image_paths: Any,
        action_samples: Optional[Sequence[str]] = None,
        vis_embeddings: Optional[Sequence[Sequence[float]]] = None,
    ) -> Dict[str, float]:
        u_vis = self.compute_u_vis(vis_embeddings)
        u_policy = self.compute_u_policy(action_samples)
        u_memory = self.compute_u_memory(memory, image_paths)

        u_total = self.w_vis * u_vis + self.w_policy * u_policy + self.w_memory * u_memory

        return {
            "u_vis": u_vis,
            "u_policy": u_policy,
            "u_memory": u_memory,
            "u_total": u_total,
        }
