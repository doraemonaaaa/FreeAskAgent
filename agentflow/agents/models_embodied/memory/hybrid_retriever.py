"""
Hybrid Retriever for A-MEM

实现BM25和语义搜索相结合的混合检索系统。
支持动态文档添加和检索。
"""

import numpy as np
from typing import List, Optional, Any, Dict

# 条件导入
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False


class HybridRetriever:
    """
    混合检索系统，结合BM25和语义搜索

    支持动态添加文档和混合检索。
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', alpha: float = 0.5,
                 local_model_path: str = None, disable_semantic_search: bool = False):
        """
        初始化混合检索器

        Args:
            model_name: SentenceTransformer模型名称
            alpha: BM25和语义搜索的权重平衡 (0.0=纯BM25, 1.0=纯语义)
            local_model_path: 本地模型路径
            disable_semantic_search: 是否禁用语义搜索
        """
        self.model_name = model_name
        self.alpha = alpha
        self.local_model_path = local_model_path
        self.disable_semantic_search = disable_semantic_search

        # 核心组件
        self.bm25 = None
        self.corpus = []
        self.embeddings = None
        self.model = None

        # 功能可用性
        self.bm25_available = BM25_AVAILABLE
        self.semantic_available = SEMANTIC_AVAILABLE and not disable_semantic_search

        # 初始化模型
        if self.semantic_available:
            self._init_model()

    def _init_model(self):
        """初始化嵌入模型"""
        try:
            if self.local_model_path and os.path.exists(self.local_model_path):
                self.model = SentenceTransformer(self.local_model_path)
            else:
                self.model = SentenceTransformer(self.model_name)
        except Exception:
            self.semantic_available = False

    def add_documents(self, documents: List[str]) -> bool:
        """批量添加文档到检索索引"""
        if not documents:
            return False

        try:
            # 初始化BM25
            if self.bm25_available:
                tokenized_docs = [self._simple_tokenize(doc.lower()) for doc in documents]
                self.bm25 = BM25Okapi(tokenized_docs)

            # 初始化语义嵌入
            if self.semantic_available and self.model:
                self.embeddings = self.model.encode(documents)

            # 更新语料库
            self.corpus = documents
            self.document_ids = {doc: idx for idx, doc in enumerate(documents)}

            return True

        except Exception:
            return False

    def retrieve(self, query: str, k: int = 5) -> List[int]:
        """执行混合检索"""
        if not self.corpus:
            return []

        try:
            bm25_scores = self._get_bm25_scores(query)
            semantic_scores = self._get_semantic_scores(query)
            hybrid_scores = self._combine_scores(bm25_scores, semantic_scores)

            k = min(k, len(self.corpus))
            top_k_indices = np.argsort(hybrid_scores)[-k:][::-1]

            return top_k_indices.tolist()

        except Exception:
            return []

    def _get_bm25_scores(self, query: str) -> np.ndarray:
        """获取BM25评分"""
        if not self.bm25_available or not self.bm25:
            return np.zeros(len(self.corpus))

        try:
            tokenized_query = self._simple_tokenize(query.lower())
            scores = np.array(self.bm25.get_scores(tokenized_query))

            # 归一化评分
            if scores.max() > scores.min():
                scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)

            return scores

        except Exception:
            return np.zeros(len(self.corpus))

    def _get_semantic_scores(self, query: str) -> np.ndarray:
        """获取语义相似度评分"""
        if not self.semantic_available or self.embeddings is None:
            return np.zeros(len(self.corpus))

        try:
            query_embedding = self.model.encode([query])[0]
            similarities = cosine_similarity([query_embedding], self.embeddings)[0]
            return similarities

        except Exception:
            return np.zeros(len(self.corpus))

    def _combine_scores(self, bm25_scores: np.ndarray, semantic_scores: np.ndarray) -> np.ndarray:
        """组合BM25和语义评分"""
        bm25_scores = np.array(bm25_scores)
        semantic_scores = np.array(semantic_scores)

        # 加权组合
        hybrid_scores = self.alpha * bm25_scores + (1 - self.alpha) * semantic_scores
        return hybrid_scores

    def _simple_tokenize(self, text: str) -> List[str]:
        """简单的文本分词"""
        import re
        tokens = re.findall(r'[\u4e00-\u9fff]{1,4}|[a-zA-Z]+|[0-9]+', text)
        return tokens

    def get_stats(self) -> Dict[str, Any]:
        """获取检索器统计信息"""
        return {
            'total_documents': len(self.corpus),
            'bm25_available': self.bm25_available,
            'semantic_available': self.semantic_available,
            'model_name': self.model_name,
            'alpha': self.alpha
        }

    def clear(self) -> None:
        """清空检索器状态"""
        self.bm25 = None
        self.corpus = []
        self.embeddings = None
        self.document_ids = {}