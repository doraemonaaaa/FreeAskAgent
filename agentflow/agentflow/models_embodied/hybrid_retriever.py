"""
Hybrid Retriever for A-MEM

实现BM25和语义搜索相结合的混合检索系统。
支持持久化存储和动态文档添加。
"""

import os
import pickle
import numpy as np
import json
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import sys

# 条件导入，增加错误处理
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("Warning: rank-bm25 not available, BM25 functionality will be disabled")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available, semantic search will be disabled")

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available, cosine similarity will be disabled")

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("Warning: python-dotenv not available, environment configuration will be disabled")

try:
    from .llm_controllers import LLMController
    LLM_CONTROLLERS_AVAILABLE = True
except ImportError:
    LLM_CONTROLLERS_AVAILABLE = False
    print("Warning: LLM controllers not available, API embedding will be disabled")


class HybridRetriever:
    """
    混合检索系统，结合BM25关键词匹配和语义向量搜索

    支持动态添加文档、持久化存储和多种检索策略。
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', alpha: float = 0.5, use_api_embedding: bool = None, local_model_path: str = None):
        """
        初始化混合检索器

        Args:
            model_name: SentenceTransformer模型名称或API模型名称
            alpha: BM25和语义搜索的权重平衡 (0.0=纯BM25, 1.0=纯语义)
            use_api_embedding: 是否使用API嵌入，None表示自动检测
            local_model_path: 本地模型路径，如果提供则使用本地模型而不是从Hugging Face下载
        """
        # 加载环境变量
        if DOTENV_AVAILABLE:
            load_dotenv()

        self.model_name = model_name
        self.alpha = alpha
        self.local_model_path = local_model_path
        self.bm25 = None
        self.corpus = []
        self.embeddings = None
        self.document_ids = {}  # document -> index 映射
        self.model = None
        self.llm_controller = None

        # 检查是否使用API嵌入
        if use_api_embedding is None:
            self.use_api_embedding = os.getenv('USE_API_EMBEDDING', 'false').lower() == 'true'
        else:
            self.use_api_embedding = use_api_embedding

        # 功能可用性标志
        self.bm25_available = BM25_AVAILABLE
        self.semantic_available = (SENTENCE_TRANSFORMERS_AVAILABLE and SKLEARN_AVAILABLE) or \
                                 (self.use_api_embedding and LLM_CONTROLLERS_AVAILABLE)

        # 延迟初始化模型
        self._init_model()

    def _init_model(self):
        """初始化嵌入模型（本地模型或API）"""
        if self.use_api_embedding:
            # 使用API嵌入
            if LLM_CONTROLLERS_AVAILABLE:
                try:
                    # 从环境变量获取API配置
                    backend = os.getenv('RETRIEVER_BACKEND', 'litellm')
                    api_model = os.getenv('EMBEDDING_MODEL', 'gpt-5')
                    api_base = os.getenv('EMBEDDING_API_BASE', 'https://yinli.one/')
                    api_key = os.getenv('EMBEDDING_API_KEY', '')

                    self.llm_controller = LLMController(
                        backend=backend,
                        model=api_model,
                        api_base=api_base,
                        api_key=api_key
                    )
                    print(f"API embedding initialized with {backend} backend, model: {api_model}")
                except Exception as e:
                    print(f"Warning: Failed to initialize API embedding: {e}")
                    self.semantic_available = False
            else:
                print("Warning: LLM controllers not available, API embedding disabled")
                self.semantic_available = False
        else:
            # 使用本地SentenceTransformer模型
            if SENTENCE_TRANSFORMERS_AVAILABLE and SKLEARN_AVAILABLE:
                try:
                    # 如果提供了本地模型路径，使用本地路径
                    if self.local_model_path and os.path.exists(self.local_model_path):
                        self.model = SentenceTransformer(self.local_model_path)
                        print(f"Local embedding model initialized from local path: {self.local_model_path}")
                    else:
                        # 设置较短的超时时间，避免长时间等待
                        import requests
                        requests.adapters.DEFAULT_RETRIES = 1
                        self.model = SentenceTransformer(self.model_name, cache_folder=os.path.expanduser("~/.cache/huggingface"))
                        print(f"Local embedding model initialized: {self.model_name}")
                except Exception as e:
                    print(f"Warning: Failed to load SentenceTransformer model: {e}")
                    print("  This may be due to network issues or missing dependencies")
                    self.semantic_available = False
            else:
                print("Semantic search disabled due to missing dependencies")
                self.semantic_available = False

    def add_documents(self, documents: List[str]) -> bool:
        """
        批量添加文档到检索索引

        Args:
            documents: 文档列表

        Returns:
            bool: 添加是否成功
        """
        if not documents:
            return False

        try:
            print(f"Adding {len(documents)} documents to retriever")

            # 初始化BM25
            if self.bm25_available:
                tokenized_docs = [self._simple_tokenize(doc.lower()) for doc in documents]
                self.bm25 = BM25Okapi(tokenized_docs)
                print(f"BM25 initialized with {len(tokenized_docs)} documents")

            # 初始化语义嵌入
            if self.semantic_available:
                if self.use_api_embedding:
                    print("Computing API embeddings...")
                    self.embeddings = self._get_api_embedding(documents)
                elif self.model:
                    print("Computing local embeddings...")
                    self.embeddings = self.model.encode(documents)

                if self.embeddings is not None:
                    print(f"Embeddings computed, shape: {self.embeddings.shape}")
                else:
                    print("Failed to compute embeddings")

            # 更新语料库和文档映射
            self.corpus = documents
            self.document_ids = {doc: idx for idx, doc in enumerate(documents)}
            print(f"Corpus updated with {len(self.corpus)} documents")

            return True

        except Exception as e:
            print(f"Error adding documents: {e}")
            return False

    def add_document(self, document: str) -> bool:
        """
        添加单个文档到检索索引

        Args:
            document: 文档内容

        Returns:
            bool: True如果成功添加，False如果已存在
        """
        # 检查文档是否已存在
        if document in self.document_ids:
            return False

        try:
            doc_idx = len(self.corpus)
            self.corpus.append(document)
            self.document_ids[document] = doc_idx

            # 更新BM25索引
            if self.bm25_available and self.bm25:
                tokenized_doc = self._simple_tokenize(document.lower())
                self.bm25.add_document(tokenized_doc)

            # 更新语义嵌入
            if self.semantic_available and self.embeddings is not None:
                if self.use_api_embedding:
                    doc_embedding = self._get_api_embedding([document])
                elif self.model:
                    doc_embedding = self.model.encode([document])
                self.embeddings = np.vstack([self.embeddings, doc_embedding])
            elif self.semantic_available and self.embeddings is None:
                # 第一个文档
                if self.use_api_embedding:
                    self.embeddings = self._get_api_embedding([document])
                elif self.model:
                    self.embeddings = self.model.encode([document])

            return True

        except Exception as e:
            print(f"Error adding single document: {e}")
            return False

    def retrieve(self, query: str, k: int = 5) -> List[int]:
        """
        执行混合检索

        Args:
            query: 查询字符串
            k: 返回的Top-K结果数量

        Returns:
            List[int]: 相关文档的索引列表
        """
        if not self.corpus:
            return []

        try:
            # 获取各种评分
            bm25_scores = self._get_bm25_scores(query)
            semantic_scores = self._get_semantic_scores(query)

            print(f"Debug: BM25 scores shape: {bm25_scores.shape}, Semantic scores shape: {semantic_scores.shape}")
            print(f"Debug: Corpus length: {len(self.corpus)}")

            # 组合评分
            hybrid_scores = self._combine_scores(bm25_scores, semantic_scores)

            print(f"Debug: Hybrid scores shape: {hybrid_scores.shape}, values: {hybrid_scores[:5]}")

            # 返回Top-K索引
            k = min(k, len(self.corpus))
            top_k_indices = np.argsort(hybrid_scores)[-k:][::-1]

            print(f"Debug: Top-K indices: {top_k_indices.tolist()}")

            return top_k_indices.tolist()

        except Exception as e:
            print(f"Error during retrieval: {e}")
            import traceback
            traceback.print_exc()
            return []

    def search(self, query: str, k: int = 5) -> List[int]:
        """
        搜索接口（与retrieve相同，用于兼容性）

        Args:
            query: 查询字符串
            k: 返回的结果数量

        Returns:
            List[int]: 相关文档的索引列表
        """
        return self.retrieve(query, k)

    def _get_bm25_scores(self, query: str) -> np.ndarray:
        """
        获取BM25评分

        Args:
            query: 查询字符串

        Returns:
            np.ndarray: BM25评分数组
        """
        if not self.bm25_available or not self.bm25:
            return np.zeros(len(self.corpus))

        try:
            tokenized_query = self._simple_tokenize(query.lower())
            scores = np.array(self.bm25.get_scores(tokenized_query))

            # 归一化评分
            if len(scores) > 0 and scores.max() > scores.min():
                scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)

            return scores

        except Exception as e:
            print(f"Error computing BM25 scores: {e}")
            return np.zeros(len(self.corpus))

    def _get_semantic_scores(self, query: str) -> np.ndarray:
        """
        获取语义相似度评分

        Args:
            query: 查询字符串

        Returns:
            np.ndarray: 语义相似度评分数组
        """
        if not self.semantic_available or self.embeddings is None:
            return np.zeros(len(self.corpus))

        try:
            if self.use_api_embedding:
                # 使用API计算查询嵌入
                query_embedding = self._get_api_embedding([query])[0]
            else:
                # 使用本地模型计算查询嵌入
                query_embedding = self.model.encode([query])[0]

            similarities = cosine_similarity([query_embedding], self.embeddings)[0]
            return similarities

        except Exception as e:
            print(f"Error computing semantic scores: {e}")
            return np.zeros(len(self.corpus))

    def _get_api_embedding(self, texts: List[str]) -> np.ndarray:
        """
        使用API获取文本嵌入向量

        Args:
            texts: 文本列表

        Returns:
            np.ndarray: 嵌入向量数组
        """
        if not self.llm_controller:
            raise ValueError("API embedding controller not initialized")

        try:
            # 使用LLM API生成嵌入向量
            # 这里使用一个简单的提示来让LLM生成可比较的向量表示
            embeddings = []

            for text in texts:
                prompt = f"""为以下文本生成10个数字的向量表示，每个数字在-1到1之间。输出格式：只输出10个用逗号分隔的数字。

文本：{text}

向量："""

                response = self.llm_controller.get_completion(
                    prompt=prompt,
                    temperature=1.0  # GPT-5要求temperature=1
                )

                try:
                    # 解析响应，期望格式如 "0.1, 0.2, 0.3, ..."
                    vector_str = response.strip()
                    # 移除可能的markdown格式
                    vector_str = vector_str.replace('```', '').replace('json', '').strip()

                    # 尝试解析为JSON数组
                    if vector_str.startswith('[') and vector_str.endswith(']'):
                        vector = json.loads(vector_str)
                    else:
                        # 解析逗号分隔的数字
                        parts = vector_str.split(',')
                        vector = []
                        for part in parts[:10]:  # 只取前10个
                            try:
                                num = float(part.strip())
                                # 确保在-1到1范围内
                                num = max(-1.0, min(1.0, num))
                                vector.append(num)
                            except ValueError:
                                vector.append(0.0)

                    # 确保向量长度为10
                    while len(vector) < 10:
                        vector.append(0.0)
                    vector = vector[:10]

                    embeddings.append(vector)
                except Exception as parse_error:
                    print(f"Failed to parse API response: {response}, error: {parse_error}")
                    # 解析失败，返回零向量
                    embeddings.append([0.0] * 10)

            return np.array(embeddings)

        except Exception as e:
            print(f"Error getting API embeddings: {e}")
            # 返回零向量
            return np.zeros((len(texts), 10))

    def _combine_scores(self, bm25_scores: np.ndarray, semantic_scores: np.ndarray) -> np.ndarray:
        """
        组合BM25和语义评分

        Args:
            bm25_scores: BM25评分
            semantic_scores: 语义评分

        Returns:
            np.ndarray: 组合后的评分
        """
        # 确保评分数组长度一致
        bm25_scores = np.array(bm25_scores)
        semantic_scores = np.array(semantic_scores)

        if len(bm25_scores) != len(semantic_scores):
            max_len = max(len(bm25_scores), len(semantic_scores))
            bm25_scores = np.pad(bm25_scores, (0, max_len - len(bm25_scores)), 'constant')
            semantic_scores = np.pad(semantic_scores, (0, max_len - len(semantic_scores)), 'constant')

        # 加权组合
        hybrid_scores = self.alpha * bm25_scores + (1 - self.alpha) * semantic_scores
        return hybrid_scores

    def _simple_tokenize(self, text: str) -> List[str]:
        """
        简单的文本分词（用于BM25）

        Args:
            text: 输入文本

        Returns:
            List[str]: 分词结果
        """
        # 简单实现：按空格分割，并处理中文字符
        import re
        # 分离中文字符序列和英文单词
        tokens = re.findall(r'[\u4e00-\u9fff]{1,4}|[a-zA-Z]+|[0-9]+', text)
        return tokens

    def save(self, cache_file: str, embeddings_file: str) -> bool:
        """
        保存检索器状态到磁盘

        Args:
            cache_file: 状态缓存文件路径
            embeddings_file: 嵌入向量文件路径

        Returns:
            bool: 保存是否成功
        """
        try:
            # 保存嵌入向量
            if self.embeddings is not None:
                np.save(embeddings_file, self.embeddings)

            # 保存其他状态
            state = {
                'alpha': self.alpha,
                'bm25': self.bm25 if self.bm25_available else None,
                'corpus': self.corpus,
                'document_ids': self.document_ids,
                'model_name': self.model_name,
                'bm25_available': self.bm25_available,
                'semantic_available': self.semantic_available
            }

            with open(cache_file, 'wb') as f:
                pickle.dump(state, f)

            return True

        except Exception as e:
            print(f"Error saving retriever: {e}")
            return False

    @classmethod
    def load(cls, cache_file: str, embeddings_file: str) -> Optional['HybridRetriever']:
        """
        从磁盘加载检索器状态

        Args:
            cache_file: 状态缓存文件路径
            embeddings_file: 嵌入向量文件路径

        Returns:
            HybridRetriever: 加载的检索器实例，如果失败返回None
        """
        try:
            # 加载状态
            with open(cache_file, 'rb') as f:
                state = pickle.load(f)

            # 创建实例
            retriever = cls(
                model_name=state.get('model_name', 'all-MiniLM-L6-v2'),
                alpha=state.get('alpha', 0.5)
            )

            # 恢复状态
            retriever.bm25 = state.get('bm25')
            retriever.corpus = state.get('corpus', [])
            retriever.document_ids = state.get('document_ids', {})

            # 恢复功能可用性
            retriever.bm25_available = state.get('bm25_available', BM25_AVAILABLE)
            retriever.semantic_available = state.get('semantic_available', SENTENCE_TRANSFORMERS_AVAILABLE and SKLEARN_AVAILABLE)

            # 加载嵌入向量
            embeddings_path = Path(embeddings_file)
            if embeddings_path.exists():
                retriever.embeddings = np.load(embeddings_file)

            return retriever

        except Exception as e:
            print(f"Error loading retriever: {e}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """
        获取检索器统计信息

        Returns:
            Dict[str, Any]: 统计信息
        """
        return {
            'total_documents': len(self.corpus),
            'bm25_available': self.bm25_available,
            'semantic_available': self.semantic_available,
            'use_api_embedding': self.use_api_embedding,
            'model_name': self.model_name,
            'alpha': self.alpha,
            'embeddings_shape': self.embeddings.shape if self.embeddings is not None else None,
            'llm_controller_available': self.llm_controller is not None
        }

    def clear(self) -> None:
        """清空检索器状态"""
        self.bm25 = None
        self.corpus = []
        self.embeddings = None
        self.document_ids = {}
