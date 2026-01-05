"""
Long Term Memory for Embodied Agent

长期记忆模块，负责跨会话的记忆存储、检索和管理。
提供A-MEM增强功能，支持内容分析和混合检索。
"""

import os
import json
import pickle
import logging
import time
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from .hybrid_retriever import HybridRetriever
from .content_analyzer import ContentAnalyzer


class LongMemory:
    """
    长期记忆类

    提供跨会话的记忆存储、检索和管理功能。
    集成内容分析器和混合检索器，支持A-MEM增强功能。
    """

    def __init__(self,
                 use_amem: bool = True,
                 retriever_config: Optional[Dict[str, Any]] = None,
                 storage_dir: str = "./memory_store",
                 enable_persistence: bool = True,
                 max_memories: int = 1000):
        """
        初始化长期记忆系统

        Args:
            use_amem: 是否启用A-MEM功能
            retriever_config: 检索器配置参数
            storage_dir: 记忆存储目录
            enable_persistence: 是否启用持久化
            max_memories: 最大记忆数量
        """
        # A-MEM配置
        self.use_amem = use_amem
        self.retriever_config = retriever_config or {}
        self.storage_dir = Path(storage_dir)
        self.enable_persistence = enable_persistence
        self.max_memories = max_memories

        # A-MEM组件
        self.retriever: Optional[HybridRetriever] = None
        self.content_analyzer: Optional[ContentAnalyzer] = None
        self.long_term_memories: List[str] = []
        self.memory_metadata: List[Dict[str, Any]] = []

        # 配置日志
        self._setup_logging()

        # 统计信息
        self.stats = {
            'total_memories': 0,
            'retrieval_count': 0,
            'avg_retrieval_time': 0.0,
            'last_save_time': None,
            'amem_enabled': self.use_amem,
            'performance_metrics': {
                'retrieval_times': [],
                'memory_usage': [],
                'error_count': 0,
                'success_rate': 1.0
            },
            'memory_types': defaultdict(int),
            'retrieval_patterns': defaultdict(int)
        }

        # 初始化A-MEM组件
        if self.use_amem:
            self._init_amem_components()

        # 创建存储目录
        if self.enable_persistence:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            self._load_state()

    def _setup_logging(self):
        """设置日志记录"""
        self.logger = logging.getLogger('LongMemory')

        # 如果还没有配置处理器，添加一个
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # 设置日志级别
        verbose = os.getenv('AMEM_VERBOSE', 'false').lower() == 'true'
        self.logger.setLevel(logging.DEBUG if verbose else logging.INFO)

        self.logger.info("LongMemory initialized")

    def _init_amem_components(self):
        """初始化A-MEM组件"""
        try:
            # 初始化检索器
            self.retriever = HybridRetriever(
                model_name=self.retriever_config.get('model_name', 'all-MiniLM-L6-v2'),
                use_api_embedding=self.retriever_config.get('use_api_embedding', False),
                alpha=self.retriever_config.get('alpha', 0.5),
                local_model_path=self.retriever_config.get('local_model_path')
            )

            # 初始化内容分析器
            self.content_analyzer = ContentAnalyzer()

            self.logger.info("✅ A-MEM components initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize A-MEM components: {e}")
            self.use_amem = False

    def add_memory(self, content: str, memory_type: str = "custom",
                   metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        添加记忆内容

        Args:
            content: 记忆内容
            memory_type: 记忆类型
            metadata: 元数据

        Returns:
            bool: 是否成功添加
        """
        if len(self.long_term_memories) >= self.max_memories:
            self.logger.warning(f"Memory limit reached ({self.max_memories}), cannot add more memories")
            return False

        start_time = time.time()
        try:
            self.logger.debug(f"Adding memory (type: {memory_type}, length: {len(content)})")

            # 分析内容（如果有分析器）
            analysis_result = None
            processed_content = content
            if self.content_analyzer:
                analysis_start = time.time()
                analysis_result = self.content_analyzer.analyze_content(content)
                analysis_time = time.time() - analysis_start
                self.logger.debug(f"Content analysis completed in {analysis_time:.3f}s")

                # 从分析结果中构建检索内容
                if isinstance(analysis_result, dict):
                    keywords = analysis_result.get('keywords', [])
                    context = analysis_result.get('context', '')
                    tags = analysis_result.get('tags', [])
                    processed_content = f"{content} {' '.join(keywords)} {context} {' '.join(tags)}"

            # 添加到记忆库
            self.long_term_memories.append(processed_content)

            # 创建元数据
            full_metadata = {
                'content': processed_content,
                'original_content': content,
                'analysis': analysis_result if isinstance(analysis_result, dict) else None,
                'type': memory_type,
                'timestamp': time.time(),
                'added_at': datetime.now().isoformat(),
                'content_length': len(processed_content) if isinstance(processed_content, str) else len(str(processed_content)),
                'amplified': True if analysis_result else False,
                **(metadata or {})
            }
            self.memory_metadata.append(full_metadata)

            # 更新检索器 - 重新构建索引以确保一致性
            if self.retriever:
                # 由于BM25和embeddings需要重新计算，我们重新构建整个索引
                self.retriever.add_documents(self.long_term_memories)
                self.logger.debug(f"Rebuilt retriever index with {len(self.long_term_memories)} documents")

            # 更新统计信息
            self.stats['total_memories'] += 1
            self.stats['memory_types'][memory_type] += 1

            total_time = time.time() - start_time
            self.logger.info(f"✅ Memory added successfully (type: {memory_type}, "
                           f"total: {self.stats['total_memories']}, took: {total_time:.3f}s)")

            return True

        except Exception as e:
            total_time = time.time() - start_time
            self.logger.error(f"Failed to add memory: {e} (took {total_time:.3f}s)")
            return False

    def retrieve_memories(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        检索长期记忆

        Args:
            query: 查询字符串
            k: 返回结果数量

        Returns:
            List[Dict]: 包含记忆内容和元数据的列表
        """
        if not self.use_amem or not self.retriever:
            self.logger.debug("A-MEM not available, returning empty results")
            return []

        start_time = time.time()
        query_hash = hash(query) % 1000  # 简单的查询模式识别

        try:
            self.logger.debug(f"Starting memory retrieval for query: '{query[:50]}...'")
            self.logger.debug(f"Corpus length: {len(self.corpus) if hasattr(self, 'corpus') and self.corpus else 0}")
            self.logger.debug(f"Retriever available: {self.retriever is not None}")
            self.logger.debug(f"A-MEM enabled: {self.use_amem}")

            # 执行检索
            indices = self.retriever.retrieve(query, k=k)
            self.logger.debug(f"Retrieved indices: {indices}")
            self.logger.debug(f"Retrieved indices: {indices}")

            # 构建结果
            self.logger.debug(f"Building results from {len(indices)} indices")
            self.logger.debug(f"long_term_memories length: {len(self.long_term_memories)}")
            self.logger.debug(f"memory_metadata length: {len(self.memory_metadata)}")
            self.logger.debug(f"retriever corpus length: {len(self.retriever.corpus) if self.retriever and hasattr(self.retriever, 'corpus') else 'no corpus'}")
            results = []
            for idx in indices:
                self.logger.debug(f"Processing index {idx}, valid range: 0-{len(self.long_term_memories)-1}")
                if 0 <= idx < len(self.long_term_memories):
                    memory_content = self.long_term_memories[idx]
                    metadata = self.memory_metadata[idx] if idx < len(self.memory_metadata) else {}

                    results.append({
                        'content': memory_content,
                        'metadata': metadata,
                        'index': idx
                    })
                    self.logger.debug(f"Added result {len(results)-1}: {memory_content[:50]}...")
                else:
                    self.logger.warning(f"Invalid index {idx} (valid range: 0-{len(self.long_term_memories)-1})")

            try:
                # 更新统计信息
                retrieval_time = time.time() - start_time
                self.stats['retrieval_count'] += 1
                self.stats['retrieval_patterns'][query_hash] += 1

                # 更新性能指标
                perf = self.stats['performance_metrics']
                perf['retrieval_times'].append(retrieval_time)

                # 保持最近100个检索时间的记录
                if len(perf['retrieval_times']) > 100:
                    perf['retrieval_times'] = perf['retrieval_times'][-100:]

                # 更新平均时间
                self.stats['avg_retrieval_time'] = sum(perf['retrieval_times']) / len(perf['retrieval_times'])

                # 计算成功率
                if perf['retrieval_times']:
                    perf['success_rate'] = (len(perf['retrieval_times']) - perf['error_count']) / len(perf['retrieval_times'])

                self.logger.info(f"Retrieval completed in {retrieval_time:.3f}s, memories_found={len(results)}")
            except Exception as stats_e:
                self.logger.error(f"Error updating stats: {stats_e}")

            return results

        except Exception as e:
            # 记录错误
            retrieval_time = time.time() - start_time
            self.stats['performance_metrics']['error_count'] += 1
            perf = self.stats['performance_metrics']

            if perf['retrieval_times']:
                perf['success_rate'] = (len(perf['retrieval_times']) - perf['error_count']) / len(perf['retrieval_times'])

            self.logger.error(f"Long-term memory retrieval failed: {e} (took {retrieval_time:.3f}s)")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """
        获取系统统计信息

        Returns:
            Dict: 完整的统计信息
        """
        current_stats = self.stats.copy()

        # 添加实时信息
        current_stats.update({
            'current_memory_count': len(self.long_term_memories),
            'amem_available': self.use_amem and self.retriever is not None,
            'persistence_enabled': self.enable_persistence,
            'storage_dir': str(self.storage_dir),
            'uptime': time.time() - getattr(self, '_start_time', time.time()),
            'memory_utilization': len(self.long_term_memories) / self.max_memories if self.max_memories > 0 else 0,
            'retriever_config': self.retriever_config if self.retriever_config else {}
        })

        # 添加性能摘要
        perf = current_stats['performance_metrics']
        if perf['retrieval_times']:
            current_stats['performance_summary'] = {
                'avg_retrieval_time': sum(perf['retrieval_times']) / len(perf['retrieval_times']),
                'max_retrieval_time': max(perf['retrieval_times']),
                'min_retrieval_time': min(perf['retrieval_times']),
                'total_retrievals': len(perf['retrieval_times']),
                'success_rate': perf['success_rate'],
                'error_rate': perf['error_count'] / len(perf['retrieval_times']) if perf['retrieval_times'] else 0
            }

        # 添加记忆类型分布
        current_stats['memory_distribution'] = dict(self.stats['memory_types'])

        self.logger.debug(f"Stats requested: {len(self.long_term_memories)} memories, "
                         f"{self.stats['retrieval_count']} retrievals")

        return current_stats

    def save_state(self) -> bool:
        """
        保存记忆状态到磁盘

        Returns:
            bool: 保存是否成功
        """
        if not self.enable_persistence:
            return True

        try:
            # 保存A-MEM状态
            if self.use_amem:
                amem_state = {
                    'long_term_memories': self.long_term_memories,
                    'memory_metadata': self.memory_metadata,
                    'stats': self.stats,
                    'retriever_config': self.retriever_config
                }

                amem_path = self.storage_dir / "amem_state.json"
                with open(amem_path, 'w', encoding='utf-8') as f:
                    json.dump(amem_state, f, ensure_ascii=False, indent=2)

                # 保存检索器状态
                if self.retriever:
                    retriever_path = self.storage_dir / "retriever.pkl"
                    with open(retriever_path, 'wb') as f:
                        pickle.dump(self.retriever, f)

            self.stats['last_save_time'] = time.time()
            print(f"✅ Memory state saved to {self.storage_dir}")
            return True

        except Exception as e:
            print(f"⚠️  Failed to save memory state: {e}")
            return False

    def _load_state(self) -> bool:
        """
        从磁盘加载记忆状态

        Returns:
            bool: 加载是否成功
        """
        if not self.enable_persistence:
            return True

        try:
            # 加载A-MEM状态
            if self.use_amem:
                amem_path = self.storage_dir / "amem_state.json"
                if amem_path.exists():
                    with open(amem_path, 'r', encoding='utf-8') as f:
                        amem_state = json.load(f)

                    self.long_term_memories = amem_state.get('long_term_memories', [])
                    self.memory_metadata = amem_state.get('memory_metadata', [])
                    self.stats.update(amem_state.get('stats', {}))

                    # 加载检索器状态
                    retriever_path = self.storage_dir / "retriever.pkl"
                    self.logger.info(f"Loading retriever from {retriever_path}, exists: {retriever_path.exists()}")
                    if retriever_path.exists():
                        try:
                            with open(retriever_path, 'rb') as f:
                                loaded_retriever = pickle.load(f)
                                self.logger.info(f"Loaded retriever with corpus length: {len(loaded_retriever.corpus) if hasattr(loaded_retriever, 'corpus') else 'no corpus'}")
                                self.logger.info(f"Expected corpus length: {len(self.long_term_memories)}")

                                # 验证corpus是否与long_term_memories同步
                                current_corpus_size = len(loaded_retriever.corpus) if hasattr(loaded_retriever, 'corpus') else 0
                                if current_corpus_size == len(self.long_term_memories):
                                    self.retriever = loaded_retriever
                                    self.logger.info("Retriever loaded successfully - corpus size matches")
                                else:
                                    self.logger.warning(f"Corpus size mismatch ({current_corpus_size} vs {len(self.long_term_memories)}), creating new retriever")
                                    # 如果不匹配，创建新的retriever
                                    if self.long_term_memories:
                                        self.retriever = HybridRetriever(
                                            model_name=self.retriever_config.get('model_name', 'all-MiniLM-L6-v2'),
                                            use_api_embedding=self.retriever_config.get('use_api_embedding', False)
                                        )
                                        self.retriever.add_documents(self.long_term_memories)
                                        self.logger.info(f"Created new retriever with {len(self.long_term_memories)} documents")
                        except Exception as e:
                            self.logger.error(f"⚠️  Failed to load retriever state: {e}")
                            import traceback
                            traceback.print_exc()
                    else:
                        self.logger.info("Retriever pickle file does not exist, will create new retriever")

            print(f"✅ Memory state loaded from {self.storage_dir}")
            return True

        except Exception as e:
            print(f"⚠️  Failed to load memory state: {e}")
            return False

    def clear_memories(self, memory_type: Optional[str] = None) -> int:
        """
        清空记忆

        Args:
            memory_type: 要清空的记忆类型，None表示清空所有

        Returns:
            int: 清空的记忆数量
        """
        if not self.use_amem:
            return 0

        if memory_type is None:
            # 清空所有记忆
            cleared_count = len(self.long_term_memories)
            self.long_term_memories.clear()
            self.memory_metadata.clear()

            # 重新初始化检索器
            if self.retriever:
                self.retriever = HybridRetriever(
                    model_name=self.retriever_config.get('model_name', 'all-MiniLM-L6-v2'),
                    use_api_embedding=self.retriever_config.get('use_api_embedding', False),
                    alpha=self.retriever_config.get('alpha', 0.5),
                    local_model_path=self.retriever_config.get('local_model_path')
                )

            self.stats['total_memories'] = 0
        else:
            # 清空指定类型的记忆
            indices_to_remove = []
            for i, metadata in enumerate(self.memory_metadata):
                if metadata.get('type') == memory_type:
                    indices_to_remove.append(i)

            # 从后往前删除以保持索引正确性
            for i in reversed(indices_to_remove):
                del self.long_term_memories[i]
                del self.memory_metadata[i]

            cleared_count = len(indices_to_remove)
            self.stats['total_memories'] -= cleared_count

        # 保存状态
        if self.enable_persistence:
            self.save_state()

        return cleared_count