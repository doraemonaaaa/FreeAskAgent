"""
Agentic Memory System - A-MEM集成层

将A-MEM (Agentic Memory for LLM Agents) 与现有FreeAskAgent Memory系统集成，
提供长期记忆、混合检索和记忆演化能力，同时保持完全向后兼容。
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

try:
    from .memory import Memory
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from memory import Memory


class AgenticMemorySystem:
    """
    A-MEM与现有Memory的集成层

    核心功能：
    - 保持与现有Memory类的完全兼容
    - 集成HybridRetriever进行长期记忆管理
    - 支持记忆持久化和跨会话保持
    - 提供配置开关控制A-MEM功能启用
    """

    def __init__(self,
                 use_amem: bool = True,
                 retriever_config: Optional[Dict[str, Any]] = None,
                 storage_dir: str = "./memory_store",
                 enable_persistence: bool = True,
                 max_memories: int = 1000):
        """
        初始化AgenticMemorySystem

        Args:
            use_amem: 是否启用A-MEM功能
            retriever_config: 检索器配置参数
            storage_dir: 记忆存储目录
            enable_persistence: 是否启用持久化
            max_memories: 最大记忆数量
        """
        # 基础Memory保持兼容
        self.basic_memory = Memory()

        # A-MEM配置
        self.use_amem = use_amem
        self.retriever_config = retriever_config or {}
        self.storage_dir = Path(storage_dir)
        self.enable_persistence = enable_persistence
        self.max_memories = max_memories

        # A-MEM组件
        self.retriever = None
        self.content_analyzer = None
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
        self.logger = logging.getLogger('AgenticMemorySystem')

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

        self.logger.info("AgenticMemorySystem initialized")

    def _init_amem_components(self):
        """初始化A-MEM组件"""
        try:
            # 尝试相对导入
            try:
                from .memory.hybrid_retriever import HybridRetriever
                from .memory.content_analyzer import ContentAnalyzer
            except ImportError:
                # 如果相对导入失败，尝试绝对导入
                import sys
                import os
                current_dir = os.path.dirname(os.path.abspath(__file__))
                memory_dir = os.path.join(current_dir, 'memory')
                if memory_dir not in sys.path:
                    sys.path.insert(0, memory_dir)

                from hybrid_retriever import HybridRetriever
                from content_analyzer import ContentAnalyzer

            # 初始化检索器
            # 如果有本地模型路径，优先使用本地模型，不使用API嵌入
            local_model_path = self.retriever_config.get('local_model_path')
            use_api = self.retriever_config.get('use_api_embedding', True)
            if local_model_path and os.path.exists(local_model_path):
                use_api = False

            self.retriever = HybridRetriever(
                model_name=self.retriever_config.get('model_name', 'all-MiniLM-L6-v2'),
                use_api_embedding=use_api,
                alpha=self.retriever_config.get('alpha', 0.5),
                local_model_path=local_model_path
            )

            # 初始化内容分析器
            self.content_analyzer = ContentAnalyzer()

            self.logger.info("✅ A-MEM components initialized successfully")
        except ImportError as e:
            self.logger.warning(f"A-MEM components not available: {e}")
            self.logger.info("Falling back to basic memory functionality")
            self.use_amem = False
        except Exception as e:
            self.logger.warning(f"Failed to initialize A-MEM components: {e}")
            self.logger.info("Falling back to basic memory functionality")
            self.use_amem = False

    # ============ 兼容现有Memory接口 ============

    def set_query(self, query: str) -> None:
        """设置查询（兼容现有接口）"""
        self.basic_memory.set_query(query)

    def add_file(self, file_name: Union[str, List[str]], description: Union[str, List[str], None] = None) -> None:
        """添加文件（兼容现有接口）"""
        self.basic_memory.add_file(file_name, description)

    def add_action(self, step_count: int, tool_name: str, sub_goal: str, command: str, result: Any) -> None:
        """添加动作（兼容现有接口 + A-MEM增强）"""
        # 基础Memory存储
        self.basic_memory.add_action(step_count, tool_name, sub_goal, command, result)

        # A-MEM增强：分析并存储到长期记忆
        if self.use_amem and self.content_analyzer:
            # A-MEM模式：使用LLM分析
            try:
                # 构建记忆内容
                action_content = f"Tool: {tool_name}, Goal: {sub_goal}, Command: {command}, Result: {str(result)[:500]}"

                # LLM分析内容
                analyzed_memory = self.content_analyzer.analyze_content(action_content)

                # 提取用于检索的内容
                if isinstance(analyzed_memory, dict):
                    # 从分析结果中构建检索内容
                    keywords = analyzed_memory.get('keywords', [])
                    context = analyzed_memory.get('context', '')
                    tags = analyzed_memory.get('tags', [])

                    # 构建检索用的字符串内容
                    search_content = f"{action_content} {' '.join(keywords)} {context} {' '.join(tags)}"
                else:
                    # 回退到原始内容
                    search_content = str(analyzed_memory) if analyzed_memory else action_content

                # 添加到长期记忆
                if search_content and len(self.long_term_memories) < self.max_memories:
                    self.long_term_memories.append(search_content)

                    # 创建元数据
                    metadata = {
                        'content': search_content,
                        'original_content': action_content,
                        'analysis': analyzed_memory if isinstance(analyzed_memory, dict) else None,
                        'type': 'action',
                        'tool_name': tool_name,
                        'sub_goal': sub_goal,
                        'timestamp': time.time(),
                        'step_count': step_count,
                        'amplified': True  # 标记为经过LLM分析
                    }
                    self.memory_metadata.append(metadata)

                    # 更新检索器
                    if self.retriever:
                        self.retriever.add_documents([search_content])

                    self.stats['total_memories'] += 1

            except Exception as e:
                self.logger.warning(f"Failed to add to A-MEM long-term memory: {e}")

        elif not self.use_amem:
            # 基础模式：直接存储动作信息到长期记忆
            try:
                # 构建基础记忆内容
                action_content = f"Tool: {tool_name}, Goal: {sub_goal}, Result: {str(result)[:300]}"

                # 检查是否已存在相似记忆，避免重复
                if action_content not in self.long_term_memories and len(self.long_term_memories) < self.max_memories:
                    self.long_term_memories.append(action_content)

                    # 创建基础元数据
                    metadata = {
                        'content': action_content,
                        'type': 'action',
                        'tool_name': tool_name,
                        'sub_goal': sub_goal,
                        'timestamp': time.time(),
                        'step_count': step_count,
                        'amplified': False  # 标记为基础模式
                    }
                    self.memory_metadata.append(metadata)

                    self.stats['total_memories'] += 1
                    self.logger.info(f"✅ Basic action memory added (tool: {tool_name})")

            except Exception as e:
                self.logger.warning(f"Failed to add basic action memory: {e}")

    def get_query(self) -> Optional[str]:
        """获取查询（兼容现有接口）"""
        return self.basic_memory.get_query()

    def get_files(self) -> List[Dict[str, str]]:
        """获取文件列表（兼容现有接口）"""
        return self.basic_memory.get_files()

    def get_actions(self) -> Dict[str, Dict[str, Any]]:
        """获取动作列表（兼容现有接口）"""
        return self.basic_memory.get_actions()

    # ============ A-MEM增强功能 ============

    def retrieve_long_term_memories(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        检索长期记忆

        Args:
            query: 查询字符串
            k: 返回结果数量

        Returns:
            List[Dict]: 包含记忆内容和元数据的列表
        """
        if not self.use_amem:
            # 基础模式：使用简单的关键词匹配
            if not self.long_term_memories:
                self.logger.debug("No memories available for retrieval")
                return []

            # 简单的关键词检索（基础模式）
            query_lower = query.lower()
            matched_results = []

            for idx, memory_content in enumerate(self.long_term_memories):
                if query_lower in memory_content.lower():
                    metadata = self.memory_metadata[idx] if idx < len(self.memory_metadata) else {}
                    matched_results.append({
                        'content': memory_content,
                        'metadata': metadata,
                        'index': idx,
                        'relevance_score': 1.0  # 基础模式下简单设置为1.0
                    })

            # 按相关性排序并限制数量
            matched_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            results = matched_results[:k]

            self.logger.debug(f"Basic mode retrieval: found {len(results)} matches for '{query}'")
            return results

        # A-MEM模式：使用完整的检索功能
        if not self.retriever:
            self.logger.debug("A-MEM retriever not available")
            return []

        start_time = time.time()
        query_hash = hash(query) % 1000  # 简单的查询模式识别

        try:
            self.logger.debug(f"Starting memory retrieval for query: '{query[:50]}...'")

            # 执行检索
            indices = self.retriever.retrieve(query, k=k)

            # 构建结果
            results = []
            for idx in indices:
                if 0 <= idx < len(self.long_term_memories):
                    memory_content = self.long_term_memories[idx]
                    metadata = self.memory_metadata[idx] if idx < len(self.memory_metadata) else {}

                    results.append({
                        'content': memory_content,
                        'metadata': metadata,
                        'index': idx
                    })

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

            self.logger.info(".3f"
                           f"memories_found={len(results)}")

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

    def add_custom_memory(self, content: str, memory_type: str = "custom",
                         metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        添加自定义记忆

        Args:
            content: 记忆内容
            memory_type: 记忆类型
            metadata: 元数据

        Returns:
            bool: 是否成功添加
        """
        # 在基础模式下，我们仍然允许添加记忆，但不使用A-MEM增强功能
        if not self.use_amem:
            # 基础模式：直接存储到long_term_memories，但不进行LLM分析或向量嵌入
            if len(self.long_term_memories) >= self.max_memories:
                self.logger.warning(f"Memory limit reached ({self.max_memories}), cannot add more memories")
                return False

            try:
                # 直接存储内容
                self.long_term_memories.append(content)

                # 创建基础元数据
                full_metadata = {
                    'content': content,
                    'type': memory_type,
                    'timestamp': time.time(),
                    'added_at': datetime.now().isoformat(),
                    'content_length': len(content),
                    'amplified': False,  # 标记为未经过A-MEM增强
                    **(metadata or {})
                }
                self.memory_metadata.append(full_metadata)

                # 更新统计信息
                self.stats['total_memories'] += 1
                self.stats['memory_types'][memory_type] += 1

                self.logger.info(f"✅ Basic memory added (type: {memory_type}, "
                               f"total: {self.stats['total_memories']})")

                return True

            except Exception as e:
                self.logger.error(f"Failed to add basic memory: {e}")
                return False

        # A-MEM模式：使用完整的增强功能
        if len(self.long_term_memories) >= self.max_memories:
            self.logger.warning(f"Memory limit reached ({self.max_memories}), cannot add more memories")
            return False

        start_time = time.time()
        try:
            self.logger.debug(f"Adding custom memory (type: {memory_type}, length: {len(content)})")

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
                else:
                    processed_content = str(analysis_result) if analysis_result else content

            # 检查内容是否重复
            if processed_content in self.long_term_memories:
                self.logger.info("Memory content already exists, skipping addition")
                return False

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

            # 更新检索器
            if self.retriever:
                self.retriever.add_documents([processed_content])

            # 更新统计信息
            self.stats['total_memories'] += 1
            self.stats['memory_types'][memory_type] += 1

            # 检查是否需要自动保存
            if self.enable_persistence and self.stats['total_memories'] % 10 == 0:
                self.save_state()

            total_time = time.time() - start_time
            self.logger.info(f"✅ Custom memory added successfully (type: {memory_type}, "
                           f"total: {self.stats['total_memories']}, took: {total_time:.3f}s)")

            return True

        except Exception as e:
            total_time = time.time() - start_time
            self.logger.error(f"Failed to add custom memory: {e} (took {total_time:.3f}s)")
            return False

    # ============ 持久化功能 ============

    def save_state(self) -> bool:
        """
        保存记忆状态到磁盘

        Returns:
            bool: 保存是否成功
        """
        if not self.enable_persistence:
            return True

        try:
            # 保存基础Memory
            basic_memory_path = self.storage_dir / "basic_memory.pkl"
            with open(basic_memory_path, 'wb') as f:
                pickle.dump(self.basic_memory, f)

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
            # 加载基础Memory
            basic_memory_path = self.storage_dir / "basic_memory.pkl"
            if basic_memory_path.exists():
                with open(basic_memory_path, 'rb') as f:
                    self.basic_memory = pickle.load(f)

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
                    if retriever_path.exists() and self.retriever:
                        try:
                            with open(retriever_path, 'rb') as f:
                                loaded_retriever = pickle.load(f)
                                # 重新添加文档到检索器
                                if self.long_term_memories:
                                    loaded_retriever.add_documents(self.long_term_memories)
                                self.retriever = loaded_retriever
                        except Exception as e:
                            print(f"⚠️  Failed to load retriever state: {e}")

            print(f"✅ Memory state loaded from {self.storage_dir}")
            return True

        except Exception as e:
            print(f"⚠️  Failed to load memory state: {e}")
            return False

    # ============ 统计和监控 ============

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

        # 添加检索模式分析（最常见的查询模式）
        if self.stats['retrieval_patterns']:
            top_patterns = sorted(
                self.stats['retrieval_patterns'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]  # Top 5 patterns
            current_stats['top_retrieval_patterns'] = top_patterns

        self.logger.debug(f"Stats requested: {len(self.long_term_memories)} memories, "
                         f"{self.stats['retrieval_count']} retrievals")

        return current_stats

    def log_performance_report(self):
        """生成并记录性能报告"""
        stats = self.get_stats()

        self.logger.info("=" * 60)
        self.logger.info("MEMORY SYSTEM PERFORMANCE REPORT")
        self.logger.info("=" * 60)

        self.logger.info(f"System Status:")
        self.logger.info(f"  - A-MEM Enabled: {stats['amem_enabled']}")
        self.logger.info(f"  - Retriever Available: {stats['amem_available']}")
        self.logger.info(f"  - Persistence Enabled: {stats['persistence_enabled']}")
        self.logger.info(f"  - Storage Directory: {stats['storage_dir']}")

        self.logger.info(f"\nMemory Statistics:")
        self.logger.info(f"  - Total Memories: {stats['total_memories']}")
        self.logger.info(f"  - Current Count: {stats['current_memory_count']}")
        self.logger.info(f"  - Memory Utilization: {stats['memory_utilization']:.1%}")
        self.logger.info(f"  - Memory Types: {stats['memory_distribution']}")

        if 'performance_summary' in stats:
            perf = stats['performance_summary']
            self.logger.info(f"\nRetrieval Performance:")
            self.logger.info(f"  - Total Retrievals: {perf['total_retrievals']}")
            self.logger.info(f"  - Average Time: {perf['avg_retrieval_time']:.3f}s")
            self.logger.info(f"  - Success Rate: {perf['success_rate']:.1%}")
            self.logger.info(f"  - Error Rate: {perf['error_rate']:.1%}")

        if 'top_retrieval_patterns' in stats:
            self.logger.info(f"\nTop Retrieval Patterns:")
            for i, (pattern, count) in enumerate(stats['top_retrieval_patterns'][:3], 1):
                self.logger.info(f"  {i}. Pattern {pattern}: {count} times")

        self.logger.info("=" * 60)

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
                self.retriever = type(self.retriever)(
                    model_name=self.retriever_config.get('model_name', 'all-MiniLM-L6-v2'),
                    use_api_embedding=self.retriever_config.get('use_api_embedding', False),  # 默认禁用API嵌入
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
