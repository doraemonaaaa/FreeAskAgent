"""
Long Term Memory for Embodied Agent

长期记忆模块，负责跨会话的记忆存储、检索和管理。
提供A-MEM增强功能，支持内容分析和混合检索。
"""

import hashlib
import json
import pickle
import logging
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from .interfaces import LongMemoryInterface, BaseMemoryComponent
from .hybrid_retriever import HybridRetriever
from .content_analyzer import ContentAnalyzer


class LongMemory(BaseMemoryComponent, LongMemoryInterface):
    """
    长期记忆类 - 简洁实现

    提供跨会话的记忆存储、检索和管理功能。
    集成内容分析器和混合检索器，支持A-MEM增强功能。
    """

    def __init__(self,
                 use_amem: bool = True,
                 retriever_config: Optional[Dict[str, Any]] = None,
                 storage_dir: str = "./memory_store",
                 enable_persistence: bool = True,
                 max_memories: int = 1000):
        super().__init__(enable_persistence, Path(storage_dir))

        # 核心配置
        self.use_amem = use_amem
        self.retriever_config = retriever_config or {}
        self.max_memories = max_memories

        # A-MEM组件
        self.retriever: Optional[HybridRetriever] = None
        self.content_analyzer: Optional[ContentAnalyzer] = None
        self.long_term_memories: List[str] = []
        self.memory_metadata: List[Dict[str, Any]] = []

        # 统计和配置
        self.stats = self._init_stats()
        self.gate_config = self.retriever_config.get("gate_config", {})
        self._dedup_hashes = set()

        # 初始化组件
        if self.use_amem:
            self._init_amem_components()

        # 加载持久化状态
        if self.enable_persistence:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            self.load_state()

    def _init_stats(self) -> Dict[str, Any]:
        """初始化统计信息"""
        return {
            'total_memories': 0,
            'retrieval_count': 0,
            'avg_retrieval_time': 0.0,
            'amem_enabled': self.use_amem,
            'performance_metrics': {
                'retrieval_times': [],
                'error_count': 0,
                'success_rate': 1.0
            },
            'memory_types': defaultdict(int),
            'retrieval_patterns': defaultdict(int)
        }

    def _init_amem_components(self):
        """初始化A-MEM组件"""
        try:
            self.retriever = HybridRetriever(
                model_name=self.retriever_config.get('model_name', 'all-MiniLM-L6-v2'),
                alpha=self.retriever_config.get('alpha', 0.5),
                local_model_path=self.retriever_config.get('local_model_path'),
                disable_semantic_search=self.retriever_config.get('disable_semantic_search', False)
            )
            self.content_analyzer = ContentAnalyzer()
        except Exception:
            self.use_amem = False

    def add_memory(self, content: str, memory_type: str = "custom",
                   metadata: Optional[Dict[str, Any]] = None) -> bool:
        """添加记忆内容"""
        if len(self.long_term_memories) >= self.max_memories:
            return False

        try:
            clean_content = self._sanitize_content(content)

            # 门控检查
            if not self._should_store(clean_content, memory_type, None):
                return False

            # 内容分析和处理
            processed_content = clean_content
            analysis_result = None

            if self.content_analyzer:
                analysis_result = self.content_analyzer.analyze_content(clean_content)
                if self._is_valid_analysis(analysis_result):
                    keywords = analysis_result.get('keywords', [])
                    context = analysis_result.get('context', '')
                    tags = analysis_result.get('tags', [])
                    processed_content = f"{clean_content}\nKEYWORDS: {' '.join(keywords)}\nCONTEXT: {context}\nTAGS: {' '.join(tags)}"

            # 添加到记忆库
            self.long_term_memories.append(processed_content)
            self.memory_metadata.append({
                'content': processed_content,
                'original_content': clean_content,
                'analysis': analysis_result,
                'type': memory_type,
                'timestamp': time.time(),
                'added_at': datetime.now().isoformat(),
                **(metadata or {})
            })

            # 更新检索器
            if self.retriever:
                self.retriever.add_documents([processed_content])

            # 更新统计
            self.stats['total_memories'] += 1
            self.stats['memory_types'][memory_type] += 1

            return True

        except Exception:
            return False

    def add_conversation_summary(self, conversation_window: List[Dict[str, Any]],
                                window_id: int, session_id: str) -> bool:
        """添加对话窗口总结"""
        if not conversation_window:
            return False

        # 格式化对话内容
        conversation_lines = []
        for msg in conversation_window:
            role = msg.get('role', 'unknown')
            content = self._sanitize_content(msg.get('content', ''))
            if content.strip():
                timestamp = datetime.fromtimestamp(msg.get('timestamp', time.time())).strftime('%H:%M:%S')
                conversation_lines.append(f"[{timestamp}] {role.upper()}: {content}")

        conversation_text = "\n".join(conversation_lines)

        # 使用内容分析器生成总结
        summary_content = conversation_text
        if self.content_analyzer:
            try:
                analysis = self.content_analyzer.analyze_content(conversation_text)
                if isinstance(analysis, dict):
                    summary_content = self._generate_conversation_summary(conversation_text, analysis)
            except Exception:
                pass

        # 构建元数据
        metadata = {
            'type': 'conversation_summary',
            'window_id': window_id,
            'session_id': session_id,
            'turn_range': f"{conversation_window[0]['turn_id']} - {conversation_window[-1]['turn_id']}",
            'message_count': len(conversation_window),
            'created_at': datetime.now().isoformat()
        }

        return self.add_memory(content=summary_content, memory_type='conversation_summary', metadata=metadata)

    def _generate_conversation_summary(self, conversation_text: str, analysis: Dict[str, Any]) -> str:
        """生成对话总结"""
        keywords = analysis.get('keywords', [])
        context = analysis.get('context', '')

        summary_parts = []
        if keywords:
            summary_parts.append(f"关键词: {', '.join(keywords[:5])}")
        if context:
            summary_parts.append(f"主要内容: {context}")
        summary_parts.append(f"对话记录: {conversation_text}")

        return " | ".join(summary_parts)

    def retrieve_memories(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """检索长期记忆"""
        if not self.use_amem or not self.retriever:
            return []

        try:
            indices = self.retriever.retrieve(query, k=k)
            results = []

            for idx in indices:
                if 0 <= idx < len(self.long_term_memories):
                    results.append({
                        'content': self.long_term_memories[idx],
                        'metadata': self.memory_metadata[idx],
                        'index': idx
                    })

            # 更新统计
            self.stats['retrieval_count'] += 1
            return results

        except Exception:
            return []

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        current_stats = self.stats.copy()
        current_stats.update({
            'current_memory_count': len(self.long_term_memories),
            'amem_available': self.use_amem and self.retriever is not None,
            'persistence_enabled': self.enable_persistence,
            'storage_dir': str(self.storage_dir)
        })
        return current_stats

    def clear_memories(self, memory_type: Optional[str] = None) -> int:
        """清空记忆"""
        if not self.use_amem:
            return 0

        if memory_type is None:
            cleared_count = len(self.long_term_memories)
            self.long_term_memories.clear()
            self.memory_metadata.clear()
            self.stats['total_memories'] = 0
        else:
            indices_to_remove = [i for i, metadata in enumerate(self.memory_metadata)
                               if metadata.get('type') == memory_type]

            for i in reversed(indices_to_remove):
                del self.long_term_memories[i]
                del self.memory_metadata[i]

            cleared_count = len(indices_to_remove)
            self.stats['total_memories'] -= cleared_count

        # 重新初始化检索器
        if self.retriever:
            self.retriever = HybridRetriever(
                model_name=self.retriever_config.get('model_name', 'all-MiniLM-L6-v2'),
                alpha=self.retriever_config.get('alpha', 0.5),
                local_model_path=self.retriever_config.get('local_model_path'),
                disable_semantic_search=self.retriever_config.get('disable_semantic_search', False)
            )

        if self.enable_persistence:
            self.save_state()

        return cleared_count

    def _sanitize_content(self, text) -> str:
        """清理内容文本"""
        if text is None:
            return ""
        if isinstance(text, str):
            t = text
        elif isinstance(text, dict):
            t = text.get('text') or text.get('content') or str(text)
        else:
            t = str(text)

        # 清理噪声
        import re
        noise_blocks = [
            r"(?is)Strategy Analysis:.*",
            r"(?is)Theory of Mind-Reasoning:.*",
            r"(?is)<<Robot Belief>>:.*",
        ]
        for pat in noise_blocks:
            t = re.sub(pat, "", t)

        return t.strip()

    def _is_valid_analysis(self, analysis: Any) -> bool:
        """检查分析结果是否有效"""
        return isinstance(analysis, dict) and not self._is_fallback_analysis(analysis)

    def _is_fallback_analysis(self, analysis_result: Any) -> bool:
        """检查是否为降级分析结果"""
        if not isinstance(analysis_result, dict):
            return True
        keywords = analysis_result.get("keywords", []) or []
        context = analysis_result.get("context", "") or ""
        tags = analysis_result.get("tags", []) or []
        return (
            (keywords == ["general"] or keywords == []) and
            ("General" in context or context.strip() == "") and
            (tags == ["general"] or tags == [])
        )

    def _should_store(self, content: str, memory_type: str, analysis_result: Any) -> bool:
        """判断是否应该存储记忆"""
        cfg = self.gate_config

        min_chars = int(cfg.get("min_chars", 30))
        deny_types = set(cfg.get("deny_types", []))
        allow_types = cfg.get("allow_types")
        skip_general = bool(cfg.get("skip_general", True))
        enable_dedup = bool(cfg.get("enable_dedup", True))

        if memory_type in deny_types:
            return False
        if allow_types is not None and memory_type not in set(allow_types):
            return False

        c = (content or "").strip()
        if len(c) < min_chars:
            return False

        if skip_general and self._is_fallback_analysis(analysis_result):
            return False

        if enable_dedup:
            h = hashlib.md5(c.encode("utf-8")).hexdigest()
            if h in self._dedup_hashes:
                return False
            self._dedup_hashes.add(h)

        return True