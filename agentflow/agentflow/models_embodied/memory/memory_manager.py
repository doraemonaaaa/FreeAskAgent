"""
Memory Manager for Embodied Agent

记忆管理器，协调短期记忆和长期记忆之间的交互。
"""

from typing import Dict, Any, List, Optional
import logging
import json
import re
from .short_memory import ShortMemory
from .long_memory import LongMemory


class MemoryManager:
    """
    记忆管理器类

    协调短期记忆和长期记忆，提供统一的记忆管理接口。
    """

    def __init__(self,
                 short_memory_config: Optional[Dict[str, Any]] = None,
                 long_memory_config: Optional[Dict[str, Any]] = None,
                 conversation_window_size: int = 3):
        """
        初始化记忆管理器

        Args:
            short_memory_config: 短期记忆配置
            long_memory_config: 长期记忆配置
            conversation_window_size: 对话窗口大小
        """
        self.conversation_window_size = conversation_window_size

        # 初始化组件
        short_config = short_memory_config or {}
        self.short_memory = ShortMemory(
            max_files=short_config.get('max_files', 100),
            max_actions=short_config.get('max_actions', 1000),
            conversation_window_size=conversation_window_size
        )

        long_config = long_memory_config or {}
        retriever_cfg = long_config.get("retriever_config") or {}
        retriever_cfg["gate_config"] = long_config.get("gate_config", {})
        self.long_memory = LongMemory(
            use_amem=long_config.get('use_amem', True),
            retriever_config=retriever_cfg,
            storage_dir=long_config.get('storage_dir', './memory_store'),
            enable_persistence=long_config.get('enable_persistence', True),
            max_memories=long_config.get('max_memories', 1000)
        )

        # 检索门控配置
        gate = long_config.get("gate_config", {})
        self.retrieve_gate_patterns = gate.get("retrieve_gate_patterns") or [
            r"长期记忆", r"记忆", r"我叫", r"我在", r"项目代号", r"偏好", r"规则"
        ]
        self.retrieve_gate_min_len = int(gate.get("retrieve_gate_min_len", 8))

        # 设置日志
        self.logger = logging.getLogger('MemoryManager')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _coerce_to_text(self, x) -> str:
        """把 content 统一转成 string，避免 join/len 崩溃。"""
        if x is None:
            return ""
        if isinstance(x, str):
            return x
        if isinstance(x, bytes):
            try:
                return x.decode("utf-8", errors="ignore")
            except Exception:
                return str(x)

        # 常见：{"type":"text","text":"..."} 或 {"text":"..."} 等
        if isinstance(x, dict):
            for k in ("text", "content", "message", "value"):
                v = x.get(k)
                if isinstance(v, str):
                    return v
                elif isinstance(v, (dict, list)):
                    # Recursively handle nested structures
                    return self._coerce_to_text(v)
            return json.dumps(x, ensure_ascii=False)

        # 常见：多段 content parts
        if isinstance(x, list):
            parts = [self._coerce_to_text(i) for i in x]
            parts = [p for p in parts if p.strip()]
            return "\n".join(parts)

        return str(x)

    def _should_retrieve(self, query: str) -> bool:
        q = (query or "").strip()
        if len(q) < self.retrieve_gate_min_len:
            return False
        return any(re.search(p, q, flags=re.IGNORECASE) for p in self.retrieve_gate_patterns)

    def _should_summarize_window(self, window_to_summarize) -> bool:
        text = "\n".join(self._coerce_to_text(m.get("content")) for m in window_to_summarize).strip()
        if not text:
            return False
        return True

    def add_message(self, role: str, content: str, turn_id: Optional[str] = None) -> bool:
        """
        添加对话消息

        Args:
            role: 消息角色 ('user' 或 'assistant')
            content: 消息内容
            turn_id: 对话轮次ID

        Returns:
            bool: 是否需要检索相关记忆（新窗口开始时）
        """
        # 添加到短期记忆
        self.short_memory.append_message(role, content, turn_id)

        # 检查是否需要总结窗口
        window_to_summarize, window_id = self.short_memory.get_window_for_summary()

        if window_to_summarize:
            # 先门控：这段窗口是否"值得"写入长期记忆
            if self._should_summarize_window(window_to_summarize):
                success = self.long_memory.add_conversation_summary(
                    window_to_summarize,
                    window_id,
                    self.short_memory.session_id
                )
                if success:
                    self.logger.info(f"Successfully summarized conversation window {window_id}")
                else:
                    self.logger.warning(f"Failed to summarize conversation window {window_id}")
            else:
                self.logger.info(f"Skip summarization for window {window_id} (gated)")

            # 新窗口开始返回 True（是否真的检索由 retrieve 门控决定）
            return True

        return False

    def retrieve_relevant_memories(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        检索相关记忆

        Args:
            query: 查询字符串
            top_k: 返回的记忆数量

        Returns:
            相关记忆列表
        """
        if not self.long_memory.use_amem:
            self.logger.debug("A-MEM not enabled, skipping memory retrieval")
            return []

        # 🔒 Retrieval gating
        if not self._should_retrieve(query):
            self.logger.info("Retrieval gated: query not memory-seeking")
            return []

        try:
            memories = self.long_memory.retrieve_memories(query, k=top_k)
            self.logger.info(f"Retrieved {len(memories)} relevant memories for query")
            return memories
        except Exception as e:
            self.logger.error(f"Failed to retrieve memories: {e}")
            return []

    def get_short_memory(self) -> ShortMemory:
        """
        获取短期记忆实例

        Returns:
            ShortMemory实例
        """
        return self.short_memory

    def get_long_memory(self) -> LongMemory:
        """
        获取长期记忆实例

        Returns:
            LongMemory实例
        """
        return self.long_memory

    def save_state(self) -> bool:
        """
        保存记忆状态

        Returns:
            bool: 保存是否成功
        """
        return self.long_memory.save_state()

    def load_state(self) -> bool:
        """
        加载记忆状态

        Returns:
            bool: 加载是否成功
        """
        return self.long_memory.load_state()

    def get_stats(self) -> Dict[str, Any]:
        """
        获取记忆统计信息

        Returns:
            包含短期和长期记忆统计的字典
        """
        return {
            'short_memory': {
                'total_messages': len(self.short_memory.conversation_history),
                'current_window_size': len(self.short_memory.current_window),
                'window_count': self.short_memory.window_count,
                'session_id': self.short_memory.session_id
            },
            'long_memory': self.long_memory.get_stats()
        }