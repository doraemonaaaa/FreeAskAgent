"""
Memory System Interfaces

定义记忆系统的基础接口和抽象类，提供统一的API设计。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Protocol
from pathlib import Path


class MemoryInterface(Protocol):
    """记忆接口协议"""

    @abstractmethod
    def add_memory(self, content: str, memory_type: str = "custom",
                   metadata: Optional[Dict[str, Any]] = None) -> bool:
        """添加记忆内容"""
        ...

    @abstractmethod
    def retrieve_memories(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """检索相关记忆"""
        ...

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        ...

    @abstractmethod
    def save_state(self) -> bool:
        """保存状态"""
        ...

    @abstractmethod
    def load_state(self) -> bool:
        """加载状态"""
        ...


class ContentAnalyzerInterface(Protocol):
    """内容分析器接口"""

    @abstractmethod
    def analyze_content(self, content: str) -> Dict[str, Any]:
        """分析内容，提取元数据"""
        ...


class RetrieverInterface(Protocol):
    """检索器接口"""

    @abstractmethod
    def add_documents(self, documents: List[str]) -> bool:
        """添加文档到索引"""
        ...

    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> List[int]:
        """检索相关文档"""
        ...


class ShortMemoryInterface(Protocol):
    """短期记忆接口"""

    @abstractmethod
    def set_query(self, query: str) -> None:
        """设置查询内容"""
        ...

    @abstractmethod
    def add_file(self, file_name: str, description: Optional[str] = None) -> None:
        """添加文件"""
        ...

    @abstractmethod
    def add_action(self, step_count: int, tool_name: str, sub_goal: str,
                   command: str, result: Any) -> None:
        """添加动作记录"""
        ...

    @abstractmethod
    def append_message(self, role: str, content: str, turn_id: Optional[str] = None) -> None:
        """添加对话消息"""
        ...

    @abstractmethod
    def get_recent(self, n: int = 5) -> List[Dict[str, Any]]:
        """获取最近的对话记录"""
        ...


class LongMemoryInterface(MemoryInterface):
    """长期记忆接口"""

    @abstractmethod
    def add_conversation_summary(self, conversation_window: List[Dict[str, Any]],
                                window_id: int, session_id: str) -> bool:
        """添加对话窗口总结"""
        ...

    @abstractmethod
    def clear_memories(self, memory_type: Optional[str] = None) -> int:
        """清空记忆"""
        ...


class MemoryManagerInterface(Protocol):
    """记忆管理器接口"""

    @abstractmethod
    def add_message(self, role: str, content: str, turn_id: Optional[str] = None) -> bool:
        """添加消息并返回是否需要检索"""
        ...

    @abstractmethod
    def retrieve_relevant_memories(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """检索相关记忆"""
        ...

    @abstractmethod
    def get_short_memory(self) -> ShortMemoryInterface:
        """获取短期记忆实例"""
        ...

    @abstractmethod
    def get_long_memory(self) -> LongMemoryInterface:
        """获取长期记忆实例"""
        ...


class BaseMemoryComponent(ABC):
    """记忆组件基础抽象类"""

    def __init__(self, enable_persistence: bool = True, storage_dir: Optional[Path] = None):
        self.enable_persistence = enable_persistence
        self.storage_dir = storage_dir or Path("./memory_store")

    @abstractmethod
    def _setup_logging(self) -> None:
        """设置日志"""
        ...

    def save_state(self) -> bool:
        """默认保存状态实现"""
        return not self.enable_persistence

    def load_state(self) -> bool:
        """默认加载状态实现"""
        return not self.enable_persistence
