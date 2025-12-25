"""
MemoryNote: A-MEM 记忆单元实现

核心数据结构定义，支持记忆的内容存储、元数据管理和序列化。
"""

from typing import Dict, List, Optional, Any
import json
import uuid
from datetime import datetime


class MemoryNote:
    """
    记忆单元类，实现A-MEM论文中的基本记忆结构

    支持内容存储、元数据管理、序列化/反序列化等功能。
    """

    def __init__(self,
                 content: str,
                 id: Optional[str] = None,
                 keywords: Optional[List[str]] = None,
                 links: Optional[List[int]] = None,
                 importance_score: Optional[float] = None,
                 retrieval_count: Optional[int] = None,
                 timestamp: Optional[str] = None,
                 last_accessed: Optional[str] = None,
                 context: Optional[str] = None,
                 evolution_history: Optional[List] = None,
                 category: Optional[str] = None,
                 tags: Optional[List[str]] = None):
        """
        初始化记忆单元

        Args:
            content: 记忆内容文本
            id: 记忆唯一标识符，自动生成如果未提供
            keywords: 关键词列表
            links: 连接到其他记忆的索引列表
            importance_score: 重要性评分 (0.0-1.0)
            retrieval_count: 被检索次数
            timestamp: 创建时间戳
            last_accessed: 最后访问时间戳
            context: 上下文信息
            evolution_history: 演化历史记录
            category: 分类标签
            tags: 标签列表
        """
        self.content = content

        # 设置默认值
        self.id = id or str(uuid.uuid4())
        self.keywords = keywords or []
        self.links = links or []
        self.importance_score = importance_score if importance_score is not None else 1.0
        self.retrieval_count = retrieval_count or 0

        # 时间戳处理
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        self.timestamp = timestamp or current_time
        self.last_accessed = last_accessed or current_time

        # 上下文处理
        self.context = context or "General"
        if isinstance(self.context, list):
            self.context = " ".join(self.context)

        self.evolution_history = evolution_history or []
        self.category = category or "Uncategorized"
        self.tags = tags or []

    def update_access_time(self) -> None:
        """更新最后访问时间"""
        self.last_accessed = datetime.now().strftime("%Y%m%d%H%M%S")

    def increment_retrieval_count(self) -> None:
        """增加检索计数"""
        self.retrieval_count += 1
        self.update_access_time()

    def add_link(self, memory_index: int) -> None:
        """添加与其他记忆的连接"""
        if memory_index not in self.links:
            self.links.append(memory_index)

    def remove_link(self, memory_index: int) -> None:
        """移除与其他记忆的连接"""
        if memory_index in self.links:
            self.links.remove(memory_index)

    def add_tag(self, tag: str) -> None:
        """添加标签"""
        if tag not in self.tags:
            self.tags.append(tag)

    def remove_tag(self, tag: str) -> None:
        """移除标签"""
        if tag in self.tags:
            self.tags.remove(tag)

    def update_keywords(self, keywords: List[str]) -> None:
        """更新关键词列表"""
        self.keywords = keywords

    def update_context(self, context: str) -> None:
        """更新上下文信息"""
        self.context = context
        self.update_access_time()

    def add_evolution_record(self, action: str, details: Dict[str, Any]) -> None:
        """添加演化历史记录"""
        record = {
            "timestamp": datetime.now().strftime("%Y%m%d%H%M%S"),
            "action": action,
            "details": details
        }
        self.evolution_history.append(record)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，用于序列化"""
        return {
            "id": self.id,
            "content": self.content,
            "keywords": self.keywords,
            "links": self.links,
            "importance_score": self.importance_score,
            "retrieval_count": self.retrieval_count,
            "timestamp": self.timestamp,
            "last_accessed": self.last_accessed,
            "context": self.context,
            "evolution_history": self.evolution_history,
            "category": self.category,
            "tags": self.tags
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryNote':
        """从字典格式反序列化"""
        return cls(
            content=data["content"],
            id=data.get("id"),
            keywords=data.get("keywords", []),
            links=data.get("links", []),
            importance_score=data.get("importance_score", 1.0),
            retrieval_count=data.get("retrieval_count", 0),
            timestamp=data.get("timestamp"),
            last_accessed=data.get("last_accessed"),
            context=data.get("context", "General"),
            evolution_history=data.get("evolution_history", []),
            category=data.get("category", "Uncategorized"),
            tags=data.get("tags", [])
        )

    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'MemoryNote':
        """从JSON字符串反序列化"""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def __str__(self) -> str:
        """字符串表示"""
        return f"MemoryNote(id={self.id}, content='{self.content[:50]}...', tags={self.tags})"

    def __repr__(self) -> str:
        """详细字符串表示"""
        return (f"MemoryNote(id='{self.id}', content='{self.content[:30]}...', "
                f"keywords={self.keywords}, tags={self.tags}, "
                f"importance={self.importance_score}, retrievals={self.retrieval_count})")

    def __eq__(self, other) -> bool:
        """相等比较"""
        if not isinstance(other, MemoryNote):
            return False
        return self.id == other.id

    def __hash__(self) -> int:
        """哈希值"""
        return hash(self.id)
