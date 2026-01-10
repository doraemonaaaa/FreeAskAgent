"""
Short Term Memory for Embodied Agent

短期记忆模块，负责当前会话的查询、文件和动作管理。
提供基础的记忆容器功能，支持文件类型自动识别和对话记录管理。
"""

from typing import Dict, Any, List, Union, Optional, Tuple
import os
import time
import uuid
import logging
import json
from pathlib import Path

from .interfaces import ShortMemoryInterface, BaseMemoryComponent


class ShortMemory(BaseMemoryComponent, ShortMemoryInterface):
    """
    短期记忆类 - 简洁实现

    负责管理当前会话的查询、文件列表和动作记录。
    支持对话窗口管理和自动滚动。
    """

    def __init__(self, max_files: int = 100, max_actions: int = 1000,
                 conversation_window_size: int = 3, enable_persistence: bool = False):
        super().__init__(enable_persistence)

        # 核心配置
        self.max_files = max_files
        self.max_actions = max_actions
        self.conversation_window_size = conversation_window_size

        # 数据存储
        self.query: Optional[str] = None
        self.files: List[Dict[str, str]] = []
        self.actions: Dict[str, Dict[str, Any]] = []
        self.conversation_history: List[Dict[str, Any]] = []
        self.current_window: List[Dict[str, Any]] = []

        # 会话管理
        self.session_id: str = str(uuid.uuid4())
        self.window_count: int = 0
        self._pending_summary: Optional[Tuple[List[Dict[str, Any]], int]] = None

        self._init_file_types()
        self._setup_logging()

    def _setup_logging(self) -> None:
        """设置日志"""
        self.logger = logging.getLogger('ShortMemory')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _init_file_types(self):
        """初始化文件类型映射"""
        self.file_types = {
            'image': ['.jpg', '.jpeg', '.png', '.gif', '.bmp'],
            'text': ['.txt', '.md'],
            'document': ['.pdf', '.doc', '.docx'],
            'code': ['.py', '.js', '.java', '.cpp', '.h'],
            'data': ['.json', '.csv', '.xml'],
        }
        self.file_type_descriptions = {
            'image': "Image file ({ext} format)",
            'text': "Text file ({ext} format)",
            'document': "Document ({ext} format)",
            'code': "Source code file ({ext} format)",
            'data': "Data file ({ext} format)",
        }

    def _get_file_description(self, file_name: str) -> str:
        """根据文件扩展名生成描述"""
        _, ext = os.path.splitext(file_name)
        ext = ext.lower()

        for file_type, extensions in self.file_types.items():
            if ext in extensions:
                return self.file_type_descriptions[file_type].format(ext=ext[1:])

        return f"File with {ext[1:]} extension"

    def set_query(self, query: str) -> None:
        """设置查询内容"""
        if not isinstance(query, str):
            raise TypeError("Query must be a string")
        self.query = query

    def add_file(self, file_name: Union[str, List[str]], description: Union[str, List[str], None] = None) -> None:
        """添加文件"""
        if isinstance(file_name, str):
            file_name = [file_name]

        if description is None:
            description = [self._get_file_description(fname) for fname in file_name]
        elif isinstance(description, str):
            description = [description]

        if len(file_name) != len(description):
            raise ValueError("Number of files and descriptions must match")

        for fname, desc in zip(file_name, description):
            # Ensure description is a safe string
            if isinstance(desc, dict):
                try:
                    desc = json.dumps(desc, ensure_ascii=False)
                except Exception:
                    desc = str(desc)
            if len(self.files) >= self.max_files:
                raise ValueError(f"Maximum files ({self.max_files}) exceeded")
            self.files.append({'file_name': fname, 'description': desc})

    def add_action(self, step_count: int, tool_name: str, sub_goal: str, command: str, result: Any) -> None:
        """添加动作记录"""
        if len(self.actions) >= self.max_actions:
            raise ValueError(f"Maximum actions ({self.max_actions}) exceeded")

        self.actions[f"Action Step {step_count}"] = {
            'tool_name': tool_name, 'sub_goal': sub_goal,
            'command': command, 'result': result
        }

    def append_message(self, role: str, content: str, turn_id: Optional[str] = None) -> None:
        """添加对话消息"""
        if role not in ['user', 'assistant']:
            raise ValueError("Role must be 'user' or 'assistant'")

        message = {
            'role': role, 'content': content, 'timestamp': time.time(),
            'turn_id': turn_id or f"turn_{len(self.conversation_history)}",
            'session_id': self.session_id, 'window_id': self.window_count
        }

        self.conversation_history.append(message)
        self.current_window.append(message)

        # 窗口滚动逻辑
        if len(self.current_window) >= self.conversation_window_size:
            self._pending_summary = (self.current_window.copy(), self.window_count)
            self.current_window = []
            self.window_count += 1

    def get_recent(self, n: int = 5) -> List[Dict[str, Any]]:
        """获取最近的对话记录"""
        return self.conversation_history[-n:] if self.conversation_history else []

    def get_window_for_summary(self) -> Tuple[List[Dict[str, Any]], int]:
        """获取待总结的窗口"""
        if self._pending_summary:
            window, window_id = self._pending_summary
            self._pending_summary = None
            return window, window_id
        return [], -1

    def clear(self) -> None:
        """清空所有记忆"""
        self.query = None
        self.files = []
        self.actions = []
        self.conversation_history = []
        self.current_window = []
        self.window_count = 0
        self.session_id = str(uuid.uuid4())
        self._pending_summary = None


