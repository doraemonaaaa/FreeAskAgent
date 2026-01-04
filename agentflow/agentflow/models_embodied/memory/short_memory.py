"""
Short Term Memory for Embodied Agent

短期记忆模块，负责当前会话的查询、文件和动作管理。
提供基础的记忆容器功能，支持文件类型自动识别。
"""

from typing import Dict, Any, List, Union, Optional
import os


class ShortMemory:
    """
    短期记忆类

    负责管理当前会话的查询、文件列表和动作记录。
    提供基本的CRUD操作接口，支持多种文件类型的自动识别。
    """

    def __init__(self, max_files: int = 100, max_actions: int = 1000):
        """
        初始化短期记忆

        Args:
            max_files: 最大文件数量限制
            max_actions: 最大动作记录数量限制
        """
        self.max_files = max_files
        self.max_actions = max_actions

        self.query: Optional[str] = None
        self.files: List[Dict[str, str]] = []
        self.actions: Dict[str, Dict[str, Any]] = {}
        self._init_file_types()

    def set_query(self, query: str) -> None:
        """
        设置查询内容

        Args:
            query: 查询字符串

        Raises:
            TypeError: 如果query不是字符串类型
        """
        if not isinstance(query, str):
            raise TypeError("Query must be a string")
        self.query = query

    def _init_file_types(self):
        """初始化文件类型映射"""
        self.file_types = {
            'image': ['.jpg', '.jpeg', '.png', '.gif', '.bmp'],
            'text': ['.txt', '.md'],
            'document': ['.pdf', '.doc', '.docx'],
            'code': ['.py', '.js', '.java', '.cpp', '.h'],
            'data': ['.json', '.csv', '.xml'],
            'spreadsheet': ['.xlsx', '.xls'],
            'presentation': ['.ppt', '.pptx'],
        }
        self.file_type_descriptions = {
            'image': "An image file ({ext} format) provided as context for the query",
            'text': "A text file ({ext} format) containing additional information related to the query",
            'document': "A document ({ext} format) with content relevant to the query",
            'code': "A source code file ({ext} format) potentially related to the query",
            'data': "A data file ({ext} format) containing structured data pertinent to the query",
            'spreadsheet': "A spreadsheet file ({ext} format) with tabular data relevant to the query",
            'presentation': "A presentation file ({ext} format) with slides related to the query",
        }

    def _get_default_description(self, file_name: str) -> str:
        """
        根据文件扩展名生成默认描述

        Args:
            file_name: 文件名

        Returns:
            文件描述字符串
        """
        _, ext = os.path.splitext(file_name)
        ext = ext.lower()

        for file_type, extensions in self.file_types.items():
            if ext in extensions:
                return self.file_type_descriptions[file_type].format(ext=ext[1:])

        return f"A file with {ext[1:]} extension, provided as context for the query"

    def add_file(self, file_name: Union[str, List[str]], description: Union[str, List[str], None] = None) -> None:
        """
        添加文件到记忆中

        Args:
            file_name: 文件名或文件名列表
            description: 文件描述或描述列表，为None时自动生成

        Raises:
            ValueError: 如果文件名和描述数量不匹配
        """
        if isinstance(file_name, str):
            file_name = [file_name]

        if description is None:
            description = [self._get_default_description(fname) for fname in file_name]
        elif isinstance(description, str):
            description = [description]

        if len(file_name) != len(description):
            raise ValueError("The number of files and descriptions must match.")

        for fname, desc in zip(file_name, description):
            if len(self.files) >= self.max_files:
                raise ValueError(f"Maximum number of files ({self.max_files}) exceeded")
            self.files.append({
                'file_name': fname,
                'description': desc
            })

    def add_action(self, step_count: int, tool_name: str, sub_goal: str, command: str, result: Any) -> None:
        """
        添加动作记录到记忆中

        Args:
            step_count: 步骤计数
            tool_name: 工具名称
            sub_goal: 子目标
            command: 执行命令
            result: 执行结果

        Raises:
            ValueError: 当动作数量超过限制时
        """
        if len(self.actions) >= self.max_actions:
            raise ValueError(f"Maximum number of actions ({self.max_actions}) exceeded")

        action = {
            'tool_name': tool_name,
            'sub_goal': sub_goal,
            'command': command,
            'result': result,
        }
        step_name = f"Action Step {step_count}"
        self.actions[step_name] = action

    def get_query(self) -> Optional[str]:
        """
        获取当前查询

        Returns:
            查询字符串或None
        """
        return self.query

    def get_files(self) -> List[Dict[str, str]]:
        """
        获取文件列表

        Returns:
            文件信息字典列表
        """
        return self.files

    def get_actions(self) -> Dict[str, Dict[str, Any]]:
        """
        获取动作记录

        Returns:
            动作记录字典
        """
        return self.actions

    def clear(self) -> None:
        """清空所有记忆内容"""
        self.query = None
        self.files = []
        self.actions = {}

