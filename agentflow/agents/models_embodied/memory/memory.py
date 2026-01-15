from typing import Dict, Any, List, Union, Optional
import os
from pathlib import Path

class Memory:
    _global_instance: Optional['Memory'] = None  # 单例实例（进程内全局共享）

    def __init__(self, max_memory_length: int = 10):
        # 防止直接实例化，必须通过 get_instance()
        if Memory._global_instance is not None:
            raise RuntimeError("Use Memory.get_instance() to get the global Memory instance.")
        
        self.query: Optional[str] = None
        self.files: List[Dict[str, str]] = []
        self.actions: Dict[str, Dict[str, Any]] = {}
        self.max_memory_length = max_memory_length
        self._init_file_types()
        print("✅ Initialized global shared Memory (single instance for the entire process)")

    @classmethod
    def get_instance(cls) -> 'Memory':
        """获取进程内唯一的 Memory 实例（所有 solve 调用共享同一个 memory）"""
        if cls._global_instance is None:
            cls._global_instance = cls()
        return cls._global_instance

    def set_query(self, query: str) -> None:
        if not isinstance(query, str):
            raise TypeError("Query must be a string")
        self.query = query

    def _init_file_types(self):
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
        _, ext = os.path.splitext(file_name)
        ext = ext.lower()

        for file_type, extensions in self.file_types.items():
            if ext in extensions:
                return self.file_type_descriptions[file_type].format(ext=ext[1:])

        return f"A file with {ext[1:]} extension, provided as context for the query"
    
    def add_file(self, file_name: Union[str, List[str]], description: Union[str, List[str], None] = None) -> None:
        if isinstance(file_name, str):
            file_name = [file_name]
        
        if description is None:
            description = [self._get_default_description(fname) for fname in file_name]
        elif isinstance(description, str):
            description = [description]
        
        if len(file_name) != len(description):
            raise ValueError("The number of files and descriptions must match.")
        
        for fname, desc in zip(file_name, description):
            self.files.append({
                'file_name': fname,
                'description': desc
            })

    def add_embodied_action(self, prev_command: str) -> None:
        # 使用 len(self.actions) 计算步数（可靠、安全）
        step_count = len(self.actions) + 1
        
        action = {
            'previous_command': prev_command
        }
        step_name = f"Action Step {step_count}"
        self.actions[step_name] = action

    def get_total_steps(self) -> int:
        """返回当前进程内已累计的总步数（兼容旧代码，如果还有地方调用）"""
        return len(self.actions)

    def get_query(self) -> Optional[str]:
        return self.query

    def get_files(self) -> List[Dict[str, str]]:
        return self.files
    
    def get_actions(self) -> Dict[str, Any]:
        """
        返回最近的 n 个动作。
        如果总步数超过 max_memory_length，则仅返回最后一部分。
        """
        total_steps = len(self.actions)
        
        # 将字典转换为按步数排序的列表
        all_steps = sorted(self.actions.items(), key=lambda x: int(x[0].split()[-1]))
        
        # 取最后 n 个
        recent_steps_list = all_steps[-self.max_memory_length:]
        recent_actions = dict(recent_steps_list)

        return {
            "total_steps": total_steps,
            "memory_window_size": self.max_memory_length,
            "actions": recent_actions 
        }