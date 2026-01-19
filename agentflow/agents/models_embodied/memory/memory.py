from typing import Dict, Any, List, Union, Optional, Tuple
import os
from pathlib import Path
import re
import shutil
import threading


class Memory:
    _global_instance: Optional['Memory'] = None  # 单例实例（进程内全局共享）

    def __init__(self, max_memory_length: int = 3, is_enable: bool = True):
        self._memory_root = Path("tmp/memory_store")
        self._memory_root.mkdir(parents=True, exist_ok=True)
        
        self.is_enable = is_enable

        # 防止直接实例化，必须通过 get_instance()
        if Memory._global_instance is not None:
            raise RuntimeError("Use Memory.get_instance() to get the global Memory instance.")
        self.query: Optional[str] = None
        self.files: List[Dict[str, str]] = []
        self.actions: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.max_memory_length = max_memory_length
        self._init_file_types()
        print("✅ Initialized global shared Memory (single instance for the entire process)")

    @classmethod
    def get_instance(
        cls,
        max_memory_length: int = 3,
        is_enable: bool = True
    ) -> 'Memory':
        if cls._global_instance is None:
            cls._global_instance = cls(
                max_memory_length=max_memory_length,
                is_enable=is_enable
            )
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

    def reset(self):
        self.query = None
        self.files.clear()
        self.actions.clear()

        if self._memory_root.exists():
            for p in self._memory_root.glob("*"):
                if p.is_dir():
                    for f in p.glob("*"):
                        f.unlink()
                    p.rmdir()
                else:
                    p.unlink()

        print("🧹 Memory reset + internal image store cleared")


    def add_embodied_action(
        self, 
        belief: str,
        intention: str,
        state: str, 
        verification: str,
        commands: List[Tuple[str, str]] = None,
        interaction_memory: Optional[str] = None,
        execution_time: Optional[float] = None
    ) -> None:

        step_count = len(self.actions) + 1
        step_name = f"Action Step {step_count}"

        action = {
            'interaction_memory': interaction_memory if interaction_memory else None,
            'belief': belief,
            'intention': intention,
            'commands': commands if commands else None,
            'state': state,
            'execution_time': execution_time,
            'verification': None   # wait for verfier to update
        }

        with self._lock:
            self.actions[step_name] = action

        return step_name

    def get_total_steps(self) -> int:
        if self.is_enable == False:
            return None
        return len(self.actions)

    def get_query(self) -> Optional[str]:
        if self.is_enable == False:
            return None
        return self.query

    def get_files(self) -> List[Dict[str, str]]:
        if self.is_enable == False:
            return None

        return self.files
    
    def get_actions(self) -> Dict[str, Any]:
        if self.is_enable == False:
            return None
        total_steps = len(self.actions)
        
        all_steps = sorted(self.actions.items(), key=lambda x: int(x[0].split()[-1]))
        recent_steps_list = all_steps[-self.max_memory_length:]
        recent_actions = dict(recent_steps_list)

        return {
            "total_steps": total_steps,
            "memory_window_size": self.max_memory_length,
            "actions": recent_actions
        }

    
    def parse_vln_output(self, output_text: str) -> Dict[str, Any]:
        """
        Parse the VLN output text to extract Belief, Intention, and State sections.
        Returns a dictionary with parsed data for each section.
        """
        log_path = Path("tmp/llm_raw_text.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"Parsing VLN output: {output_text}\n" + "-"*80 + "\n")

        def extract_section(section_name: str, pattern: str):
            match = re.search(pattern, output_text, re.IGNORECASE | re.DOTALL)
            if match:
                section_text = match.group(1).strip()
                print(f"Extracted {section_name} Text: {section_text}")
                return section_text
            else:
                print(f"Label '{section_name}:' not found in output.")
                return None

        parsed_data = {}

        # Parse Belief
        belief_text = extract_section(
            "Belief",
            r"(?:\*\*Belief\*\*|Belief)\s*:\s*(.*?)(\n(?:Intention|State|Action|Description|$))"
        )
        if belief_text:
            belief_dict = {}
            lines = belief_text.splitlines()
            for line in lines:
                if line.strip().startswith('-'):
                    key_value = line.strip()[1:].strip().split(':', 1)
                    if len(key_value) == 2:
                        key = key_value[0].strip()
                        value = key_value[1].strip()
                        belief_dict[key] = value
            parsed_data['belief'] = belief_dict

        # Parse Intention
        intention_text = extract_section(
            "Intention",
            r"(?:\*\*Intention\*\*|Intention)\s*:\s*(.*?)(\n(?:State|Action|Belief|Description|$))"
        )
        if intention_text:
            reasoning_match = re.search(r"\[Next step reasoning\]:\s*(.*?)(?=\[Area of interest\]|$)", intention_text, re.DOTALL)
            area_match = re.search(r"\[Area of interest\]:\s*(.*)", intention_text, re.DOTALL)
            intention = {
                "reasoning": reasoning_match.group(1).strip() if reasoning_match else None,
                "area": area_match.group(1).strip() if area_match else None
            }
            parsed_data['intention'] = intention

        # Parse State
        state_text = extract_section(
            "State",
            r"(?:\*\*State\*\*|State)\s*:\s*(.*?)(\n(?:Action|Intention|Belief|Description|$))"
        )
        if state_text:
            state_match = re.search(r"<(.*?)>", state_text, re.DOTALL)
            if state_match:
                parsed_data['state'] = state_match.group(1).strip()

        return parsed_data
