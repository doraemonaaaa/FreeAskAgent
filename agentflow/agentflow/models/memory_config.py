"""
记忆系统配置管理

提供A-MEM相关参数的配置管理，支持环境变量和配置文件两种方式。
"""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path


class MemoryConfig:
    """
    记忆系统配置管理类

    支持从环境变量、配置文件加载配置，并提供默认值。
    """

    def __init__(self, config_file: Optional[str] = None):
        """
        初始化配置管理器

        Args:
            config_file: 配置文件路径，默认使用环境变量或标准路径
        """
        if config_file is None:
            # 默认配置文件路径
            self.config_file = os.getenv(
                'AMEM_CONFIG_FILE',
                '/root/autodl-tmp/FreeAskAgent/agentflow/agentflow/models/memory_config.json'
            )
        else:
            self.config_file = config_file

        # 默认配置
        self._defaults = {
            # A-MEM功能开关
            'use_amem': True,

            # 检索器配置
            'retriever': {
                'backend': 'litellm',
                'model': 'gpt-5',
                'api_base': 'https://yinli.one/v1',
                'alpha': 0.5,  # BM25与语义搜索权重
                'use_api_embedding': True,
                'max_tokens': 1000,
                'temperature': 0.0
            },

            # 记忆系统配置
            'memory': {
                'storage_dir': './memory_store',
                'enable_persistence': True,
                'max_memories': 1000,
                'auto_save_interval': 10,  # 每10个记忆自动保存
                'backup_enabled': True
            },

            # 内容分析配置
            'content_analysis': {
                'enabled': True,
                'model': 'gpt-5',
                'max_content_length': 1000,
                'analysis_prompt_template': None
            },

            # 性能配置
            'performance': {
                'cache_enabled': True,
                'max_cache_size': 100,
                'retrieval_timeout': 30.0,
                'batch_size': 10
            },

            # 调试配置
            'debug': {
                'verbose': False,
                'log_level': 'INFO',
                'enable_metrics': True,
                'profile_memory_usage': False
            }
        }

        # 当前配置（从默认值开始）
        self._config = self._defaults.copy()

        # 加载配置
        self._load_config()

    def _load_config(self):
        """从多种来源加载配置"""
        # 1. 从环境变量加载
        self._load_from_env()

        # 2. 从配置文件加载（如果存在）
        self._load_from_file()

        # 3. 验证配置
        self._validate_config()

    def _load_from_env(self):
        """从环境变量加载配置"""
        # A-MEM功能开关
        if 'USE_AMEM' in os.environ:
            self._config['use_amem'] = os.getenv('USE_AMEM', 'true').lower() == 'true'

        # 检索器配置
        retriever_env = {
            'retriever.backend': 'AMEM_RETRIEVER_BACKEND',
            'retriever.model': 'AMEM_RETRIEVER_MODEL',
            'retriever.api_base': 'AMEM_RETRIEVER_API_BASE',
            'retriever.alpha': 'AMEM_RETRIEVER_ALPHA',
            'retriever.use_api_embedding': 'AMEM_USE_API_EMBEDDING',
            'retriever.max_tokens': 'AMEM_MAX_TOKENS',
            'retriever.temperature': 'AMEM_TEMPERATURE'
        }

        for config_key, env_key in retriever_env.items():
            if env_key in os.environ:
                keys = config_key.split('.')
                if keys[0] == 'retriever':
                    if keys[1] in ['alpha', 'max_tokens', 'temperature']:
                        self._config['retriever'][keys[1]] = float(os.environ[env_key])
                    elif keys[1] == 'use_api_embedding':
                        self._config['retriever'][keys[1]] = os.environ[env_key].lower() == 'true'
                    else:
                        self._config['retriever'][keys[1]] = os.environ[env_key]

        # 记忆系统配置
        memory_env = {
            'memory.storage_dir': 'AMEM_STORAGE_DIR',
            'memory.enable_persistence': 'AMEM_ENABLE_PERSISTENCE',
            'memory.max_memories': 'AMEM_MAX_MEMORIES',
            'memory.auto_save_interval': 'AMEM_AUTO_SAVE_INTERVAL'
        }

        for config_key, env_key in memory_env.items():
            if env_key in os.environ:
                keys = config_key.split('.')
                if keys[1] in ['enable_persistence']:
                    self._config['memory'][keys[1]] = os.environ[env_key].lower() == 'true'
                elif keys[1] in ['max_memories', 'auto_save_interval']:
                    self._config['memory'][keys[1]] = int(os.environ[env_key])
                else:
                    self._config['memory'][keys[1]] = os.environ[env_key]

        # 调试配置
        if 'AMEM_VERBOSE' in os.environ:
            self._config['debug']['verbose'] = os.environ['AMEM_VERBOSE'].lower() == 'true'

    def _load_from_file(self):
        """从配置文件加载配置"""
        config_path = Path(self.config_file)
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)

                # 深度合并配置
                self._deep_merge(self._config, file_config)
                print(f"✅ Loaded configuration from {config_path}")

            except Exception as e:
                print(f"⚠️  Failed to load config file {config_path}: {e}")
        else:
            print(f"ℹ️  Config file {config_path} not found, using defaults and environment variables")

    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]):
        """深度合并字典"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def _validate_config(self):
        """验证配置的合理性"""
        # 验证alpha值范围
        alpha = self._config['retriever']['alpha']
        if not 0.0 <= alpha <= 1.0:
            print(f"⚠️  Invalid alpha value {alpha}, resetting to 0.5")
            self._config['retriever']['alpha'] = 0.5

        # 验证max_memories
        max_mem = self._config['memory']['max_memories']
        if max_mem <= 0:
            print(f"⚠️  Invalid max_memories value {max_mem}, resetting to 1000")
            self._config['memory']['max_memories'] = 1000

        # 验证存储目录
        storage_dir = Path(self._config['memory']['storage_dir'])
        if not storage_dir.is_absolute():
            # 转换为绝对路径
            self._config['memory']['storage_dir'] = str(storage_dir.resolve())

    def get_config(self) -> Dict[str, Any]:
        """获取完整配置"""
        return self._config.copy()

    def get_amem_config(self) -> Dict[str, Any]:
        """获取A-MEM相关配置"""
        return {
            'use_amem': self._config['use_amem'],
            'retriever_config': self._config['retriever'],
            'memory_config': self._config['memory'],
            'content_analysis_config': self._config['content_analysis'],
            'performance_config': self._config['performance'],
            'debug_config': self._config['debug']
        }

    def get_retriever_config(self) -> Dict[str, Any]:
        """获取检索器配置"""
        return self._config['retriever'].copy()

    def get_memory_config(self) -> Dict[str, Any]:
        """获取记忆系统配置"""
        return self._config['memory'].copy()

    def save_config(self, config: Optional[Dict[str, Any]] = None):
        """
        保存配置到文件

        Args:
            config: 要保存的配置，如果为None则保存当前配置
        """
        if config is not None:
            # 验证新配置
            old_config = self._config.copy()
            self._deep_merge(self._config, config)
            try:
                self._validate_config()
            except Exception as e:
                print(f"⚠️  Invalid config, reverting: {e}")
                self._config = old_config
                return

        config_path = Path(self.config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, ensure_ascii=False, indent=2)
            print(f"✅ Configuration saved to {config_path}")
        except Exception as e:
            print(f"⚠️  Failed to save config: {e}")

    def update_config(self, updates: Dict[str, Any]):
        """
        更新配置

        Args:
            updates: 配置更新字典
        """
        self._deep_merge(self._config, updates)
        self._validate_config()
        print("✅ Configuration updated")

    def reset_to_defaults(self):
        """重置为默认配置"""
        self._config = self._defaults.copy()
        print("✅ Configuration reset to defaults")

    def print_config(self):
        """打印当前配置"""
        print("🔧 Current Memory Configuration:")
        print("=" * 50)

        for section, values in self._config.items():
            print(f"\n📋 {section.upper()}:")
            if isinstance(values, dict):
                for key, value in values.items():
                    print(f"   {key}: {value}")
            else:
                print(f"   {values}")

        print("\n" + "=" * 50)

    # 便捷方法
    @property
    def use_amem(self) -> bool:
        """是否启用A-MEM"""
        return self._config['use_amem']

    @property
    def storage_dir(self) -> str:
        """存储目录"""
        return self._config['memory']['storage_dir']

    @property
    def verbose(self) -> bool:
        """是否启用详细输出"""
        return self._config['debug']['verbose']


# 全局配置实例
_default_config = None

def get_memory_config() -> MemoryConfig:
    """获取全局记忆配置实例"""
    global _default_config
    if _default_config is None:
        _default_config = MemoryConfig()
    return _default_config

def reload_memory_config() -> MemoryConfig:
    """重新加载记忆配置"""
    global _default_config
    _default_config = MemoryConfig()
    return _default_config

