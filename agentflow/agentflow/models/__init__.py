"""
FreeAskAgent Models Package

This package contains all the model components for the FreeAskAgent system,
including A-MEM (Agentic Memory) integration components.
"""

# Core memory components
from .memory import Memory
from .agentic_memory_system import AgenticMemorySystem

# Planning and verification components
from .planner import Planner
from .verifier import Verifier

# Configuration and utilities
from .memory_config import MemoryConfig
from .formatters import NextStep, QueryAnalysis, MemoryVerification
from .initializer import Initializer
from .executor import Executor

# Version information
__version__ = "1.0.0"

__all__ = [
    # Core classes
    "Memory",
    "AgenticMemorySystem",
    "Planner",
    "Verifier",
    "MemoryConfig",

    # Utilities
    "NextStep",
    "QueryAnalysis",
    "MemoryVerification",
    "Initializer",
    "Executor",

    # Version
    "__version__",
]
