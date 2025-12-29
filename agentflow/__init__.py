__version__ = "0.1.2"

from .client import AgentFlowClient, DevTaskLoader
from .config import flow_cli
from .litagent import LitAgent
from .logging import configure_logger
from .reward import reward
from .server import AgentFlowServer
from .trainer import Trainer
from .types import *

# Import subpackage for convenience
from . import agentflow as _agentflow
# Re-export key classes from subpackage
from .agentflow import Memory, AgenticMemorySystem, Planner, Verifier

# Create models attribute for convenience
import sys
sys.modules['agentflow.models'] = sys.modules['agentflow.agentflow.models']
