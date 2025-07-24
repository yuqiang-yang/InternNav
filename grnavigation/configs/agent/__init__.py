from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class AgentCfg(BaseModel):
    server_host: str = 'localhost'
    server_port: int = 5000
    model_name: str
    ckpt_path: str
    model_settings: Dict[str, Any]


class InitRequest(BaseModel, extra='allow'):
    agent_config: AgentCfg


class StepRequest(BaseModel, extra='allow'):
    observation: Any


class ResetRequest(BaseModel):
    reset_index: Optional[List]


__all__ = ['AgentCfg', 'InitRequest', 'StepRequest', 'ResetRequest']
