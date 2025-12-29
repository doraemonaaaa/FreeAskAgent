
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# class Attributes(BaseModel):
#     model_config = ConfigDict(extra="forbid")

#     color: str | None = None
#     material: str | None = None
#     state: str | None = None
#     # 若未来需要更多字段，请在此补充，保持 extra=forbid 以满足 OpenAI schema 要求


# class Position(BaseModel):
#     model_config = ConfigDict(extra="forbid")

#     x: float | None = None
#     y: float | None = None
#     z: float | None = None


# class SceneObject(BaseModel):
#     model_config = ConfigDict(extra="forbid")

#     object_id: str
#     label: str
#     category: str | None = None
#     # For structured outputs, forbid extra keys in nested dicts to satisfy OpenAI schema rules
#     attributes: Attributes | None = None
#     bbox: list[float] | None = None
#     position: Position | None = Field(default=None, json_schema_extra={"additionalProperties": False})
#     is_static: bool = True
#     confidence: float | None = None
#     source_frames: list[str] | None = None


# class SceneRelation(BaseModel):
#     model_config = ConfigDict(extra="forbid")

#     subject_id: str
#     predicate: str
#     object_id: str
#     confidence: float | None = None


# class SceneGraph(BaseModel):
#     model_config = ConfigDict(extra="forbid")

#     objects: list[SceneObject]
#     relations: list[SceneRelation] = Field(default_factory=list)
#     frames: list[str] | None = None

# Planner: QueryAnalysis
class QueryAnalysis(BaseModel):
    concise_summary: str
    required_skills: str
    relevant_tools: str
    additional_considerations: str

    def __str__(self):
        return f"""
Concise Summary: {self.concise_summary}

Required Skills:
{self.required_skills}

Relevant Tools:
{self.relevant_tools}

Additional Considerations:
{self.additional_considerations}
"""

# Planner: NextStep
class NextStep(BaseModel):
    justification: str
    context: str
    sub_goal: str
    tool_name: str

# Executor: MemoryVerification
class MemoryVerification(BaseModel):
    analysis: str
    stop_signal: bool

# Executor: ToolCommand
class ToolCommand(BaseModel):
    analysis: str
    explanation: str
    command: str
