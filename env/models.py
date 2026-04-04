from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Literal
from enum import Enum


class ActionType(str, Enum):
    fix_value = "fix_value"
    drop_row = "drop_row"
    fill_missing = "fill_missing"
    normalize_column = "normalize_column"
    mark_done = "mark_done"


class ColumnTransform(str, Enum):
    lowercase = "lowercase"
    strip = "strip"
    to_int = "to_int"
    to_float = "to_float"
    to_date = "to_date"


class ValidationError(BaseModel):
    row_index: Optional[int] = None
    column: Optional[str] = None
    error_type: str
    description: str
    severity: Literal["low", "medium", "high"] = "medium"


class SchemaField(BaseModel):
    name: str
    expected_type: str
    nullable: bool = False
    constraints: Optional[Dict[str, Any]] = None


class Observation(BaseModel):
    """OpenEnv typed Observation model"""
    dataset: List[Dict[str, Any]] = Field(description="Current table rows")
    schema_fields: List[SchemaField] = Field(description="Expected schema")
    errors: List[ValidationError] = Field(description="Current validation errors")
    step_count: int = Field(default=0)
    task_id: str = Field(description="Active task identifier")
    task_description: str = Field(description="What the agent must accomplish")
    total_errors_initial: int = Field(description="Errors at episode start")
    errors_fixed_so_far: int = Field(default=0)
    done: bool = Field(default=False)


class Action(BaseModel):
    """OpenEnv typed Action model"""
    action_type: ActionType
    row_index: Optional[int] = Field(default=None, description="Target row (0-indexed)")
    column: Optional[str] = Field(default=None, description="Target column name")
    new_value: Optional[Any] = Field(default=None, description="Replacement value")
    column_transform: Optional[ColumnTransform] = Field(default=None)


class Reward(BaseModel):
    """OpenEnv typed Reward model"""
    value: float = Field(ge=0.0, le=1.0, description="Reward value [0.0, 1.0]")
    errors_fixed: int = Field(default=0)
    collateral_damage: int = Field(default=0, description="Correct cells broken")
    partial_progress: float = Field(default=0.0)
    reason: str = Field(default="")


class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class TaskInfo(BaseModel):
    task_id: str
    name: str
    difficulty: Literal["easy", "medium", "hard"]
    max_steps: int
    description: str


class EnvState(BaseModel):
    task_id: str
    step_count: int
    done: bool
    errors_remaining: int
    errors_fixed: int
    total_errors_initial: int
    score: float
    dataset_snapshot: List[Dict[str, Any]]
