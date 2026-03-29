from pydantic import BaseModel, field_validator
from typing import Dict, Any, List, Optional, Union

class Action(BaseModel):
    type: str # book/reschedule/cancel/waitlist
    patient_id: Union[int, str]
    doctor_id: Optional[Union[int, str]] = None
    slot_id: Optional[Union[int, str]] = None

class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]

class ResetResponse(BaseModel):
    observation: Dict[str, Any]
    info: Dict[str, Any] = {}

class StateResponse(BaseModel):
    state: Dict[str, Any]

class TaskResponse(BaseModel):
    name: str
    difficulty: str
    description: str

class GraderResponse(BaseModel):
    task_scores: Dict[str, float]

class BaselineResponse(BaseModel):
    task_scores: Dict[str, float]
    observations: List[Dict[str, Any]]
