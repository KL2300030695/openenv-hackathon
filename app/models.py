from pydantic import BaseModel
from typing import Dict, Any, List, Optional

class Action(BaseModel):
    type: str # book/reschedule/cancel/waitlist
    patient_id: int
    doctor_id: Optional[int] = None
    slot_id: Optional[int] = None

class StepResponse(BaseModel):
    obs: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]

class ResetResponse(BaseModel):
    obs: Dict[str, Any]

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
