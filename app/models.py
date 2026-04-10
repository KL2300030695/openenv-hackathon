"""
Data models for the Healthcare Scheduling Environment.

Uses OpenEnv base types for framework compatibility.
Fallback to plain Pydantic if openenv-core is not installed.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

# ── OpenEnv base types (with graceful fallback) ────────────────────────
try:
    from openenv.core.env_server.types import Action, Observation
except ImportError:
    # Fallback: define minimal base classes so the env still works standalone
    class Action(BaseModel):
        """Base action class (fallback when openenv-core is not installed)."""
        pass

    class Observation(BaseModel):
        """Base observation class (fallback when openenv-core is not installed)."""
        done: bool = False
        reward: float = 0.0


# ── Environment Action ────────────────────────────────────────────────
class HealthcareAction(Action):
    """Action for the Healthcare Scheduling environment."""

    type: str = Field(
        ...,
        description="Action type: 'book', 'reschedule', 'cancel', or 'waitlist'",
    )
    patient_id: Union[int, str] = Field(
        ...,
        description="ID of the patient",
    )
    doctor_id: Optional[Union[int, str]] = Field(
        default=None,
        description="ID of the doctor (required for 'book' and 'reschedule')",
    )
    slot_id: Optional[Union[int, str]] = Field(
        default=None,
        description="ID of the time slot (required for 'book' and 'reschedule')",
    )


# ── Environment Observation ──────────────────────────────────────────
class HealthcareObservation(Observation):
    """Observation from the Healthcare Scheduling environment."""

    doctor_slots: Optional[Dict[str, List[bool]]] = Field(
        default=None,
        description="Availability of each doctor for each time slot (True = available)",
    )
    patients: Optional[Dict[str, Dict[str, Any]]] = Field(
        default=None,
        description="Patient information: priority, preferred doctor, status",
    )
    waiting_queue: Optional[List[str]] = Field(
        default=None,
        description="List of patient IDs currently waiting",
    )
    current_step: Optional[int] = Field(
        default=None,
        description="Current step number in the episode",
    )
    info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional info (errors, status messages)",
    )


# ── API Response Models (for FastAPI endpoint compatibility) ──────────
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
