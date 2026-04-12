"""
Data models for the Healthcare Scheduling Environment.

Typed Pydantic models extending the OpenEnv base classes for
actions, observations, and state management.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import Field
from openenv.core.env_server import Action, Observation, State


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

    doctor_slots: Optional[Dict[str, List[bool]]] = None
    doctor_specialties: Optional[Dict[str, str]] = None
    patients: Optional[Dict[str, Dict[str, Any]]] = None
    waiting_queue: Optional[List[str]] = None
    current_step: Optional[int] = None
    max_steps: Optional[int] = None
    info: Optional[Dict[str, Any]] = None
    score: Optional[float] = None


# ── Environment State ────────────────────────────────────────────────
class HealthcareState(State):
    """Custom state for the Healthcare Scheduling environment."""
    difficulty: str = "easy"
    score: float = 0.01
