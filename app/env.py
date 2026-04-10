"""
Healthcare Appointment Scheduling Environment.

Implements the OpenEnv Environment interface for a hospital appointment
booking system with doctors, patients, and time slots.

Design:
    - Multi-step episodes: reset() initialises state, step() processes actions
    - 3 doctors × 5 time slots = 15 total bookable slots
    - 10 patients with random priorities (1=High, 2=Medium, 3=Low)
    - Episode ends when waiting queue is empty or max steps reached

Rewards:
    +1.0  successful booking
    +0.2  bonus for preferred doctor match
    +0.5  successful reschedule
    +0.3  waitlisting (partial progress)
    -0.2  cancellation
    -1.0  invalid action / conflict
"""

import random
from typing import Any, Dict, List, Optional
from uuid import uuid4

# ── OpenEnv base classes (with graceful fallback) ────────────────────
try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import State
    _HAS_OPENENV = True
except ImportError:
    # Fallback so the environment works standalone without openenv-core
    _HAS_OPENENV = False

    class Environment:
        """Minimal Environment stub when openenv-core is not installed."""
        SUPPORTS_CONCURRENT_SESSIONS = False

    class State:
        """Minimal State stub."""
        def __init__(self, episode_id: str = "", step_count: int = 0):
            self.episode_id = episode_id
            self.step_count = step_count

try:
    from app.models import HealthcareAction, HealthcareObservation
except ImportError:
    from .models import HealthcareAction, HealthcareObservation


class HealthcareEnvironment(Environment):
    """
    Healthcare Appointment Scheduling RL Environment.

    Simulates a hospital appointment booking system with doctors, patients,
    and time slots. Follows the OpenEnv Environment interface.

    Example:
        >>> env = HealthcareEnvironment()
        >>> obs = env.reset()
        >>> print(obs.waiting_queue)     # ['0', '1', ..., '9']
        >>> obs = env.step(HealthcareAction(type='book', patient_id=0, doctor_id=0, slot_id=0))
        >>> print(obs.reward)            # 1.0 or 1.2
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, seed: Optional[int] = None):
        """
        Initialise the Healthcare Scheduling environment.

        Args:
            seed: Optional random seed for reproducibility.
        """
        self.num_doctors = 3
        self.num_slots = 5       # 5 slots per doctor (e.g. 9 AM to 2 PM)
        self.num_patients = 10
        self.priorities = [1, 2, 3]  # 1: High, 2: Medium, 3: Low

        self._seed = seed
        self._state = State(episode_id=str(uuid4()), step_count=0)

        # Internal state (populated by reset)
        self.doctor_slots: Dict[int, List[bool]] = {}
        self.patients: Dict[int, Dict[str, Any]] = {}
        self.waiting_queue: List[int] = []
        self.current_step: int = 0
        self.max_steps: int = 0
        self.done: bool = False

        # Tracking for grading
        self._action_history: List[Dict[str, Any]] = []

    # ── OpenEnv Interface ─────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> HealthcareObservation:
        """
        Reset the environment to an initial state.

        Args:
            seed: Random seed for reproducibility.
            episode_id: Optional episode ID.

        Returns:
            HealthcareObservation with initial state, done=False, reward=0.0
        """
        if seed is not None:
            self._seed = seed
        if self._seed is not None:
            random.seed(self._seed)

        # Doctors availability: True means slot is free
        self.doctor_slots = {
            i: [True] * self.num_slots for i in range(self.num_doctors)
        }

        # Patients with random priorities and preferred doctors
        self.patients = {}
        for i in range(self.num_patients):
            self.patients[i] = {
                "id": i,
                "priority": random.choice(self.priorities),
                "preferred_doctor": random.randint(0, self.num_doctors - 1),
                "status": "waiting",
                "booked_slot": None,
                "booked_doctor": None,
            }

        self.waiting_queue = list(self.patients.keys())
        self.current_step = 0
        self.max_steps = self.num_patients * 2  # buffer for rescheduling
        self.done = False
        self._action_history = []

        # Update episode state
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        return HealthcareObservation(
            doctor_slots={str(k): v for k, v in self.doctor_slots.items()},
            patients={str(k): v for k, v in self.patients.items()},
            waiting_queue=[str(p) for p in self.waiting_queue],
            current_step=self.current_step,
            info={},
            done=False,
            reward=0.0,
        )

    def step(self, action: HealthcareAction) -> HealthcareObservation:
        """
        Execute an action in the environment.

        Args:
            action: HealthcareAction with type, patient_id, doctor_id, slot_id.

        Returns:
            HealthcareObservation with updated state, reward, and done flag.
        """
        if hasattr(self, '_state'):
            self._state.step_count += 1

        if self.done:
            return HealthcareObservation(
                doctor_slots={str(k): v for k, v in self.doctor_slots.items()},
                patients={str(k): v for k, v in self.patients.items()},
                waiting_queue=[str(p) for p in self.waiting_queue],
                current_step=self.current_step,
                info={"error": "Environment done"},
                done=True,
                reward=0.0,
            )

        action_type = action.type

        # Robustly convert IDs to integers
        try:
            patient_id = int(action.patient_id) if action.patient_id is not None else None
            doctor_id = int(action.doctor_id) if action.doctor_id is not None else None
            slot_id = int(action.slot_id) if action.slot_id is not None else None
        except (ValueError, TypeError):
            return self._make_obs(-1.0, {"error": "Invalid ID format (must be numeric)"})

        reward = 0.0
        info: Dict[str, Any] = {}

        if action_type == "book":
            reward, info = self._handle_book(patient_id, doctor_id, slot_id)
        elif action_type == "reschedule":
            reward, info = self._handle_reschedule(patient_id, doctor_id, slot_id)
        elif action_type == "cancel":
            reward, info = self._handle_cancel(patient_id)
        elif action_type == "waitlist":
            reward, info = self._handle_waitlist(patient_id)
        else:
            reward = -1.0
            info = {"error": "Invalid action type"}

        self._action_history.append({
            "type": action_type,
            "patient_id": patient_id,
            "doctor_id": doctor_id,
            "slot_id": slot_id,
            "reward": reward,
            "info": info,
        })

        self.current_step += 1
        if self.current_step >= self.max_steps or not self.waiting_queue:
            self.done = True

        return self._make_obs(reward, info)

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            State with episode_id and step_count.
        """
        return self._state

    # ── Legacy compatibility (dict-based step) ───────────────────────

    def step_dict(self, action_dict: Dict[str, Any]) -> tuple:
        """
        Dict-based step for backward compatibility with legacy endpoints.

        Args:
            action_dict: Action as a plain dictionary.

        Returns:
            Tuple of (obs_dict, reward, done, info).
        """
        action = HealthcareAction(
            type=action_dict.get("type", "waitlist"),
            patient_id=action_dict.get("patient_id", 0),
            doctor_id=action_dict.get("doctor_id"),
            slot_id=action_dict.get("slot_id"),
        )
        obs = self.step(action)
        obs_dict = {
            "doctor_slots": obs.doctor_slots,
            "patients": obs.patients,
            "waiting_queue": obs.waiting_queue,
            "current_step": obs.current_step,
        }
        return obs_dict, obs.reward, obs.done, obs.info or {}

    def reset_dict(self) -> Dict[str, Any]:
        """Reset and return observation as a plain dict."""
        obs = self.reset()
        return {
            "doctor_slots": obs.doctor_slots,
            "patients": obs.patients,
            "waiting_queue": obs.waiting_queue,
            "current_step": obs.current_step,
        }

    # ── Internal helpers ─────────────────────────────────────────────

    def _make_obs(self, reward: float, info: Dict[str, Any]) -> HealthcareObservation:
        """Build an observation from current internal state."""
        return HealthcareObservation(
            doctor_slots={str(k): v for k, v in self.doctor_slots.items()},
            patients={str(k): v for k, v in self.patients.items()},
            waiting_queue=[str(p) for p in self.waiting_queue],
            current_step=self.current_step,
            info=info,
            done=self.done,
            reward=reward,
        )

    def _handle_book(self, patient_id, doctor_id, slot_id) -> tuple:
        if patient_id not in self.patients or self.patients[patient_id]["status"] != "waiting":
            return -1.0, {"error": f"Patient ID {patient_id} not available for booking"}

        if doctor_id not in self.doctor_slots or slot_id is None or slot_id < 0 or slot_id >= self.num_slots:
            return -1.0, {"error": "Invalid doctor or slot ID"}

        if not self.doctor_slots[doctor_id][slot_id]:
            return -1.0, {"error": "Slot already booked (conflict)"}

        # Successful booking
        self.doctor_slots[doctor_id][slot_id] = False
        self.patients[patient_id].update({
            "status": "booked",
            "booked_slot": slot_id,
            "booked_doctor": doctor_id,
        })
        if patient_id in self.waiting_queue:
            self.waiting_queue.remove(patient_id)

        reward = 1.0
        if self.patients[patient_id]["preferred_doctor"] == doctor_id:
            reward += 0.2  # Bonus for preferred doctor

        return reward, {"status": "success"}

    def _handle_reschedule(self, patient_id, doctor_id, slot_id) -> tuple:
        if patient_id not in self.patients or self.patients[patient_id]["status"] != "booked":
            return -1.0, {"error": "Patient not currently booked"}

        if doctor_id not in self.doctor_slots or slot_id is None or slot_id < 0 or slot_id >= self.num_slots:
            return -1.0, {"error": "Invalid doctor or slot ID"}

        if not self.doctor_slots[doctor_id][slot_id]:
            return -1.0, {"error": "Target slot already booked (conflict)"}

        # Free old slot
        old_doctor = self.patients[patient_id]["booked_doctor"]
        old_slot = self.patients[patient_id]["booked_slot"]
        self.doctor_slots[old_doctor][old_slot] = True

        # Book new slot
        self.doctor_slots[doctor_id][slot_id] = False
        self.patients[patient_id].update({
            "booked_slot": slot_id,
            "booked_doctor": doctor_id,
        })

        return 0.5, {"status": "success"}

    def _handle_cancel(self, patient_id) -> tuple:
        if patient_id not in self.patients or self.patients[patient_id]["status"] == "waiting":
            return -1.0, {"error": "Patient not booked"}

        # Free slot if booked
        if self.patients[patient_id]["status"] == "booked":
            doctor_id = self.patients[patient_id]["booked_doctor"]
            slot_id = self.patients[patient_id]["booked_slot"]
            self.doctor_slots[doctor_id][slot_id] = True

        self.patients[patient_id].update({
            "status": "waiting",
            "booked_slot": None,
            "booked_doctor": None,
        })
        if patient_id not in self.waiting_queue:
            self.waiting_queue.append(patient_id)

        return -0.2, {"status": "cancelled"}

    def _handle_waitlist(self, patient_id) -> tuple:
        if patient_id not in self.patients or self.patients[patient_id]["status"] != "waiting":
            return -1.0, {"error": "Patient not in waiting queue"}

        return 0.3, {"status": "waitlisted"}


# ── Backward-compatible alias ────────────────────────────────────────
HealthcareEnv = HealthcareEnvironment
