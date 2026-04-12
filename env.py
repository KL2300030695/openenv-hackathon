"""
Healthcare Appointment Scheduling Environment.

Implements the OpenEnv Environment interface for a hospital appointment
booking system with multiple departments/specialties, doctors, patients
with varying priorities and medical needs, and time slots.

Key Features:
  - Specialty matching: patients have required specialties, doctors specialise
  - Priority scheduling: urgent patients (priority 1) should be booked first
  - Time preferences: patients prefer morning or afternoon slots
  - Conflict handling: double-booking detection and prevention
  - Pre-existing appointments create realistic scarcity
"""

import random
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server import Environment
from models import HealthcareAction, HealthcareObservation, HealthcareState


class HealthcareEnvironment(Environment):
    """
    Healthcare Appointment Scheduling RL Environment.

    Simulates a hospital with multiple departments, each staffed by a
    specialist doctor. Patients arrive with specific medical needs (requiring
    a particular specialty), varying urgency levels, and time preferences.
    The agent must schedule appointments efficiently while respecting all
    constraints.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # Hospital configuration
    SPECIALTIES = [
        "General Medicine",
        "Cardiology",
        "Orthopedics",
        "Pediatrics",
        "Dermatology",
    ]

    SLOT_TIMES = [
        "9:00 AM", "9:30 AM", "10:00 AM", "10:30 AM",   # morning  (0-3)
        "2:00 PM", "2:30 PM", "3:00 PM", "3:30 PM",      # afternoon (4-7)
    ]

    def __init__(self, seed: Optional[int] = None,
                 num_doctors: int = 5, num_slots: int = 8,
                 num_patients: int = 15):
        self.num_doctors = num_doctors
        self.num_slots = num_slots
        self.num_patients = num_patients
        self.priorities = [1, 2, 3]

        self._seed = seed
        self._state = HealthcareState(episode_id=str(uuid4()), step_count=0)

        # Internal state
        self.doctor_slots: Dict[int, List[bool]] = {}
        self.doctor_specialties: Dict[int, str] = {}
        self.patients: Dict[int, Dict[str, Any]] = {}
        self.waiting_queue: List[int] = []
        self.current_step: int = 0
        self.max_steps: int = 0
        self.done: bool = False
        self._action_history: List[Dict[str, Any]] = []

        # Auto-reset so the env is usable immediately after __init__
        self.reset()

    # ── OpenEnv Interface ─────────────────────────────────────────────

    def reset(self, seed=None, episode_id=None, **kwargs) -> HealthcareObservation:
        """Reset the environment to a fresh hospital scenario."""
        if seed is not None:
            self._seed = seed
        if self._seed is not None:
            random.seed(self._seed)

        # ── Assign specialties to doctors ──
        self.doctor_specialties = {
            i: self.SPECIALTIES[i % len(self.SPECIALTIES)]
            for i in range(self.num_doctors)
        }

        # ── Initialise all slots as available ──
        self.doctor_slots = {
            i: [True] * self.num_slots for i in range(self.num_doctors)
        }

        # ── Pre-book some slots to create realistic scarcity ──
        num_preexisting = random.randint(3, 8)
        for _ in range(num_preexisting):
            doc = random.randint(0, self.num_doctors - 1)
            slot = random.randint(0, self.num_slots - 1)
            self.doctor_slots[doc][slot] = False

        # ── Create patients with diverse medical needs ──
        self.patients = {}
        specialty_weights = [0.30, 0.20, 0.20, 0.15, 0.15]

        for i in range(self.num_patients):
            required_specialty = random.choices(
                self.SPECIALTIES, weights=specialty_weights, k=1
            )[0]

            # Preferred doctor has the matching specialty
            matching_docs = [
                d for d, s in self.doctor_specialties.items()
                if s == required_specialty
            ]
            preferred_doctor = (
                random.choice(matching_docs) if matching_docs
                else random.randint(0, self.num_doctors - 1)
            )

            time_preference = random.choice(["morning", "afternoon"])

            self.patients[i] = {
                "id": i,
                "priority": random.choice(self.priorities),
                "preferred_doctor": preferred_doctor,
                "required_specialty": required_specialty,
                "time_preference": time_preference,
                "status": "waiting",
                "booked_slot": None,
                "booked_doctor": None,
                "wait_steps": 0,
            }

        self.waiting_queue = list(self.patients.keys())
        self.current_step = 0
        self.max_steps = self.num_patients * 2
        self.done = False
        self._action_history = []

        self._state = HealthcareState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            score=0.1,
        )

        return HealthcareObservation(
            doctor_slots={str(k): v for k, v in self.doctor_slots.items()},
            doctor_specialties={str(k): v for k, v in self.doctor_specialties.items()},
            patients={str(k): v for k, v in self.patients.items()},
            waiting_queue=[str(p) for p in self.waiting_queue],
            current_step=self.current_step,
            max_steps=self.max_steps,
            info={},
            done=False,
            reward=0.1,
            score=0.1,
        )

    def step(self, action: HealthcareAction, **kwargs) -> HealthcareObservation:
        """Execute a scheduling action in the environment."""
        self._state.step_count += 1

        if self.done:
            return self._make_obs(0.1, {"error": "Environment done"})

        action_type = action.type

        try:
            patient_id = int(action.patient_id) if action.patient_id is not None else None
            doctor_id = int(action.doctor_id) if action.doctor_id is not None else None
            slot_id = int(action.slot_id) if action.slot_id is not None else None
        except (ValueError, TypeError):
            return self._make_obs(0.1, {"error": "Invalid ID format (must be numeric)"})

        reward = 0.1
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
            reward = 0.1
            info = {"error": "Invalid action type"}

        self._action_history.append({
            "step": self.current_step,
            "type": action_type,
            "patient_id": patient_id,
            "doctor_id": doctor_id,
            "slot_id": slot_id,
            "reward": reward,
            "info": info,
        })

        # Increment wait counter for patients still waiting
        for pid in self.waiting_queue:
            self.patients[pid]["wait_steps"] += 1

        self.current_step += 1
        if self.current_step >= self.max_steps or not self.waiting_queue:
            self.done = True

        # Update running score
        booked = sum(1 for p in self.patients.values() if p["status"] == "booked")
        self._state.score = min(max(booked / self.num_patients, 0.1), 0.9)

        return self._make_obs(reward, info)

    @property
    def state(self) -> HealthcareState:
        """Get the current environment state."""
        return self._state

    # ── Legacy compatibility (dict-based step) ───────────────────────

    def step_dict(self, action_dict: Dict[str, Any]) -> tuple:
        action = HealthcareAction(
            type=action_dict.get("type", "waitlist"),
            patient_id=action_dict.get("patient_id", 0),
            doctor_id=action_dict.get("doctor_id"),
            slot_id=action_dict.get("slot_id"),
        )
        obs = self.step(action)
        obs_dict = {
            "doctor_slots": obs.doctor_slots,
            "doctor_specialties": obs.doctor_specialties,
            "patients": obs.patients,
            "waiting_queue": obs.waiting_queue,
            "current_step": obs.current_step,
        }
        return obs_dict, obs.reward, obs.done, obs.info or {}

    def reset_dict(self) -> Dict[str, Any]:
        obs = self.reset()
        return {
            "doctor_slots": obs.doctor_slots,
            "doctor_specialties": obs.doctor_specialties,
            "patients": obs.patients,
            "waiting_queue": obs.waiting_queue,
            "current_step": obs.current_step,
        }

    # ── Internal helpers ─────────────────────────────────────────────

    def _make_obs(self, reward: float, info: Dict[str, Any]) -> HealthcareObservation:
        """Create observation. Rewards clamped to [0.1, 0.9] for validator compliance."""
        clamped_reward = min(max(float(reward), 0.1), 0.9)
        return HealthcareObservation(
            doctor_slots={str(k): v for k, v in self.doctor_slots.items()},
            doctor_specialties={str(k): v for k, v in self.doctor_specialties.items()},
            patients={str(k): v for k, v in self.patients.items()},
            waiting_queue=[str(p) for p in self.waiting_queue],
            current_step=self.current_step,
            max_steps=self.max_steps,
            info=info,
            done=self.done,
            reward=clamped_reward,
            score=self._state.score,
        )

    def _handle_book(self, patient_id, doctor_id, slot_id) -> tuple:
        """Handle booking with specialty validation and multi-factor rewards."""
        # ── Validate patient ──
        if patient_id not in self.patients:
            return 0.1, {"error": f"Patient {patient_id} does not exist"}
        if self.patients[patient_id]["status"] != "waiting":
            return 0.1, {"error": f"Patient {patient_id} not in waiting queue"}

        # ── Validate doctor and slot ──
        if doctor_id not in self.doctor_slots:
            return 0.1, {"error": f"Doctor {doctor_id} does not exist"}
        if slot_id is None or slot_id < 0 or slot_id >= self.num_slots:
            return 0.1, {"error": f"Invalid slot {slot_id}"}

        # ── Conflict detection ──
        if not self.doctor_slots[doctor_id][slot_id]:
            return 0.15, {
                "error": "Slot already booked (scheduling conflict)",
                "conflict": True,
            }

        # ── Specialty validation ──
        patient_specialty = self.patients[patient_id].get(
            "required_specialty", "General Medicine"
        )
        doctor_specialty = self.doctor_specialties.get(doctor_id, "General Medicine")

        # Matching rules:
        #   - Exact specialty match always OK
        #   - General Medicine doctor can see any patient (catch-all)
        #   - General Medicine patient can see any doctor (flexible)
        specialty_match = (
            patient_specialty == doctor_specialty
            or doctor_specialty == "General Medicine"
            or patient_specialty == "General Medicine"
        )

        if not specialty_match:
            return 0.15, {
                "error": (
                    f"Specialty mismatch: patient needs {patient_specialty}, "
                    f"doctor offers {doctor_specialty}"
                ),
                "specialty_mismatch": True,
            }

        # ── SUCCESS — execute the booking ──
        self.doctor_slots[doctor_id][slot_id] = False
        self.patients[patient_id].update({
            "status": "booked",
            "booked_slot": slot_id,
            "booked_doctor": doctor_id,
        })
        if patient_id in self.waiting_queue:
            self.waiting_queue.remove(patient_id)

        # ── Multi-factor reward ──
        reward = 0.60                                          # base booking

        if self.patients[patient_id]["preferred_doctor"] == doctor_id:
            reward += 0.15                                     # preferred doctor bonus

        if patient_specialty == doctor_specialty:
            reward += 0.10                                     # exact specialty match

        time_pref = self.patients[patient_id].get("time_preference", "morning")
        is_morning_slot = slot_id < (self.num_slots // 2)
        if (time_pref == "morning" and is_morning_slot) or \
           (time_pref == "afternoon" and not is_morning_slot):
            reward += 0.05                                     # time preference match

        return reward, {
            "status": "success",
            "specialty_match": patient_specialty == doctor_specialty,
            "preferred_doctor": self.patients[patient_id]["preferred_doctor"] == doctor_id,
        }

    def _handle_reschedule(self, patient_id, doctor_id, slot_id) -> tuple:
        """Handle rescheduling with specialty validation."""
        if patient_id not in self.patients:
            return 0.1, {"error": "Patient does not exist"}
        if self.patients[patient_id]["status"] != "booked":
            return 0.1, {"error": "Patient not currently booked"}
        if doctor_id not in self.doctor_slots:
            return 0.1, {"error": "Doctor does not exist"}
        if slot_id is None or slot_id < 0 or slot_id >= self.num_slots:
            return 0.1, {"error": "Invalid slot"}
        if not self.doctor_slots[doctor_id][slot_id]:
            return 0.15, {"error": "Target slot already booked", "conflict": True}

        # Specialty check for new doctor
        patient_specialty = self.patients[patient_id].get(
            "required_specialty", "General Medicine"
        )
        doctor_specialty = self.doctor_specialties.get(doctor_id, "General Medicine")
        if not (patient_specialty == doctor_specialty
                or doctor_specialty == "General Medicine"
                or patient_specialty == "General Medicine"):
            return 0.15, {
                "error": "Specialty mismatch for reschedule",
                "specialty_mismatch": True,
            }

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

        reward = 0.50
        if self.patients[patient_id]["preferred_doctor"] == doctor_id:
            reward += 0.10
        return reward, {"status": "success"}

    def _handle_cancel(self, patient_id) -> tuple:
        """Handle appointment cancellation."""
        if patient_id not in self.patients:
            return 0.1, {"error": "Patient does not exist"}
        if self.patients[patient_id]["status"] != "booked":
            return 0.1, {"error": "Patient not booked"}

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

        return 0.20, {"status": "cancelled"}

    def _handle_waitlist(self, patient_id) -> tuple:
        """Handle placing a patient on the waitlist."""
        if patient_id not in self.patients:
            return 0.1, {"error": "Patient does not exist"}
        if self.patients[patient_id]["status"] != "waiting":
            return 0.1, {"error": "Patient not in waiting queue"}
        return 0.25, {"status": "waitlisted"}


# ── Backward-compatible alias ────────────────────────────────────────
HealthcareEnv = HealthcareEnvironment
