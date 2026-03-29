import random
from typing import Dict, List, Any, Optional, Union

class HealthcareEnv:
    """
    Healthcare Appointment Scheduling RL Environment.
    Simulates a hospital appointment booking system with doctors, patients, and time slots.
    """
    def __init__(self):
        self.num_doctors = 3
        self.num_slots = 5  # 5 slots per doctor (e.g., 9 AM to 2 PM)
        self.num_patients = 10
        self.priorities = [1, 2, 3]  # 1: High, 2: Medium, 3: Low
        self.reset()

    def reset(self):
        """Resets the environment to initial state."""
        # Doctors availability: {doctor_id: [slot_0_available, slot_1_available, ...]}
        # True means slot is available (free), False means booked.
        self.doctor_slots = {i: [True] * self.num_slots for i in range(self.num_doctors)}
        
        # Patients: {patient_id: {"priority": p, "preferred_doctor": d, "status": "waiting/booked/cancelled"}}
        self.patients = {}
        for i in range(self.num_patients):
            self.patients[i] = {
                "id": i,
                "priority": random.choice(self.priorities),
                "preferred_doctor": random.randint(0, self.num_doctors - 1),
                "status": "waiting",
                "booked_slot": None,
                "booked_doctor": None
            }
        
        self.waiting_queue = list(self.patients.keys())
        self.current_step = 0
        self.max_steps = self.num_patients * 2  # Allow some buffer for rescheduling
        self.done = False
        
        return self._get_obs()

    def _get_obs(self) -> Dict[str, Any]:
        """Returns the current observation of the environment with string keys for JSON serialization."""
        return {
            "doctor_slots": {str(k): v for k, v in self.doctor_slots.items()},
            "patients": {str(k): v for k, v in self.patients.items()},
            "waiting_queue": [str(p) for p in self.waiting_queue],
            "current_step": self.current_step
        }

    def state(self) -> Dict[str, Any]:
        """Returns the full state of the environment (OpenEnv API)."""
        return self._get_obs()

    def step(self, action: Dict[str, Any]) -> tuple:
        """
        Performs an action in the environment.
        Action format: {"type": "book/reschedule/cancel/waitlist", "patient_id": pid, "doctor_id": did, "slot_id": sid}
        Robustly handles string IDs from external clients/LLMs.
        """
        if self.done:
            return self._get_obs(), 0, True, {"error": "Environment done"}

        action_type = action.get("type")
        
        # Robustly convert IDs to integers
        try:
            patient_id = int(action.get("patient_id")) if action.get("patient_id") is not None else None
            doctor_id = int(action.get("doctor_id")) if action.get("doctor_id") is not None else None
            slot_id = int(action.get("slot_id")) if action.get("slot_id") is not None else None
        except (ValueError, TypeError):
            return self._get_obs(), -1, self.done, {"error": "Invalid ID format (must be numeric)"}

        reward = 0
        info = {}

        if action_type == "book":
            reward, info = self._handle_book(patient_id, doctor_id, slot_id)
        elif action_type == "reschedule":
            reward, info = self._handle_reschedule(patient_id, doctor_id, slot_id)
        elif action_type == "cancel":
            reward, info = self._handle_cancel(patient_id)
        elif action_type == "waitlist":
            reward, info = self._handle_waitlist(patient_id)
        else:
            reward = -1
            info = {"error": "Invalid action type"}

        self.current_step += 1
        if self.current_step >= self.max_steps or not self.waiting_queue:
            self.done = True

        return self._get_obs(), reward, self.done, info

    def _handle_book(self, patient_id, doctor_id, slot_id) -> tuple:
        if patient_id not in self.patients or self.patients[patient_id]["status"] != "waiting":
            return -1, {"error": f"Patient ID {patient_id} not available for booking"}
        
        if doctor_id not in self.doctor_slots or slot_id is None or slot_id < 0 or slot_id >= self.num_slots:
            return -1, {"error": "Invalid doctor or slot ID"}

        if not self.doctor_slots[doctor_id][slot_id]:
            return -1, {"error": "Slot already booked (conflict)"}

        # Successful booking
        self.doctor_slots[doctor_id][slot_id] = False
        self.patients[patient_id].update({
            "status": "booked",
            "booked_slot": slot_id,
            "booked_doctor": doctor_id
        })
        if patient_id in self.waiting_queue:
            self.waiting_queue.remove(patient_id)
        
        # Reward based on priority and preference
        reward = 1.0
        if self.patients[patient_id]["preferred_doctor"] == doctor_id:
            reward += 0.2  # Bonus for preferred doctor
        
        return reward, {"status": "success"}

    def _handle_reschedule(self, patient_id, doctor_id, slot_id) -> tuple:
        if patient_id not in self.patients or self.patients[patient_id]["status"] != "booked":
            return -1, {"error": "Patient not currently booked"}

        if doctor_id not in self.doctor_slots or slot_id is None or slot_id < 0 or slot_id >= self.num_slots:
            return -1, {"error": "Invalid doctor or slot ID"}

        if not self.doctor_slots[doctor_id][slot_id]:
            return -1, {"error": "Target slot already booked (conflict)"}

        # Free old slot
        old_doctor = self.patients[patient_id]["booked_doctor"]
        old_slot = self.patients[patient_id]["booked_slot"]
        self.doctor_slots[old_doctor][old_slot] = True

        # Book new slot
        self.doctor_slots[doctor_id][slot_id] = False
        self.patients[patient_id].update({
            "booked_slot": slot_id,
            "booked_doctor": doctor_id
        })
        
        return 0.5, {"status": "success"}

    def _handle_cancel(self, patient_id) -> tuple:
        if patient_id not in self.patients or self.patients[patient_id]["status"] == "waiting":
             return -1, {"error": "Patient not booked"}

        # Free slot if booked
        if self.patients[patient_id]["status"] == "booked":
            doctor_id = self.patients[patient_id]["booked_doctor"]
            slot_id = self.patients[patient_id]["booked_slot"]
            self.doctor_slots[doctor_id][slot_id] = True

        self.patients[patient_id].update({
            "status": "waiting",
            "booked_slot": None,
            "booked_doctor": None
        })
        if patient_id not in self.waiting_queue:
            self.waiting_queue.append(patient_id)
            
        return -0.2, {"status": "cancelled"}

    def _handle_waitlist(self, patient_id) -> tuple:
        if patient_id not in self.patients or self.patients[patient_id]["status"] != "waiting":
            return -1, {"error": "Patient not in waiting queue"}
        
        return 0.3, {"status": "waitlisted"}
