"""
Rule-based baseline agent for the Healthcare Scheduling environment.

Strategy:
    1. Sort waiting patients by priority (1 = highest).
    2. For each patient, try their preferred doctor first.
    3. If preferred doctor has no free slots, try any other doctor.
    4. If no slots at all, waitlist the patient.
"""

from typing import Any, Dict, List


class BaselineAgent:
    """
    Rule-based agent for appointment scheduling.

    Works with both the typed HealthcareEnvironment and the dict-based
    observation format.
    """

    def __init__(self, env=None):
        self.env = env

    def select_action(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select the best action given the current observation.

        Args:
            obs: Dict with doctor_slots, patients, waiting_queue.

        Returns:
            Action dict with type, patient_id, doctor_id, slot_id.
        """
        waiting_queue = obs.get("waiting_queue", [])
        if not waiting_queue:
            return {"type": "waitlist", "patient_id": -1}

        patients = obs.get("patients", {})
        doctor_slots = obs.get("doctor_slots", {})

        # Sort waiting patients by priority (1 is highest)
        waiting_patients = [patients[pid] for pid in waiting_queue if pid in patients]
        waiting_patients.sort(key=lambda x: x["priority"])

        for patient in waiting_patients:
            pid = patient["id"]
            preferred_doctor = patient["preferred_doctor"]

            # Convert preferred_doctor to string key if needed
            pref_key = str(preferred_doctor) if str(preferred_doctor) in doctor_slots else preferred_doctor

            # 1. Try preferred doctor
            if pref_key in doctor_slots:
                slots = doctor_slots[pref_key]
                for sid, available in enumerate(slots):
                    if available:
                        return {
                            "type": "book",
                            "patient_id": pid,
                            "doctor_id": preferred_doctor,
                            "slot_id": sid,
                        }

            # 2. Try any other doctor
            for did, slots in doctor_slots.items():
                did_int = int(did) if isinstance(did, str) else did
                if did_int == preferred_doctor:
                    continue
                for sid, available in enumerate(slots):
                    if available:
                        return {
                            "type": "book",
                            "patient_id": pid,
                            "doctor_id": did_int,
                            "slot_id": sid,
                        }

        # 3. No slots available, waitlist the first patient
        first_pid = waiting_queue[0]
        pid_int = int(first_pid) if isinstance(first_pid, str) else first_pid
        return {"type": "waitlist", "patient_id": pid_int}
