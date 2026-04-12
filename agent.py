"""
Rule-based baseline agent for the Healthcare Scheduling environment.

Strategy:
    1. Sort waiting patients by priority (1 = highest urgency).
    2. Match patient's required_specialty to a compatible doctor.
    3. Within matching doctors, try preferred doctor first.
    4. Match time preference (morning slots 0-3, afternoon slots 4-7).
    5. If no matching slots, try any compatible doctor.
    6. If all else fails, waitlist the patient.
"""

from typing import Any, Dict, List


class BaselineAgent:
    """
    Rule-based agent for appointment scheduling.

    Handles specialty matching, priority ordering, and time preferences.
    Works with both the typed HealthcareEnvironment and dict-based
    observation format.
    """

    def __init__(self, env=None):
        self.env = env

    def select_action(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select the best scheduling action given the current observation.

        Args:
            obs: Dict with doctor_slots, doctor_specialties, patients,
                 waiting_queue.

        Returns:
            Action dict with type, patient_id, doctor_id, slot_id.
        """
        waiting_queue = obs.get("waiting_queue", [])
        if not waiting_queue:
            return {"type": "waitlist", "patient_id": -1}

        patients = obs.get("patients", {})
        doctor_slots = obs.get("doctor_slots", {})
        doctor_specialties = obs.get("doctor_specialties", {})

        # Sort waiting patients by priority (1 is highest)
        waiting_patients = [
            patients[pid] for pid in waiting_queue if pid in patients
        ]
        waiting_patients.sort(key=lambda x: x.get("priority", 3))

        for patient in waiting_patients:
            pid = patient["id"]
            preferred_doctor = patient.get("preferred_doctor", 0)
            required_specialty = patient.get("required_specialty", "General Medicine")
            time_preference = patient.get("time_preference", "morning")

            # ── Find doctors with compatible specialty ──
            matching_docs = []
            for did, spec in doctor_specialties.items():
                if (spec == required_specialty
                        or spec == "General Medicine"
                        or required_specialty == "General Medicine"):
                    matching_docs.append(did)

            # Fallback: all doctors
            if not matching_docs:
                matching_docs = list(doctor_slots.keys())

            # Sort: preferred doctor first
            pref_key = str(preferred_doctor)
            matching_docs.sort(key=lambda d: (0 if str(d) == pref_key else 1))

            # ── Determine slot preference ──
            num_slots = (
                max(len(slots) for slots in doctor_slots.values())
                if doctor_slots else 8
            )
            half = num_slots // 2
            if time_preference == "morning":
                pref_slots = list(range(0, half))
                other_slots = list(range(half, num_slots))
            else:
                pref_slots = list(range(half, num_slots))
                other_slots = list(range(0, half))

            for did in matching_docs:
                did_key = str(did) if str(did) in doctor_slots else did
                if did_key not in doctor_slots:
                    continue
                slots = doctor_slots[did_key]
                did_int = int(did)

                # Try preferred time slots first
                for sid in pref_slots:
                    if sid < len(slots) and slots[sid]:
                        return {
                            "type": "book",
                            "patient_id": pid,
                            "doctor_id": did_int,
                            "slot_id": sid,
                        }

                # Then try other time slots
                for sid in other_slots:
                    if sid < len(slots) and slots[sid]:
                        return {
                            "type": "book",
                            "patient_id": pid,
                            "doctor_id": did_int,
                            "slot_id": sid,
                        }

        # No slots available anywhere, waitlist the first patient
        first_pid = waiting_queue[0]
        pid_int = int(first_pid) if isinstance(first_pid, str) else first_pid
        return {"type": "waitlist", "patient_id": pid_int}
