from typing import Dict, Any, List

class BaselineAgent:
    """
    Rule-based agent for appointment scheduling.
    Strategy: 
    1. Iterate through waiting queue.
    2. Sort patients by priority.
    3. Book the patient's preferred doctor if a slot is available.
    4. If not available, book any available slot with any doctor.
    5. If no slots, waitlist.
    """
    def __init__(self, env):
        self.env = env

    def select_action(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        waiting_queue = obs.get("waiting_queue", [])
        if not waiting_queue:
            return {"type": "waitlist", "patient_id": -1}

        patients = obs.get("patients", [])
        doctor_slots = obs.get("doctor_slots", [])
        
        # Sort waiting patients by priority (1 is highest)
        waiting_patients = [patients[pid] for pid in waiting_queue]
        waiting_patients.sort(key=lambda x: x["priority"])
        
        for patient in waiting_patients:
            pid = patient["id"]
            preferred_doctor = patient["preferred_doctor"]
            
            # 1. Try preferred doctor
            if 0 <= preferred_doctor < len(doctor_slots):
                for sid, available in enumerate(doctor_slots[preferred_doctor]):
                    if available:
                        return {
                            "type": "book",
                            "patient_id": pid,
                            "doctor_id": preferred_doctor,
                            "slot_id": sid
                        }
            
            # 2. Try any other doctor
            for did, slots in enumerate(doctor_slots):
                if did == preferred_doctor:
                    continue
                for sid, available in enumerate(slots):
                    if available:
                        return {
                            "type": "book",
                            "patient_id": pid,
                            "doctor_id": did,
                            "slot_id": sid
                        }

        # 3. No slots available, waitlist the first patient in queue
        return {"type": "waitlist", "patient_id": waiting_queue[0]}
