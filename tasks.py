from typing import List, Dict, Any

class Task:
    def __init__(self, name: str, difficulty: str, description: str):
        self.name = name
        self.difficulty = difficulty
        self.description = description

class TaskGrader:
    """
    Grader for Healthcare Scheduling Tasks.
    
    IMPORTANT: OpenEnv requirements specify that scores MUST be strictly 
    between 0 and 1 (not 0.0 and not 1.0). We use 0.01 and 0.99 as bounds.
    """
    def __init__(self, env):
        self.env = env

    def _clamp(self, score: float) -> float:
        """Clamp score to strictly (0, 1) range — never exactly 0.0 or 1.0."""
        if score != score:  # NaN check
            return 0.5
        return min(max(float(score), 0.01), 0.99)

    def grade_task_1(self) -> float:
        """
        Task 1 (Easy): successfully book an appointment when slots are available.
        Score based on number of waiting patients booked.
        """
        if not self.env.patients:
            return 0.5  # No patients = default mid-range score
        booked_count = sum(1 for p in self.env.patients.values() if p.get("status") == "booked")
        total_patients = len(self.env.patients)
        if total_patients == 0:
            return 0.5
        raw_score = booked_count / (total_patients * 0.5)
        return self._clamp(raw_score)

    def grade_task_2(self) -> float:
        """
        Task 2 (Medium): Handle scheduling conflicts and rescheduling.
        Score based on rescheduling success and slot utilization.
        """
        if not self.env.patients:
            return 0.5
        booked_count = sum(1 for p in self.env.patients.values() if p.get("status") == "booked")
        total_patients = len(self.env.patients)
        if total_patients == 0:
            return 0.5
        raw_score = booked_count / (total_patients * 0.7)
        return self._clamp(raw_score)

    def grade_task_3(self) -> float:
        """
        Task 3 (Hard): Optimize scheduling for multiple patients with different priorities.
        Score based on priority completion. Priority 1 should be booked FIRST.
        """
        if not self.env.patients:
            return 0.5
        p1_total = sum(1 for p in self.env.patients.values() if p.get("priority") == 1)
        p1_booked = sum(1 for p in self.env.patients.values() if p.get("priority") == 1 and p.get("status") == "booked")
        
        p2_total = sum(1 for p in self.env.patients.values() if p.get("priority") == 2)
        p2_booked = sum(1 for p in self.env.patients.values() if p.get("priority") == 2 and p.get("status") == "booked")
        
        if p1_total == 0:
            p1_score = 0.5
        else:
            p1_score = p1_booked / p1_total

        if p2_total == 0:
            p2_score = 0.5
        else:
            p2_score = p2_booked / p2_total
            
        raw_score = (p1_score * 0.7) + (p2_score * 0.3)
        return self._clamp(raw_score)

TASKS = [
    Task("Task 1", "Easy", "Successfully book an appointment when slots are available."),
    Task("Task 2", "Medium", "Handle scheduling conflicts and rescheduling."),
    Task("Task 3", "Hard", "Optimize scheduling for multiple patients with different priorities.")
]
