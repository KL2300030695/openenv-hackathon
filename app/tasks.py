from typing import List, Dict, Any

class Task:
    def __init__(self, name: str, difficulty: str, description: str):
        self.name = name
        self.difficulty = difficulty
        self.description = description

class TaskGrader:
    """
    Grader for Healthcare Scheduling Tasks.
    Returns a score between 0.0 and 1.0.
    """
    def __init__(self, env):
        self.env = env

    def grade_task_1(self) -> float:
        """
        Task 1 (Easy): successfully book an appointment when slots are available.
        Score based on number of waiting patients booked.
        """
        booked_count = sum(1 for p in self.env.patients if p["status"] == "booked")
        total_patients = len(self.env.patients)
        return min(1.0, booked_count / (total_patients * 0.5))  # 50% booking is enough for full score here

    def grade_task_2(self) -> float:
        """
        Task 2 (Medium): Handle scheduling conflicts and rescheduling.
        Score based on rescheduling success and slot utilization.
        """
        # We can track successful reschedules in the info dict if we want,
        # but for simplicity, let's use efficiency.
        booked_count = sum(1 for p in self.env.patients if p["status"] == "booked")
        # In Task 2, we expect some rescheduling to happen.
        return min(1.0, booked_count / (len(self.env.patients) * 0.7))

    def grade_task_3(self) -> float:
        """
        Task 3 (Hard): Optimize scheduling for multiple patients with different priorities.
        Score based on priority completion. Priority 1 should be booked FIRST.
        """
        p1_total = sum(1 for p in self.env.patients if p["priority"] == 1)
        p1_booked = sum(1 for p in self.env.patients if p["priority"] == 1 and p["status"] == "booked")
        
        p2_total = sum(1 for p in self.env.patients if p["priority"] == 2)
        p2_booked = sum(1 for p in self.env.patients if p["priority"] == 2 and p["status"] == "booked")
        
        if p1_total == 0:
            p1_score = 1.0
        else:
            p1_score = p1_booked / p1_total

        if p2_total == 0:
            p2_score = 1.0
        else:
            p2_score = p2_booked / p2_total
            
        return (p1_score * 0.7) + (p2_score * 0.3)

TASKS = [
    Task("Task 1", "Easy", "Successfully book an appointment when slots are available."),
    Task("Task 2", "Medium", "Handle scheduling conflicts and rescheduling."),
    Task("Task 3", "Hard", "Optimize scheduling for multiple patients with different priorities.")
]
