"""
Task graders for Healthcare Scheduling Environment.

OpenEnv expects graders to be standalone callable functions that can be
imported via the `module:function` path format.  Each grader:
  - Creates / receives an environment instance
  - Evaluates the task
  - Returns a float score strictly in (0, 1)  — never 0.0 or 1.0
"""

from typing import Any, Dict, List


# ── Helper ───────────────────────────────────────────────────────────
def _clamp(score: float) -> float:
    """Clamp score to strictly (0, 1) — never exactly 0.0 or 1.0."""
    if score != score:  # NaN check
        return 0.5
    return min(max(float(score), 0.01), 0.99)


# ── Standalone grader functions (referenced by openenv.yaml) ─────────

def grade_task_1(env) -> float:
    """
    Task 1 (Easy): Successfully book appointments when slots are available.
    Score based on fraction of patients booked.
    """
    if not env.patients:
        return 0.5
    booked_count = sum(1 for p in env.patients.values() if p.get("status") == "booked")
    total_patients = len(env.patients)
    if total_patients == 0:
        return 0.5
    raw_score = booked_count / (total_patients * 0.5)
    return _clamp(raw_score)


def grade_task_2(env) -> float:
    """
    Task 2 (Medium): Handle scheduling conflicts and rescheduling.
    Score based on rescheduling success and slot utilization.
    """
    if not env.patients:
        return 0.5
    booked_count = sum(1 for p in env.patients.values() if p.get("status") == "booked")
    total_patients = len(env.patients)
    if total_patients == 0:
        return 0.5
    raw_score = booked_count / (total_patients * 0.7)
    return _clamp(raw_score)


def grade_task_3(env) -> float:
    """
    Task 3 (Hard): Optimize scheduling for multiple patients with priorities.
    Score weighted toward priority-1 patients being booked first.
    """
    if not env.patients:
        return 0.5
    p1_total = sum(1 for p in env.patients.values() if p.get("priority") == 1)
    p1_booked = sum(1 for p in env.patients.values() if p.get("priority") == 1 and p.get("status") == "booked")

    p2_total = sum(1 for p in env.patients.values() if p.get("priority") == 2)
    p2_booked = sum(1 for p in env.patients.values() if p.get("priority") == 2 and p.get("status") == "booked")

    p1_score = (p1_booked / p1_total) if p1_total > 0 else 0.5
    p2_score = (p2_booked / p2_total) if p2_total > 0 else 0.5

    raw_score = (p1_score * 0.7) + (p2_score * 0.3)
    return _clamp(raw_score)


# ── Task metadata ────────────────────────────────────────────────────

class Task:
    def __init__(self, name: str, difficulty: str, description: str):
        self.name = name
        self.difficulty = difficulty
        self.description = description


TASKS = [
    Task("Task 1", "easy", "Successfully book an appointment when slots are available."),
    Task("Task 2", "medium", "Handle scheduling conflicts and rescheduling."),
    Task("Task 3", "hard", "Optimize scheduling for multiple patients with different priorities."),
]


# ── Backward-compatible class wrapper (used by inference.py) ─────────

class TaskGrader:
    """Class wrapper for backward compat with inference.py."""
    def __init__(self, env):
        self.env = env

    def grade_task_1(self) -> float:
        return grade_task_1(self.env)

    def grade_task_2(self) -> float:
        return grade_task_2(self.env)

    def grade_task_3(self) -> float:
        return grade_task_3(self.env)
