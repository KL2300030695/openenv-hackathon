"""
Task 2 (Medium): Scheduling Conflicts Grader.
Handle scheduling conflicts and rescheduling efficiently.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _clamp(score: float) -> float:
    """Clamp score to strictly (0, 1) with safe margins."""
    if score != score:
        return 0.5
    return min(max(float(score), 0.1), 0.9)


def grade(env=None) -> float:
    """Grade task 2: conflict management."""
    if env is None:
        from env import HealthcareEnvironment
        env = HealthcareEnvironment(seed=42)

    if not env.patients:
        return 0.5
    booked_count = sum(1 for p in env.patients.values() if p.get("status") == "booked")
    total_patients = len(env.patients)
    if total_patients == 0:
        return 0.5
    raw_score = booked_count / (total_patients * 0.7)
    return _clamp(raw_score)
