"""
Task 3 (Hard): Priority Scheduling Grader.
Optimize scheduling for patients with different priorities.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _clamp(score: float) -> float:
    """Clamp score to strictly (0, 1) with safe margins."""
    if score != score:
        return 0.5
    return min(max(float(score), 0.1), 0.9)


def grade(env=None) -> float:
    """Grade task 3: priority-based scheduling."""
    if env is None:
        from env import HealthcareEnvironment
        env = HealthcareEnvironment(seed=42)

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
