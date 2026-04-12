"""
Task 1 (Easy): Basic Appointment Booking Grader.

Evaluates the agent's ability to successfully book appointments
for waiting patients into available slots.

Scoring breakdown:
  - 50%: Booking success rate (patients booked / total patients)
  - 25%: Action efficiency (successful actions / total actions taken)
  - 25%: Error avoidance (1 - error_rate)
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _clamp(score: float) -> float:
    """Clamp score to [0.1, 0.9] for validator compliance."""
    if score != score:  # NaN check
        return 0.5
    return min(max(float(score), 0.1), 0.9)


def grade(env=None) -> float:
    """Grade task 1: basic appointment booking."""
    if env is None:
        from env import HealthcareEnvironment
        env = HealthcareEnvironment(seed=42)

    if not env.patients:
        return 0.5

    total_patients = len(env.patients)
    booked_count = sum(
        1 for p in env.patients.values()
        if p.get("status") == "booked"
    )

    # ── Metric 1 (50%): Booking success rate ──
    booking_rate = booked_count / total_patients if total_patients > 0 else 0.0

    # ── Metric 2 (25%): Action efficiency ──
    history = getattr(env, "_action_history", [])
    total_actions = len(history)
    successful_actions = sum(
        1 for a in history
        if a.get("info", {}).get("status") == "success"
    )
    efficiency = successful_actions / total_actions if total_actions > 0 else 0.0

    # ── Metric 3 (25%): Error avoidance ──
    error_actions = sum(
        1 for a in history
        if a.get("info", {}).get("error") is not None
    )
    error_rate = error_actions / total_actions if total_actions > 0 else 0.0
    error_avoidance = 1.0 - error_rate

    raw_score = (
        0.50 * booking_rate
        + 0.25 * efficiency
        + 0.25 * error_avoidance
    )
    return _clamp(raw_score)
