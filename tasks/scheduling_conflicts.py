"""
Task 2 (Medium): Scheduling Conflict Management Grader.

Evaluates how well the agent handles scheduling conflicts:
double-booking prevention, recovery after errors, avoiding repeated
mistakes, and specialty compliance.

Scoring breakdown:
  - 35%: Booking success rate despite constraints
  - 30%: Conflict recovery (successful bookings after encountering errors)
  - 20%: No repeated identical errors (agent learns from mistakes)
  - 15%: Specialty compliance (correct specialty matching)
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
    """Grade task 2: conflict management and error recovery."""
    if env is None:
        from env import HealthcareEnvironment
        env = HealthcareEnvironment(seed=123)

    if not env.patients:
        return 0.5

    total_patients = len(env.patients)
    booked_count = sum(
        1 for p in env.patients.values()
        if p.get("status") == "booked"
    )
    history = getattr(env, "_action_history", [])
    total_actions = len(history)

    if total_actions == 0:
        return 0.5

    # ── Metric 1 (35%): Booking rate ──
    booking_rate = booked_count / total_patients

    # ── Metric 2 (30%): Conflict recovery ──
    # After encountering an error, does the agent successfully act within
    # the next 3 steps?
    error_indices = [
        i for i, a in enumerate(history)
        if a.get("info", {}).get("error") is not None
    ]

    recoveries = 0
    for ei in error_indices:
        for j in range(ei + 1, min(ei + 4, total_actions)):
            if history[j].get("info", {}).get("status") == "success":
                recoveries += 1
                break

    recovery_rate = (
        recoveries / len(error_indices)
        if error_indices
        else 1.0  # no errors = perfect recovery
    )

    # ── Metric 3 (20%): No repeated identical errors ──
    # An agent that keeps trying the same bad action is poor at conflict mgmt
    error_signatures = []
    for a in history:
        if a.get("info", {}).get("error"):
            sig = (a.get("type"), a.get("doctor_id"), a.get("slot_id"))
            error_signatures.append(sig)

    unique_errors = len(set(error_signatures))
    total_errors = len(error_signatures)
    repeat_avoidance = (
        unique_errors / total_errors
        if total_errors > 0
        else 1.0  # no errors = perfect
    )

    # ── Metric 4 (15%): Specialty compliance ──
    specialty_mismatches = sum(
        1 for a in history
        if a.get("info", {}).get("specialty_mismatch")
    )
    specialty_compliance = 1.0 - (specialty_mismatches / total_actions)

    raw_score = (
        0.35 * booking_rate
        + 0.30 * recovery_rate
        + 0.20 * repeat_avoidance
        + 0.15 * specialty_compliance
    )
    return _clamp(raw_score)
