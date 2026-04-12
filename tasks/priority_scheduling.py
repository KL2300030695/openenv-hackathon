"""
Task 3 (Hard): Priority-Based Scheduling Optimisation Grader.

Evaluates the agent's ability to prioritise high-urgency patients, match
specialties and preferred doctors, and optimise overall scheduling quality.

Scoring breakdown:
  - 35%: Priority ordering (P1 booked rate >= P2 >= P3)
  - 25%: High-priority completion (all priority-1 patients booked)
  - 20%: Preferred doctor match rate for booked patients
  - 20%: Overall quality (booking rate + time-preference matching)
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
    """Grade task 3: priority-based scheduling optimisation."""
    if env is None:
        from env import HealthcareEnvironment
        env = HealthcareEnvironment(seed=456)

    if not env.patients:
        return 0.5

    patients = env.patients

    # ── Group patients by priority level ──
    by_priority = {1: [], 2: [], 3: []}
    for p in patients.values():
        pri = p.get("priority", 3)
        if pri in by_priority:
            by_priority[pri].append(p)

    def booking_rate(patient_list):
        if not patient_list:
            return 1.0  # vacuously satisfied
        booked = sum(1 for p in patient_list if p.get("status") == "booked")
        return booked / len(patient_list)

    p1_rate = booking_rate(by_priority[1])
    p2_rate = booking_rate(by_priority[2])
    p3_rate = booking_rate(by_priority[3])

    # ── Metric 1 (35%): Priority ordering ──
    # Higher-priority patients should have equal or better booking rates
    ordering_score = 0.0
    if p1_rate >= p2_rate:
        ordering_score += 0.5
    if p2_rate >= p3_rate:
        ordering_score += 0.5
    # Bonus: perfect priority-1 completion
    if p1_rate == 1.0:
        ordering_score = min(ordering_score + 0.3, 1.0)

    # ── Metric 2 (25%): High-priority completion ──
    p1_completion = p1_rate

    # ── Metric 3 (20%): Preferred doctor match rate ──
    booked_patients = [
        p for p in patients.values() if p.get("status") == "booked"
    ]
    if booked_patients:
        pref_matches = sum(
            1 for p in booked_patients
            if p.get("booked_doctor") == p.get("preferred_doctor")
        )
        pref_match_rate = pref_matches / len(booked_patients)
    else:
        pref_match_rate = 0.0

    # ── Metric 4 (20%): Overall quality ──
    total_patients = len(patients)
    overall_booking_rate = (
        len(booked_patients) / total_patients if total_patients > 0 else 0.0
    )

    # Time-preference matching
    time_matches = 0
    num_slots = getattr(env, "num_slots", 8)
    for p in booked_patients:
        slot = p.get("booked_slot")
        time_pref = p.get("time_preference", "morning")
        if slot is not None:
            is_morning = slot < (num_slots // 2)
            if (time_pref == "morning" and is_morning) or \
               (time_pref == "afternoon" and not is_morning):
                time_matches += 1
    time_match_rate = (
        time_matches / len(booked_patients) if booked_patients else 0.0
    )

    quality_score = 0.60 * overall_booking_rate + 0.40 * time_match_rate

    raw_score = (
        0.35 * ordering_score
        + 0.25 * p1_completion
        + 0.20 * pref_match_rate
        + 0.20 * quality_score
    )
    return _clamp(raw_score)
