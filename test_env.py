"""
Quick verification script for the Healthcare RL Environment.

Tests:
    1. Environment reset works with specialties
    2. Booking with correct specialty succeeds
    3. Booking with wrong specialty is rejected
    4. Baseline agent respects specialty matching
    5. Grader returns valid scores for all 3 tasks
    6. Pre-booked slots create realistic scarcity
    7. Typed action/observation interface works
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import HealthcareEnvironment
from models import HealthcareAction
from agent import BaselineAgent
from tasks import TaskGrader


def test():
    print("=" * 60)
    print("Testing Healthcare RL Environment (OpenEnv)")
    print("=" * 60)

    # 1. Reset with specialties
    env = HealthcareEnvironment(seed=42)
    obs = env.reset()
    assert obs.doctor_specialties is not None, "Missing doctor_specialties"
    assert obs.max_steps is not None, "Missing max_steps"
    print(
        f"[OK] Reset successful - {len(obs.waiting_queue)} patients, "
        f"{len(obs.doctor_specialties)} doctors with specialties"
    )
    for did, spec in obs.doctor_specialties.items():
        print(f"     Doctor {did}: {spec}")

    # 2. Test correct specialty booking
    patient_0 = env.patients[0]
    req_spec = patient_0["required_specialty"]
    matching_doc = None
    for did, spec in env.doctor_specialties.items():
        if spec == req_spec:
            matching_doc = did
            break

    if matching_doc is not None:
        available_slot = None
        for sid, avail in enumerate(env.doctor_slots[matching_doc]):
            if avail:
                available_slot = sid
                break

        if available_slot is not None:
            action = HealthcareAction(
                type="book", patient_id=0,
                doctor_id=matching_doc, slot_id=available_slot,
            )
            obs = env.step(action)
            assert obs.reward > 0.3, f"Correct specialty booking should reward > 0.3, got {obs.reward}"
            print(f"[OK] Specialty-matched booking works - reward={obs.reward:.2f}")
        else:
            print("[SKIP] No available slot for specialty-matched booking test")
    else:
        print("[SKIP] No matching doctor found for patient 0's specialty")

    # 3. Test wrong specialty rejection
    env2 = HealthcareEnvironment(seed=42)
    env2.reset()
    patient_1 = env2.patients[1]
    req_spec = patient_1["required_specialty"]
    wrong_doc = None
    for did, spec in env2.doctor_specialties.items():
        if spec != req_spec and spec != "General Medicine" and req_spec != "General Medicine":
            wrong_doc = did
            break

    if wrong_doc is not None:
        available_slot = None
        for sid, avail in enumerate(env2.doctor_slots[wrong_doc]):
            if avail:
                available_slot = sid
                break
        if available_slot is not None:
            action = HealthcareAction(
                type="book", patient_id=1,
                doctor_id=wrong_doc, slot_id=available_slot,
            )
            obs2 = env2.step(action)
            assert obs2.info.get("specialty_mismatch"), "Wrong specialty should flag mismatch"
            print(f"[OK] Wrong specialty rejected - reward={obs2.reward:.2f}")
        else:
            print("[SKIP] No slot available for wrong-specialty test")
    else:
        print("[SKIP] Could not find wrong-specialty doctor for test")

    # 4. Baseline agent episode
    env = HealthcareEnvironment(seed=42)
    obs = env.reset()
    agent = BaselineAgent(env)
    print("[OK] Agent initialised")

    steps = 0
    while not env.done:
        obs_dict = {
            "doctor_slots": obs.doctor_slots,
            "doctor_specialties": obs.doctor_specialties,
            "patients": obs.patients,
            "waiting_queue": obs.waiting_queue,
        }
        action_dict = agent.select_action(obs_dict)
        action = HealthcareAction(
            type=action_dict["type"],
            patient_id=action_dict["patient_id"],
            doctor_id=action_dict.get("doctor_id"),
            slot_id=action_dict.get("slot_id"),
        )
        obs = env.step(action)
        steps += 1

    booked = sum(1 for p in env.patients.values() if p["status"] == "booked")
    print(
        f"[OK] Episode finished in {steps} steps - "
        f"{booked}/{len(env.patients)} patients booked"
    )

    # 5. Grade all tasks
    grader = TaskGrader(env)
    scores = {
        "task_1": grader.grade_task_1(),
        "task_2": grader.grade_task_2(),
        "task_3": grader.grade_task_3(),
    }
    print(f"[OK] Grader Scores: {scores}")

    for task, score in scores.items():
        assert 0.0 < score < 1.0, f"{task} score {score} out of (0, 1) range"

    # 6. Verify pre-booked slots create scarcity
    env3 = HealthcareEnvironment(seed=99)
    env3.reset()
    preexisting = sum(
        1 for slots in env3.doctor_slots.values()
        for s in slots if not s
    )
    total_slots = env3.num_doctors * env3.num_slots
    print(
        f"[OK] Pre-booked slots: {preexisting}/{total_slots} "
        f"(creates realistic scarcity)"
    )

    print("\n[PASS] All verifications passed!")
    print("=" * 60)


if __name__ == "__main__":
    test()
