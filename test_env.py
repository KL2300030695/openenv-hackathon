"""
Quick verification script for the Healthcare RL Environment.

Tests:
    1. Environment reset works
    2. Baseline agent can play a full episode
    3. Grader returns positive scores for all 3 tasks
    4. Typed action/observation interface works
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.env import HealthcareEnvironment
from app.models import HealthcareAction
from app.agent import BaselineAgent
from app.tasks import TaskGrader


def test():
    print("=" * 60)
    print("Testing Healthcare RL Environment (OpenEnv)")
    print("=" * 60)

    # 1. Reset
    env = HealthcareEnvironment(seed=42)
    obs = env.reset()
    print(f"[OK] Reset successful - {len(obs.waiting_queue)} patients waiting")

    # 2. Test typed step
    action = HealthcareAction(type="book", patient_id=0, doctor_id=0, slot_id=0)
    obs = env.step(action)
    print(f"[OK] Typed step works - reward={obs.reward:.2f}, done={obs.done}")

    # 3. Reset and run baseline agent
    env = HealthcareEnvironment(seed=42)
    obs = env.reset()
    agent = BaselineAgent(env)
    print("[OK] Agent initialized")

    steps = 0
    while not env.done:
        obs_dict = {
            "doctor_slots": obs.doctor_slots,
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

    print(f"[OK] Environment finished in {steps} steps")

    # 4. Grade
    grader = TaskGrader(env)
    scores = {
        "task_1": grader.grade_task_1(),
        "task_2": grader.grade_task_2(),
        "task_3": grader.grade_task_3(),
    }

    print(f"[OK] Grader Scores: {scores}")

    assert scores["task_1"] > 0, f"task_1 score should be > 0, got {scores['task_1']}"
    assert scores["task_2"] > 0, f"task_2 score should be > 0, got {scores['task_2']}"
    assert scores["task_3"] > 0, f"task_3 score should be > 0, got {scores['task_3']}"

    print("\n[PASS] All verifications passed!")
    print("=" * 60)


if __name__ == "__main__":
    test()
