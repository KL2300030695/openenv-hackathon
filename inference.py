"""
Inference Script — Healthcare Appointment Scheduling
=====================================================
MANDATORY
- Environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- Defaults are set only for API_BASE_URL and MODEL_NAME.
- Uses OpenAI Client for all LLM calls.

STDOUT FORMAT
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
import sys
import json
from typing import List, Optional

from openai import OpenAI
from app.env import HealthcareEnv
from app.tasks import TaskGrader

# ── Environment variables (match sample exactly) ──────────────────────────
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

BENCHMARK = "healthcare-scheduling"
MAX_STEPS = 20

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


# ── Structured logging helpers (match sample exactly) ─────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM action selection ─────────────────────────────────────────────────
def get_action_from_llm(obs):
    """
    Asks the LLM to choose an action based on the current environment state.
    """
    prompt = f"""You are an expert healthcare administrator and scheduling assistant. 
Your goal is to book appointments efficiently while respecting doctor availability and patient priorities.

Current environment state:
```json
{json.dumps(obs, indent=2)}
```

Instructions:
1. Analyze which patients are in the 'waiting_queue'.
2. Check 'doctor_slots' to find available slots (True means available).
3. Select a patient and an available slot.
4. Respond ONLY with a valid JSON object.

JSON Keys:
- "type": "book", "reschedule", "cancel", or "waitlist"
- "patient_id": Integer ID of the patient
- "doctor_id": Integer ID of the doctor
- "slot_id": Integer ID of the time slot

Example Response:
{{"type": "book", "patient_id": 0, "doctor_id": 1, "slot_id": 2}}

Reply with only the JSON object:"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.1,
        )
        text = response.choices[0].message.content.strip()

        # Strip potential markdown code blocks
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:].strip()
            text = text.strip()

        return json.loads(text)
    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", flush=True)
        waiting_q = obs.get("waiting_queue", [])
        p_id = int(waiting_q[0]) if waiting_q else 0
        return {"type": "waitlist", "patient_id": p_id, "doctor_id": 0, "slot_id": 0}


# ── Format action as a compact string for [STEP] logging ─────────────────
def action_to_str(action: dict) -> str:
    """Convert action dict to a compact string like book(patient=0,doctor=1,slot=2)"""
    a_type = action.get("type", "unknown")
    p = action.get("patient_id", "?")
    d = action.get("doctor_id", "?")
    s = action.get("slot_id", "?")
    return f"{a_type}(patient={p},doctor={d},slot={s})"


# ── Main inference loop ──────────────────────────────────────────────────
def run_inference():
    # Task definitions from openenv.yaml
    task_defs = [
        {"name": "book_appointment", "grader": "grade_task_1"},
        {"name": "scheduling_conflicts", "grader": "grade_task_2"},
        {"name": "priority_scheduling", "grader": "grade_task_3"},
    ]

    env = HealthcareEnv()
    obs = env.reset()

    # ── Run the agent loop and collect step data ──
    steps_taken = 0
    rewards: List[float] = []
    actions: List[str] = []
    dones: List[bool] = []
    errors: List[Optional[str]] = []

    while not env.done and steps_taken < MAX_STEPS:
        action = get_action_from_llm(obs)
        obs, reward, done, info = env.step(action)

        steps_taken += 1
        rewards.append(reward)
        actions.append(action_to_str(action))
        dones.append(done)
        errors.append(info.get("error"))

    # ── Grade all tasks ──
    grader = TaskGrader(env)

    # ── Emit one structured [START]/[STEP]/[END] block per task ──
    for task_def in task_defs:
        task_name = task_def["name"]
        grade_fn = getattr(grader, task_def["grader"])
        task_score = min(max(grade_fn(), 0.0), 1.0)  # clamp to [0, 1]
        success = task_score > 0.0

        log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

        for i in range(steps_taken):
            log_step(
                step=i + 1,
                action=actions[i],
                reward=rewards[i],
                done=dones[i],
                error=errors[i],
            )

        log_end(success=success, steps=steps_taken, score=task_score, rewards=rewards)

    sys.exit(0)


if __name__ == "__main__":
    run_inference()