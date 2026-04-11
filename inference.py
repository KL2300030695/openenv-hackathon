"""
Inference Script - Healthcare Appointment Scheduling
=====================================================
MANDATORY
- Environment variables:
    API_BASE_URL   The API endpoint for the LLM (must have default).
    MODEL_NAME     The model identifier to use for inference (must have default).
    HF_TOKEN       Your Hugging Face / API key (mandatory, no default).

- Uses OpenAI Client for all LLM calls.

STDOUT FORMAT (exact official spec):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

import os
import sys
import json
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ── Import environment components ────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import HealthcareEnvironment
from models import HealthcareAction
from tasks import TaskGrader

# ── Environment variables (per submission guidelines) ────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

BENCHMARK = "healthcare_scheduling"
MAX_STEPS = 20

# Initialize OpenAI client
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


# ── Structured logging helpers (EXACT official spec format) ─────────
def log_start(task: str, env: str, model: str) -> None:
    """Emit [START] line at episode begin."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    """Emit [STEP] line immediately after env.step() returns."""
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    """Emit [END] line after episode ends."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ── LLM action selection (with memory of past failures) ─────────────
def get_action_from_llm(
    obs: Dict[str, Any],
    action_history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Asks the LLM to choose an action based on the current environment state.
    Includes history of past actions so the LLM avoids repeating mistakes.
    """
    # Build a concise summary of past actions for the LLM
    history_lines = ""
    if action_history:
        history_lines = "\n\nPrevious actions taken this episode:\n"
        for h in action_history:
            status = "SUCCESS" if h.get("reward", 0) > 0 else "FAILED"
            err = h.get("error", "")
            err_str = f" - Error: {err}" if err else ""
            history_lines += f"  - {h['action_str']} => {status} (reward={h['reward']:.2f}){err_str}\n"
        history_lines += "\nDO NOT repeat any failed actions. Choose a DIFFERENT doctor_id and slot_id.\n"

    # Build a compact view of which slots are actually available
    available_slots = []
    doctor_slots = obs.get("doctor_slots", {})
    for did, slots in doctor_slots.items():
        for sid, avail in enumerate(slots):
            if avail:
                available_slots.append(f"doctor={did}, slot={sid}")

    available_str = "\n".join(f"  - {s}" for s in available_slots) if available_slots else "  NONE - all slots are booked!"

    # Build compact patient info for waiting patients only
    waiting_queue = obs.get("waiting_queue", [])
    patients = obs.get("patients", {})
    waiting_info = []
    for pid in waiting_queue:
        p = patients.get(str(pid), patients.get(pid, {}))
        waiting_info.append(
            f"  - Patient {pid}: priority={p.get('priority','?')}, preferred_doctor={p.get('preferred_doctor','?')}"
        )
    waiting_str = "\n".join(waiting_info) if waiting_info else "  NONE - all patients are scheduled!"

    prompt = f"""You are an expert healthcare appointment scheduler. Book appointments efficiently.

AVAILABLE SLOTS (True = free):
{available_str}

WAITING PATIENTS (need appointments):
{waiting_str}
{history_lines}
RULES:
- ONLY book slots that are available (True).
- Prioritize patients with lower priority number (1=highest priority).
- Match patients with their preferred_doctor when possible for bonus reward.
- If no slots are available for a patient, use "waitlist".
- NEVER book a slot that is False (already taken).

Respond with ONLY a JSON object:
{{"type": "book", "patient_id": <int>, "doctor_id": <int>, "slot_id": <int>}}

Your action:"""

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
    except Exception:
        # Silently log as fallback; do not print debug info to STDOUT
        # Smart fallback: try to find an available slot for a waiting patient
        return _fallback_action(obs)


def _fallback_action(obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rule-based fallback when LLM fails. Picks the best available action.
    """
    waiting_q = obs.get("waiting_queue", [])
    if not waiting_q:
        return {"type": "waitlist", "patient_id": 0, "doctor_id": 0, "slot_id": 0}

    patients = obs.get("patients", {})
    doctor_slots = obs.get("doctor_slots", {})

    # Sort by priority (1 = highest)
    sorted_patients = []
    for pid in waiting_q:
        p = patients.get(str(pid), patients.get(pid, {}))
        sorted_patients.append((pid, p.get("priority", 3), p.get("preferred_doctor", 0)))
    sorted_patients.sort(key=lambda x: x[1])

    for pid, priority, pref_doc in sorted_patients:
        pid_int = int(pid)
        # Try preferred doctor first
        pref_key = str(pref_doc)
        if pref_key in doctor_slots:
            for sid, avail in enumerate(doctor_slots[pref_key]):
                if avail:
                    return {"type": "book", "patient_id": pid_int, "doctor_id": pref_doc, "slot_id": sid}
        # Try any doctor
        for did, slots in doctor_slots.items():
            did_int = int(did)
            for sid, avail in enumerate(slots):
                if avail:
                    return {"type": "book", "patient_id": pid_int, "doctor_id": did_int, "slot_id": sid}

    # No slots left
    p_id = int(waiting_q[0])
    return {"type": "waitlist", "patient_id": p_id, "doctor_id": 0, "slot_id": 0}


# ── Format action as a compact string for [STEP] logging ────────────
def action_to_str(action: Dict[str, Any]) -> str:
    """Convert action dict to a compact string like book(patient=0,doctor=1,slot=2)"""
    a_type = action.get("type", "unknown")
    p = action.get("patient_id", "?")
    d = action.get("doctor_id", "?")
    s = action.get("slot_id", "?")
    return f"{a_type}(patient={p},doctor={d},slot={s})"


# ── Main inference loop ─────────────────────────────────────────────
def run_inference():
    # Task definitions
    task_defs = [
        {"name": "book_appointment", "grader": "grade_task_1"},
        {"name": "scheduling_conflicts", "grader": "grade_task_2"},
        {"name": "priority_scheduling", "grader": "grade_task_3"},
    ]

    env = HealthcareEnvironment(seed=42)
    obs = env.reset()

    # Convert observation to dict for the LLM prompt
    obs_dict = {
        "doctor_slots": obs.doctor_slots,
        "patients": obs.patients,
        "waiting_queue": obs.waiting_queue,
        "current_step": obs.current_step,
    }

    # ── Grading functions for the final [END] line ──
    grader = TaskGrader(env)

    # ── Reporting Setup ──
    # Using the primary task name as the default reporter
    primary_task = "healthcare_appointment_scheduling"
    
    try:
        # Emit [START] at the very beginning
        log_start(task=primary_task, env=BENCHMARK, model=MODEL_NAME)

        # ── Run the agent loop ──
        steps_taken = 0
        rewards: List[float] = []
        action_history: List[Dict[str, Any]] = []

        while not env.done and steps_taken < MAX_STEPS:
            action_dict = get_action_from_llm(obs_dict, action_history)

            # Step using typed action
            action = HealthcareAction(
                type=action_dict.get("type", "waitlist"),
                patient_id=action_dict.get("patient_id", 0),
                doctor_id=action_dict.get("doctor_id"),
                slot_id=action_dict.get("slot_id"),
            )
            obs = env.step(action)

            # Update observation for next step
            obs_dict = {
                "doctor_slots": obs.doctor_slots,
                "patients": obs.patients,
                "waiting_queue": obs.waiting_queue,
                "current_step": obs.current_step,
            }

            step_error = obs.info.get("error") if obs.info else "null"
            step_action_type = action_dict.get("type", "waitlist")
            
            steps_taken += 1
            rewards.append(obs.reward)

            # Emit [STEP] line IMMEDIATELY after env.step() returns
            log_step(
                step=steps_taken,
                action=action_to_str(action_dict),
                reward=obs.reward,
                done=obs.done,
                error=step_error,
            )

            # Track history for LLM
            action_history.append({
                "action_str": action_to_str(action_dict),
                "reward": obs.reward,
                "error": step_error if step_error != "null" else None,
            })

        # Calculate final success (based on Task 1 logic or similar)
        # We'll use the average of available graders or just the primary one
        task_score = grader.grade_task_1()
        success = task_score > 0.5

        # Emit [END] line after episode ends
        log_end(success=success, steps=steps_taken, rewards=rewards)

    except Exception:
        # Fallback to ensure [END] is emitted even on failure
        log_end(success=False, steps=steps_taken if 'steps_taken' in locals() else 0, rewards=rewards if 'rewards' in locals() else [0.0])
    finally:
        sys.exit(0)


if __name__ == "__main__":
    run_inference()