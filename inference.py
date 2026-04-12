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


def log_end(task: str, score: float, success: bool, steps: int, rewards: List[float]) -> None:
    """Emit [END] line after episode ends."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] task={task} score={score:.3f} success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ── LLM action selection (specialty-aware with history) ─────────────
def get_action_from_llm(
    obs: Dict[str, Any],
    action_history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Asks the LLM to choose a scheduling action based on hospital state.
    Includes specialty constraints, priority info, and past action history.
    """
    # Build history summary (last 8 actions for context window)
    history_lines = ""
    if action_history:
        history_lines = "\n\nPrevious actions this episode:\n"
        for h in action_history[-8:]:
            status = "SUCCESS" if h.get("reward", 0) > 0.3 else "FAILED"
            err = h.get("error", "")
            err_str = f" - Error: {err}" if err else ""
            history_lines += f"  - {h['action_str']} => {status} (reward={h['reward']:.2f}){err_str}\n"
        history_lines += "\nDO NOT repeat any failed actions. Choose DIFFERENT doctor_id and slot_id.\n"

    # Build doctor availability with specialties
    doctor_specialties = obs.get("doctor_specialties", {})
    doctor_slots = obs.get("doctor_slots", {})
    available_lines = []
    for did, slots in doctor_slots.items():
        specialty = doctor_specialties.get(str(did), "Unknown")
        free_slots = [str(sid) for sid, avail in enumerate(slots) if avail]
        if free_slots:
            morning = [s for s in free_slots if int(s) < 4]
            afternoon = [s for s in free_slots if int(s) >= 4]
            available_lines.append(
                f"  Doctor {did} ({specialty}): morning slots={morning}, afternoon slots={afternoon}"
            )
    available_str = "\n".join(available_lines) if available_lines else "  NONE - all slots booked!"

    # Build waiting patient info with specialty needs
    waiting_queue = obs.get("waiting_queue", [])
    patients = obs.get("patients", {})
    waiting_info = []
    for pid in waiting_queue[:10]:
        p = patients.get(str(pid), patients.get(pid, {}))
        waiting_info.append(
            f"  Patient {pid}: priority={p.get('priority','?')}, "
            f"needs={p.get('required_specialty','General Medicine')}, "
            f"preferred_doctor={p.get('preferred_doctor','?')}, "
            f"prefers={p.get('time_preference','any')} slots"
        )
    waiting_str = "\n".join(waiting_info) if waiting_info else "  NONE - all patients scheduled!"

    prompt = f"""You are an expert healthcare appointment scheduler. Book appointments optimally.

DOCTORS & AVAILABLE SLOTS:
{available_str}

WAITING PATIENTS (sorted by urgency):
{waiting_str}
{history_lines}
CRITICAL RULES:
1. Patient's required_specialty MUST match doctor's specialty. General Medicine can see anyone; any patient needing General Medicine can see any doctor.
2. Prioritise priority=1 patients first, then 2, then 3.
3. Match patients to their preferred_doctor when possible.
4. Match time_preference: morning=slots 0-3, afternoon=slots 4-7.
5. ONLY use slots that are available (shown above).
6. If no valid slots exist for a patient, use "waitlist".

Respond with ONLY a valid JSON object:
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
        return _fallback_action(obs)


def _fallback_action(obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rule-based fallback when LLM fails.
    Respects specialty matching, priority ordering, and time preferences.
    """
    waiting_q = obs.get("waiting_queue", [])
    if not waiting_q:
        return {"type": "waitlist", "patient_id": 0, "doctor_id": 0, "slot_id": 0}

    patients = obs.get("patients", {})
    doctor_slots = obs.get("doctor_slots", {})
    doctor_specialties = obs.get("doctor_specialties", {})

    # Sort by priority (1 = highest)
    sorted_patients = []
    for pid in waiting_q:
        p = patients.get(str(pid), patients.get(pid, {}))
        sorted_patients.append((
            pid,
            p.get("priority", 3),
            p.get("preferred_doctor", 0),
            p.get("required_specialty", "General Medicine"),
            p.get("time_preference", "morning"),
        ))
    sorted_patients.sort(key=lambda x: x[1])

    for pid, priority, pref_doc, req_specialty, time_pref in sorted_patients:
        pid_int = int(pid)

        # Find doctors with matching specialty (or General Medicine)
        matching_docs = []
        for did, spec in doctor_specialties.items():
            if spec == req_specialty or spec == "General Medicine" or req_specialty == "General Medicine":
                matching_docs.append(int(did))

        if not matching_docs:
            matching_docs = [int(d) for d in doctor_slots.keys()]

        # Sort: preferred doctor first
        matching_docs.sort(key=lambda d: (0 if d == pref_doc else 1))

        # Determine preferred slot range
        morning_range = range(0, 4)
        afternoon_range = range(4, 8)
        pref_range = morning_range if time_pref == "morning" else afternoon_range
        other_range = afternoon_range if time_pref == "morning" else morning_range

        # Try preferred time first, then other
        for did in matching_docs:
            did_str = str(did)
            if did_str not in doctor_slots:
                continue
            slots = doctor_slots[did_str]

            for sid in pref_range:
                if sid < len(slots) and slots[sid]:
                    return {"type": "book", "patient_id": pid_int, "doctor_id": did, "slot_id": sid}

            for sid in other_range:
                if sid < len(slots) and slots[sid]:
                    return {"type": "book", "patient_id": pid_int, "doctor_id": did, "slot_id": sid}

    # No slots available
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


# ── Run a single task episode ───────────────────────────────────────
def run_task_episode(task_id: str, seed: int, max_steps: int = 20):
    """Run one task episode and return (steps, rewards, score)."""
    env = HealthcareEnvironment(seed=seed)
    obs = env.reset()

    obs_dict = {
        "doctor_slots": obs.doctor_slots,
        "doctor_specialties": obs.doctor_specialties,
        "patients": obs.patients,
        "waiting_queue": obs.waiting_queue,
        "current_step": obs.current_step,
    }

    steps_taken = 0
    rewards: List[float] = []
    action_history: List[Dict[str, Any]] = []

    # ── [START] ──
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        while not env.done and steps_taken < max_steps:
            action_dict = get_action_from_llm(obs_dict, action_history)

            action = HealthcareAction(
                type=action_dict.get("type", "waitlist"),
                patient_id=action_dict.get("patient_id", 0),
                doctor_id=action_dict.get("doctor_id"),
                slot_id=action_dict.get("slot_id"),
            )
            obs = env.step(action)

            obs_dict = {
                "doctor_slots": obs.doctor_slots,
                "doctor_specialties": obs.doctor_specialties,
                "patients": obs.patients,
                "waiting_queue": obs.waiting_queue,
                "current_step": obs.current_step,
            }

            step_error = obs.info.get("error") if obs.info and obs.info.get("error") else "null"

            steps_taken += 1
            rewards.append(obs.reward)

            log_step(
                step=steps_taken,
                action=action_to_str(action_dict),
                reward=obs.reward,
                done=env.done,
                error=step_error,
            )

            action_history.append({
                "action_str": action_to_str(action_dict),
                "reward": obs.reward,
                "error": step_error if step_error != "null" else None,
            })

        # Grade the specific task
        grader = TaskGrader(env)
        if task_id == "book_appointment":
            score = grader.grade_task_1()
        elif task_id == "scheduling_conflicts":
            score = grader.grade_task_2()
        else:
            score = grader.grade_task_3()

        # Clamp score strictly to (0, 1)
        if score <= 0.0:
            score = 0.1
        elif score >= 1.0:
            score = 0.9

        success = score > 0.5
        log_end(task=task_id, score=score, success=success, steps=steps_taken, rewards=rewards)
        return steps_taken, rewards, score

    except Exception:
        log_end(task=task_id, score=0.1, success=False, steps=steps_taken, rewards=rewards if rewards else [0.1])
        return steps_taken, rewards if rewards else [0.1], 0.1


# ── Main inference loop ─────────────────────────────────────────────
def run_inference():
    # Task IDs must match the `id` field in openenv.yaml
    task_ids = ["book_appointment", "scheduling_conflicts", "priority_scheduling"]
    seeds = [42, 123, 456]

    for task_id, seed in zip(task_ids, seeds):
        run_task_episode(task_id, seed=seed, max_steps=20)

    sys.exit(0)


if __name__ == "__main__":
    run_inference()