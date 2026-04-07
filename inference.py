import os
import sys
import json
from openai import OpenAI
from app.env import HealthcareEnv
from app.tasks import TaskGrader

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
# Using HF_TOKEN or API_KEY, default to dummy for local testing
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

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
            temperature=0.1 # Low temperature for more deterministic/stable responses
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
        print(f"Error parsing LLM response or calling API: {e}")
        # Fallback to a safe 'waitlist' action with valid numeric IDs if queue is not empty
        waiting_q = obs.get("waiting_queue", [])
        p_id = int(waiting_q[0]) if waiting_q else 0
        return {"type": "waitlist", "patient_id": p_id, "doctor_id": 0, "slot_id": 0}

def run_inference():
    """
    Runs LLM-based inference on the Healthcare Scheduling environment
    and prints structured [START]/[STEP]/[END] output required by the OpenEnv validator.
    """
    # Define the tasks we will report on
    task_defs = [
        {"name": "Book an Appointment",       "grader": "grade_task_1"},
        {"name": "Handle Scheduling Conflicts", "grader": "grade_task_2"},
        {"name": "Priority Scheduling",        "grader": "grade_task_3"},
    ]

    env = HealthcareEnv()
    obs = env.reset()

    # ---------- Run the agent loop ----------
    step_num = 0
    cumulative_reward = 0.0
    step_log = []  # collect (step, reward) for each step

    while not env.done and step_num < 20:
        action = get_action_from_llm(obs)
        obs, reward, done, info = env.step(action)
        cumulative_reward += reward
        step_num += 1
        step_log.append((step_num, reward))

    # ---------- Grade ----------
    grader = TaskGrader(env)

    # ---------- Emit structured output per task ----------
    for task_def in task_defs:
        task_name = task_def["name"]
        grade_fn = getattr(grader, task_def["grader"])
        task_score = grade_fn()

        # [START] block
        print(f"[START] task={task_name}", flush=True)

        # [STEP] blocks — replay the step log for each task
        for s, r in step_log:
            print(f"[STEP] step={s} reward={r}", flush=True)

        # [END] block
        print(f"[END] task={task_name} score={task_score:.4f} steps={step_num}", flush=True)

    # Also print a human-readable summary (after structured blocks)
    print(f"\nInference finished in {step_num} steps.", flush=True)
    print(f"Total Cumulative Reward: {cumulative_reward:.2f}", flush=True)
    sys.exit(0)

if __name__ == "__main__":
    run_inference()