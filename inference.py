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
    print("Starting OpenEnv LLM Inference...")
    env = HealthcareEnv()
    obs = env.reset()
    print("Environment reset complete.")

    steps = 0
    total_reward = 0

    while not env.done and steps < 20:
        print(f"\n--- Step {steps + 1} ---")
        action = get_action_from_llm(obs)
        print(f"Action Chosen: {action}")
        
        obs, reward, done, info = env.step(action)
        print(f"Reward: {reward}, Done: {done}")
        if "error" in info:
            print(f"Info Error: {info['error']}")
            
        total_reward += reward
        steps += 1

    print("\n" + "="*30)
    print(f"Inference finished in {steps} steps.")
    print(f"Total Cumulative Reward: {total_reward:.2f}")

    grader = TaskGrader(env)
    scores = {
        "task_1": grader.grade_task_1(),
        "task_2": grader.grade_task_2(),
        "task_3": grader.grade_task_3()
    }
    print(f"Final Task Scores: {scores}")
    print("="*30)
    sys.exit(0)

if __name__ == "__main__":
    run_inference()