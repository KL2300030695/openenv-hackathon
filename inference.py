import sys
import os
from app.env import HealthcareEnv
from app.agent import BaselineAgent
from app.tasks import TaskGrader

def run_inference():
    print("Starting OpenEnv Inference...")
    env = HealthcareEnv()
    obs = env.reset()
    print("Environment reset complete.")
    
    agent = BaselineAgent(env)
    print("Baseline Agent initialized.")
    
    steps = 0
    total_reward = 0
    while not env.done:
        action = agent.select_action(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
    
    print(f"Inference finished in {steps} steps.")
    print(f"Total Reward: {total_reward}")
    
    grader = TaskGrader(env)
    scores = {
        "task_1": grader.grade_task_1(),
        "task_2": grader.grade_task_2(),
        "task_3": grader.grade_task_3()
    }
    
    print(f"Final Grader Scores: {scores}")
    
    # Ensure scores are valid
    if all(s >= 0 for s in scores.values()):
        print("Inference successful!")
        sys.exit(0)
    else:
        print("Inference failed: Negative scores detected.")
        sys.exit(1)

if __name__ == "__main__":
    run_inference()
