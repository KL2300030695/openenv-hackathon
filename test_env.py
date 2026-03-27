import sys
import os
sys.path.append('d:/Meta_Hackathon')

from app.env import HealthcareEnv
from app.agent import BaselineAgent
from app.tasks import TaskGrader

def test():
    print("Testing Healthcare RL Environment...")
    env = HealthcareEnv()
    obs = env.reset()
    print("Reset successful.")
    
    agent = BaselineAgent(env)
    print("Agent initialized.")
    
    steps = 0
    while not env.done:
        action = agent.select_action(obs)
        obs, reward, done, info = env.step(action)
        steps += 1
    
    print(f"Environment finished in {steps} steps.")
    
    grader = TaskGrader(env)
    scores = {
        "task_1": grader.grade_task_1(),
        "task_2": grader.grade_task_2(),
        "task_3": grader.grade_task_3()
    }
    
    print(f"Grader Scores: {scores}")
    
    assert scores["task_1"] > 0
    assert scores["task_2"] > 0
    assert scores["task_3"] > 0
    
    print("Verification successful!")

if __name__ == "__main__":
    test()
