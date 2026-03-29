from fastapi import FastAPI, HTTPException
from app.env import HealthcareEnv
from app.agent import BaselineAgent
from app.tasks import TASKS, TaskGrader
from app.models import Action, StepResponse, ResetResponse, StateResponse, TaskResponse, GraderResponse, BaselineResponse
from typing import List, Dict, Any

app = FastAPI(title="Healthcare Appointment Scheduling RL Environment")

# Global environment instance
env = HealthcareEnv()

@app.get("/")
def read_root():
    return {"message": "Healthcare Appointment Scheduling RL Environment API"}

@app.post("/reset", response_model=ResetResponse)
def reset_env():
    obs = env.reset()
    return {
        "observation": obs,
        "reward": 0.0,
        "done": False,
        "info": {}
    }

@app.post("/step", response_model=StepResponse)
def step_env(action: Action):
    obs, reward, done, info = env.step(action.dict())
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/state", response_model=StateResponse)
def get_state():
    return {"state": env.state()}

@app.get("/tasks", response_model=List[TaskResponse])
def list_tasks():
    return [task.__dict__ for task in TASKS]

@app.get("/grader", response_model=GraderResponse)
def get_grader_scores():
    grader = TaskGrader(env)
    return {
        "task_scores": {
            "task_1": grader.grade_task_1(),
            "task_2": grader.grade_task_2(),
            "task_3": grader.grade_task_3()
        }
    }

@app.get("/baseline", response_model=BaselineResponse)
def run_baseline():
    """ Runs the rule-based baseline agent and returns scores. """
    env.reset()
    agent = BaselineAgent(env)
    observations = []
    
    while not env.done:
        obs = env._get_obs()
        observations.append(obs)
        action = agent.select_action(obs)
        env.step(action)
    
    grader = TaskGrader(env)
    return {
        "task_scores": {
            "task_1": grader.grade_task_1(),
            "task_2": grader.grade_task_2(),
            "task_3": grader.grade_task_3()
        },
        "observations": observations
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
