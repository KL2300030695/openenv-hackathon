from fastapi import FastAPI, HTTPException
from app.env import HealthcareEnv
from app.agent import BaselineAgent
from app.tasks import TASKS, TaskGrader
from app.models import Action, StepResponse, ResetResponse, StateResponse, TaskResponse, GraderResponse, BaselineResponse
from typing import List, Dict, Any

DESCRIPTION = """
## OpenEnv Environment HTTP API

HTTP API for interacting with OpenEnv environments through a standardized interface.

### Features
- **Environment Reset**: Initialize or restart episodes
- **Action Execution**: Send actions and receive observations
- **State Inspection**: Query current environment state
- **Schema Access**: Retrieve JSON schemas for actions and observations

### Workflow
1. Call `/reset` to start a new episode and get initial observation
2. Call `/step` repeatedly with actions to interact with environment
3. Episode ends when observation returns `done: true`
4. Call `/state` anytime to inspect current environment state

### Documentation
- **Swagger UI**: Available at `/docs`
- **ReDoc**: Available at `/redoc`
- **OpenAPI Schema**: Available at `/openapi.json`

[OpenEnv Team - Website](https://openenv.org) | BSD-3-Clause
"""

app = FastAPI(
    title="OpenEnv Environment HTTP API",
    version="1.0.0",
    description=DESCRIPTION,
    docs_url="/",
)

# Global environment instance
env = HealthcareEnv()

@app.api_route("/reset", methods=["GET", "POST"], response_model=ResetResponse)
def reset_env():
    obs = env.reset()
    return {
        "observation": obs,
        "info": {}
    }

@app.post("/step", response_model=StepResponse)
def step_env(action: Action):
    # Convert Pydantic model to dict for environment handling
    action_dict = action.model_dump() if hasattr(action, "model_dump") else action.dict()
    obs, reward, done, info = env.step(action_dict)
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
