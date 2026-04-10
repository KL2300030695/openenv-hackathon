"""
FastAPI application entry point for the Healthcare Scheduling environment.

This module creates the HTTP server using the OpenEnv framework's create_app()
factory, which automatically provides standardised endpoints:
    - POST /reset     : Reset the environment
    - POST /step      : Execute an action
    - GET  /state     : Get current environment state
    - GET  /health    : Health check
    - GET  /schema    : Action/Observation JSON schemas
    - WS   /ws        : WebSocket for persistent sessions
    - GET  /web       : Interactive web interface
    - GET  /docs      : Swagger/OpenAPI documentation

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ── Ensure project root is importable ───────────────────────────────
SERVER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SERVER_DIR.parent

# Insert PROJECT_ROOT first so 'app' resolves to the app/ package,
# not server/app.py (which would shadow it).
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Import environment and models ───────────────────────────────────
from app.env import HealthcareEnvironment
from app.models import HealthcareAction, HealthcareObservation

# ── Build the FastAPI app ───────────────────────────────────────────
try:
    from openenv.core.env_server.http_server import create_app

    app = create_app(
        HealthcareEnvironment,
        HealthcareAction,
        HealthcareObservation,
        env_name="healthcare_scheduling",
        max_concurrent_envs=1,
    )

except ImportError:
    # Fallback: build a plain FastAPI app when openenv-core isn't available
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse, JSONResponse

    from app.agent import BaselineAgent
    from app.tasks import TASKS, TaskGrader

    DESCRIPTION = """\
    ## Healthcare Appointment Scheduling — OpenEnv Environment

    RL environment simulating a hospital appointment booking system.

    ### Endpoints
    - `POST /reset` — Start a new episode
    - `POST /step` — Execute an action
    - `GET  /state` — Current observation
    - `GET  /health` — Container health check
    - `GET  /tasks` — Task definitions
    - `GET  /grader` — Grading scores
    - `GET  /baseline` — Baseline agent performance
    """

    app = FastAPI(
        title="Healthcare Scheduling — OpenEnv",
        version="1.0.0",
        description=DESCRIPTION,
    )

    # Serve static files
    STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
    if os.path.isdir(STATIC_DIR):
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/", include_in_schema=False)
        def serve_frontend():
            return FileResponse(os.path.join(STATIC_DIR, "index.html"))

    # Global environment instance
    env = HealthcareEnvironment()

    @app.get("/health")
    def health_check():
        return {"status": "healthy", "service": "healthcare_scheduling"}

    @app.api_route("/reset", methods=["GET", "POST"])
    def reset_env():
        obs = env.reset()
        return {
            "observation": obs.model_dump() if hasattr(obs, "model_dump") else obs.__dict__,
            "info": {},
        }

    @app.post("/step")
    def step_env(action: HealthcareAction):
        obs = env.step(action)
        return {
            "observation": obs.model_dump() if hasattr(obs, "model_dump") else obs.__dict__,
            "reward": obs.reward,
            "done": obs.done,
            "info": obs.info or {},
        }

    @app.get("/state")
    def get_state():
        return {
            "state": {
                "doctor_slots": {str(k): v for k, v in env.doctor_slots.items()},
                "patients": {str(k): v for k, v in env.patients.items()},
                "waiting_queue": [str(p) for p in env.waiting_queue],
                "current_step": env.current_step,
            }
        }

    @app.get("/schema")
    def get_schema():
        return {
            "action_schema": HealthcareAction.model_json_schema(),
            "observation_schema": HealthcareObservation.model_json_schema(),
        }

    @app.get("/tasks")
    def list_tasks():
        return [task.__dict__ for task in TASKS]

    @app.get("/grader")
    def get_grader_scores():
        grader = TaskGrader(env)
        return {
            "task_scores": {
                "task_1": grader.grade_task_1(),
                "task_2": grader.grade_task_2(),
                "task_3": grader.grade_task_3(),
            }
        }

    @app.get("/baseline")
    def run_baseline():
        env.reset()
        agent = BaselineAgent(env)

        while not env.done:
            obs_dict = {
                "doctor_slots": {str(k): v for k, v in env.doctor_slots.items()},
                "patients": {str(k): v for k, v in env.patients.items()},
                "waiting_queue": [str(p) for p in env.waiting_queue],
            }
            action = agent.select_action(obs_dict)
            env.step_dict(action)

        grader = TaskGrader(env)
        return {
            "task_scores": {
                "task_1": grader.grade_task_1(),
                "task_2": grader.grade_task_2(),
                "task_3": grader.grade_task_3(),
            }
        }


# ── Direct execution ────────────────────────────────────────────────
def main(host: str = "0.0.0.0", port: int | None = None):
    """Run the Healthcare Scheduling server."""
    import uvicorn

    if port is None:
        port = int(os.getenv("API_PORT", "7860"))

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main(port=args.port)