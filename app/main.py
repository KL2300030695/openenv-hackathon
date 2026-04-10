"""
Main FastAPI application — backward-compatible entry point.

The primary server entry point is now ``server.app``, which uses the OpenEnv
``create_app()`` factory.  This module re-exports the app for legacy
``python app/main.py`` usage and adds the custom ``/tasks``, ``/grader``,
and ``/baseline`` endpoints.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server.app import app  # re-export the canonical app

# ── Additional domain-specific endpoints ─────────────────────────────
# These are NOT part of the OpenEnv spec but are useful for the hackathon
# grader and for debugging.

from app.env import HealthcareEnvironment
from app.agent import BaselineAgent
from app.tasks import TASKS, TaskGrader

_env = HealthcareEnvironment()

# Only register these routes if they don't already exist (avoids duplicates
# when server.app's fallback path already added them).
_existing = {r.path for r in app.routes}

if "/tasks" not in _existing:
    @app.get("/tasks")
    def list_tasks():
        return [task.__dict__ for task in TASKS]

if "/grader" not in _existing:
    @app.get("/grader")
    def get_grader_scores():
        grader = TaskGrader(_env)
        return {
            "task_scores": {
                "task_1": grader.grade_task_1(),
                "task_2": grader.grade_task_2(),
                "task_3": grader.grade_task_3(),
            }
        }

if "/baseline" not in _existing:
    @app.get("/baseline")
    def run_baseline():
        _env.reset()
        agent = BaselineAgent(_env)
        while not _env.done:
            obs_dict = {
                "doctor_slots": {str(k): v for k, v in _env.doctor_slots.items()},
                "patients": {str(k): v for k, v in _env.patients.items()},
                "waiting_queue": [str(p) for p in _env.waiting_queue],
            }
            action = agent.select_action(obs_dict)
            _env.step_dict(action)
        grader = TaskGrader(_env)
        return {
            "task_scores": {
                "task_1": grader.grade_task_1(),
                "task_2": grader.grade_task_2(),
                "task_3": grader.grade_task_3(),
            }
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
