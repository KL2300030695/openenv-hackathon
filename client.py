"""
Remote client for the Healthcare Scheduling environment.

Follows the OpenEnv EnvClient pattern for async/sync remote access.

Usage (async):
    async with HealthcareEnvClient(base_url="http://localhost:7860") as env:
        result = await env.reset()
        result = await env.step(HealthcareAction(type="book", patient_id=0, doctor_id=0, slot_id=0))

Usage (sync):
    with HealthcareEnvClient(base_url="http://localhost:7860").sync() as env:
        result = env.reset()
        result = env.step(HealthcareAction(type="book", patient_id=0, doctor_id=0, slot_id=0))

Usage (Docker):
    env = HealthcareEnvClient.from_docker_image("healthcare-scheduling:latest")
    result = env.reset()
    env.close()
"""

from __future__ import annotations

from models import HealthcareAction, HealthcareObservation

try:
    from openenv.core.env_client import EnvClient

    class HealthcareEnvClient(EnvClient[HealthcareAction, HealthcareObservation]):
        """
        Remote async client for the Healthcare Scheduling environment.

        Connects to the environment server over WebSocket for low-latency
        persistent sessions. Falls back to HTTP if WebSocket is unavailable.
        """

        ACTION_TYPE = HealthcareAction
        OBSERVATION_TYPE = HealthcareObservation

except ImportError:
    # Fallback: lightweight HTTP-only client using requests
    import requests
    from typing import Optional, Dict, Any

    class HealthcareEnvClient:
        """
        Simple HTTP client for Healthcare Scheduling (fallback, no openenv-core).
        """

        def __init__(self, base_url: str = "http://localhost:7860"):
            self.base_url = base_url.rstrip("/")

        def reset(self) -> Dict[str, Any]:
            resp = requests.post(f"{self.base_url}/reset")
            resp.raise_for_status()
            return resp.json()

        def step(self, action: HealthcareAction) -> Dict[str, Any]:
            payload = action.model_dump() if hasattr(action, "model_dump") else action.dict()
            resp = requests.post(f"{self.base_url}/step", json=payload)
            resp.raise_for_status()
            return resp.json()

        def state(self) -> Dict[str, Any]:
            resp = requests.get(f"{self.base_url}/state")
            resp.raise_for_status()
            return resp.json()

        def health(self) -> Dict[str, Any]:
            resp = requests.get(f"{self.base_url}/health")
            resp.raise_for_status()
            return resp.json()

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()
