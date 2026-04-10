---
title: Healthcare Appointment Scheduling
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Healthcare Appointment Scheduling — OpenEnv Environment

An OpenEnv-compatible reinforcement learning environment simulating a real-world hospital appointment booking system. The environment exposes scheduling tasks through the standard OpenEnv `reset()`/`step()`/`state()` interface.

## Quick Start

### Docker (Recommended)

```bash
cd Meta_Hackathon
docker build -t healthcare-scheduling:latest .
docker run --rm -p 7860:7860 healthcare-scheduling:latest
```

Verify the server is healthy:

```bash
curl http://localhost:7860/health
```

Expected response:

```json
{"status": "healthy", "service": "healthcare_scheduling"}
```

### Without Docker

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Deploy to Hugging Face Spaces

```bash
openenv push
```

Or specify options:

```bash
openenv push --namespace my-org --private
```

## Environment Design

The environment simulates a hospital with **3 doctors × 5 time slots** and **10 patients** with varying priorities. The agent's goal is to efficiently schedule appointments while respecting constraints.

### Observation Space

`HealthcareObservation` contains:

| Field | Type | Description |
|-------|------|-------------|
| `doctor_slots` | `Dict[str, List[bool]]` | Availability per doctor per slot (True = free) |
| `patients` | `Dict[str, Dict]` | Patient info: priority, preferred doctor, status |
| `waiting_queue` | `List[str]` | Patient IDs currently awaiting appointment |
| `current_step` | `int` | Current step number |

### Action Space

`HealthcareAction` fields:

| Field | Type | Description |
|-------|------|-------------|
| `type` | `str` | `"book"`, `"reschedule"`, `"cancel"`, or `"waitlist"` |
| `patient_id` | `int` | ID of the target patient |
| `doctor_id` | `int` | ID of the doctor (for book/reschedule) |
| `slot_id` | `int` | ID of the time slot (for book/reschedule) |

### Reward Function

| Action | Reward | Condition |
|--------|--------|-----------|
| Book | **+1.0** | Successful booking |
| Book (preferred) | **+1.2** | Booking with preferred doctor |
| Reschedule | **+0.5** | Successful reschedule |
| Waitlist | **+0.3** | Partial progress |
| Cancel | **-0.2** | Cancellation |
| Invalid/Conflict | **-1.0** | Bad action |

## Tasks & Grading

| Task | Difficulty | Description | Score Range |
|------|-----------|-------------|-------------|
| Task 1 | Easy | Successfully book appointments | 0.0 – 1.0 |
| Task 2 | Medium | Handle conflicts and rescheduling | 0.0 – 1.0 |
| Task 3 | Hard | Priority-based scheduling (1 > 2 > 3) | 0.0 – 1.0 |

## Client Usage

### Python (sync)

```python
from client import HealthcareEnvClient
from app.models import HealthcareAction

with HealthcareEnvClient(base_url="http://localhost:7860") as env:
    result = env.reset()
    print(result)

    result = env.step(HealthcareAction(
        type="book",
        patient_id=0,
        doctor_id=1,
        slot_id=2
    ))
    print(result)
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Container health check |
| `POST` | `/reset` | Reset environment, get initial observation |
| `POST` | `/step` | Execute action, get next observation |
| `GET` | `/state` | Get current environment state |
| `GET` | `/schema` | Action/Observation JSON schemas |
| `GET` | `/tasks` | List task definitions |
| `GET` | `/grader` | Get grading scores |
| `GET` | `/baseline` | Run baseline agent |
| `GET` | `/docs` | Swagger/OpenAPI docs |
| `WS` | `/ws` | WebSocket for persistent sessions |

## Project Structure

```
healthcare_scheduling/
├── openenv.yaml          # OpenEnv manifest (spec_version: 1)
├── pyproject.toml         # Dependencies and package config
├── requirements.txt       # Docker build dependencies
├── Dockerfile             # Container image definition
├── client.py              # EnvClient for remote access
├── inference.py           # LLM inference script
├── test_env.py            # Verification script
├── app/
│   ├── __init__.py        # Package exports
│   ├── models.py          # HealthcareAction, HealthcareObservation
│   ├── env.py             # HealthcareEnvironment (OpenEnv interface)
│   ├── agent.py           # Baseline rule-based agent
│   └── tasks.py           # Task definitions and grader
├── server/
│   ├── __init__.py
│   └── app.py             # FastAPI app via create_app()
└── static/
    └── index.html         # Web frontend
```

## Running Inference

```bash
export HF_TOKEN=your_token_here
python inference.py
```

The script outputs structured `[START]`/`[STEP]`/`[END]` blocks for each task.

## License

MIT
