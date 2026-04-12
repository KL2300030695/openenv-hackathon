---
title: Healthcare Appointment Scheduling
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Healthcare Appointment Scheduling — OpenEnv Environment

An OpenEnv-compatible reinforcement learning environment simulating a real-world
hospital appointment booking system with **specialty matching**, **priority scheduling**,
and **time-preference optimisation**. The environment exposes scheduling tasks through
the standard OpenEnv `reset()`/`step()`/`state()` interface.

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

The environment simulates a hospital with **5 specialist departments**, each staffed
by a doctor. **15 patients** arrive with varying medical needs (requiring specific
specialties), urgency levels (priority 1-3), and time-of-day preferences. Some
appointment slots are pre-booked to create realistic scarcity.

### Key Mechanics

- **Specialty matching**: Patients require a specific specialty (e.g. Cardiology).
  They can be booked with a matching specialist or a General Medicine doctor.
  Booking with the wrong specialist is rejected.
- **Priority scheduling**: Priority-1 (urgent) patients should be booked before
  lower-priority patients. The agent must learn to triage.
- **Time preferences**: Patients prefer morning (slots 0-3) or afternoon (slots 4-7).
  Matching time preferences earns bonus reward.
- **Pre-existing appointments**: Random slots are pre-booked at reset, creating
  realistic scarcity and forcing the agent to handle conflicts.

### Observation Space

`HealthcareObservation` contains:

| Field | Type | Description |
|-------|------|-------------|
| `doctor_slots` | `Dict[str, List[bool]]` | Availability per doctor per slot (True = free) |
| `doctor_specialties` | `Dict[str, str]` | Specialty of each doctor |
| `patients` | `Dict[str, Dict]` | Patient info: priority, required_specialty, preferred_doctor, time_preference, status |
| `waiting_queue` | `List[str]` | Patient IDs currently awaiting appointment |
| `current_step` | `int` | Current step number |
| `max_steps` | `int` | Maximum steps in the episode |

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
| Book | **0.60** | Successful booking (base) |
| + preferred doctor | **+0.15** | Booking with preferred doctor |
| + specialty match | **+0.10** | Exact specialty match (not GM catch-all) |
| + time preference | **+0.05** | Morning/afternoon slot matches preference |
| Reschedule | **0.50** | Successful reschedule |
| Waitlist | **0.25** | Partial progress |
| Cancel | **0.20** | Cancellation |
| Conflict (slot taken) | **0.15** | Attempting to book occupied slot |
| Specialty mismatch | **0.15** | Wrong specialty for patient |
| Invalid action | **0.10** | Bad patient/doctor/slot ID |

## Tasks & Grading

| Task | Difficulty | What It Tests | Metrics |
|------|-----------|---------------|---------|
| Task 1 | Easy | Basic booking ability | Booking rate, efficiency, error avoidance |
| Task 2 | Medium | Conflict management | Conflict recovery, no repeated errors, specialty compliance |
| Task 3 | Hard | Priority optimisation | Priority ordering, P1 completion, preferred doctor match, time preferences |

Each grader returns a score in `(0, 1)` — analysing the environment state and
the full action history to evaluate multiple dimensions of agent behaviour.

## Client Usage

### Python (sync)

```python
from client import HealthcareEnvClient
from models import HealthcareAction

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
├── env.py                 # HealthcareEnvironment (OpenEnv interface)
├── models.py              # HealthcareAction, HealthcareObservation, HealthcareState
├── agent.py               # Baseline rule-based agent
├── client.py              # EnvClient for remote access
├── inference.py           # LLM inference script
├── test_env.py            # Verification script
├── server/
│   ├── __init__.py
│   └── app.py             # FastAPI app via create_app()
├── tasks/
│   ├── __init__.py        # TaskGrader wrapper
│   ├── book_appointment.py       # Task 1 grader (easy)
│   ├── scheduling_conflicts.py   # Task 2 grader (medium)
│   └── priority_scheduling.py    # Task 3 grader (hard)
└── static/
    └── index.html         # Interactive web frontend
```

## Running Inference

```bash
export HF_TOKEN=your_token_here
python inference.py
```

The script outputs structured `[START]`/`[STEP]`/`[END]` blocks for each task.

## License

MIT
