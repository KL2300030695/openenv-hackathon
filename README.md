# Healthcare Appointment Scheduling RL Environment

This project implements a complete OpenEnv-compatible reinforcement learning environment simulating a real-world hospital appointment booking system.

## Environment Design

The environment simulates a hospital with multiple doctors and patients. The goal is to efficiently schedule appointments while considering patient priorities, doctor availability, and patient preferences.

### Observation Space
The state is a dictionary containing:
- `doctor_slots`: Availability of each doctor for each time slot.
- `patients`: Information about patients (priority, preferred doctor, status).
- `waiting_queue`: List of patient IDs currently waiting for an appointment.

### Action Space
Actions are dictionaries:
- `type`: "book", "reschedule", "cancel", or "waitlist".
- `patient_id`: ID of the patient.
- `doctor_id`: ID of the doctor (required for "book" and "reschedule").
- `slot_id`: ID of the time slot (required for "book" and "reschedule").

### Reward Function
- **+1.0**: Successful booking.
- **+0.2 bonus**: Booking with preferred doctor.
- **+0.3**: Waitlisting a patient (partial progress).
- **-1.0**: Conflicts (trying to book an occupied slot) or invalid actions.
- **-0.2**: Cancellations.
- **+0.5**: Successful rescheduling.

## Tasks & Grading

1.  **Task 1 (Easy)**: Successfully book an appointment when slots are available.
2.  **Task 2 (Medium)**: Handle scheduling conflicts and rescheduling efficiently.
3.  **Task 3 (Hard)**: Optimize scheduling for multiple patients with different priorities (Priority 1 > Priority 2 > Priority 3).

The `/grader` endpoint returns a score between 0.0 and 1.0 for each task based on efficiency and priority fulfillment.

## Setup & Usage

### Local Development

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the FastAPI server:
    ```bash
    python app/main.py
    ```
4.  The API will be available at `http://localhost:7860`.

### Running with Docker

```bash
docker build -t healthcare-rl-env .
docker run -p 7860:7860 healthcare-rl-env
```

### Deployment

This environment is fully compatible with Hugging Face Spaces. Simply upload the files to a Space and select "Docker" configuration.

## API Endpoints

-   `GET /`: Root endpoint.
-   `POST /reset`: Resets the environment.
-   `POST /step`: Performs an action.
-   `GET /state`: Gets current state.
-   `GET /tasks`: Lists all available tasks.
-   `GET /grader`: Returns scores for each task.
-   `GET /baseline`: Runs a rule-based baseline agent and returns its performance.

## License

MIT
