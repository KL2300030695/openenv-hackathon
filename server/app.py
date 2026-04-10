import sys
import os

# Add both parent directory and server directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

from openenv.core.env_server import create_fastapi_app
from env import HealthcareEnvironment
from models import HealthcareAction, HealthcareObservation

app = create_fastapi_app(HealthcareEnvironment, HealthcareAction, HealthcareObservation)

@app.get("/")
def read_root():
    return {
        "name": "healthcare_scheduling",
        "status": "running",
        "version": "1.0.0",
    }

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=True)

if __name__ == '__main__':
    main()