import os
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add the project root (WORKDIR) to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def client():
    from app.main import app

    os.environ["TESTING"] = "1"
    # Using context manager ensures lifespan runs before the first request
    with TestClient(app) as c:
        yield c
