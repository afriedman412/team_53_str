import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Make project root importable (so `from app.main import app` works)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def client():
    # Import inside the fixture; isort won't reorder this file-level
    from app.main import app
    with TestClient(app) as c:   # ensures startup/lifespan runs
        yield c
