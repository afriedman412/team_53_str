import pytest
import json
from pathlib import Path
from conftest import DEMOGRAPHICS_ETC, DISTANCE_FIELDS, PREDICTION_FIELDS


def load_expected(file_name: str):
    """Load expected JSON from tests/test_data/<file_name>."""
    HERE = Path(__file__).resolve().parent
    TEST_DATA = HERE / "test_data"
    with open(TEST_DATA / file_name, "r") as f:
        return json.load(f)


@pytest.mark.parametrize(
    "url,expected_file",
    [
        (
            "https://www.zillow.com/homedetails/3052-N-Oakley-Ave-1-Chicago-IL-60618/2145898185_zpid/",
            "oakley_test.json",
        ),
        (
            "https://www.zillow.com/homedetails/4004-N-Clarendon-Ave-1-Chicago-IL-60613/3707103_zpid/",
            "clarendon_test.json",
        ),
    ],
)
def test_from_url_predictions_census_and_distances(client, url, expected_file):
    expected = load_expected(expected_file)

    from app.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        resp = c.post("/api/from_url", data={"url": url})
        assert resp.status_code == 200, resp.text
        data = resp.json()

    data = resp.json()
    inputs = data.get("inputs", {})

    # predictions
    for field in PREDICTION_FIELDS:
        assert inputs.get(field) == expected["inputs"][field], f"{field} mismatch"

    # census stats
    for field in DEMOGRAPHICS_ETC:
        assert inputs.get(field) == expected["inputs"][field], f"{field} mismatch"

    # distances and neighborhood price context
    for field in DISTANCE_FIELDS:
        assert inputs.get(field) == expected["inputs"][field], f"{field} mismatch"
