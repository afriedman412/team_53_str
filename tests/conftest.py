# tests/conftest.py
from pathlib import Path
import json
import pytest
import sys
from fastapi.testclient import TestClient

PREDICTION_FIELDS = ["price_pred", "occ_pred", "rev_pred"]

DEMOGRAPHICS_ETC = [
    "median_income",
    "median_gross_rent",
    "population",
    "median_home_value",
    "education_bachelors",
    "median_age",
    "race_white",
    "race_black",
    "race_asian",
    "race_other",
    "median_year_built",
    "total_housing_units",
    "labor_force",
    "unemployed",
    "commute_time_mean",
    "gini_index",
    "percent_foreign_born",
    "unemployment_rate",
    "percent_owner_occupied",
    "percent_public_transport",
    "percent_work_from_home",
    "vacancy_rate",
    "avg_price",
    "med_price",
]
DISTANCE_FIELDS = [
    "dist_to_airport_km",
    "dist_to_train_km",
    "dist_to_park_km",
    "dist_to_university_km",
    "dist_to_bus_km",
    "dist_to_city_center_km",
]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def client():
    from app.main import app

    """Reusable FastAPI test client."""
    return TestClient(app)


@pytest.fixture(scope="session")
def test_data_dir():
    """Return the path to the test data folder."""
    return Path(__file__).parent / "test_data"
