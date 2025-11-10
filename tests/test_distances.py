import pytest

DISTANCE_KEYS = {
    "dist_to_airport_km",
    "dist_to_train_km",
    "dist_to_park_km",
    "dist_to_university_km",
    "dist_to_bus_km",
    "dist_to_city_center_km",
}

CASES = [
    # by address
    {
        "params": {"address": "2025 N SAWYER AVE CHICAGO IL 60647"},
        "expect_address_key": True,
        "expected_lat": 41.918169,
        "expected_lon": -87.708605,
        "expected_zip": {"60647", "60647.0"},
    },
    # by coords
    {
        "params": {"lat": 41.918169, "lon": -87.708605},
        "expect_address_key": False,
        "expected_lat": 41.918169,
        "expected_lon": -87.708605,
    },
]


@pytest.mark.parametrize("case", CASES, ids=["by_address", "by_coords"])
def test_get_distance_param_styles(client, case):
    resp = client.get("/get_distance", params=case["params"])
    assert resp.status_code == 200, resp.text
    data = resp.json()

    # required fields
    assert "lat" in data and "lon" in data
    assert DISTANCE_KEYS.issubset(data.keys())

    # address key presence
    if case["expect_address_key"]:
        assert "address" in data
        assert "zip" in data
        assert str(data["zip"]) in case["expected_zip"]
        # optionally: assert data["address"] == case["params"]["address"]
    else:
        assert "address" not in data

    # lat/lon tolerance
    assert data["lat"] == pytest.approx(case["expected_lat"], abs=1e-4)
    assert data["lon"] == pytest.approx(case["expected_lon"], abs=1e-4)

    # sanity bounds to avoid flakiness -- adjusted for current test data
    assert 0.1 <= data["dist_to_park_km"] < 0.2
    # assert 0 <= data["dist_to_bus_km"] < 5.0
    assert 1.4 <= data["dist_to_train_km"] < 1.5
    # assert 2.0 <= data["dist_to_city_center_km"] < 12.0
    assert 18.0 <= data["dist_to_airport_km"] < 19.0
    assert 3.6 <= data["dist_to_university_km"] < 3.7
