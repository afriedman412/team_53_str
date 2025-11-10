import pytest
from fastapi.testclient import TestClient


def test_input_from_url_zillow(client):
    url = "https://www.zillow.com/homedetails/1318-N-Sutton-Pl-Chicago-IL-60610/3856577_zpid/"

    resp = client.post("/input/from_url", data={"url": url})
    assert resp.status_code == 200, resp.text
