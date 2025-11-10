# app/api/distances.py
from fastapi import APIRouter, Query, HTTPException
from app.utils.distance_calc import calc_distances, distances_from_address

router = APIRouter()


@router.get("/get_distance")
def get_distance(
    address: str | None = Query(None, description="Full Chicago address"),
    lat: float | None = Query(None, description="Latitude in WGS84"),
    lon: float | None = Query(None, description="Longitude in WGS84"),
):
    """
    Return distances (in km) to airports, train stations, parks, etc.
    Can be called with either:
      - address=<string>
      - lat=<float>&lon=<float>
    """
    try:
        if address:
            dists = distances_from_address(address)
            return {"address": address, **dists}
        elif lat is not None and lon is not None:
            dists = calc_distances(lat, lon)
            return {"lat": lat, "lon": lon, **dists}
        else:
            raise HTTPException(
                status_code=400,
                detail="Must provide either 'address' or both 'lat' and 'lon'."
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
