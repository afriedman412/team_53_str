# app/api/distances.py
from fastapi import APIRouter, Query, Body, HTTPException
from app.utils.distance_calc import calc_distances, validate_address_data
from app.utils.helpers import extract_address_from_url
from app.schemas.property import AddressData

router = APIRouter()


@router.post("/get_distance")
def get_distance_generic(
    address: str | AddressData | None = Body(None, description="Full Chicago address"),
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
            address = validate_address_data(address)
            dists = calc_distances(address.latitude, address.longitude)
            return {"address": address, **dists}
        elif lat is not None and lon is not None:
            dists = calc_distances(lat, lon)
            return {"lat": lat, "lon": lon, **dists}
        else:
            raise HTTPException(
                status_code=400, detail="Must provide either 'address' or both 'lat' and 'lon'."
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/url_to_address")
def url_to_address(url: str) -> AddressData:
    try:
        address = extract_address_from_url(url)
        return address
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
