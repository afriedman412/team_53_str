# app/utils/distance_calc.py
from __future__ import annotations
from typing import Dict, Any, Tuple, Optional

import geopandas as gpd
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
from app.core.store import DataStore
from app.core.config import CITY_CENTERS

# from app.utils.zip_lookup import zip_for_latlon
from app.schemas.property import AddressData
from app.utils.helpers import haversine
from app.core.registry import get_store


def geocode_address(address: str | AddressData) -> Tuple[float, float]:
    """
    Convert an address to (lat, lon) using the shared Nominatim client.
    """
    store = get_store()
    address_ = str(address).replace(",", "")
    print("*** YOU ARE GETTING LAT LON FOR THIS:", address_)
    loc = store.geolocator.geocode(address_, exactly_one=True)
    # if not loc:
    #     raise ValueError(f"Address not found: {address}")
    return (loc.latitude, loc.longitude)


def _iter_feature_sets(store: DataStore):
    """
    Yield (name, tree, geoms) pairs.

    If store.meta["feature_sets"] is present, use those per-category STRtrees.
    Otherwise, fall back to a single set named 'all_features' using the global tree.
    """
    meta = store.meta or {}
    feature_sets: Optional[Dict[str, Dict[str, Any]]] = meta.get("feature_sets")

    if feature_sets:
        for name, data in feature_sets.items():
            yield name, data["tree"], data["geoms"]
    else:
        # Single global tree over store.gdf_features.geometry
        yield "all_features", store.feature_index, store.gdf_features.geometry.values


def calc_distances(
    lat: float,
    lon: float,
) -> Dict[str, float]:
    store = get_store()
    pt_m = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(3857).iloc[0]

    dists: Dict[str, float] = {}
    for name, tree, geoms in _iter_feature_sets(store):
        nearest_geom = tree.nearest(pt_m)
        if nearest_geom is None:
            dists[name] = float("nan")
            continue
        if not isinstance(nearest_geom, BaseGeometry):
            nearest_geom = geoms[int(nearest_geom)]
        dists[name] = float(pt_m.distance(nearest_geom)) / 1000.0
    return dists


def calc_city_center_distance(address: AddressData) -> float:
    """
    Hard-coded for Chicago right now

    Excluded from calc_distances because of input diff

    Will fix later.
    """
    c_lat, c_lon = CITY_CENTERS["chicago-il"]
    cc_distance = haversine(c_lat, c_lon, address.latitude, address.longitude)
    return cc_distance


def validate_address_data(address: AddressData) -> AddressData:
    """
    Fills out lat, lon if needed in an AddressData object.
    """
    if address.latitude is None or address.longitude is None:
        address.latitude, address.longitude = geocode_address(address)

    return address
