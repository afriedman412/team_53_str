# app/utils/distance_calc.py

from __future__ import annotations
from typing import Dict, Any, Tuple, Optional

import geopandas as gpd
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from app.core.store import DataStore
from app.core.deps import get_store
from app.utils.zip_lookup import zip_for_latlon


# ---------------------------------------------------------------------------
#  Geocoding (uses shared Nominatim from store)
# ---------------------------------------------------------------------------

def geocode_address(address: str, store: Optional[DataStore] = None) -> Tuple[float, float]:
    """
    Convert an address to (lat, lon) using the shared Nominatim client.
    """
    loc = store.geolocator.geocode(address, exactly_one=True)
    if not loc:
        raise ValueError(f"Address not found: {address}")
    return (loc.latitude, loc.longitude)


# ---------------------------------------------------------------------------
#  Distance calculator (nearest feature distances)
# ---------------------------------------------------------------------------

def _iter_feature_sets(store: DataStore):
    """
    Yield (name, tree, geoms) pairs.

    If store.meta["feature_sets"] is present, use those per-category STRtrees.
    Otherwise, fall back to a single set named 'all_features' using the global tree.
    """
    meta = store.meta or {}
    feature_sets: Optional[Dict[str, Dict[str, Any]]
                           ] = meta.get("feature_sets")

    if feature_sets:
        for name, data in feature_sets.items():
            yield name, data["tree"], data["geoms"]
    else:
        # Single global tree over store.gdf_features.geometry
        yield "all_features", store.feature_index, store.gdf_features.geometry.values


def calc_distances(lat: float, lon: float, store: Optional[DataStore] = None) -> Dict[str, float]:
    store = store or get_store()
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


# ---------------------------------------------------------------------------
#  Combined wrapper
# ---------------------------------------------------------------------------

def distances_from_address(address: str, store: Optional[DataStore] = None) -> Dict[str, Any]:
    store = store or get_store()
    lat, lon = geocode_address(address, store)
    zip_code = zip_for_latlon(lat, lon, store)
    return {"lat": lat, "lon": lon, "zip": zip_code, **calc_distances(lat, lon, store)}
