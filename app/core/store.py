# app/core/store.py
from dataclasses import dataclass
import geopandas as gpd
import numpy as np
from shapely.strtree import STRtree
from typing import Optional, Dict, Any
from geopy.geocoders import Nominatim


@dataclass(frozen=True)
class DataStore:
    gdf_features: gpd.GeoDataFrame
    zips: gpd.GeoDataFrame
    trees: Optional[gpd.GeoDataFrame]
    feature_index: STRtree
    zip_bounds_minx: np.ndarray
    zip_bounds_miny: np.ndarray
    zip_bounds_maxx: np.ndarray
    zip_bounds_maxy: np.ndarray
    geolocator: Nominatim                 # ← add the shared geocoder
    meta: Dict[str, Any]
