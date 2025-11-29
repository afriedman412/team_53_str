# app/core/store.py
from dataclasses import dataclass, field
import geopandas as gpd
import numpy as np
from shapely.strtree import STRtree
from typing import Optional, Dict, Any
from cachetools import TTLCache
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from app.core.config import (
    GEOCODER_DOMAIN,
    GEOCODER_SCHEME,
    GEOCODER_TIMEOUT,
    GEOCODER_USER_AGENT,
    GEOCODER_CACHE_SIZE,
    GEOCODER_CACHE_TTL_SEC,
    GEOCODER_MIN_INTERVAL_SEC,
)
import threading
import time


@dataclass(frozen=True)
class DataStore:
    pipeline: Any
    gdf_features: gpd.GeoDataFrame
    trees: Optional[gpd.GeoDataFrame]
    feature_index: STRtree
    meta: Dict[str, Any]

    # internal mutable helpers (excluded from equality/repr; OK with frozen dataclass)
    _geo_lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)
    _rate_lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)
    _last_call: float = field(default=0.0, repr=False, compare=False)
    _cache: TTLCache = field(
        default_factory=lambda: TTLCache(
            maxsize=GEOCODER_CACHE_SIZE,
            ttl=GEOCODER_CACHE_TTL_SEC,
        ),
        repr=False,
        compare=False,
    )
    _geolocator_instance: Optional[Nominatim] = field(default=None, repr=False, compare=False)

    # ---- Lazy geolocator (one per process) ---------------------------------
    @property
    def geolocator(self) -> Nominatim:
        val = self._geolocator_instance
        if val is None:
            print("🌍 Initializing Nominatim geocoder...")
            with self._geo_lock:
                val = self._geolocator_instance
                if val is None:
                    val = Nominatim(
                        user_agent=GEOCODER_USER_AGENT,
                        timeout=GEOCODER_TIMEOUT,
                        domain=GEOCODER_DOMAIN,
                        scheme=GEOCODER_SCHEME,
                    )
                    object.__setattr__(self, "_geolocator_instance", val)
        return val

    def geocode_address(self, address: str) -> tuple[float, float] | None:
        # cache
        if address in self._cache:
            return self._cache[address]

        def _rate():
            dt = time.time() - self._last_call
            if dt < GEOCODER_MIN_INTERVAL_SEC:
                time.sleep(GEOCODER_MIN_INTERVAL_SEC - dt)
            object.__setattr__(self, "_last_call", time.time())

        # helpers
        def _hit(loc):
            if loc:
                coords = (loc.latitude, loc.longitude)
                self._cache[address] = coords
                return coords
            return None

        # 1) Structured Nominatim (street/city/state/zip)
        try:
            _rate()
            parts = {
                "street": address,  # full line still helps parser
                "city": "Chicago",
                "state": "IL",
                "postalcode": "",
                "countrycodes": "us",
            }
            loc = self.geolocator.geocode(parts, exactly_one=True, addressdetails=False)
            if (coords := _hit(loc)) is not None:
                return coords
        except (GeocoderTimedOut, GeocoderUnavailable):
            pass

        # 2) Unstructured, with commas
        try:
            _rate()
            q = (
                address
                if "," in address
                else address.replace(" IL ", ", IL ").replace(" Chicago ", ", Chicago ")
            )
            loc = self.geolocator.geocode(q, exactly_one=True, addressdetails=False)
            if (coords := _hit(loc)) is not None:
                return coords
        except (GeocoderTimedOut, GeocoderUnavailable):
            pass

        # 3) Bias to Chicago bounding box (viewbox + bounded=1)
        # Chicago bbox approx: (minlon, minlat, maxlon, maxlat)
        try:
            _rate()
            viewbox = (-87.94, 41.64, -87.52, 42.02)
            loc = self.geolocator.geocode(
                address,
                exactly_one=True,
                viewbox=viewbox,
                bounded=True,
                addressdetails=False,
                country_codes="us",
            )
            if (coords := _hit(loc)) is not None:
                return coords
        except (GeocoderTimedOut, GeocoderUnavailable):
            pass

        # 4) Final fallback: ArcGIS (no key, good US coverage)
        try:
            from geopy.geocoders import ArcGIS

            _rate()
            ags = getattr(self, "_arcgis_instance", None)
            if ags is None:
                ags = ArcGIS(timeout=GEOCODER_TIMEOUT)
                object.__setattr__(self, "_arcgis_instance", ags)
            loc = ags.geocode(address)
            if (coords := _hit(loc)) is not None:
                return coords
        except Exception:
            pass

        return None
