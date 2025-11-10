# app/utils/geocode.py

from cachetools import TTLCache
import threading
import time
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

# 24h in-memory cache (max 10k addresses)
_cache = TTLCache(maxsize=10_000, ttl=60 * 60 * 24)

# Simple thread lock + rate limit control
_lock = threading.Lock()
_last_call = 0.0
_MIN_INTERVAL = 1.0  # Nominatim etiquette: 1 request/sec


def geocode_address(address: str, store) -> tuple[float, float] | None:
    """
    Geocode an address using the shared Nominatim instance in store.
    Includes local caching and polite rate-limiting.
    """
    # Check cache first
    if address in _cache:
        return _cache[address]

    global _last_call
    with _lock:
        dt = time.time() - _last_call
        if dt < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - dt)

        try:
            loc = store.geolocator.geocode(address)
        except (GeocoderTimedOut, GeocoderUnavailable):
            loc = None
        _last_call = time.time()

    if loc:
        coords = (loc.latitude, loc.longitude)
        _cache[address] = coords
        return coords

    return None
