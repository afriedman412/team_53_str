
# scripts/download_osm_chicago.py
import osmnx as ox
from pathlib import Path

# Directory to save data
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "osm"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Tell OSMnx to cache requests
ox.settings.use_cache = True
ox.settings.cache_folder = str(DATA_DIR)

# Define what to download
TAGS = {
    "aeroway": ["aerodrome"],                       # airports
    "railway": ["station"],                         # train stations
    "leisure": ["park"],                            # parks
    "amenity": ["university", "bus_station"],       # unis, bus terminals
    "place": ["city_centre"]                        # downtown-like tags
}

print("📡 Downloading OSM features for Chicago...")
gdf = ox.features_from_place("Chicago, Illinois, USA", tags=TAGS)
print(f"✅ Retrieved {len(gdf)} features")

# Save locally for reuse
out_path = DATA_DIR / "chicago_features.gpkg"
gdf.to_file(out_path, driver="GPKG")
print(f"💾 Saved to {out_path}")
