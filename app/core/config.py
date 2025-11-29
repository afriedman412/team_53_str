# app/core/config.py
import os
from pathlib import Path

# Check if running in Cloud Run
IS_CLOUD = os.getenv("K_SERVICE") is not None

if IS_CLOUD:
    # Paths in the container
    ROOT_DIR = Path("/app")
else:
    # Local development paths
    ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"

DATA_PATHS = {
    "osm": DATA_DIR / "osm" / "chicago_features.gpkg",
    "census": DATA_DIR / "census_data.parquet",
}

PIPELINE_PATH = ROOT_DIR / "models" / "pipeline.joblib"

POPS_PATH = ROOT_DIR / "app" / "models" / "pops_"

# for loading
PIPELINE_PATHS = {
    "properties": DATA_DIR / "df_clean_city_subset_111725.csv",
    "embedder": PIPELINE_PATH / "embedder.pkl",
    "embeddings": PIPELINE_PATH / "embeddings.pkl",
    "assembler": PIPELINE_PATH / "assembler.pkl",
}

# for revenue predictions
PRED_OPTIONS = {
    "pool": [True, False],
    "housekeeping": [True, False],
    "gym": [True, False],
    "free_parking": [True, False],
}

CSG_PALETTE = [
    "#C9A84A",  # gold
    "#1E2A39",  # dark navy
    "#A9C7E4",  # light blue
    "#78D0F8",  # cyan
    "#E8E8E8",  # light gray
    "#1A1A1A",  # near black
    "#FFFFFF",  # white
]


# Geocoder settings
# consider adding contact info per Nominatim policy
GEOCODER_USER_AGENT = "afriedman412@gmail.com"
GEOCODER_TIMEOUT = 10  # seconds
GEOCODER_DOMAIN = "nominatim.openstreetmap.org"
GEOCODER_SCHEME = "https"
GEOCODER_MIN_INTERVAL_SEC = 1.1
GEOCODER_CACHE_TTL_SEC = 24 * 60 * 60
GEOCODER_CACHE_SIZE = 10000

CITY_SUBSET = [
    "boston-ma",
    "washington-dc",
    "denver-co",
    "columbus-oh",
    "twin-cities-mn",
    "chicago-il",
    "austin-tx",
    "nashville-tn",
]

# inferred city center data
CITY_CENTERS = {
    "albany-ny": (42.6526, -73.7562),
    "austin-tx": (30.2672, -97.7431),
    "boston-ma": (42.3601, -71.0589),
    "bozeman-mt": (45.6770, -111.0429),
    "dade-county-fl": (25.7617, -80.1918),  # Miami
    "cambridge-ma": (42.3736, -71.1097),
    "chicago-il": (41.8781, -87.6298),
    "las-vegas-nv": (36.1699, -115.1398),
    "columbus-oh": (39.9612, -82.9988),
    "dallas-tx": (32.7767, -96.7970),
    "denver-co": (39.7392, -104.9903),
    "ft-worth-tx": (32.7555, -97.3308),
    "honolulu-hi": (21.3099, -157.8581),
    "jersey-city-nj": (40.7178, -74.0431),
    "los-angeles-ca": (34.0522, -118.2437),
    "nashville-tn": (36.1627, -86.7816),
    "new-orleans-no": (29.9511, -90.0715),
    "new-york-ny": (40.7128, -74.0060),
    "newark-nj": (40.7357, -74.1724),
    "oakland-ca": (37.8044, -122.2712),
    "pacific-grove-ca": (36.6177, -121.9166),
    "portland-or": (45.5152, -122.6784),
    "rhode-island": (41.8240, -71.4128),  # Providence
    "rochester-ny": (43.1566, -77.6088),
    "salem-or": (44.9429, -123.0351),
    "san-diego-ca": (32.7157, -117.1611),
    "san-francisco-ca": (37.7749, -122.4194),
    "santa-clara-county-ca": (37.3541, -121.9552),  # San Jose area
    "san-mateo-county-ca": (37.5630, -122.3255),
    "santa-cruz-county-ca": (36.9741, -122.0308),
    "seattle-wa": (47.6062, -122.3321),
    "twin-cities-mn": (44.9778, -93.2650),  # Minneapolis
    "washington-dc": (38.9072, -77.0369),
}
