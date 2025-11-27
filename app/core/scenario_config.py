from typing import List, Dict, Any

# These are the knobs we’re allowed to change
CONTROLLABLE_COLS = [
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms",
    "pool",  # or pool_type if you’ve encoded it that way
    "hot_tub",
    "gym",
    "housekeeping",
    "free_parking",
    "paid_parking",
    "privacy_private",  # or a single "privacy" categorical if you refactored
    "privacy_shared",
    "privacy_room in",
]

SIZE_PROTOTYPES: List[Dict[str, Any]] = [
    # Studio / small
    {"name": "studio_basic", "bedrooms": 0, "bathrooms": 1.0, "accommodates": 2, "beds": 1},
    {"name": "small_1br", "bedrooms": 1, "bathrooms": 1.0, "accommodates": 2, "beds": 1},
    {"name": "large_1br", "bedrooms": 1, "bathrooms": 1.0, "accommodates": 4, "beds": 2},
    {"name": "small_2br", "bedrooms": 2, "bathrooms": 1.5, "accommodates": 4, "beds": 3},
    {"name": "large_2br", "bedrooms": 2, "bathrooms": 2.0, "accommodates": 6, "beds": 4},
    {"name": "three_br", "bedrooms": 3, "bathrooms": 2.0, "accommodates": 6, "beds": 4},
    {"name": "four_br", "bedrooms": 4, "bathrooms": 3.0, "accommodates": 8, "beds": 5},
]

AMENITY_BUNDLES: List[Dict[str, Any]] = [
    {
        "name": "amenity_none",
        "pool": 0,
        "hot_tub": 0,
        "gym": 0,
        "housekeeping": 0,
        "free_parking": 0,
        "paid_parking": 0,
        # privacy flags as “entire home” by default
        "privacy_private": 1,
        "privacy_shared": 0,
        "privacy_room in": 0,
    },
    {
        "name": "amenity_mid",
        "pool": 0,
        "hot_tub": 0,
        "gym": 1,
        "housekeeping": 0,
        "free_parking": 1,
        "paid_parking": 0,
        "privacy_private": 1,
        "privacy_shared": 0,
        "privacy_room in": 0,
    },
    {
        "name": "amenity_premium",
        "pool": 1,
        "hot_tub": 1,
        "gym": 1,
        "housekeeping": 1,
        "free_parking": 1,
        "paid_parking": 0,
        "privacy_private": 1,
        "privacy_shared": 0,
        "privacy_room in": 0,
    },
]
