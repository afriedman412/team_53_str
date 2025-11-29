import pandas as pd
import numpy as np
import itertools
from functools import reduce
import operator
from app.core.config import PRED_OPTIONS

SCENARIO_DEFAULTS = {
    # Independent structural sliders
    "bedrooms": 1,
    "beds": 1,
    "accommodates": 2,
    "bathrooms": 1.5,
    # Amenities
    "air_conditioning": 1,
    "heating": 1,
    "free_parking": 1,
    "paid_parking": 0,
    "gym": 0,
    "housekeeping": 0,
    "pool": 0,
    "pool_private": 0,
    "pool_shared": 0,
    "pool_indoor": 0,
    "pool_outdoor": 0,
    "hot_tub": 0,
    "hot_tub_private": 0,
    "hot_tub_shared": 0,
    # Privacy (mutually exclusive)
    "privacy_private": 1,
    "privacy_room_in": 0,
    "privacy_shared": 0,
    # Room type (mutually exclusive)
    "room_type_entire": 1,
    "room_type_private_room": 0,
    "room_type_shared_room": 0,
    "room_type_hotel_room": 0,
}


def scenario_df_verify(scenario_df, city="chicago-il"):
    scenario_df["city"] = "chicago-il"
    scenario_df["slice"] = "chicago-il"
    scenario_df["avg_price"] = 577.59
    scenario_df["med_price"] = 169.0
    return scenario_df


def dict_permutations(options_dict):
    keys = list(options_dict.keys())
    values_product = itertools.product(*(options_dict[k] for k in keys))
    return [dict(zip(keys, combo)) for combo in values_product]


def assemble_options(base, input_values):
    base.update(input_values)
    options = {
        "accommodates": [
            base["accommodates"] + n for n in [0, 1, 2]
        ],  # default, default +1, default +2
    }
    options.update(PRED_OPTIONS)

    permo = dict_permutations(options)
    permo_df = pd.DataFrame(permo)
    for k in base:
        if k not in permo_df:
            permo_df[k] = base[k]
    permo_df = scenario_df_verify(permo_df)

    return permo_df


def get_original_match_mask(permo_df, input_values):
    blank_ = {
        "pool": 0,
        "hot_tub": 0,
        "gym": 0,
        "housekeeping": 0,
        "free_parking": 0,
    }
    input_values.update(blank_)
    mask_parts = []
    for col, val in input_values.items():
        if col not in permo_df.columns:
            raise KeyError(f"{col} not in preds columns")
        # use np.isclose for floats
        if isinstance(val, float):
            mask_parts.append(np.isclose(permo_df[col], val))
        else:
            mask_parts.append(permo_df[col] == val)

    mask = reduce(operator.and_, mask_parts)
    return mask
