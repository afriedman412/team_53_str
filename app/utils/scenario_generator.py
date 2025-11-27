"""
Scenario generator for hypothetical Airbnb listings.

Key principles:
- All inputs start from an AddressData object.
- Static (location-only) features come from build_scenario_base().
- Controllable features (beds, baths, amenities, etc.) are set by permutations.
- ATTOM is completely removed.
- No predictions are computed here — this is pure input generation.
"""

import itertools
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from app.core.registry import get_store
from app.schemas.property import AddressData, ScenarioControls
from app.utils.input_builder import build_scenario_base_from_address
from app.core.scenario_config import SIZE_PROTOTYPES, AMENITY_BUNDLES, CONTROLLABLE_COLS
from app.utils.helpers import short_term_scenario_cleaning, COL_FIXES

# ============================================================
# CONTROLLABLE FEATURE DOMAIN (SCENARIO SPACE)
# ============================================================

SCENARIO_SPACE = {
    # Independent structural sliders
    "bedrooms": [0, 1, 2, 3, 4],
    "beds": [1, 2, 3, 4, 5, 6],
    "accommodates": [1, 2, 3, 4, 5, 6, 8],
    "bathrooms": [1.0, 1.5, 2.0, 3.0],
    # Amenities
    "air_conditioning": [0, 1],
    "heating": [0, 1],
    "free_parking": [0, 1],
    "paid_parking": [0, 1],
    "gym": [0, 1],
    "housekeeping": [0, 1],
    "pool": [0, 1],
    "pool_private": [0, 1],
    "pool_shared": [0, 1],
    "pool_indoor": [0, 1],
    "pool_outdoor": [0, 1],
    "hot_tub": [0, 1],
    "hot_tub_private": [0, 1],
    "hot_tub_shared": [0, 1],
    # Privacy (mutually exclusive)
    "privacy_private": [0, 1],
    "privacy_room_in": [0, 1],
    "privacy_shared": [0, 1],
    # Room type (mutually exclusive)
    "room_type_entire": [0, 1],
    "room_type_private_room": [0, 1],
    "room_type_shared_room": [0, 1],
    "room_type_hotel_room": [0, 1],
}


# ============================================================
# SCENARIO VALIDATION
# ============================================================


def enforce_constraints(row: Dict[str, Any]) -> Dict[str, Any] | None:
    """
    Apply logical constraints:
    - Bedrooms/beds/accommodates are independent but must be minimally consistent.
    - Privacy and room_type are one-hot exclusive.
    """

    # --- Basic validity ---
    row["bedrooms"] = max(0, row["bedrooms"])
    row["beds"] = max(1, row["beds"])
    row["accommodates"] = max(1, row["accommodates"])

    if row["accommodates"] < row["beds"]:
        row["accommodates"] = row["beds"]

    # --- Privacy exclusivity ---
    priv_cols = ["privacy_private", "privacy_room in", "privacy_shared"]
    if sum(row.get(k, 0) for k in priv_cols) > 1:
        return None

    # --- Room type exclusivity ---
    rt_cols = [
        "room_type_entire",
        "room_type_private_room",
        "room_type_shared_room",
        "room_type_hotel_room",
    ]
    if sum(row.get(k, 0) for k in rt_cols) > 1:
        return None

    return row


# ============================================================
# SCENARIO PERMUTATION GENERATOR
# ============================================================


def generate_scenarios_from_address(
    address: AddressData,
    max_scenarios: int = 600,
) -> pd.DataFrame:
    """
    Full scenario generation pipeline.
    - Build static (location-only) base
    - Enumerate controllable permutations
    - Enforce constraints
    - Output DataFrame for Pops embedding + prediction

    Doing this with dicts for now!
    """

    base = build_scenario_base_from_address(address)
    base_dict = base.model_dump()

    control_keys = list(SCENARIO_SPACE.keys())
    control_values = [SCENARIO_SPACE[k] for k in control_keys]

    scenarios: List[Dict[str, Any]] = []
    count = 0

    for values in itertools.product(*control_values):
        control_map = dict(zip(control_keys, values))

        # (Optional) validate/control types using ScenarioControls
        controls = ScenarioControls(**control_map)
        control_dict = controls.model_dump()

        # merge base + controls
        row = {**base_dict, **control_dict}

        row = enforce_constraints(row)
        if row is None:
            continue

        scenarios.append(row)
        count += 1
        if max_scenarios and count >= max_scenarios:
            break

    return pd.DataFrame(scenarios)


def build_prototype_grid_from_address(address: AddressData) -> pd.DataFrame:
    """
    Build prototype grid from:
    - static location features (census + distances)
    - default controllable features (all set to 0)
    - size prototypes
    - amenity bundles
    """
    store = get_store()
    pops = store.pipeline

    base = build_scenario_base_from_address(address)
    base_dict = base.model_dump()

    #
    # NEW: inject all controllable columns with safe defaults
    #
    for col in SCENARIO_SPACE:
        if col not in base_dict:
            # use first entry as “default” (usually 0)
            base_dict[col] = SCENARIO_SPACE[col][0]

    rows: List[Dict[str, Any]] = []

    for size_proto in SIZE_PROTOTYPES:
        for amenity_proto in AMENITY_BUNDLES:
            row = base_dict.copy()
            row.update(size_proto)
            row.update(amenity_proto)

            row_obj = enforce_constraints(row)
            if row_obj is None:
                continue

            row_clean = row_obj.model_dump() if hasattr(row_obj, "model_dump") else row_obj

            rows.append(row_clean)

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("No valid prototypes generated.")

    # df = short_term_scenario_cleaning(df)
    df["city"] = "chicago-il"
    df["avg_price"] = 577.59
    df["med_price"] = 169.0
    df.rename(
        columns=COL_FIXES,
        inplace=True,
    )

    # 4. Inject embeddings EXACTLY as Pops does before prediction
    df_emb = pops.embedder.transform(df)

    return df_emb


def compute_location_shap_top_levers(
    address: AddressData,
    top_k: int = 6,
) -> List[str]:
    """
    For a given address, build a grid of prototype properties,
    compute SHAP values for the revenue model, and return the
    top_k most important controllable features.
    """

    df_proto = build_prototype_grid_from_address(address)
    store = get_store()

    pops = store.pipeline

    # Make sure we only feed modeling columns to the model:
    X = df_proto[pops.modeling_cols]

    exp = pops.shap_rev_explainer(X)  # shap.Explanation

    # exp.values: (n_samples, n_features)
    vals = np.abs(exp.values).mean(axis=0)  # mean |SHAP| per feature
    feature_names = np.array(exp.feature_names)

    importance = pd.Series(vals, index=feature_names)

    # restrict to controllables that actually exist in this model
    controllables_present = [c for c in CONTROLLABLE_COLS if c in importance.index]
    if not controllables_present:
        raise ValueError("No controllable columns found in SHAP importance index.")

    importance_ctrl = importance.loc[controllables_present]
    top = importance_ctrl.sort_values(ascending=False).head(top_k)

    return list(top.index)


def generate_scenarios_guided_by_shap(
    address: AddressData,
    max_scenarios: int = 200,
    top_k: int = 6,
) -> pd.DataFrame:
    """
    Use SHAP over prototypes to pick top controllable levers,
    then generate scenarios varying only those levers.
    """
    top_levers = compute_location_shap_top_levers(address, top_k=top_k)

    base = build_scenario_base_from_address(address)
    base_dict = base.model_dump()

    # Only vary keys that are in both SHAP top levers and SCENARIO_SPACE
    active_keys = [k for k in top_levers if k in SCENARIO_SPACE]
    if not active_keys:
        # fall back to original behavior or raise
        raise ValueError("No overlap between SHAP top levers and SCENARIO_SPACE.")

    control_values = [SCENARIO_SPACE[k] for k in active_keys]

    scenarios: List[Dict[str, Any]] = []
    count = 0

    for values in itertools.product(*control_values):
        row = base_dict.copy()
        for k, v in zip(active_keys, values):
            row[k] = v

        row_obj = enforce_constraints(row)
        if row_obj is None:
            continue

        row_clean = row_obj.model_dump() if hasattr(row_obj, "model_dump") else row_obj
        scenarios.append(row_clean)
        count += 1

        if max_scenarios and count >= max_scenarios:
            break

    if not scenarios:
        raise ValueError("No valid SHAP-guided scenarios generated for this address.")

    df_scen = pd.DataFrame(scenarios)
    return df_scen
