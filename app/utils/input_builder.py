import pandas as pd
import json
from app.utils.distance_calc import calc_distances, calc_city_center_distance, validate_address_data
from app.core.config import DATA_PATHS
from app.utils.helpers import build_fget, extract_address_from_url
from app.schemas.property import PropertyGeographic, AddressData


def get_distance_features(address: AddressData) -> dict:
    distances = calc_distances(address.latitude, address.longitude)
    distances["dist_to_city_center_km"] = calc_city_center_distance(address)

    fget = build_fget(distances)

    return {
        "dist_to_airport_km": fget("dist_to_airport_km"),
        "dist_to_train_km": fget("dist_to_train_km"),
        "dist_to_park_km": fget("dist_to_park_km"),
        "dist_to_university_km": fget("dist_to_university_km"),
        "dist_to_bus_km": fget("bus"),
        "dist_to_city_center_km": fget("dist_to_city_center_km"),
    }


def get_census_features(zipcode: str) -> dict:
    census = pd.read_parquet(DATA_PATHS["census"])
    row = census.loc[census["zip"].astype(str) == str(zipcode)]
    data = row.iloc[0].to_dict() if not row.empty else {}

    fget = build_fget(data)

    return {
        "median_income": fget("median_income"),
        "median_gross_rent": fget("median_gross_rent"),
        "population": fget("population"),
        "median_home_value": fget("median_home_value"),
        "education_bachelors": fget("education_bachelors"),
        "median_age": fget("median_age"),
        "race_white": fget("race_white"),
        "race_black": fget("race_black"),
        "race_asian": fget("race_asian"),
        "race_other": fget("race_other"),
        "median_year_built": fget("median_year_built"),
        "total_housing_units": fget("total_housing_units"),
        "labor_force": fget("labor_force"),
        "unemployed": fget("unemployed"),
        "commute_time_mean": fget("commute_time_mean"),
        "gini_index": fget("gini_index"),
        "percent_foreign_born": fget("percent_foreign_born"),
        "unemployment_rate": fget("unemployment_rate"),
        "percent_owner_occupied": fget("percent_owner_occupied"),
        "percent_public_transport": fget("percent_public_transport"),
        "percent_work_from_home": fget("percent_work_from_home"),
        "vacancy_rate": fget("vacancy_rate"),
        "poverty_rate": fget("poverty_rate"),
        "percent_over_65": fget("percent_over_65"),
    }


def build_scenario_location_base(address: AddressData) -> dict:
    """
    Build a clean, ATTOM-free static feature dict:
    - latitude / longitude
    - state / city / zip
    - distance features
    - census features
    """

    # ensure lat/lon exist
    if address.latitude is None or address.longitude is None:
        address = validate_address_data(address)

    base = {
        "latitude": address.latitude,
        "longitude": address.longitude,
        "state": address.state,
        "zipcode": address.zipcode,
    }

    # distances
    base.update(get_distance_features(address))

    # census
    base.update(get_census_features(address.zipcode))

    return base


def build_scenario_base_from_address(address: AddressData) -> PropertyGeographic:
    base_dict = build_scenario_location_base(address)
    return PropertyGeographic(**base_dict)


def janky_url_loader(url, json_path):
    with open(json_path) as f:
        jjj = f.read()
        j = json.loads(jjj)

    address = extract_address_from_url(url)

    for l in ["latitude", "longitude"]:
        if address.__getattribute__(l) is None:
            address.__setattr__(l, j["address"][l])

    distances = {
        k: float(j["inputs"][k].split()[0])
        for k in [
            "dist_to_airport_km",
            "dist_to_train_km",
            "dist_to_park_km",
            "dist_to_university_km",
            "dist_to_bus_km",
            "dist_to_city_center_km",
        ]
    }

    distances["dist_to_bus_km"] = 5

    base = {
        "latitude": address.latitude,
        "longitude": address.longitude,
        "state": address.state,
        "zipcode": address.zipcode,
    }

    base.update(distances)
    base.update(get_census_features(address.zipcode))
    return base


def validate_scenario_df(scenario_df, city="chicago-il"):
    scenario_df["city"] = "chicago-il"
    scenario_df["slice"] = "chicago-il"
    scenario_df["avg_price"] = 577.59
    scenario_df["med_price"] = 169.0
    return scenario_df
