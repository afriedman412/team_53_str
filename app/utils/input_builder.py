from pathlib import Path
import pandas as pd
from app.utils.distance_calc import calc_distances
from app.utils.helpers import call_attom_property_detail
from app.schemas.input import InputData

CENSUS_PATH = Path(__file__).resolve(
).parents[2] / "data" / "census_data.parquet"


def build_input(address: str) -> InputData:
    """
    Build a fully populated InputData instance from a Zillow/Redfin URL.
    Combines ATTOM, distances, and ZIP-level census data.
    """

    attom = call_attom_property_detail(*address)
    prop = attom["property"][0]

    # --- Property-level values ---
    lat = float(prop["location"]["latitude"])
    lon = float(prop["location"]["longitude"])
    zip_code = prop["address"]["postal1"]
    state = prop["address"]["countrySubd"].lower()

    building = prop.get("building", {})
    rooms = building.get("rooms", {})
    utilities = prop.get("utilities", {})
    parking = building.get("parking", {})

    air_conditioning = utilities.get(
        "coolingtype", "").upper().startswith("CENTRAL")
    heating = bool(utilities)
    free_parking = bool(parking)
    beds = float(rooms.get("beds", 0) or 0)
    baths = float(rooms.get("bathstotal", 0) or 0)
    accommodates = max(1.0, beds * 2 or 2.0)

    # 2️⃣ Distances
    dists = calc_distances(lat, lon)
    # if you have this calc elsewhere, insert it
    dist_to_city_center = float("nan")

    # 3️⃣ Census data by ZIP
    census = pd.read_parquet(CENSUS_PATH)
    row = census.loc[census["zip"].astype(str) == str(zip_code)]
    census_dict = row.iloc[0].to_dict() if not row.empty else {}

    # Helper to get float safely
    def fget(d, key):
        return float(d.get(key, float("nan")))

    # 4️⃣ Return validated InputData
    return InputData(
        neighbourhood_cleansed="unknown",
        latitude=lat,
        longitude=lon,
        room_type="Entire home/apt",
        accommodates=accommodates,
        bathrooms=baths,
        bedrooms=beds,
        beds=beds,
        privacy="entire",
        state=state,
        air_conditioning=air_conditioning,
        heating=heating,
        free_parking=free_parking,
        gym=False,
        housekeeping=False,
        pool=False,
        pool_shared=False,
        pool_private=False,
        pool_indoor=False,
        pool_outdoor=False,
        hot_tub=False,
        hot_tub_shared=False,
        hot_tub_private=False,
        paid_parking=False,
        avg_price=float("nan"),
        med_price=float("nan"),
        dist_to_airport_km=fget(dists, "dist_to_airport_km"),
        dist_to_train_km=fget(dists, "dist_to_train_km"),
        dist_to_park_km=fget(dists, "dist_to_park_km"),
        dist_to_university_km=fget(dists, "dist_to_university_km"),
        dist_to_bus_km=fget(dists, "bus"),
        dist_to_city_center_km=dist_to_city_center,
        median_income=fget(census_dict, "median_income"),
        median_gross_rent=fget(census_dict, "median_gross_rent"),
        population=fget(census_dict, "population"),
        median_home_value=fget(census_dict, "median_home_value"),
        education_bachelors=fget(census_dict, "education_bachelors"),
        median_age=fget(census_dict, "median_age"),
        race_white=fget(census_dict, "race_white"),
        race_black=fget(census_dict, "race_black"),
        race_asian=fget(census_dict, "race_asian"),
        race_other=fget(census_dict, "race_other"),
        median_year_built=fget(census_dict, "median_year_built"),
        total_housing_units=fget(census_dict, "total_housing_units"),
        labor_force=fget(census_dict, "labor_force"),
        unemployed=fget(census_dict, "unemployed"),
        commute_time_mean=fget(census_dict, "commute_time_mean"),
        gini_index=fget(census_dict, "gini_index"),
        percent_foreign_born=fget(census_dict, "percent_foreign_born"),
        unemployment_rate=fget(census_dict, "unemployment_rate"),
        percent_owner_occupied=fget(census_dict, "percent_owner_occupied"),
        percent_public_transport=fget(census_dict, "percent_public_transport"),
        percent_work_from_home=fget(census_dict, "percent_work_from_home"),
        vacancy_rate=fget(census_dict, "vacancy_rate"),
        poverty_rate=fget(census_dict, "poverty_rate"),
        percent_over_65=fget(census_dict, "percent_over_65"),
    )
