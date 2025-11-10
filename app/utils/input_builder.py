import pandas as pd
from app.utils.distance_calc import calc_distances, calc_city_center_distance, validate_address_data
from app.utils.helpers import call_attom_property_detail
from app.core.config import DATA_PATHS
from app.schemas.property import PropertyData, AddressData


def parse_attom(attom):
    """
    Parses the output of a call to the ATTOM API
    """
    prop = attom["property"][0]

    attom_dict = {}
    attom_dict["building"] = prop.get("building", {})
    attom_dict["rooms"] = attom_dict["building"].get("rooms", {})
    attom_dict["utilities"] = prop.get("utilities", {})
    attom_dict["parking"] = attom_dict["building"].get("parking", {})

    attom_dict["air_conditioning"] = (
        attom_dict["utilities"].get("coolingtype", "").upper().startswith("CENTRAL")
    )
    attom_dict["heating"] = bool(attom_dict["utilities"])
    attom_dict["free_parking"] = bool(attom_dict["parking"])
    attom_dict["beds"] = float(attom_dict["rooms"].get("beds", 0) or 0)
    attom_dict["baths"] = float(attom_dict["rooms"].get("bathstotal", 0) or 0)
    attom_dict["accommodates"] = max(1.0, attom_dict["beds"] * 2 or 2.0)
    return attom_dict


def assemble_prop_data(address: AddressData) -> PropertyData:
    """
    Build a fully populated PropertyData instance from an AddressData object.
    Combines ATTOM, distances, and ZIP-level census data.
    """

    attom_data = call_attom_property_detail(address)
    attom_dict = parse_attom(attom_data)

    # get lat and lon if needed
    if address.latitude is None or address.latitude is None:
        address = validate_address_data(address)
    distance_calcs = calc_distances(address.latitude, address.longitude)
    distance_calcs["dist_to_city_center_km"] = calc_city_center_distance(address)

    # 3️⃣ Census data by ZIP
    census = pd.read_parquet(DATA_PATHS["census"])
    row = census.loc[census["zip"].astype(str) == str(address.zipcode)]
    census_dict = row.iloc[0].to_dict() if not row.empty else {}

    # Helper to get float safely
    def fget(d, key):
        try:
            return float(d.get(key, float("nan")))
        except Exception:
            return 9675722

    return PropertyData(
        neighbourhood_cleansed="TBD!!!",
        latitude=address.latitude,
        longitude=address.longitude,
        room_type="Entire home/apt",
        accommodates=attom_dict["accommodates"],
        bathrooms=attom_dict["baths"],
        bedrooms=attom_dict["beds"],
        beds=attom_dict["beds"],
        privacy="entire",
        state=address.state,
        air_conditioning=attom_dict["air_conditioning"],
        heating=attom_dict["heating"],
        free_parking=attom_dict["free_parking"],
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
        avg_price=577.59,  # chicago hard-wired
        med_price=169.0,  # chicago hard-wired
        dist_to_airport_km=fget(distance_calcs, "dist_to_airport_km"),
        dist_to_train_km=fget(distance_calcs, "dist_to_train_km"),
        dist_to_park_km=fget(distance_calcs, "dist_to_park_km"),
        dist_to_university_km=fget(distance_calcs, "dist_to_university_km"),
        dist_to_bus_km=fget(distance_calcs, "bus"),
        dist_to_city_center_km=fget(distance_calcs, "dist_to_city_center_km"),
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
