# app/schemas/input.py
from pydantic import BaseModel, ConfigDict


class InputData(BaseModel):
    # Strict schema matching your features; expand/adjust as needed
    neighbourhood_cleansed: str
    latitude: float
    longitude: float
    # or Literal["Entire home/apt","Private room","Shared room","Hotel room"]
    room_type: str
    accommodates: float
    bathrooms: float
    bedrooms: float
    beds: float
    # or Literal["entire","private","shared"]
    privacy: str
    state: str  # or Literal["ma","il","ny", ...]
    air_conditioning: bool
    heating: bool
    free_parking: bool
    gym: bool
    housekeeping: bool
    pool: bool
    pool_shared: bool
    pool_private: bool
    pool_indoor: bool
    pool_outdoor: bool
    hot_tub: bool
    hot_tub_shared: bool
    hot_tub_private: bool
    paid_parking: bool
    avg_price: float
    med_price: float
    dist_to_airport_km: float
    dist_to_train_km: float
    dist_to_park_km: float
    dist_to_university_km: float
    dist_to_bus_km: float
    dist_to_city_center_km: float
    median_income: float
    median_gross_rent: float
    population: float
    median_home_value: float
    education_bachelors: float
    median_age: float
    race_white: float
    race_black: float
    race_asian: float
    race_other: float
    median_year_built: float
    total_housing_units: float
    labor_force: float
    unemployed: float
    commute_time_mean: float
    gini_index: float
    percent_foreign_born: float
    unemployment_rate: float
    percent_owner_occupied: float
    percent_public_transport: float
    percent_work_from_home: float
    vacancy_rate: float
    poverty_rate: float
    percent_over_65: float

    # Pydantic config: reject unexpected fields to catch typos early
    model_config = ConfigDict(extra="forbid")
