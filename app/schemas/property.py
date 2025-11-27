# app/schemas/input.py
from pydantic import BaseModel, ConfigDict, Field


class AddressData(BaseModel):
    street_tokens: list[str] = Field(default_factory=list)
    city: str = ""
    state: str = ""
    zipcode: str = ""
    address1: str = ""
    address2: str = ""
    address: str = ""
    latitude: float | None = None
    longitude: float | None = None

    # Make the instance callable -> returns the full address string
    def __call__(self) -> str:
        return self.address

    # Make str(instance) also return the full address
    def __str__(self) -> str:
        return self.address

    # Nice repr for logs
    def __repr__(self) -> str:
        return f"AddressData({self.address!r})"


class PropertyGeographic(BaseModel):
    latitude: float
    longitude: float
    state: str
    zipcode: str

    # Distances
    dist_to_airport_km: float
    dist_to_train_km: float
    dist_to_park_km: float
    dist_to_university_km: float
    dist_to_bus_km: float
    dist_to_city_center_km: float

    # Census
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

    model_config = {"extra": "ignore"}  # base is strict but ignore extra fields


class PropertyControls(BaseModel):
    bedrooms: int
    beds: int
    accommodates: int
    bathrooms: float

    air_conditioning: bool
    heating: bool
    free_parking: bool
    paid_parking: bool

    gym: bool
    housekeeping: bool

    pool: bool
    pool_private: bool
    pool_shared: bool
    pool_indoor: bool
    pool_outdoor: bool

    hot_tub: bool
    hot_tub_private: bool
    hot_tub_shared: bool

    # Privacy (one-hot)
    privacy_private: bool = False
    privacy_room_in: bool = False
    privacy_shared: bool = False

    # Room type (one-hot)
    room_type_entire: bool = False
    room_type_private_room: bool = False
    room_type_shared_room: bool = False
    room_type_hotel_room: bool = False

    model_config = {"extra": "ignore"}


class PropertyData(PropertyGeographic, PropertyControls):
    model_config = {"extra": "ignore"}
