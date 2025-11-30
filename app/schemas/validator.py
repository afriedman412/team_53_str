import pandera.pandas as pa
from pandera import DataFrameSchema, Column, Check
from app.model.helpers import merge_schemas


# ================================================================
# 1. Census / Neighborhood Features
# ================================================================

CensusFeaturesSchema = DataFrameSchema(
    {
        # --- Housing & Units ---
        "median_home_value": Column(float, Check.ge(0)),
        "median_gross_rent": Column(float, Check.ge(0)),
        "median_year_built": Column(float, Check.ge(0)),
        "total_housing_units": Column(float, Check.ge(0)),
        "vacancy_rate": Column(float, Check.ge(0), nullable=True),
        # --- Income / Economics ---
        "median_income": Column(float, Check.ge(0)),
        "labor_force": Column(float, Check.ge(0)),
        "unemployed": Column(float, Check.ge(0)),
        "unemployment_rate": Column(float, Check.ge(0), nullable=True),
        "poverty_rate": Column(float, Check.ge(0), nullable=True),
        "gini_index": Column(float, Check.ge(0)),
        # --- Education ---
        "education_bachelors": Column(float, Check.ge(0)),
        # --- Demographics ---
        "race_white": Column(float, Check.ge(0)),
        "race_black": Column(float, Check.ge(0)),
        "race_asian": Column(float, Check.ge(0)),
        "race_other": Column(float, Check.ge(0)),
        "percent_foreign_born": Column(float, Check.ge(0), nullable=True),
        # --- Age ---
        "median_age": Column(float, Check.ge(0)),
        "percent_over_65": Column(float, Check.ge(0), nullable=True),
        # --- Population ---
        "population": Column(float, Check.ge(0)),
        # --- Commute / Lifestyle ---
        "commute_time_mean": Column(float, Check.ge(0)),
        "percent_public_transport": Column(float, Check.ge(0), nullable=True),
        "percent_owner_occupied": Column(float, Check.ge(0), nullable=True),
        "percent_work_from_home": Column(float, Check.ge(0), nullable=True),
        # --- Distances ---
        "dist_to_bus_km": Column(float, Check.ge(0)),
        "dist_to_park_km": Column(float, Check.ge(0)),
        "dist_to_train_km": Column(float, Check.ge(0)),
        "dist_to_airport_km": Column(float, Check.ge(0)),
        "dist_to_university_km": Column(float, Check.ge(0)),
        "dist_to_city_center_km": Column(float, Check.ge(0)),
    }
)


# ================================================================
# 2. Performance features (used ONLY for embedding training)
# ================================================================

PerfFeaturesSchema = DataFrameSchema(
    {
        "estimated_occupancy_l365d": Column(float),
        "estimated_revenue_l365d": Column(float),
        "price_capped": Column(float),
        "availability_30": Column(float),
        "availability_60": Column(float),
        "availability_90": Column(float),
        "reviews_per_month": Column(float),
        "number_of_reviews_ltm": Column(float),
        "review_scores_rating": Column(float),
        "number_of_reviews": Column(float),
        "number_of_reviews_l30d": Column(float),
        "number_of_reviews_ly": Column(float),
        "days_as_host": Column(float),
        "days_since_first_review": Column(float),
        "days_since_last_review": Column(float),
    }
)


# ================================================================
# 3. Coordinates (also required for embedding)
# ================================================================

CoordinatesSchema = DataFrameSchema(
    {
        "latitude": Column(float),
        "longitude": Column(float),
    }
)


# ================================================================
# 4. City (embedder grouping)
# ================================================================

CitySchema = DataFrameSchema({"city": Column(str)})


# ================================================================
# 5. Perf Embeddings (perf_emb_0 .. perf_emb_31)
# ================================================================

PerfEmbeddingSchema = DataFrameSchema({**{f"perf_emb_{i}": Column(float) for i in range(32)}})


# ================================================================
# 6. Structural Features (amenities + listing structure + census)
# ================================================================

StructuralCoreSchema = DataFrameSchema(
    {
        # --- Amenities ---
        "hot_tub": Column(bool),
        "air_conditioning": Column(bool),
        "housekeeping": Column(bool),
        "pool_shared": Column(bool),
        "paid_parking": Column(bool),
        "hot_tub_shared": Column(bool),
        "pool_outdoor": Column(bool),
        "pool_indoor": Column(bool),
        "hot_tub_private": Column(bool),
        "heating": Column(bool),
        "gym": Column(bool),
        "free_parking": Column(bool),
        "pool_private": Column(bool),
        "pool": Column(bool),
        # --- Listing structure ---
        "beds": Column(float, nullable=True),
        "bathrooms": Column(float, nullable=True),
        "bedrooms": Column(float, nullable=True),
        "accommodates": Column(float),
        # --- Pricing ---
        "avg_price": Column(float, Check.ge(0)),
        "med_price": Column(float, Check.ge(0)),
        # --- Room types ---
        "room_type_hotel_room": Column(bool),
        "room_type_private_room": Column(bool),
        "room_type_shared_room": Column(bool),
        # --- Privacy ---
        "privacy_private": Column(bool),
        "privacy_room_in": Column(bool),
        "privacy_shared": Column(bool),
    }
)


StructuralSchema = merge_schemas(
    StructuralCoreSchema,
    CensusFeaturesSchema,
    CoordinatesSchema,
)

# Full training features (structural + embeddings)
FullTrainingSchema = merge_schemas(
    StructuralSchema,
    PerfEmbeddingSchema,
)

# Embedding step: city + perf features + structural
EmbeddingSchema = merge_schemas(
    CitySchema,
    PerfFeaturesSchema,
    StructuralSchema,
)

# Price model training
PriceTrainingSchema = merge_schemas(
    FullTrainingSchema,
    DataFrameSchema({"price_capped": Column(float)}),
)

# Occupancy model training
OccupancyTrainingSchema = merge_schemas(
    FullTrainingSchema,
    DataFrameSchema({"estimated_occupancy_l365d": Column(float)}),
)

# Revenue model training
RevenueTrainingSchema = merge_schemas(
    FullTrainingSchema,
    DataFrameSchema(
        {
            "rev_base_log": Column(float),
            # "estimated_revenue_l365d": Column(float),
        }
    ),
)

# Prediction schema
PredictSchema = FullTrainingSchema
