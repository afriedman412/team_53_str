# Distance band cutpoints (in miles)
DISTANCE_BANDS = [1, 2, 4]

# PCA dimensionality
PCA_COMPONENTS = 10

DF_PATH = "data/df_clean_city_subset_111725.csv"

EMBEDDING_CONFIG = {
    "k_neighbors_graph": 20,
    "k_neighbors_infer": 10,
    "dimensions": 32,
    "min_city_size": 200,
}

MODEL_DATA = {
    "price": {"target": "price_capped", "params": {"num_leaves": 63, "learning_rate": 0.03}},
    "occupancy": {
        "target": "estimated_occupancy_l365d",
        "params": {
            "objective": "tweedie",
            "tweedie_variance_power": 1.3,
            "learning_rate": 0.03,
            "num_leaves": 63,
        },
    },
    "revenue_corr": {
        "target": "rev_corr_target",
        "params": {
            "num_leaves": 31,
            "learning_rate": 0.03,
            "feature_fraction": 0.8,
        },
    },
}
