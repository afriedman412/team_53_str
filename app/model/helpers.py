import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from app.core.col_control import STRUCTURAL_FEATS


def get_modeling_columns(df):
    perf_cols = df.filter(regex="perf").columns
    modeling_cols = STRUCTURAL_FEATS + list(perf_cols)
    return modeling_cols


def some_cleaning(df, cat_df):
    df["price_capped"] = (
        df.groupby("slice")["price"]
        .transform(lambda s: np.minimum(s, s.quantile(0.99)))
        .clip(upper=5000)  # global hard cap
    )

    df["privacy"] = df["privacy"].fillna("entire")

    df.loc[:, cat_df.query("category=='PROXIMITY'").feature.unique()] = df.loc[
        :, cat_df.query("category=='PROXIMITY'").feature.unique()
    ].fillna(9999)

    df = df[df["zip"].notnull()]

    df["host_has_profile_pic"] = df["host_has_profile_pic"].map({"t": 1, "f": 0}).astype(bool)

    df = pd.concat(
        [df, pd.get_dummies(df[["privacy", "room_type", "host_response_time"]], drop_first=True)],
        axis=1,
    ).drop(columns=["privacy", "room_type", "host_response_time", "amenities", "price"])
    return df


def feature_plot(fi, n=25):
    # Order features by mean importance (descending) and take top 25
    order = fi.groupby("feature")["importance"].mean().sort_values(ascending=False).head(n).index

    plt.figure(figsize=(7, 5))
    sns.barplot(
        data=fi[fi["feature"].isin(order)],
        x="importance",
        y="feature",
        order=order,  # sorted by mean importance
        orient="h",
        palette="Set2",
        errorbar=None,
        hue="feature",
        legend=False,
    )
    plt.xlabel("Importance (gain)")
    plt.ylabel("")
    plt.title("Top 25 Features by Importance (per-fold distribution)")
    plt.tight_layout()
    plt.show()
