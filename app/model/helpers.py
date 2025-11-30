import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from app.core.col_control import STRUCTURAL_FEATS
from app.core.model_config import MODEL_DATA
from typing import Dict, Any, Iterable
import pandera.pandas as pa
from pandera.errors import SchemaErrors
from pandera import DataFrameSchema


def get_modeling_columns(df, transform=True):
    perf_cols = df.filter(regex="perf").columns
    modeling_cols = STRUCTURAL_FEATS + list(perf_cols)
    if transform:
        return df[modeling_cols]
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


def merge_schemas(*schemas: Iterable[pa.DataFrameSchema]) -> pa.DataFrameSchema:
    """
    Merge multiple DataFrameSchema objects into a single schema by
    combining their columns. Later schemas override earlier ones
    if there are name collisions.
    """
    merged_columns = {}
    for schema in schemas:
        merged_columns.update(schema.columns)
    return pa.DataFrameSchema(merged_columns)


def canonical_data_cleaning(df):
    df_ = df[df["estimated_revenue_l365d"] > 0]
    df_["dist_to_bus_km"] = df_["dist_to_bus_km"].clip(upper=150)

    idx = []
    for k in [
        "gini_index",
        "median_year_built",
        "median_age",
        "median_income",
        "median_home_value",
        "median_gross_rent",
    ]:
        ll = list(df_[df_[k] < 0].index)
        idx += ll

    df_ = df_[~df_.index.isin(idx)]

    return df_


def coerce_df_to_match_schema(
    df: pd.DataFrame,
    schema: DataFrameSchema,
    verbose: bool = True,
):
    """
    Gently coerce dataframe column dtypes to match a Pandera schema.

    Rules:
      - If schema expects bool:
          * If column values are subset of {0, 1, True, False, NaN} → cast to bool
      - If schema expects float:
          * If column is int or bool → cast to float
      - If schema expects int:
          * If column is float and all non-null values are integral (e.g. 1.0, 2.0) → cast to int
      - Otherwise: leave as-is.

    Returns
    -------
    coerced_df : pd.DataFrame
    changes    : list of str  (human-readable descriptions of what was changed)
    """
    df = df.copy()
    changes = []

    for col_name, col_schema in schema.columns.items():
        if col_name not in df.columns:
            continue  # nothing to do for missing columns here

        s = df[col_name]
        expected = str(col_schema.dtype).lower()

        # -------------------------
        # 1) Booleans
        # -------------------------
        if "bool" in expected:
            non_null = s.dropna()
            # Only coerce if it's clearly bool-like
            allowed = {0, 1, True, False}
            if len(non_null) and set(non_null.unique()).issubset(allowed):
                if not pd.api.types.is_bool_dtype(s):
                    df[col_name] = s.astype(bool)
                    changes.append(f"{col_name}: coerced {s.dtype} -> bool")
                    if verbose:
                        print(f"🔧 {col_name}: coerced {s.dtype} -> bool")
            else:
                if verbose:
                    print(
                        f"⚠️ {col_name}: expected bool but values not bool-like, leaving as {s.dtype}"
                    )
            continue

        # -------------------------
        # 2) Floats
        # -------------------------
        if "float" in expected or "double" in expected:
            if pd.api.types.is_integer_dtype(s) or pd.api.types.is_bool_dtype(s):
                df[col_name] = s.astype("float64")
                changes.append(f"{col_name}: coerced {s.dtype} -> float64")
                if verbose:
                    print(f"🔧 {col_name}: coerced {s.dtype} -> float64")
            # if it's already float, do nothing
            continue

        # -------------------------
        # 3) Ints
        # -------------------------
        if "int" in expected:
            # Only coerce float→int if they’re truly integral
            if pd.api.types.is_float_dtype(s):
                non_null = s.dropna()
                if len(non_null) and np.all(np.floor(non_null) == non_null):
                    df[col_name] = non_null.astype("int64").reindex(s.index)
                    changes.append(f"{col_name}: coerced float -> int64 (all values integral)")
                    if verbose:
                        print(f"🔧 {col_name}: coerced float -> int64 (all values integral)")
                else:
                    if verbose:
                        print(
                            f"⚠️ {col_name}: expected int but non-integral floats present, leaving as {s.dtype}"
                        )
            # if it's already int, do nothing
            continue

        # For non-numeric / non-bool expectations, we’re intentionally conservative.
        # (e.g., strings, categoricals, etc.)
        # You can expand this section later if needed.

    return df, changes


def check_schema_compatibility(
    df: pd.DataFrame,
    schema: pa.DataFrameSchema,
    df_name: str = "df",
    raise_on_error: bool = True,
) -> Dict[str, Any]:
    """
    Compare a DataFrame to a pandera schema and report:
      - missing columns
      - extra columns
      - pandera validation errors (dtypes, ranges, nullability, etc.)

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to check.
    schema : pa.DataFrameSchema
        The pandera schema to validate against.
    df_name : str, optional
        Friendly name for the dataframe in messages.
    raise_on_error : bool, optional
        If True, re-raise SchemaErrors after collecting info.

    Returns
    -------
    Dict[str, Any]
        {
          "missing_columns": [...],
          "extra_columns": [...],
          "schema_errors": list of error dicts (may be empty)
        }
    """
    result: Dict[str, Any] = {
        "missing_columns": [],
        "extra_columns": [],
        "schema_errors": [],
    }

    schema_cols = set(schema.columns.keys())
    df_cols = set(df.columns)

    # Column-level comparison
    missing = sorted(schema_cols - df_cols)
    extra = sorted(df_cols - schema_cols)

    result["missing_columns"] = missing
    result["extra_columns"] = extra

    # Only run full validation if we at least have the schema columns present
    if not missing:
        try:
            # lazy=True collects all violations instead of failing fast
            schema.validate(df, lazy=True)
        except SchemaErrors as err:
            # Pandera gives a rich errors dataframe; convert to dicts for easy logging
            result["schema_errors"] = err.failure_cases.to_dict(orient="records")
            if raise_on_error:
                raise

    # Optional: nice log printout
    print(f"🔎 Schema check for {df_name}:")
    if missing:
        print(f"  ❌ Missing columns: {missing}")
    if extra:
        print(f"  ⚠️ Extra columns (ignored by schema): {extra}")
    if result["schema_errors"]:
        print(
            f"  ❌ {len(result['schema_errors'])} schema violations (see result['schema_errors'])"
        )
    if not missing and not extra and not result["schema_errors"]:
        print("  ✅ DataFrame matches schema.")

    return result
