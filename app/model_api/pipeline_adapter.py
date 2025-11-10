# app/models/pipeline_adapter.py
import numpy as np
import pandas as pd
from typing import Optional
from app.core.store import DataStore
from app.core.registry import get_store


def predict_from_saved(df_input: pd.DataFrame, store: Optional[DataStore] = None) -> pd.DataFrame:
    store = store or get_store()
    pipeline = store.pipeline
    X_eval = df_input[pipeline.feature_cols].copy()

    price_pred = pipeline.m_price.predict(X_eval)
    occ_pred = pipeline.m_occ.predict(X_eval)

    if getattr(pipeline, "make_cross_features", False):
        X_eval["occ_x_price"] = occ_pred * price_pred
        X_eval["log_occ_x_price"] = np.log1p(X_eval["occ_x_price"])

    rev_pred = pipeline.m_rev.predict(X_eval)

    preds = {
        "price_pred": price_pred.item(),
        "occ_pred": occ_pred.item(),
        "rev_pred": rev_pred.item(),
    }

    return preds
