# app/routers/scenario_test.py

import json
from fastapi import APIRouter, Form
from app.utils.scenario_generator import build_prototype_grid_from_address
from app.utils.helpers import extract_address_from_url, COL_FIXES
from app.schemas.property import AddressData
from app.core.registry import get_store

router = APIRouter(prefix="/debug_scenarios", tags=["scenario-debug"])


@router.post("/prototypes")
def debug_prototypes(url: str = Form(..., description="Zillow or Redfin listing URL")):
    """
    Accepts a Zillow/Redfin URL via POST form or JSON body,
    returns structured property data and predictions.
    """
    address = extract_address_from_url(url)
    df = build_prototype_grid_from_address(address)
    df.rename(columns=COL_FIXES, inplace=True)

    sample = json.loads(df.head(10).to_json(orient="records"))

    return {"n_rows": len(df), "columns": list(df.columns), "sample": sample}


@router.post("/shap_raw")
def debug_shap(url: str = Form(..., description="Zillow or Redfin listing URL")):
    address = extract_address_from_url(url)
    df_proto = build_prototype_grid_from_address(address)

    store = get_store()
    pops = store.pipeline

    proto_preds = pops.predict(df_proto)

    exp = pops.shap_rev_explainer(pops.df_with_embeds)

    # Return mean absolute shap for ALL features
    abs_mean = exp.abs.mean(0)  # shap.Explanation supports this syntax

    result = {
        "n_prototypes": len(df_proto),
        "top_20_features": {f: float(abs_mean[f]) for f in abs_mean.argsort()[-20:][::-1]},
    }
    return result
