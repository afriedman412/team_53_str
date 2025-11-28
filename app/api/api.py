import pandas as pd
from fastapi import APIRouter, HTTPException, Form
from fastapi.responses import ORJSONResponse
from app.utils.helpers import (
    extract_address_from_url,
    format_property_data,
)
from app.utils.input_builder import build_scenario_location_base
from app.utils.scenario_generator import generate_scenarios_guided_by_shap
from app.core.registry import get_store

router = APIRouter(prefix="/api")


@router.post("/from_url", response_class=ORJSONResponse)
async def api_from_url(url: str = Form(..., description="Zillow or Redfin listing URL")):
    """
    Accepts a Zillow/Redfin URL via POST form or JSON body,
    returns structured property data and predictions.
    """
    try:
        address = extract_address_from_url(url)
        prop_dict = build_scenario_location_base(address)
        formatted_prop_dict = format_property_data(prop_dict)

        result = {
            "source_url": url,
            "address": address.model_dump(),
            "inputs": formatted_prop_dict,  # raw extracted features
        }
        return ORJSONResponse(content=result, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process URL: {e}")


@router.post("/scenario_from_url", response_class=ORJSONResponse)
async def scenario_from_url(url: str = Form(..., description="Zillow or Redfin listing URL")):
    """
    Same as above but a whole scenario
    """
    # try:
    address = extract_address_from_url(url)

    informed_scenarios_df = generate_scenarios_guided_by_shap(address)

    store = get_store()
    preds = store.pipeline.predict(informed_scenarios_df)
    print(preds.shape)
    df = pd.concat([informed_scenarios_df, preds], axis=1)

    scenarios_json = df.to_json()

    return ORJSONResponse(content=scenarios_json, status_code=200)

    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=f"Failed to process URL: {e}")
