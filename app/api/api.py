from fastapi import APIRouter, HTTPException, Form
from fastapi.responses import ORJSONResponse
from app.utils.helpers import (
    extract_address_from_url,
    format_property_data,
)
from app.core.registry import get_store
from app.utils.input_builder import build_base
from app.utils.perm_builder import assemble_options, get_original_match_mask, SCENARIO_DEFAULTS

router = APIRouter(prefix="/api")


@router.post("/from_url", response_class=ORJSONResponse)
async def api_from_url(url: str = Form(..., description="Zillow or Redfin listing URL")):
    """
    Accepts a Zillow/Redfin URL via POST form or JSON body,
    returns structured property data and predictions.
    """
    try:
        address = extract_address_from_url(url)
        prop_dict = build_base(address)
        formatted_prop_dict = format_property_data(prop_dict)

        result = {
            "source_url": url,
            "address": address.model_dump(),
            "inputs": formatted_prop_dict,  # raw extracted features
        }
        return ORJSONResponse(content=result, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process URL: {e}")


@router.post("/perms_from_url", response_class=ORJSONResponse)
async def scenario_from_url(
    url: str = Form(..., description="Zillow or Redfin listing URL"),
    bedrooms: int = Form(..., description="Number of bedrooms"),
    bathrooms: float = Form(..., description="Number of bathrooms"),
    accommodates: int = Form(..., description="Max guest capacity"),
):
    """
    Build a base listing from a Zillow/Redfin URL and user-provided overrides,
    generate permutations, run prediction + uplift, and return a
    frontend-friendly payload.
    """
    try:
        store = get_store()

        # 1) Get address (and possibly other metadata) from the URL
        address = extract_address_from_url(url)

        # 2) Build the base row, injecting user overrides
        # Adjust this to match your actual build_base signature
        base = build_base(address=address)
        base.update(SCENARIO_DEFAULTS)

        input_values = {
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "accommodates": accommodates,
            "beds": bedrooms,
        }

        # 3) Generate permutations + run predictions + explanations
        permo_df = assemble_options(base, input_values)
        preds_and_exp = store.pipeline.predict_and_explain(permo_df)

        # 4) Shape a clean response for the UI
        idx = get_original_match_mask(permo_df, input_values).index[0]
        content = {
            "address": str(address),
            "price_pred": preds_and_exp["preds"].at[idx, "price_pred"],
            "occ_pred": int(preds_and_exp["preds"].at[idx, "occ_pred"]),
            "revenue": preds_and_exp["preds"].at[idx, "rev_final_pred"],
            "uplift_table": preds_and_exp["uplift_table"].to_dict(),
            "uplift_chart_png": preds_and_exp["uplift_char_png"],
        }

        return ORJSONResponse(content=content, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process URL: {e}")
