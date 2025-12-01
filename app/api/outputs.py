# app/routers/output.py
from fastapi import APIRouter, HTTPException, Form, Request
from fastapi.responses import ORJSONResponse
import pandas as pd
from urllib.parse import quote
from app.core.registry import get_store
from app.utils.helpers import extract_address_from_url
from app.core.templates import templates
from app.utils.input_builder import build_base
from app.utils.ai_analyzer import PropertyInvestmentAnalyzer
from app.model.helpers import coerce_df_to_match_schema
from app.schemas.validator import StructuralSchema
from app.utils.perm_builder import (
    SCENARIO_DEFAULTS,
    scenario_df_verify,
)


router = APIRouter(prefix="/output")


@router.post("/preds_w_AI", response_class=ORJSONResponse)
async def preds_with_ai(
    request: Request,
    url: str = Form(..., description="Zillow or Redfin listing URL"),
    bedrooms: int = Form(..., description="Number of bedrooms"),
    bathrooms: float = Form(..., description="Number of bathrooms"),
    accommodates: int = Form(..., description="Max guest capacity"),
):
    if not url:
        raise HTTPException(400, "Missing url")

    store = get_store()

    address = extract_address_from_url(url)
    base = build_base(address=address)
    base.update(SCENARIO_DEFAULTS)

    input_values = {
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "accommodates": accommodates,
        "beds": bedrooms,
    }

    base.update(input_values)
    base_df = scenario_df_verify(pd.DataFrame(base, index=[0]))
    base_df["dist_to_bus_km"] = 50
    base_df, changes = coerce_df_to_match_schema(base_df, StructuralSchema)
    prop_dict = base_df.iloc[0].to_dict()
    # 3) Generate predictions
    pred_df = store.pipeline.predict(base_df)
    preds = pred_df.iloc[0].to_dict()

    lat = address.latitude
    lon = address.longitude

    # Extract raw numeric values for AI (not formatted strings)
    price_pred_raw = preds.get('price_pred', 0)
    occ_pred_raw = preds.get("occ_pred", 0)
    rev_pred_raw = preds.get("rev_final_pred", 0)

    maps_url = (
        f"https://www.google.com/maps?q={lat},{lon}"
        if lat is not None and lon is not None
        else f"https://www.google.com/maps/search/{quote(address)}"
    )

    # Generate AI investment summary
    ai_summary = None
    ai_error = None
    try:
        analyzer = PropertyInvestmentAnalyzer()

        ai_summary = analyzer.generate_investment_summary(
            address=address.address,
            price_pred=price_pred_raw,
            occ_pred=occ_pred_raw,
            rev_pred=rev_pred_raw,
            property_features=prop_dict
        )
    except Exception as e:
        # If AI fails, continue without it (ML predictions still show)
        ai_error = f"AI analysis unavailable: {str(e)}"

    return templates.TemplateResponse(
        "output.html",
        {
            "request": request,
            "prop_dict": prop_dict,
            "price_pred": price_pred_raw,
            "occ_pred": occ_pred_raw,
            "rev_pred": rev_pred_raw,
            "address": address,
            "maps_url": maps_url,
            "ai_summary": ai_summary,
            "ai_error": ai_error,
        },
    )
