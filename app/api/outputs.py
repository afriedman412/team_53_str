# app/routers/output.py
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse
from app.utils.helpers import extract_address_from_url, format_property_data
from app.core.templates import templates
from app.utils.input_builder import build_base
from app.utils.ai_analyzer import PropertyInvestmentAnalyzer
from urllib.parse import quote

router = APIRouter(prefix="/output")


@router.get("", response_class=HTMLResponse)
def show_result(request: Request, url: str):
    if not url:
        raise HTTPException(400, "Missing url")
    address = extract_address_from_url(url)
    prop_dict = build_base(address)
    formatted_prop_dict = format_property_data(prop_dict)
    lat = prop_dict.get("latitude")
    lon = prop_dict.get("longitude")

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
        
        # Extract raw numeric values for AI (not formatted strings)
        price_pred_raw = prop_dict.get("price_pred")
        occ_pred_raw = prop_dict.get("occ_pred")
        rev_pred_raw = prop_dict.get("rev_pred")
        
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
            "prop_dict": formatted_prop_dict,
            "price_pred": formatted_prop_dict.get("price_pred"),
            "occ_pred": formatted_prop_dict.get("occ_pred"),
            "rev_pred": formatted_prop_dict.get("rev_pred"),
            "address": address,
            "maps_url": maps_url,
            "ai_summary": ai_summary,
            "ai_error": ai_error,
        },
    )
