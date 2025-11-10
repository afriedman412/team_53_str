# app/routers/output.py
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse
from app.utils.helpers import extract_address_from_url, format_property_data
from app.utils.processing import process_address
from app.core.templates import templates
from app.core.config import percent_keys

router = APIRouter(prefix="/output")


@router.get("", response_class=HTMLResponse)
def show_result(request: Request, url: str):
    if not url:
        raise HTTPException(400, "Missing url")
    address = extract_address_from_url(url)
    joined_address = ", ".join(address)
    prop_dict = process_address(address)
    formatted_prop_dict = format_property_data(prop_dict)
    # ... inside build_input_endpoint after you have prop_dict
    lat = prop_dict.get("latitude")
    lon = prop_dict.get("longitude")

    maps_url = (
        f"https://www.google.com/maps?q={lat},{lon}" if lat is not None and lon is not None
        else f"https://www.google.com/maps/search/{quote(joined_address)}"
    )
    return templates.TemplateResponse("output.html", {
        "request": request,
        "prop_dict": formatted_prop_dict,
        "price_pred": formatted_prop_dict.get("price_pred"),
        "occ_pred": formatted_prop_dict.get("occ_pred"),
        "rev_pred": formatted_prop_dict.get("rev_pred"),
        "percent_keys": percent_keys,
        "address": joined_address,
        "maps_url": maps_url,
    })
