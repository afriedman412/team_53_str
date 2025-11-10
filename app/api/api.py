from fastapi import APIRouter, HTTPException, Form
from fastapi.responses import ORJSONResponse
from app.utils.helpers import extract_address_from_url, format_property_data
from app.utils.processing import process_address

router = APIRouter(prefix="/api")


@router.post("/from_url", response_class=ORJSONResponse)
async def api_from_url(url: str = Form(..., description="Zillow or Redfin listing URL")):
    """
    Accepts a Zillow/Redfin URL via POST form or JSON body,
    returns structured property data and predictions.
    """
    try:
        address = extract_address_from_url(url)
        prop_dict = process_address(address)
        formatted_prop_dict = format_property_data(prop_dict)

        result = {
            "source_url": url,
            "address": address.model_dump(),
            "inputs": formatted_prop_dict,  # raw extracted features
        }
        return ORJSONResponse(content=result, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process URL: {e}")
