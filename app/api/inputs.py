from fastapi import APIRouter, Form
from fastapi.responses import RedirectResponse
from urllib.parse import urlencode

router = APIRouter(prefix="/input", tags=["input"])


@router.post("/from_url")
async def from_url(url: str = Form(...)):
    # minimal: just pass original URL to output and let it compute there
    qs = urlencode({"url": url})
    return RedirectResponse(url=f"/output?{qs}", status_code=303)
