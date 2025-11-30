import pandas as pd
from fastapi import APIRouter, Form
from fastapi.responses import ORJSONResponse
from app.core.registry import get_store
from app.utils.input_builder import build_base
from app.utils.helpers import (
    extract_address_from_url,
)
from app.utils.perm_builder import (
    SCENARIO_DEFAULTS,
    scenario_df_verify,
)

router = APIRouter()


@router.post("/get_embeddings", response_class=ORJSONResponse)
async def get_embeddings(
    url: str = Form(..., description="Zillow or Redfin listing URL"),
):
    store = get_store()
    address = extract_address_from_url(url)

    # 2) Build the base row, injecting user overrides
    # Adjust this to match your actual build_base signature
    base = build_base(address=address)
    base.update(SCENARIO_DEFAULTS)
    base_df = scenario_df_verify(pd.DataFrame(base, index=[0]))

    embo_df = store.pipeline.embedder.transform(base_df)
    embo_dict = embo_df.to_dict("records")
    return ORJSONResponse(content=embo_dict, status_code=200)


@router.get("/_store_info")
def store_info():
    store = get_store()
    provider = (
        type(store._geolocator_instance).__name__
        if store._geolocator_instance is not None
        else None
    )
    return {
        "gdf_features_rows": len(store.gdf_features),
        "trees_rows": 0 if store.trees is None else len(store.trees),
        "meta_keys": list(store.meta.keys()),
        "feature_index_type": "STRtree",
        "geocoder_provider": provider,
        "has_geocoder": provider is not None,
    }
