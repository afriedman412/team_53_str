from fastapi import APIRouter
from app.core.registry import get_store

router = APIRouter()


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
