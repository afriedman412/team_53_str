from fastapi import APIRouter
from app.core.deps import get_store

router = APIRouter()


@router.get("/_store_info")
def get_store_info():
    """Return summary info about the currently loaded DataStore."""
    try:
        store = get_store()
    except RuntimeError as e:
        return {"error": str(e)}

    # summarize what’s loaded
    info = {
        "gdf_features_rows": len(store.gdf_features),
        "zips_rows": len(store.zips),
        "trees_rows": len(store.trees) if store.trees is not None else 0,
        "meta_keys": list(store.meta.keys()) if store.meta else [],
        "geolocator": getattr(store.geolocator, "user_agent", None),
        "feature_index_type": type(store.feature_index).__name__,
    }
    return info
