import numpy as np
from typing import Dict, Any, Optional
from functools import lru_cache

import geopandas as gpd
from shapely.geometry import Point

from app.core.store import DataStore
from app.core.registry import get_store

# --- small per-store cache for row-wise arrays --------------------------------
_ZIP_CTX: Dict[int, Dict[str, Any]] = {}


def _ensure_zip_ctx(store) -> None:
    sid = id(store)
    if sid in _ZIP_CTX:
        return

    zips: gpd.GeoDataFrame = store.zips
    zip_col = "zcta"

    # Ensure projected CRS is Web Mercator (meters) since you query in 3857
    if zips.crs is None or int(
        gpd.GeoSeries([Point(0, 0)], crs=4326).to_crs(3857).crs.to_epsg() or 3857
    ) != int(zips.crs.to_epsg() or 0):
        # If your stored file is already 3857 this is a no-op; otherwise convert once.
        zips = zips.to_crs(3857)

    b = zips.geometry.bounds  # DataFrame with minx, miny, maxx, maxy
    c = zips.geometry.centroid

    _ZIP_CTX[sid] = {
        "zips": zips,  # GeoDataFrame (3857)
        "zip_col": zip_col,
        "minx": b["minx"].to_numpy(),  # per-row arrays
        "miny": b["miny"].to_numpy(),
        "maxx": b["maxx"].to_numpy(),
        "maxy": b["maxy"].to_numpy(),
        "cx": c.x.to_numpy(),  # centroids (for fast rough nearest)
        "cy": c.y.to_numpy(),
    }


def _get_ctx(store):
    _ensure_zip_ctx(store)
    return _ZIP_CTX[id(store)]


# --- main lookup --------------------------------------------------------------


@lru_cache(maxsize=4096)
def _zip_for_xy_3857(x: float, y: float, sid: int) -> Optional[str]:
    # sid is just to partition the cache by store identity
    store = get_store() if sid != id(get_store()) else get_store()
    ctx = _get_ctx(store)

    zips = ctx["zips"]
    ZIP_COL = ctx["zip_col"]

    # 1) Row-wise bbox prefilter
    mask = (x >= ctx["minx"]) & (x <= ctx["maxx"]) & (y >= ctx["miny"]) & (y <= ctx["maxy"])
    if mask.any():
        cand = zips.loc[mask]
        # 2) Robust: covers includes boundary points (contains would exclude them)
        pt = Point(x, y)
        inside = cand[cand.geometry.covers(pt)]
        if not inside.empty:
            return str(inside.iloc[0][ZIP_COL]).zfill(5)

    # 3) Nearest fallback (tighten candidate set using a small ring bbox first)
    try:
        # meters (Web Mercator). 250–500m is a good search radius.
        r = 300.0
        rb = (x - r, y - r, x + r, y + r)
        near_mask = (
            (ctx["minx"] <= rb[2])
            & (ctx["maxx"] >= rb[0])
            & (ctx["miny"] <= rb[3])
            & (ctx["maxy"] >= rb[1])
        )
        if not near_mask.any():
            # if nothing overlaps the ring bbox, relax (use all)
            near_mask = mask if mask.any() else np.ones(len(zips), dtype=bool)

        idx = np.nonzero(near_mask)[0]
        if idx.size:
            dx = ctx["cx"][idx] - x
            dy = ctx["cy"][idx] - y
            j_local = int(np.argmin(dx * dx + dy * dy))
            return str(zips.iloc[idx[j_local]][ZIP_COL]).zfill(5)
    except Exception:
        pass
    return None


def zip_for_latlon(lat: float, lon: float, store: Optional["DataStore"] = None) -> Optional[str]:
    # Project lon/lat -> 3857 to match zips
    store = store or get_store()
    zips = store.zips
    pt = gpd.GeoSeries([Point(lon, lat)], crs=4326)

    # Project only if needed
    if zips.crs and zips.crs.to_epsg() != 4326:
        pt = pt.to_crs(zips.crs)

    geom = pt.iloc[0]
    _ensure_zip_ctx(store)  # make sure context is ready for this store
    return _zip_for_xy_3857(geom.x, geom.y, id(store))
