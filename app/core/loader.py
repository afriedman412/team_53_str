# app/core/loader.py
import geopandas as gpd
import numpy as np
from shapely.strtree import STRtree
from app.core.config import DATA_PATHS
from app.core.store import DataStore
from app.model_api.model_loader import load_model


def load_store() -> DataStore:
    # --- OSM features ---
    gdf = gpd.read_file(DATA_PATHS["osm"])
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid & ~gdf.geometry.is_empty].copy()
    gdf = gdf.to_crs(3857)

    # --- ZIPs ---
    zips = gpd.read_parquet(DATA_PATHS["zips"])
    if zips.crs is None:
        zips = zips.set_crs(3857, allow_override=True)
    elif zips.crs.to_epsg() != 3857:
        zips = zips.to_crs(3857)

    zips = zips[["zcta", "geometry"]].copy()

    # Precompute bounds arrays for quick bbox masking
    bounds = zips.geometry.bounds
    minx = bounds["minx"].to_numpy()
    miny = bounds["miny"].to_numpy()
    maxx = bounds["maxx"].to_numpy()
    maxy = bounds["maxy"].to_numpy()

    # --- Optional: trees layer (if present) ---
    try:
        trees = gpd.read_parquet(DATA_PATHS["trees"])
        if trees.crs is None:
            trees = trees.set_crs(3857, allow_override=True)
        elif trees.crs.to_epsg() != 3857:
            trees = trees.to_crs(3857)
    except Exception:
        trees = None

    # --- Split OSM features into categories ---
    def _subset(gdf, keywords):
        """
        Find rows whose text columns contain any keyword.
        Works across multiple possible schema columns (e.g. amenity, name, leisure, highway).
        """
        # Pick text-like columns to search
        text_cols = [c for c in gdf.columns if gdf[c].dtype == object or "str" in str(gdf[c].dtype)]
        if not text_cols:
            return gdf.iloc[0:0]  # empty GeoDataFrame

        mask = np.zeros(len(gdf), dtype=bool)
        for kw in keywords:
            for col in text_cols:
                mask |= gdf[col].astype(str).str.contains(kw, case=False, na=False)
        return gdf[mask]

    airports = _subset(gdf, ["airport", "aerodrome"])
    trains = _subset(gdf, ["station", "train", "metra"])
    parks = _subset(gdf, ["park"])
    universities = _subset(gdf, ["university", "college"])
    buses = _subset(gdf, ["bus", "transit"])
    city_center = _subset(gdf, ["city", "downtown", "loop"])

    feature_sets = {
        "dist_to_airport_km": {
            "tree": STRtree(airports.geometry.values),
            "geoms": airports.geometry.values,
        },
        "dist_to_train_km": {
            "tree": STRtree(trains.geometry.values),
            "geoms": trains.geometry.values,
        },
        "dist_to_park_km": {
            "tree": STRtree(parks.geometry.values),
            "geoms": parks.geometry.values,
        },
        "dist_to_university_km": {
            "tree": STRtree(universities.geometry.values),
            "geoms": universities.geometry.values,
        },
        "dist_to_bus_km": {
            "tree": STRtree(buses.geometry.values),
            "geoms": buses.geometry.values,
        },
        "dist_to_city_center_km": {
            "tree": STRtree(city_center.geometry.values),
            "geoms": city_center.geometry.values,
        },
    }

    meta = {
        "n_features": len(gdf),
        "n_zips": len(zips),
        "has_trees": trees is not None,
        "feature_sets": feature_sets,
    }

    pipeline = load_model()

    return DataStore(
        gdf_features=gdf,
        zips=zips,
        trees=trees,
        feature_index=STRtree(gdf.geometry.values),
        zip_bounds_minx=minx,
        zip_bounds_miny=miny,
        zip_bounds_maxx=maxx,
        zip_bounds_maxy=maxy,
        meta=meta,
        pipeline=pipeline,
    )
