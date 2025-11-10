#!/usr/bin/env python3
"""
Prepare and cache national ZIP Code Tabulation Areas (ZCTAs) for fast lookups.
Reads the TIGER/Line shapefile and saves reprojected copies.
"""

import geopandas as gpd
from pathlib import Path
import numpy as np

RAW_PATH = Path(__file__).resolve(
).parents[1] / "data" / "geo" / "raw_zip_data" / "tl_2023_us_zcta520.shp"
OUT_DIR = RAW_PATH.parent
CACHE_GPKG = OUT_DIR / "zcta_3857.gpkg"
CACHE_PARQUET = OUT_DIR / "zcta_3857.parquet"


def main():
    print(f"📦 Loading raw ZCTA shapefile:\n   {RAW_PATH}")

    zips = gpd.read_file(RAW_PATH)
    print(f"✅ Loaded {len(zips):,} polygons; columns: {list(zips.columns)}")

    # Ensure CRS before reprojecting
    if zips.crs is None:
        print("⚠️  No CRS found — assuming EPSG:4269 (NAD83).")
        zips = zips.set_crs("EPSG:4269", allow_override=True)
    else:
        print(f"🗺  Input CRS: {zips.crs}")

    # Reproject to 3857 for distance queries
    print("♻️  Reprojecting to EPSG:3857 (Web Mercator)...")
    zips = zips.to_crs(epsg=3857)

    # Clean invalid or empty geometries
    zips = zips[zips.geometry.notna() & zips.geometry.is_valid & ~
                zips.geometry.is_empty].copy()

    zips = zips.rename(columns={"ZCTA5CE20": "zcta"})
    zips["zcta"] = zips["zcta"].astype(str).str.zfill(5)

    print(zips.columns)
    print(zips[zips['zcta'].str.startswith("6")].head())

    # Save both cache formats
    print("💾 Saving GeoPackage and Parquet versions...")
    zips.to_file(CACHE_GPKG, driver="GPKG")
    zips[["zcta", "geometry"]].to_parquet(CACHE_PARQUET, index=False)

    print("✅ Done. Cached reprojected ZCTAs:")
    print(f"   • {CACHE_GPKG.name}")
    print(f"   • {CACHE_PARQUET.name}")
    print(f"   Total: {len(zips):,} polygons")

    zips_reload = gpd.read_parquet(CACHE_PARQUET)
    print(zips_reload.shape)
    print(zips_reload.columns)


if __name__ == "__main__":
    main()
