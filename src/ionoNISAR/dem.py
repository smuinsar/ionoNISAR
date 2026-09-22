"""Download and save a DEM GeoTIFF for a given bounding box."""

import argparse
import io
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import rasterio


# Taken from dem_stitcher itself rather than hardcoded, so a DEM it adds is offered here
# without an edit.  The fallback keeps --help working when dem_stitcher is absent; the
# missing-package error is raised at fetch time, not at argument parsing.
try:
    from dem_stitcher.datasets import DATASETS as _DEM_DATASETS
    AVAILABLE_DEMS = sorted(_DEM_DATASETS)
except Exception:
    AVAILABLE_DEMS = [
        "3dep",
        "glo_30",
        "glo_90",
        "glo_90_missing",
        "nasadem",
        "nisar_dem",
        "srtm_v3",
    ]

# OPERA burst-ID geometry database (simplified footprints, WGS84).
BURST_DB_URL = (
    "https://github.com/opera-adt/burst_db/releases/download/"
    "v0.17.0/burst-id-geometries-simple-0.17.0.geojson.zip"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Download a DEM GeoTIFF for given geographic bounds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    region = parser.add_mutually_exclusive_group(required=True)
    region.add_argument(
        "--bounds",
        nargs=4,
        type=float,
        metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
        help="Bounding box in EPSG:4326: xmin ymin xmax ymax",
    )
    region.add_argument(
        "--burstIDs",
        nargs="+",
        metavar="BURST_ID",
        help="OPERA burst IDs in JPL format (e.g. t064_135523_iw2). The DEM "
             "coverage is the bounding box encompassing all listed bursts, "
             "looked up in the OPERA burst-ID geometry database. Mutually "
             "exclusive with --bounds.",
    )
    parser.add_argument(
        "--buffer",
        type=float,
        default=0.1,
        metavar="DEGREES",
        help="Pad the --burstIDs bounding box by this many degrees on every "
             "side (default: 0.1). Ignored when --bounds is given.",
    )
    parser.add_argument(
        "--dem",
        default="glo_30",
        choices=AVAILABLE_DEMS,
        metavar="DEM_NAME",
        help=f"DEM source to use. Choices: {', '.join(AVAILABLE_DEMS)}. Default: glo_30",
    )
    parser.add_argument(
        "--output",
        default="dem.tif",
        help="Output GeoTIFF file path (default: dem.tif)",
    )
    parser.add_argument(
        "--ellipsoidal-height",
        action="store_true",
        default=True,
        help="Convert geoid heights to ellipsoidal heights (default: True)",
    )
    parser.add_argument(
        "--no-ellipsoidal-height",
        dest="ellipsoidal_height",
        action="store_false",
        help="Keep geoid heights (orthometric), do not convert to ellipsoidal",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=None,
        metavar="DEGREES",
        help="Output resolution in degrees. Default: native tile resolution",
    )
    parser.add_argument(
        "--threads-download",
        type=int,
        default=5,
        metavar="N",
        help="Number of parallel download threads (default: 5)",
    )
    parser.add_argument(
        "--threads-reproject",
        type=int,
        default=4,
        metavar="N",
        help="Number of reprojection threads (default: 4)",
    )
    parser.add_argument(
        "--fill-glo30-gaps",
        action="store_true",
        default=True,
        help="Fill missing GLO-30 tiles (Armenia/Azerbaijan) with GLO-90 (default: True)",
    )
    parser.add_argument(
        "--no-fill-glo30-gaps",
        dest="fill_glo30_gaps",
        action="store_false",
        help="Do not fill missing GLO-30 tiles",
    )
    parser.add_argument(
        "--fill-voids",
        action="store_true",
        default=True,
        help="Fill nodata voids (ocean, missing tiles) so the DEM has no NaN "
             "gaps, via stitch_dem's merge_nodata_value=0. Voids are set to 0 m "
             "orthometric; with ellipsoidal output (the default) they become the "
             "geoid-ellipsoid separation, i.e. the sea-surface ellipsoidal "
             "height, not a literal 0. Prevents downstream rdr2geo "
             "layover/shadow hangs and ionosphere interpn errors over voids. "
             "(default: True)",
    )
    parser.add_argument(
        "--no-fill-voids",
        dest="fill_voids",
        action="store_false",
        help="Leave nodata voids as NaN.",
    )
    parser.add_argument(
        "--tile-dir",
        default=None,
        metavar="DIR",
        help="Download source DEM tiles to DIR and process them locally "
             "instead of streaming via /vsicurl. More robust over flaky "
             "networks; downloaded tiles are kept and reused on later runs.",
    )
    parser.add_argument(
        "--overwrite-tiles",
        action="store_true",
        help="When --tile-dir is set, re-download tiles that already exist "
             "on disk (default: reuse).",
    )
    parser.add_argument(
        "--keep-tile-dir",
        action="store_true",
        help="When --tile-dir is set, keep the tile directory and its "
             "downloaded files after processing. Default: remove it.",
    )
    return parser.parse_args(argv)


def bounds_from_burst_ids(burst_ids, buffer=0.1):
    """Return [xmin, ymin, xmax, ymax] (EPSG:4326) covering all burst IDs."""
    try:
        import geopandas as gpd
        import requests
    except ImportError:
        print("Error: --burstIDs requires geopandas and requests.\n"
              "Install them with:  conda install -c conda-forge geopandas requests",
              file=sys.stderr)
        sys.exit(1)

    # Normalize to the DB's burst_id_jpl format: lowercase with underscores.
    wanted = {bid.strip().lower().replace("-", "_") for bid in burst_ids}

    print("Downloading burst-ID geometry database ...")
    resp = requests.get(BURST_DB_URL, allow_redirects=True)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        gdf_bursts = gpd.read_file(io.BytesIO(zf.read(zf.namelist()[0])))

    ids = gdf_bursts["burst_id_jpl"].str.lower().str.replace("-", "_", regex=False)
    gdf_match = gdf_bursts[ids.isin(wanted)]

    found = set(ids[ids.isin(wanted)])
    missing = sorted(wanted - found)
    if missing:
        print(f"Warning: {len(missing)} burst ID(s) not found in the database:",
              file=sys.stderr)
        for bid in missing:
            print(f"  {bid}", file=sys.stderr)

    if gdf_match.empty:
        print("Error: none of the given burst IDs were found in the database.",
              file=sys.stderr)
        sys.exit(1)

    xmin, ymin, xmax, ymax = gdf_match.total_bounds
    print(f"Matched {len(gdf_match)} burst(s) covering "
          f"[{xmin:.4f}, {ymin:.4f}, {xmax:.4f}, {ymax:.4f}]")

    if buffer:
        xmin, ymin = xmin - buffer, ymin - buffer
        xmax, ymax = xmax + buffer, ymax + buffer
        print(f"Buffered by {buffer} deg -> "
              f"[{xmin:.4f}, {ymin:.4f}, {xmax:.4f}, {ymax:.4f}]")

    return [float(xmin), float(ymin), float(xmax), float(ymax)]


def main(argv=None):
    args = parse_args(argv)

    if args.burstIDs:
        args.bounds = bounds_from_burst_ids(args.burstIDs, buffer=args.buffer)

    # Validate bounds
    xmin, ymin, xmax, ymax = args.bounds
    if xmin >= xmax or ymin >= ymax:
        print(f"Error: invalid bounds [{xmin}, {ymin}, {xmax}, {ymax}]. "
              "Ensure xmin < xmax and ymin < ymax.", file=sys.stderr)
        sys.exit(1)

    try:
        from dem_stitcher import stitch_dem
    except ImportError:
        print("Error: dem_stitcher is not installed.\n"
              "Install it with:  conda install -c conda-forge dem_stitcher\n"
              "              or:  pip install dem_stitcher", file=sys.stderr)
        sys.exit(1)

    print(f"Fetching DEM: {args.dem}")
    print(f"  Bounds     : {args.bounds}")
    print(f"  Output     : {args.output}")
    print(f"  Ellipsoidal: {args.ellipsoidal_height}")
    if args.resolution:
        print(f"  Resolution : {args.resolution} deg")
    if args.tile_dir:
        print(f"  Tile dir   : {args.tile_dir} (download-then-process)")

    X, profile = stitch_dem(
        args.bounds,
        dem_name=args.dem,
        dst_ellipsoidal_height=args.ellipsoidal_height,
        dst_area_or_point="Area",
        dst_resolution=args.resolution,
        n_threads_reproj=args.threads_reproject,
        n_threads_downloading=args.threads_download,
        fill_in_glo_30=args.fill_glo30_gaps,
        merge_nodata_value=0 if args.fill_voids else np.nan,
        dst_tile_dir=Path(args.tile_dir) if args.tile_dir else None,
        overwrite_existing_tiles=args.overwrite_tiles,
    )

    with rasterio.open(args.output, "w", **profile) as ds:
        ds.write(X, 1)
        ds.update_tags(AREA_OR_POINT="Area", DEM_SOURCE=args.dem)

    print(f"Saved: {args.output}")
    print(f"  Shape : {X.shape[1]} x {X.shape[0]} pixels")
    print(f"  CRS   : {profile['crs']}")
    print(f"  Dtype : {profile['dtype']}")

    if args.tile_dir and not args.keep_tile_dir:
        tile_path = Path(args.tile_dir)
        if tile_path.exists():
            shutil.rmtree(tile_path)
            print(f"Removed tile dir: {tile_path}")


if __name__ == "__main__":
    main()
