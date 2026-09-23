"""Mask water areas in GeoTIFF files using ESA WorldCover data."""

import os
import argparse
import hashlib
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.transform import from_bounds
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import requests
from pyproj import Transformer


def download_worldcover_tiles(corners, output_dir='worldcover', year=2021):
    """Download ESA WorldCover tiles covering the given corner coordinates."""
    import geopandas as gpd
    from shapely.geometry import box

    os.makedirs(output_dir, exist_ok=True)
    version = 'v200' if year == 2021 else 'v100'
    s3_url_prefix = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"

    # Select every tile the footprint INTERSECTS, not just the ones containing
    # the corner points.  Tiles are 3x3 degrees, so a scene wider than 3 degrees
    # has interior tiles that no corner falls in; missing those leaves holes that
    # reproject to class 0 and, being treated as nodata, mask out real land.
    lons = [c[0] for c in corners]
    lats = [c[1] for c in corners]
    footprint = box(min(lons), min(lats), max(lons), max(lats))

    # The tile index is a 3x3-degree lattice named for each tile's lower-left corner, so it
    # can be computed.  Try the authoritative grid first and fall back to arithmetic, so a
    # fully cached run does not depend on the network.
    tile_names = None
    try:
        print("Downloading WorldCover grid file...")
        grid_url = (f'{s3_url_prefix}/v100/2020/'
                    f'esa_worldcover_2020_grid.geojson')
        grid = gpd.read_file(grid_url)
        tile_names = set(grid[grid.intersects(footprint)].ll_tile)
    except Exception as e:
        import math
        print(f"  grid fetch failed ({type(e).__name__}); computing the 3-degree tile "
              f"lattice instead")
        lo = lambda v: int(math.floor(v / 3.0) * 3)
        tile_names = set()
        for la in range(lo(min(lats)), lo(max(lats)) + 3, 3):
            for ln in range(lo(min(lons)), lo(max(lons)) + 3, 3):
                tile_names.add(f"{'N' if la >= 0 else 'S'}{abs(la):02d}"
                               f"{'E' if ln >= 0 else 'W'}{abs(ln):03d}")
        missing = [t for t in sorted(tile_names)
                   if not os.path.exists(os.path.join(
                       output_dir, f"ESA_WorldCover_10m_{year}_{version}_{t}_Map.tif"))]
        if missing:
            raise RuntimeError(
                f"the WorldCover grid could not be fetched and these computed tiles are not "
                f"cached in {output_dir}: {', '.join(missing)}.  Re-run when the network is "
                f"back, or fetch them by hand.") from e
        print(f"  all {len(tile_names)} computed tiles are already cached")

    if not tile_names:
        raise ValueError(
            f"No WorldCover tiles intersect lon {min(lons):.4f}..{max(lons):.4f} "
            f"lat {min(lats):.4f}..{max(lats):.4f}")

    print(f"  WorldCover tiles needed: {len(tile_names)} "
          f"({', '.join(sorted(tile_names))})")

    # Download each tile
    tile_files = []
    for tile_name in sorted(tile_names):
        file_name = (f"ESA_WorldCover_10m_{year}_{version}_"
                     f"{tile_name}_Map.tif")
        url = f"{s3_url_prefix}/{version}/{year}/map/{file_name}"
        local_file = os.path.join(output_dir, file_name)

        if os.path.exists(local_file):
            print(f"  Tile {tile_name} already exists at {local_file}")
        else:
            print(f"  Downloading WorldCover tile {tile_name}...")
            response = requests.get(url, allow_redirects=True)
            response.raise_for_status()
            with open(local_file, 'wb') as f:
                f.write(response.content)
            print(f"  Downloaded tile to {local_file}")
        tile_files.append(local_file)

    # If only one tile, return it directly
    if len(tile_files) == 1:
        return tile_files[0]

    # Mosaic as a VRT, not a materialised GeoTIFF: merging reads every tile into memory and
    # writes the union, which is tens of GB at 10 m over a frame.  A VRT is a few kB, builds
    # instantly, and reprojects identically because GDAL reads only the windows it needs.
    # The name is keyed to the exact tile set, or a second frame silently reuses this one.
    tag = hashlib.md5(",".join(sorted(tile_names)).encode()).hexdigest()[:10]
    merged_file = os.path.join(
        output_dir, f"ESA_WorldCover_10m_{year}_{len(tile_files)}t_{tag}.vrt")
    if os.path.exists(merged_file):
        print(f"  Mosaic VRT already exists at {merged_file}")
        return merged_file

    print(f"  Building mosaic VRT over {len(tile_files)} WorldCover tiles...")
    from osgeo import gdal
    vrt = gdal.BuildVRT(merged_file, tile_files)
    if vrt is None:
        raise RuntimeError(f"gdal.BuildVRT failed for {tile_files}")
    vrt.FlushCache()
    vrt = None
    print(f"  Mosaic VRT saved to {merged_file}")
    return merged_file


def get_geotiff_corners(src, densify=0):
    """Get the boundary coordinates of a GeoTIFF in lon/lat (EPSG:4326)."""
    bounds = src.bounds
    if densify > 0:
        xs = np.linspace(bounds.left, bounds.right, densify + 2)
        ys = np.linspace(bounds.bottom, bounds.top, densify + 2)
        native_corners = ([(x, bounds.bottom) for x in xs] +
                          [(x, bounds.top) for x in xs] +
                          [(bounds.left, y) for y in ys] +
                          [(bounds.right, y) for y in ys])
    else:
        native_corners = [
            (bounds.left, bounds.bottom),
            (bounds.left, bounds.top),
            (bounds.right, bounds.bottom),
            (bounds.right, bounds.top),
        ]

    if src.crs != 'EPSG:4326':
        transformer = Transformer.from_crs(
            src.crs, "EPSG:4326", always_xy=True)
        corners_lonlat = [
            transformer.transform(x, y) for x, y in native_corners]
    else:
        corners_lonlat = native_corners

    return corners_lonlat


def create_water_mask(input_geotiff, worldcover_file):
    """Create a water mask from ESA WorldCover data, reprojected to match input GeoTIFF."""
    # Get target parameters from input GeoTIFF
    dst_shape = (input_geotiff.height, input_geotiff.width)
    dst_transform = input_geotiff.transform
    dst_crs = input_geotiff.crs

    # Initialize output array
    water_mask_data = np.zeros(dst_shape, dtype=np.uint8)

    print("Reprojecting WorldCover to match input GeoTIFF...")
    with rasterio.open(worldcover_file) as src:
        # Perform reprojection
        reproject(
            source=rasterio.band(src, 1),
            destination=water_mask_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest
        )

    # ESA WorldCover: class 80 = water, nodata = 0 (areas outside coverage)
    water_mask = (water_mask_data == 80) | (water_mask_data == 0)
    n_water = int(np.sum(water_mask_data == 80))
    n_nodata = int(np.sum(water_mask_data == 0))
    n_total = int(np.sum(water_mask))

    frac_nodata = n_nodata / water_mask.size
    print(f"Water mask created: {n_total} pixels "
          f"({100 * n_total / water_mask.size:.2f}% of total area) — "
          f"{n_water} water ({100 * n_water / water_mask.size:.2f}%), "
          f"{n_nodata} nodata ({100 * frac_nodata:.2f}%)")

    # Class 0 means "no WorldCover data here", which is masked along with water.
    # A large nodata fraction therefore silently deletes real land, and always
    # means the tile set does not cover the scene -- say so instead of returning
    # a mask that blanks the product.
    if frac_nodata > 0.05:
        print(f"  WARNING: {100 * frac_nodata:.1f}% of the scene has no WorldCover "
              f"coverage and will be masked as nodata.\n"
              f"  WARNING: this usually means the downloaded tiles do not span "
              f"the scene; check the tile list above against the footprint.")

    return water_mask


def apply_water_mask(input_file, output_file, worldcover_file=None, worldcover_dir='worldcover',
                     year=2021, nodata_value=None, visualize=True, fig_dir='figures'):
    """Apply water mask to input GeoTIFF and save masked output."""
    print(f"Opening input GeoTIFF: {input_file}")

    with rasterio.open(input_file) as src:
        # Read input data
        input_data = src.read(1)
        profile = src.profile.copy()

        print(f"Input GeoTIFF info:")
        print(f"  - Size: {src.width} x {src.height}")
        print(f"  - CRS: {src.crs}")
        print(f"  - Bounds: {src.bounds}")
        print(f"  - Data type: {src.dtypes[0]}")

        # Download WorldCover if not provided
        if worldcover_file is None:
            corners_lonlat = get_geotiff_corners(src, densify=20)
            lons = [c[0] for c in corners_lonlat]
            lats = [c[1] for c in corners_lonlat]
            print(f"Scene footprint (lon, lat): "
                  f"{min(lons):.4f}..{max(lons):.4f}, "
                  f"{min(lats):.4f}..{max(lats):.4f}")
            worldcover_file = download_worldcover_tiles(
                corners_lonlat, output_dir=worldcover_dir, year=year)

        # Create water mask
        water_mask = create_water_mask(src, worldcover_file)

        # Prepare output data
        output_data = input_data.copy()

        # Determine nodata value
        if nodata_value is None:
            if src.nodata is not None:
                nodata_value = src.nodata
            else:
                # Use a default based on data type
                if np.issubdtype(output_data.dtype, np.integer):
                    nodata_value = -9999
                else:
                    nodata_value = np.nan

        print(f"Setting water pixels to nodata value: {nodata_value}")

        # Apply water mask
        output_data[water_mask] = nodata_value

        # Update profile for output
        profile.update(nodata=nodata_value)

        # Save output GeoTIFF
        print(f"Saving masked GeoTIFF to: {output_file}")
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)

        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(output_data, 1)

        print(f"Successfully saved masked GeoTIFF")

        # Visualization
        if visualize:
            visualize_results(input_data, output_data, water_mask,
                            input_file, output_file, src.nodata, nodata_value, fig_dir)


def visualize_results(input_data, output_data, water_mask, input_file, output_file,
                      input_nodata, output_nodata, fig_dir='figures'):
    """Visualize input, water mask, and output side by side."""
    print("Creating visualization...")

    os.makedirs(fig_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Prepare data for visualization (mask nodata values)
    input_display = input_data.copy()
    output_display = output_data.copy()

    if input_nodata is not None:
        if np.isnan(input_nodata):
            input_display = np.where(np.isnan(input_display), np.nan, input_display)
        else:
            input_display = np.where(input_display == input_nodata, np.nan, input_display)

    if output_nodata is not None:
        if np.isnan(output_nodata):
            output_display = np.where(np.isnan(output_display), np.nan, output_display)
        else:
            output_display = np.where(output_display == output_nodata, np.nan, output_display)

    # Plot 1: Input data
    ax1 = axes[0, 0]
    im1 = ax1.imshow(input_display, cmap='viridis')
    ax1.set_title(f'Input GeoTIFF\n{os.path.basename(input_file)}')
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Plot 2: Water mask
    ax2 = axes[0, 1]
    # Create custom colormap for water mask
    colors = ['lightgray', 'darkblue']
    cmap = ListedColormap(colors)
    im2 = ax2.imshow(water_mask, cmap=cmap)
    ax2.set_title('Water Mask\n(Blue = Water)')
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, ticks=[0, 1])
    cbar2.ax.set_yticklabels(['Land', 'Water'])

    # Plot 3: Output data
    ax3 = axes[1, 0]
    im3 = ax3.imshow(output_display, cmap='viridis')
    ax3.set_title(f'Output (Water Masked)\n{os.path.basename(output_file)}')
    ax3.set_xlabel('Column')
    ax3.set_ylabel('Row')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # Plot 4: Difference visualization
    ax4 = axes[1, 1]
    difference = np.isnan(output_display) & ~np.isnan(input_display)
    im4 = ax4.imshow(difference, cmap='Reds')
    ax4.set_title(f'Masked Pixels\n(Red = Water pixels set to nodata)')
    ax4.set_xlabel('Column')
    ax4.set_ylabel('Row')
    water_pixel_count = np.sum(difference)
    total_pixels = difference.size
    ax4.text(0.02, 0.98, f'Masked: {water_pixel_count:,} pixels\n'
                         f'({100*water_pixel_count/total_pixels:.2f}% of total)',
             transform=ax4.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save figure
    output_basename = os.path.splitext(os.path.basename(output_file))[0]
    fig_path = os.path.join(fig_dir, f'water_mask_comparison_{output_basename}.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to: {fig_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Mask water areas in GeoTIFF files using ESA WorldCover data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - auto-download WorldCover
  python -m ionoNISAR.water -i input.tif -o output_masked.tif

  # Use existing WorldCover file
  python -m ionoNISAR.water -i input.tif -o output_masked.tif -w worldcover/tile.tif

  # Skip visualization
  python -m ionoNISAR.water -i input.tif -o output_masked.tif --no-visualize

  # Use custom nodata value
  python -m ionoNISAR.water -i input.tif -o output_masked.tif --nodata -9999

Notes:
  - Input GeoTIFF can be in any coordinate system (UTM, geographic, etc.)
  - ESA WorldCover water class (80) will be used for masking
  - Output will preserve input CRS and resolution
        """
    )

    parser.add_argument('-i', '--input', required=True,
                       help='Input GeoTIFF file path')
    parser.add_argument('-o', '--output', required=True,
                       help='Output GeoTIFF file path')
    parser.add_argument('-w', '--worldcover-file', type=str, default=None,
                       help='Input ESA WorldCover GeoTIFF file path. If not provided, will be downloaded automatically.')
    parser.add_argument('--worldcover-dir', default='worldcover',
                       help='Directory for WorldCover downloads (default: worldcover)')
    parser.add_argument('--year', type=int, default=2021, choices=[2020, 2021],
                       help='Year of WorldCover data (default: 2021)')
    parser.add_argument('--nodata', type=float, default=None,
                       help='No-data value for masked pixels (default: use input nodata or -9999/nan)')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Skip visualization step')
    parser.add_argument('--fig-dir', default='figures',
                       help='Directory to save figures (default: figures)')

    args = parser.parse_args()

    # Validate input file exists
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    # Apply water mask
    apply_water_mask(
        input_file=args.input,
        output_file=args.output,
        worldcover_file=args.worldcover_file,
        worldcover_dir=args.worldcover_dir,
        year=args.year,
        nodata_value=args.nodata,
        visualize=not args.no_visualize,
        fig_dir=args.fig_dir
    )

    print("\nWater masking completed successfully!")


if __name__ == '__main__':
    main()
