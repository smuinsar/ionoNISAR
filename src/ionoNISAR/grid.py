"""The map grid every product of a track/frame shares: an empty 120 m EPSG:3413 raster."""
import argparse
import os
import re
import sys

import numpy as np


def bounding_polygon(h5):
    import h5py
    with h5py.File(h5, "r") as h:
        p = h["/science/LSAR/identification/boundingPolygon"][()]
    p = p.decode() if isinstance(p, bytes) else str(p)
    pts = [(float(x), float(y)) for x, y, *_ in re.findall(r"(-?\d+\.?\d*) (-?\d+\.?\d*)(?: (-?\d+\.?\d*))?", p)]
    if len(pts) < 3:
        raise SystemExit(f"{h5}: could not parse boundingPolygon: {p[:120]}")
    return pts


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from", dest="src", required=True, help="a granule of the track/frame (RSLC or GSLC)")
    p.add_argument("--out", required=True)
    p.add_argument("--posting", type=float, default=120.0)
    p.add_argument("--epsg", type=int, default=3413)
    p.add_argument("--pad-km", type=float, default=10.0, help="margin around the footprint (products are cropped to data anyway)")
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    from osgeo import gdal, osr
    gdal.UseExceptions()
    if os.path.exists(a.out) and not a.force:
        ds = gdal.Open(a.out); gt = ds.GetGeoTransform()
        print(f"{a.out} exists: {ds.RasterXSize} x {ds.RasterYSize} @ {gt[1]:g} m, origin ({gt[0]:.0f}, {gt[3]:.0f}) -- kept")
        return 0
    pts = bounding_polygon(a.src)
    s = osr.SpatialReference(); s.ImportFromEPSG(4326); s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    t = osr.SpatialReference(); t.ImportFromEPSG(a.epsg); t.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    # densify the polygon edges: a frame edge is a great-circle-ish curve that bows in EPSG:3413
    dense = []
    for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]):
        for f in np.linspace(0.0, 1.0, 20, endpoint=False):
            dense.append((x0 + f * (x1 - x0), y0 + f * (y1 - y0)))
    xy = np.array(osr.CoordinateTransformation(s, t).TransformPoints(dense))[:, :2]
    d, pad = a.posting, a.pad_km * 1000.0
    x0 = np.floor((xy[:, 0].min() - pad) / d) * d
    x1 = np.ceil((xy[:, 0].max() + pad) / d) * d
    y0 = np.floor((xy[:, 1].min() - pad) / d) * d
    y1 = np.ceil((xy[:, 1].max() + pad) / d) * d
    nx, ny = int(round((x1 - x0) / d)), int(round((y1 - y0) / d))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(a.out, nx, ny, 1, gdal.GDT_Float32, options=["COMPRESS=DEFLATE", "TILED=YES"])
    ds.SetGeoTransform((x0, d, 0.0, y1, 0.0, -d))
    ds.SetProjection(t.ExportToWkt())
    b = ds.GetRasterBand(1); b.SetNoDataValue(float("nan"))
    b.WriteArray(np.full((ny, nx), np.nan, np.float32)); ds = None
    print(f"wrote {a.out}: {nx} x {ny} @ {d:g} m EPSG:{a.epsg}, x {x0:.0f}..{x1:.0f}, y {y0:.0f}..{y1:.0f} "
          f"(footprint of {os.path.basename(a.src)} + {a.pad_km:g} km)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
