"""Put an offset-lattice screen onto the output map grid."""
import os

import numpy as np


def strided_lonlat(scratch, rows, cols):
    """rdr2geo lon/lat at the given crop rows/cols.  Row reads are contiguous, so this
    touches a few per cent of a 23 GB raster rather than all of it."""
    import re
    vrt = os.path.join(scratch, "rdr2geo", "topo.vrt")
    m = re.search(r'rasterXSize="(\d+)"\s+rasterYSize="(\d+)"', open(vrt).read())
    nr, na = int(m.group(1)), int(m.group(2))
    out = []
    for nm in ("x", "y"):
        mm = np.memmap(os.path.join(scratch, "rdr2geo", f"{nm}.rdr"), np.float64, "r",
                       shape=(na, nr))
        out.append(np.stack([np.asarray(mm[r, :])[cols] for r in rows]))
    return out[0], out[1]                        # lon, lat


def _to_grid(wkt):
    """Coordinate transform from lon/lat (EPSG:4326) to the projection given as WKT."""
    from osgeo import osr
    s = osr.SpatialReference(); s.ImportFromEPSG(4326)
    s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    t = osr.SpatialReference(); t.ImportFromWkt(wkt)
    t.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return osr.CoordinateTransformation(s, t)


def to_map(vals, lon, lat, gt, shape, wkt):
    """Area-average scattered radar cells onto the map grid described by gt/shape/wkt."""
    tr = _to_grid(wkt)
    g = np.isfinite(vals) & np.isfinite(lon) & np.isfinite(lat)
    pts = np.array(tr.TransformPoints(np.column_stack([lon[g].ravel(), lat[g].ravel()]).tolist()))
    jx = ((pts[:, 0] - gt[0]) / gt[1]).astype(np.int64)
    iy = ((pts[:, 1] - gt[3]) / gt[5]).astype(np.int64)
    ok = (jx >= 0) & (jx < shape[1]) & (iy >= 0) & (iy < shape[0])
    idx = iy[ok] * shape[1] + jx[ok]
    v = vals[g].ravel()[ok]
    num = np.bincount(idx, weights=v, minlength=shape[0] * shape[1])
    cnt = np.bincount(idx, minlength=shape[0] * shape[1])
    out = np.where(cnt > 0, num / np.maximum(cnt, 1), np.nan)
    return out.reshape(shape)


def lattice_to_map_tif(npz, out_dir, tag, scratch, grid_from):
    """Write the screen as applied and its validity mask as GeoTIFFs on the grid's own lattice."""
    from osgeo import gdal
    from scipy.ndimage import distance_transform_edt
    z = np.load(npz)
    scr = z["screen"].astype(np.float64)
    val = (np.asarray(z["valid"]).astype(np.float64) if "valid" in z.files
           else np.isfinite(scr).astype(np.float64))
    w = [int(v) for v in z["window"]]; sk = [int(v) for v in z["skip"]]
    ws = [int(v) for v in z["winsize"]]; se = [int(v) for v in z["search"]]
    ds = gdal.Open(grid_from)
    if ds is None:
        raise SystemExit(f"cannot open the output grid {grid_from}")
    gt, shape, wkt = ds.GetGeoTransform(), (ds.RasterYSize, ds.RasterXSize), ds.GetProjection()
    if not wkt:
        raise SystemExit(f"{grid_from} carries no projection; the screen cannot be placed")
    lon, lat = strided_lonlat(scratch,
                              w[0] + se[0] + ws[0] // 2 + sk[0] * np.arange(scr.shape[0]) - w[0],
                              w[1] + se[1] + ws[1] // 2 + sk[1] * np.arange(scr.shape[1]) - w[1])
    outs = {}
    for name, field, kind in (("iono_screen", scr, "screen"),
                              ("iono_screen_valid", val, "valid")):
        g = to_map(field, lon, lat, gt, shape, wkt)
        # the scatter leaves speckle where no lattice node lands in a map cell; close only
        # that, out to 4 px.  Past the lattice the field stays NaN.
        bad = ~np.isfinite(g)
        if bad.any() and (~bad).any():
            dist, idx = distance_transform_edt(bad, return_distances=True, return_indices=True)
            g = np.where(bad & (dist <= 4.0), g[tuple(idx)], g)
        if kind == "valid":
            g = np.where(np.isfinite(g), (g > 0.5).astype(np.float64), np.nan)
        out = os.path.join(out_dir, f"{name}_{tag}.tif")
        dst = gdal.GetDriverByName("GTiff").Create(out, shape[1], shape[0], 1, gdal.GDT_Float32,
                                                   ["COMPRESS=DEFLATE", "TILED=YES"])
        dst.SetGeoTransform(gt); dst.SetProjection(wkt)
        b = dst.GetRasterBand(1); b.WriteArray(g.astype(np.float32))
        b.SetNoDataValue(float("nan")); dst.FlushCache(); dst = None
        outs[kind] = out
        fin = np.isfinite(g)
        extra = f", measured on {100 * np.nanmean(g):.1f} % of it" if kind == "valid" else ""
        print(f"wrote {out}  ({100 * fin.mean():.1f} % of the grid{extra})")
    return outs
