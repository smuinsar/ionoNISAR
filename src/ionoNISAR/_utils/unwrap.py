"""Phase unwrapping and water masking."""
from __future__ import annotations
import os
import time
import numpy as np
from osgeo import gdal


def _gtiff(path, arr, dtype):
    """Minimal single-band GTiff writer (no georef needed for unwrapping)."""
    length, width = arr.shape
    ds = gdal.GetDriverByName("GTiff").Create(path, width, length, 1, dtype)
    ds.GetRasterBand(1).WriteArray(arr)
    ds.FlushCache()
    ds = None


def unwrap_ifg(ifg, coh, mask, method="snaphu", coh_thresh=0.2, cache_dir=".",
               nlooks=None, snaphu_cost="smooth", snaphu_init="mcf",
               snaphu_ntiles=(1, 1), snaphu_tile_overlap=200, snaphu_nproc=1):
    """Unwrap a (filtered) interferogram. Returns (unw_phase [float32], conn or None)."""
    assert method in ("snaphu", "phass", "icu"), method
    corr = np.clip(np.nan_to_num(coh), 0, 1).astype(np.float32)
    corr[~mask] = 0.0
    conn = None
    t0 = time.time()

    if method in ("phass", "icu"):
        from isce3.io import Raster
        from isce3.unwrap import Phass, ICU
        co = os.path.join(cache_dir, "unw_corr_in.tif")
        _gtiff(co, corr, gdal.GDT_Float32)
        # Phass consumes the real phase; ICU consumes the complex interferogram
        # (matching how nisar.workflows.unwrap drives each solver).
        if method == "phass":
            ph = np.angle(ifg).astype(np.float32)
            ph[~mask] = 0.0
            src = os.path.join(cache_dir, "unw_phase_in.tif")
            _gtiff(src, ph, gdal.GDT_Float32)
        else:
            cx = ifg.astype(np.complex64).copy()
            cx[~mask] = 0
            src = os.path.join(cache_dir, "unw_ifg_in.tif")
            _gtiff(src, cx, gdal.GDT_CFloat32)
        unw_p = os.path.join(cache_dir, "unw_tmp.tif")
        conn_p = os.path.join(cache_dir, "conn_tmp.tif")
        in_r, co_r = Raster(src), Raster(co)
        width, length = int(in_r.width), int(in_r.length)
        unw_r = Raster(unw_p, width, length, 1, gdal.GDT_Float32, "GTiff")
        conn_r = Raster(conn_p, width, length, 1, gdal.GDT_UInt32, "GTiff")
        if method == "phass":
            solver = Phass()
            solver.correlation_threshold = coh_thresh
            solver.unwrap(in_r, co_r, unw_r, conn_r)      # (phase, corr, unw, label)
        else:
            solver = ICU()
            solver.init_corr_thr = coh_thresh
            solver.unwrap(unw_r, conn_r, in_r, co_r)      # (unw, label, igram, corr)
        del in_r, co_r, unw_r, conn_r                     # flush isce3 rasters to disk
        unw = gdal.Open(unw_p).ReadAsArray().astype(np.float32)
        conn = gdal.Open(conn_p).ReadAsArray()
    elif method == "snaphu":
        import snaphu
        if nlooks is None:
            raise ValueError("method='snaphu' needs nlooks (the equivalent number of "
                             "independent looks behind `coh`)")
        cx = ifg.astype(np.complex64).copy()
        cx[~mask] = 0
        unw_a, conn = snaphu.unwrap(
            cx, corr, nlooks=float(nlooks), cost=snaphu_cost, init=snaphu_init,
            mask=mask.astype(np.uint8), ntiles=tuple(snaphu_ntiles),
            tile_overlap=int(snaphu_tile_overlap), nproc=int(snaphu_nproc),
            scratchdir=cache_dir, delete_scratch=True)
        unw = np.asarray(unw_a, np.float32)
        conn = np.asarray(conn)
        print(f"snaphu: nlooks {float(nlooks):.0f}, cost {snaphu_cost}, init "
              f"{snaphu_init}, tiles {tuple(snaphu_ntiles)}; "
              f"{int((conn == 0)[mask].sum())} of {int(mask.sum())} masked px left "
              f"unlabelled (component 0)")
    unw[~mask] = np.nan
    print(f"{method} unwrap: {time.time() - t0:.1f}s")
    component_report(unw, conn, mask)
    return unw, conn


def component_report(unw, conn, mask, warn_below=0.95):
    """Say out loud that the components carry INDEPENDENT integer cycle datums."""
    if conn is None:
        return
    lab = np.asarray(conn)
    ok = np.asarray(mask, bool) & np.isfinite(unw) & (lab != 0)
    if not ok.any():
        return
    ids, cnt = np.unique(lab[ok], return_counts=True)
    tot = int(ok.sum())
    share = cnt.max() / tot
    print(f"[unwrap] {len(ids)} connected component(s); the largest holds "
          f"{100 * share:.1f} % of the unwrapped pixels")
    if share < warn_below and len(ids) > 1:
        big = int((cnt / tot >= 0.005).sum())
        print(f"[unwrap] WARNING: {100 * (1 - share):.1f} % of the unwrapped pixels sit in "
              f"{len(ids) - 1} other component(s), {big - 1} of them over 0.5 %.  Each "
              f"carries its OWN arbitrary integer cycle datum and this function does not "
              f"remove it -- the consumer must re-reference or restrict, or the datum goes "
              f"into the product.  See component_report's docstring.")


def _worldcover_bounds(src, n=64):
    """Lon/lat envelope of an open rasterio dataset, from a densified boundary."""
    from pyproj import Transformer
    b = src.bounds
    ex = np.linspace(b.left, b.right, n)
    ey = np.linspace(b.bottom, b.top, n)
    xs = np.concatenate([ex, ex, np.full(n, b.left), np.full(n, b.right)])
    ys = np.concatenate([np.full(n, b.bottom), np.full(n, b.top), ey, ey])
    if src.crs.to_epsg() != 4326:
        tr = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
        xs, ys = tr.transform(xs, ys)
    return float(np.min(xs)), float(np.min(ys)), float(np.max(xs)), float(np.max(ys))


def _worldcover_tiles(bounds):
    """Names of the 3-degree ESA WorldCover tiles overlapping a (lon, lat) envelope."""
    lon0, lat0, lon1, lat1 = bounds
    lons = range(int(np.floor(lon0 / 3)) * 3, int(np.floor(lon1 / 3)) * 3 + 1, 3)
    lats = range(int(np.floor(lat0 / 3)) * 3, int(np.floor(lat1 / 3)) * 3 + 1, 3)
    return [f"{'N' if la >= 0 else 'S'}{abs(la):02d}{'E' if lo >= 0 else 'W'}{abs(lo):03d}"
            for la in lats for lo in lons]


def _download_worldcover(bounds, output_dir, year=2021):
    """Download the ESA WorldCover tiles covering the (lon, lat) envelope `bounds`
    (merging if >1). Returns a GeoTIFF path."""
    import requests
    import rasterio
    from rasterio.merge import merge as rio_merge

    os.makedirs(output_dir, exist_ok=True)
    version = "v200" if year == 2021 else "v100"
    s3 = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"

    files, tiles = [], []
    for t in _worldcover_tiles(bounds):
        fn = f"ESA_WorldCover_10m_{year}_{version}_{t}_Map.tif"
        local = os.path.join(output_dir, fn)
        if not os.path.exists(local):
            r = requests.get(f"{s3}/{version}/{year}/map/{fn}", allow_redirects=True, stream=True)
            if r.status_code == 404:
                print(f"  WorldCover tile {t}: not published (all ocean), skipped")
                continue
            r.raise_for_status()
            print(f"  downloading WorldCover tile {t} ...")
            with open(local + ".part", "wb") as f:       # atomic: a killed download must not
                for chunk in r.iter_content(1 << 20):    # leave a truncated tile that later runs
                    f.write(chunk)                       # accept (it would decode as water)
            os.replace(local + ".part", local)
        files.append(local)
        tiles.append(t)
    if not files:
        raise ValueError(f"No ESA WorldCover tiles published for lon/lat envelope {bounds}")
    if len(files) == 1:
        return files[0]

    # the tile list goes in the mosaic name: a stale mosaic built from a different (or smaller)
    # tile set would silently mask real land as water
    merged = os.path.join(output_dir, f"ESA_WorldCover_10m_{year}_{'_'.join(tiles)}_merged.tif")
    if not os.path.exists(merged):
        ds = [rasterio.open(f) for f in files]
        mosaic, transform = rio_merge(ds)
        prof = ds[0].profile.copy()
        prof.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=transform)
        for d in ds:
            d.close()
        with rasterio.open(merged + ".part.tif", "w", **prof) as dst:
            dst.write(mosaic)
        os.replace(merged + ".part.tif", merged)
    return merged


def mask_water(unw_tif, out_tif, worldcover_dir, year=2021):
    """Copy `unw_tif` to `out_tif` with ESA-WorldCover water pixels set to NaN."""
    import rasterio
    from rasterio.warp import reproject, Resampling

    with rasterio.open(unw_tif) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        wc_file = _download_worldcover(_worldcover_bounds(src), worldcover_dir, year)
        wc = np.zeros((src.height, src.width), dtype=np.uint8)
        with rasterio.open(wc_file) as wsrc:
            reproject(source=rasterio.band(wsrc, 1), destination=wc,
                      src_transform=wsrc.transform, src_crs=wsrc.crs,
                      dst_transform=src.transform, dst_crs=src.crs,
                      resampling=Resampling.nearest)
    water = (wc == 80) | (wc == 0)
    data[water] = np.nan
    profile.update(nodata=float("nan"))
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(data, 1)
    print(f"  water-masked {int(water.sum())} px ({100 * water.mean():.1f}%) -> {out_tif}")
    return out_tif
