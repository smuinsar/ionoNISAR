"""Per-pair configuration and the checks that run before a pair is processed."""
from __future__ import annotations

import glob
import os
import shutil
import types


def load(path):
    """Read a pair configuration file and fill in everything derived from it."""
    import yaml

    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    for k in ("track", "frame", "dates"):
        if k not in raw:
            raise SystemExit(f"{path}: '{k}' is required")
    dates = [str(d) for d in raw["dates"]]
    if len(dates) != 2:
        raise SystemExit(f"{path}: 'dates' must be the two acquisition dates of one pair")

    c = types.SimpleNamespace(
        path=os.path.abspath(path),
        track=int(raw["track"]),
        frame=int(raw["frame"]),
        dates=dates,
        pol=str(raw.get("pol", "HH")),
        grid_epsg=int(raw.get("grid_epsg", 3413)),
        posting=float(raw.get("posting", 120.0)),
        looks=tuple(raw.get("looks", (24, 16))),
        gpus=str(raw.get("gpus", "0")),
        cache=raw.get("cache", "cache"),
        coreg_out=raw.get("coreg_out", "outputs_offsets"),
        routes_out=raw.get("routes_out", "outputs_unified"),
        refocus_km=str(raw.get("refocus_km", "auto")),
        hybrid_cut_km=float(raw.get("hybrid_cut_km", 32.0)),
        fill=dict(raw.get("fill", {})),
        outlier=dict(raw.get("outlier", {})),
        split=dict(raw.get("split", {})),
    )
    c.tag = raw.get("tag") or f"t{c.track:03d}_{dates[0]}_{dates[1]}"
    c.scratch = raw.get("scratch", f"offsets_scratch_{dates[0]}_{dates[1]}")
    c.granules = os.path.join(c.cache, "granules")
    c.dem = raw.get("dem", os.path.join(c.cache, f"dem_insar_t{c.track:03d}.tif"))
    c.grid = raw.get("grid", os.path.join(
        c.cache, f"grid_t{c.track:03d}_f{c.frame:03d}_{c.posting:.0f}m_{c.grid_epsg}.tif"))
    c.glacier_mask = raw.get("glacier_mask", os.path.join(
        c.cache, f"rgi_glacier_t{c.track:03d}_ifggrid.tif"))
    c.offsets = os.path.join(c.coreg_out, f"offsets_{c.tag}.npz")
    c.ref = raw.get("ref") or find_granule(c.granules, "RSLC", c.track, c.frame, dates[0])
    c.sec = raw.get("sec") or find_granule(c.granules, "RSLC", c.track, c.frame, dates[1])
    return c


def find_granule(where, level, track, frame, date):
    """The one granule of this level, track, frame and date in a download directory."""
    pat = os.path.join(where, f"NISAR_L1_PR_{level}_*_{track:03d}_*_{frame:03d}_*_{date}T*.h5")
    hits = sorted(glob.glob(pat))
    if len(hits) > 1:
        raise SystemExit(f"{len(hits)} {level} granules match {date} in {where}/; "
                         f"name the one wanted as 'ref'/'sec' in the config")
    return hits[0] if hits else None


def carriers(h5, pol="HH"):
    """Which frequencies this RSLC holds for the given polarisation."""
    import h5py
    out = []
    try:
        with h5py.File(h5, "r") as f:
            for fr in ("A", "B"):
                if f"/science/LSAR/RSLC/swaths/frequency{fr}/{pol}" in f:
                    out.append(fr)
    except OSError:
        pass
    return out


def dem_covers(ref, dem):
    """How many of the frame polygon's vertices fall inside the DEM."""
    from osgeo import gdal

    from . import grid as G
    gdal.UseExceptions()
    ds = gdal.Open(dem)
    gt = ds.GetGeoTransform()
    x0, y0 = gt[0], gt[3]
    x1, y1 = x0 + gt[1] * ds.RasterXSize, y0 + gt[5] * ds.RasterYSize
    lo_x, hi_x = min(x0, x1), max(x0, x1)
    lo_y, hi_y = min(y0, y1), max(y0, y1)
    pts = G.bounding_polygon(ref)
    inside = sum(1 for lon, lat in pts if lo_x <= lon <= hi_x and lo_y <= lat <= hi_y)
    return inside, len(pts)


def coreg_extra(c):
    """The fill, outlier and refocus settings as coregister arguments."""
    out = []
    for k, flag in (("cutoff", "--fill-cutoff"), ("robust", "--fill-robust"),
                    ("hole_cutoff", "--fill-hole-cutoff"), ("hole_km", "--fill-hole-km"),
                    ("aniso", "--fill-aniso"), ("aniso_direction", "--fill-aniso-direction"),
                    ("trust_km", "--fill-trust-km")):
        if k in c.fill:
            out += [flag, str(c.fill[k])]
    for k, flag in (("spike", "--outlier-spike"), ("min_neighbors", "--outlier-min-neighbors"),
                    ("mad", "--outlier-mad")):
        if k in c.outlier:
            out += [flag, str(c.outlier[k])]
    if c.refocus_km:
        out += ["--refocus-km", c.refocus_km]
    return out


def preflight(c, need_gpu=True, min_free_gb=300):
    """Report everything that would stop this pair, rather than failing on the first one."""
    problems, notes = [], []

    for mod in ("isce3", "asf_search", "geopandas", "rasterio", "h5py", "snaphu"):
        try:
            __import__(mod)
        except ImportError:
            problems.append(f"python package '{mod}' is not importable")

    netrc = os.path.expanduser("~/.netrc")
    if not (os.path.exists(netrc)
            and "urs.earthdata.nasa.gov" in open(netrc, errors="replace").read()):
        problems.append("~/.netrc has no urs.earthdata.nasa.gov entry; the granules, the DEM "
                        "and the glacier outlines all need an Earthdata login")

    if need_gpu:
        if shutil.which("nvidia-smi") is None:
            problems.append("nvidia-smi is not on PATH; the dense offset estimation needs CUDA")

    free = shutil.disk_usage(".").free / 1e9
    if free < min_free_gb:
        problems.append(f"{free:.0f} GB free here, and one pair needs about {min_free_gb} GB "
                        f"of granules and scratch")
    else:
        notes.append(f"{free:.0f} GB free")

    for what, p in (("reference granule", c.ref), ("secondary granule", c.sec),
                    ("DEM", c.dem), ("map grid", c.grid), ("glacier mask", c.glacier_mask)):
        if p is None or not os.path.exists(p):
            notes.append(f"{what} not built yet ({p})")
        else:
            notes.append(f"{what}: {os.path.basename(p)}")

    # both carriers, because the split-spectrum screen differences them
    for what, p in (("reference", c.ref), ("secondary", c.sec)):
        if p and os.path.exists(p):
            have = carriers(p, c.pol)
            if "A" not in have:
                problems.append(f"the {what} granule has no frequency A / {c.pol}")
            elif "B" not in have:
                problems.append(f"the {what} granule has no frequency B / {c.pol}; the split "
                                f"screen differences the two carriers and cannot run without it")
            else:
                notes.append(f"{what} carries frequency A and B / {c.pol}")

    # a DEM that does not reach the frame edge costs a run, so say so before one starts
    if c.ref and os.path.exists(c.ref) and os.path.exists(c.dem):
        inside, total = dem_covers(c.ref, c.dem)
        if inside == total:
            notes.append(f"DEM covers all {total} frame-polygon vertices")
        else:
            notes.append(f"DEM covers {inside}/{total} frame-polygon vertices -- the gap is "
                         f"usually open water, where the product has no data either; check a "
                         f"quicklook before assuming it is harmless")

    print(f"pair {c.tag}   track {c.track} frame {c.frame}   {c.dates[0]} / {c.dates[1]}")
    for n in notes:
        print(f"  - {n}")
    for p in problems:
        print(f"  PROBLEM: {p}")
    return problems
