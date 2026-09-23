"""Masking, outlier rejection and hole filling for offset fields."""
from __future__ import annotations

import os

import numpy as np

# fill_holes' bulb check: the robust reference fit, and how deep into a hole a bulb counts
BULB_HOLE_CUTOFF = 100.0
BULB_DEPTH_KM = 0.5


# --- geocoded products -------------------------------------------------------

def mask_geotiff(src, dst, args, gl=None):
    """Write a water/glacier-masked copy of one geocoded GeoTIFF."""
    import rasterio

    wc_dir = os.path.join(args.cache_dir, "worldcover")
    with rasterio.open(src) as s:
        arr = s.read(1).astype(np.float32)
        profile = s.profile.copy()
    a = arr.copy()

    if getattr(args, "mask_water", False):
        print(f"[mask] water: {os.path.basename(src)}")
        try:
            # the tile fetcher and mask this directory's products were built with
            from . import water as MW
            MW.apply_water_mask(src, dst, worldcover_dir=wc_dir, year=args.water_year,
                                nodata_value=np.nan, visualize=False)
        except ImportError:
            # the water module is optional; the unwrap helpers do the same job
            # thing (WorldCover class 80 = water, class 0 = out of coverage) and is always
            # there.  Only --mask-where rdr genuinely needs the former, for its tile VRT.
            from ._utils import unwrap
            unwrap.mask_water(src, dst, wc_dir, args.water_year)
        with rasterio.open(dst) as s:
            a = s.read(1).astype(np.float32)
        # A WorldCover coverage gap is masked as nodata exactly like water, so an
        # incomplete tile set blanks the product instead of erroring.  Compare what
        # survived against what went in and refuse to be quiet.
        kept, had = np.isfinite(a).mean(), np.isfinite(arr).mean()
        if had > 0 and kept < 0.5 * had:
            print(f"[mask] WARNING: {os.path.basename(dst)} kept only "
                  f"{100 * kept:.1f}% valid pixels, down from {100 * had:.1f}% before "
                  f"water masking.\n"
                  f"[mask] WARNING: water is a few % of a land frame, so this points at "
                  f"missing WorldCover tiles rather than real water; check the tile list "
                  f"above and delete {wc_dir} to refetch.")

    if getattr(args, "mask_glacier", False) and getattr(
            args, "mask_glacier_in_products", True):
        if gl is None:
            gl = glacier_mask(src, args)
        a[gl] = np.nan

    profile.update(driver="GTiff", dtype="float32", count=1, nodata=float("nan"))
    with rasterio.open(dst, "w", **profile) as d:
        d.write(a.astype(np.float32), 1)
    print(f"[mask] {os.path.basename(dst)}: {100 * np.isfinite(a).mean():.1f} % valid, "
          f"from {100 * np.isfinite(arr).mean():.1f} % unmasked")
    return a, gl


# --- glacier mask ------------------------------------------------------------
# RGI 7.0 C (glacier complexes) from NSIDC, EarthData-authenticated.
RGI_DIR = ("https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0770_rgi_v7/"
           "regional_files/RGI2000-v7.0-C/")


def glacier_mask(tif, args):
    """Boolean raster, True on RGI glacier complexes, on `tif`'s grid."""
    import rasterio
    from . import water as MW

    with rasterio.open(tif) as s:
        # densified so the reprojected footprint follows the curved EPSG:3413 edges
        # rather than cutting corners off a polar frame
        corners = MW.get_geotiff_corners(s, densify=20)
        transform, shape, crs = s.transform, (s.height, s.width), s.crs
    lons, lats = [c[0] for c in corners], [c[1] for c in corners]
    return rgi_raster(transform, shape, crs, (min(lons), min(lats),
                                              max(lons), max(lats)), args)


def rgi_raster(transform, shape, crs, bbox, args):
    """Boolean raster, True on RGI glacier complexes, for the given grid."""
    import geopandas as gpd
    import pandas as pd
    from rasterio.features import rasterize

    session = _rgi_login(args)
    regions = ([args.rgi_region] if args.rgi_region
               else _rgi_regions_for(bbox, session, args))
    if not regions:
        print(f"[mask] WARNING: no RGI region covers {bbox}; glacier mask is empty")
        return np.zeros(shape, bool)

    parts = []
    for reg in regions:
        stem = f"RGI2000-v7.0-C-{reg}"
        # read straight out of the remote zip; nothing is downloaded
        g = gpd.read_file(f"/vsizip//vsicurl/{RGI_DIR}{stem}.zip/{stem}.shp",
                          bbox=bbox, engine="pyogrio")
        print(f"[mask] RGI {reg}: {len(g)} glacier complexes in the footprint")
        parts.append(g)
    gdf = pd.concat(parts).to_crs(crs) if len(parts) > 1 else parts[0].to_crs(crs)
    if gdf.empty:
        return np.zeros(shape, bool)
    geom = gdf.geometry
    if args.mask_buffer:
        # in the projected CRS, so the buffer is metres on the ground
        geom = geom.buffer(args.mask_buffer)
        print(f"[mask] glacier polygons buffered by {args.mask_buffer:g} m")

    m = rasterize(geom, out_shape=shape, transform=transform,
                  fill=0, default_value=1, all_touched=True).astype(bool)
    print(f"[mask] glacier: {m.sum()} px ({100 * m.mean():.2f} % of the raster)")
    return m


def _rgi_login(args):
    """EarthData login, plus the GDAL settings /vsicurl needs to read NSIDC."""
    import earthaccess
    earthaccess.login(strategy="netrc")
    ck = os.path.join(args.cache_dir, "rgi", "cookies.txt")
    os.makedirs(os.path.dirname(ck), exist_ok=True)
    open(ck, "a").close()
    # os.environ, not gdal.SetConfigOption: pyogrio may be linked against its own
    # GDAL, and only the environment reaches both.
    os.environ.update(GDAL_HTTP_NETRC="YES", GDAL_HTTP_COOKIEFILE=ck,
                      GDAL_HTTP_COOKIEJAR=ck,
                      # NSIDC answers HEAD with 401 even when authorised and
                      # /vsicurl opens with a HEAD -- make it use ranged GETs
                      CPL_VSIL_CURL_USE_HEAD="NO")
    return earthaccess.get_requests_https_session()


def _rgi_regions_for(bbox, session, args):
    """RGI region names whose extent overlaps `bbox`, e.g. ['01_alaska']."""
    import json
    import re
    import struct
    from osgeo import gdal

    cache = os.path.join(args.cache_dir, "rgi", "region_extents.json")
    if os.path.exists(cache):
        with open(cache) as f:
            ext = json.load(f)
    else:
        names = sorted(set(re.findall(r'href="RGI2000-v7\.0-C-(\d\d_[^"]+)\.zip"',
                                      session.get(RGI_DIR, timeout=60).text)))
        if not names:
            raise SystemExit(f"no RGI region zips listed at {RGI_DIR}")
        print(f"[mask] indexing {len(names)} RGI regions (cached in {cache})")
        ext = {}
        for n in names:
            stem = f"RGI2000-v7.0-C-{n}"
            f = gdal.VSIFOpenL(f"/vsizip//vsicurl/{RGI_DIR}{stem}.zip/{stem}.shp", "rb")
            if f is None:
                raise SystemExit(f"cannot read {stem}.shp -- EarthData login ok?")
            gdal.VSIFSeekL(f, 36, 0)
            b = gdal.VSIFReadL(1, 32, f)
            gdal.VSIFCloseL(f)
            ext[n] = list(struct.unpack("<4d", b))
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, "w") as fh:
            json.dump(ext, fh, indent=1)

    x0, y0, x1, y1 = bbox
    return [n for n, (a, b, c, d) in sorted(ext.items())
            if a <= x1 and c >= x0 and b <= y1 and d >= y0]


# --- radar-lattice geolocation and masking -----------------------------------

def _save_atomic(path, X, Y):
    """Write the cache through a temporary file, so jobs sharing it never read a
    half-written one -- the whole point of the cache is running methods in parallel."""
    # the tmp name has to end in .npz, else np.savez appends it and os.replace below
    # looks for a file that was never written
    tmp = f"{path}.{os.getpid()}.tmp.npz"
    np.savez(tmp, X=X, Y=Y)
    os.replace(tmp, path)


def geolocate(gi, gj, ii, jj, rgp, orbit, demI, epsg, label=""):
    """Map coordinates of the window centres (gi, gj), via rdr2geo on the DEM."""
    import isce3
    from pyproj import Transformer

    ellip = isce3.core.Ellipsoid()
    tr = Transformer.from_crs(4326, epsg, always_xy=True)
    xs = np.empty(gi.size)
    ys = np.empty(gi.size)
    ok = np.zeros(gi.size, bool)
    for k in range(gi.size):
        t = rgp.sensing_start + ii[gi[k]] / rgp.prf
        r = rgp.starting_range + jj[gj[k]] * rgp.range_pixel_spacing
        try:
            xyz = isce3.geometry.rdr2geo_bracket(t, r, orbit, rgp.lookside, 0.0,
                                                 rgp.wavelength, dem=demI)
        except Exception:
            continue
        # rdr2geo_bracket returns ECEF XYZ in metres, not lon/lat -- convert
        llh = ellip.xyz_to_lon_lat(xyz)
        lon, lat = np.degrees(llh[0]), np.degrees(llh[1])
        xs[k], ys[k] = tr.transform(lon, lat)
        ok[k] = True
        if k % 5000 == 0:
            print(f"  rdr2geo{label} {k}/{gi.size}", flush=True)
    print(f"  geolocated{label} {ok.sum()} of {gi.size}")
    return xs, ys, ok


def rdr_mask_samples(x, y, gt, epsg, args):
    """True for the window centres (x, y) that sit on water or ice."""
    pad = 2
    rows, cols = (y - gt[3]) / gt[5], (x - gt[0]) / gt[1]
    r0, r1 = int(np.floor(rows.min())) - pad, int(np.ceil(rows.max())) + pad
    c0, c1 = int(np.floor(cols.min())) - pad, int(np.ceil(cols.max())) + pad
    ny, nx = r1 - r0, c1 - c0
    gt_m = (gt[0] + c0 * gt[1], gt[1], gt[2], gt[3] + r0 * gt[5], gt[4], gt[5])
    transform, crs, corners, bbox = grid_footprint(gt_m, (ny, nx), epsg)

    bad = np.zeros((ny, nx), bool)
    if args.mask_water:
        bad |= water_raster(transform, (ny, nx), crs, corners, args)
    if args.mask_glacier:
        bad |= rgi_raster(transform, (ny, nx), crs, bbox, args)

    ri = np.clip(np.floor(rows).astype(int) - r0, 0, ny - 1)
    ci = np.clip(np.floor(cols).astype(int) - c0, 0, nx - 1)
    return bad[ri, ci]


def grid_footprint(gt, shape, epsg):
    """(affine, crs, densified lon/lat outline, lon/lat bbox) for one map grid."""
    from pyproj import Transformer
    from rasterio.crs import CRS
    from rasterio.transform import Affine

    ny, nx = shape
    ex = np.linspace(gt[0], gt[0] + nx * gt[1], 25)
    ey = np.linspace(gt[3], gt[3] + ny * gt[5], 25)
    bx = np.concatenate([ex, ex, np.full(25, ex[0]), np.full(25, ex[-1])])
    by = np.concatenate([np.full(25, ey[0]), np.full(25, ey[-1]), ey, ey])
    lon, lat = Transformer.from_crs(epsg, 4326, always_xy=True).transform(bx, by)
    return (Affine.from_gdal(*gt), CRS.from_epsg(epsg), list(zip(lon, lat)),
            (min(lon), min(lat), max(lon), max(lat)))


def water_raster(transform, shape, crs, corners, args):
    """Boolean raster, True on ESA WorldCover permanent water, for the given grid."""
    from types import SimpleNamespace
    from . import water as MW

    ny, nx = shape
    vrt = MW.download_worldcover_tiles(
        corners, output_dir=os.path.join(args.cache_dir, "worldcover"),
        year=args.water_year)
    m = MW.create_water_mask(
        SimpleNamespace(height=ny, width=nx, transform=transform, crs=crs), vrt)
    print(f"[mask] water: {m.sum()} px ({100 * m.mean():.2f} % of the raster)")
    return m


# --- outliers and filling ----------------------------------------------------

def _blockwise(a, block):
    """View `a` as (ny/block, block, nx/block, block), NaN-padded to fit."""
    ny, nx = a.shape
    ap = np.pad(a, ((0, (-ny) % block), (0, (-nx) % block)), constant_values=np.nan)
    return ap.reshape(ap.shape[0] // block, block, ap.shape[1] // block, block)


def _upsample(coarse, block, shape):
    """Nearest-neighbour expansion of a block statistic back to the full grid."""
    return np.repeat(np.repeat(coarse, block, 0), block, 1)[:shape[0], :shape[1]]


def reject_outliers(az, rg, keep, se_a, se_r, k, block=8):
    """Measured windows that cannot be real ground offsets, dropped before filling."""
    bad = np.zeros_like(keep)
    lim = keep & ((np.abs(az) >= 0.9 * se_a) | (np.abs(rg) >= 0.9 * se_r))
    bad |= lim
    print(f"[outlier] {int(lim.sum())} windows at the search limit "
          f"(|az|>={0.9 * se_a:.0f} or |rg|>={0.9 * se_r:.0f} px)")

    for name, v in (("azimuth", az), ("range", rg)):
        ok = keep & ~bad
        w = np.where(ok, v, np.nan)
        # Block median AND spread on the same coarse grid: the scatter varies across the
        # frame, so one frame-wide sigma cuts into good data.
        blocks = _blockwise(w, block)
        bm = _upsample(np.nanmedian(blocks, axis=(1, 3)), block, w.shape)
        r = v - bm
        rb = _blockwise(np.where(ok, r, np.nan), block)
        sig = 1.4826 * np.nanmedian(np.abs(rb), axis=(1, 3))
        # a floor, so a block that happens to be perfectly flat does not reject its
        # neighbours' ordinary noise
        floor = np.nanmedian(sig)
        sig = _upsample(np.fmax(sig, floor), block, w.shape)
        hit = ok & np.isfinite(sig) & (np.abs(r) > k * sig)
        bad |= hit
        print(f"[outlier] {name}: local sigma {floor:.3f} px (frame median), "
              f"{int(hit.sum())} windows beyond {k:g} sigma")
    print(f"[outlier] {int(bad.sum())} of {int(keep.sum())} measured windows dropped "
          f"({100 * bad.sum() / max(keep.sum(), 1):.2f} %)")
    return bad


def _decimate(a):
    """2x2 mean, edge-padded to an even shape."""
    ny, nx = a.shape
    p = np.pad(a, ((0, ny % 2), (0, nx % 2)), mode="edge")
    return 0.25 * (p[0::2, 0::2] + p[1::2, 0::2] + p[0::2, 1::2] + p[1::2, 1::2])


def reject_spikes(az, rg, keep, k_px, min_neighbors, box=7, max_passes=4):
    """Kept windows that disagree with their own neighbourhood by an ABSOLUTE amount."""
    from scipy.ndimage import uniform_filter
    bad = np.zeros_like(keep)
    h = box // 2
    for p_ in range(1, max_passes + 1):
        before = int(bad.sum())
        ok = keep & ~bad
        k = ok.astype(np.float64)
        n = np.rint(uniform_filter(k, box, mode="constant") * box * box - k).astype(int)
        if min_neighbors > 0:
            iso = ok & (n < min_neighbors)
            bad |= iso
            print(f"[outlier] pass {p_}: {int(iso.sum())} windows with fewer than "
                  f"{min_neighbors} kept neighbours in {box} x {box} dropped as unverifiable")
        if k_px > 0:
            for name, v in (("azimuth", az), ("range", rg)):
                w = np.where(keep & ~bad, v, np.nan).astype(np.float32)
                p = np.pad(w, h, constant_values=np.nan)
                nb = np.stack([p[h + di:h + di + w.shape[0], h + dj:h + dj + w.shape[1]]
                               for di in range(-h, h + 1) for dj in range(-h, h + 1)
                               if (di, dj) != (0, 0)])
                cnt = np.isfinite(nb).sum(axis=0)
                with np.errstate(all="ignore"):
                    med = np.nanmedian(nb, axis=0)
                del nb, p
                hit = keep & ~bad & (cnt >= 3) & (np.abs(v - med) > k_px)
                bad |= hit
                print(f"[outlier] pass {p_}: {name}: {int(hit.sum())} windows more than "
                      f"{k_px:g} px from the median of their kept {box} x {box} neighbours "
                      f"dropped")
        if int(bad.sum()) == before:
            break
    print(f"[outlier] spikes: {int(bad.sum())} of {int(keep.sum())} kept windows dropped "
          f"({100 * bad.sum() / max(keep.sum(), 1):.3f} %) in {p_} pass(es)")
    return bad


def regional_level(v, good, levels=9, thresh=0.02):
    """The level the surrounding measurements define, everywhere, bounded by them."""
    from scipy.ndimage import zoom

    num, den = [np.where(good, v, 0.0).astype(np.float32)], [good.astype(np.float32)]
    for _ in range(levels):
        num.append(_decimate(num[-1]))
        den.append(_decimate(den[-1]))
    est = np.where(den[-1] > 0, num[-1] / np.maximum(den[-1], 1e-9), float(v[good].mean()))
    for lv in range(levels - 1, -1, -1):
        ny, nx = num[lv].shape
        up = zoom(est, (ny / est.shape[0], nx / est.shape[1]), order=1,
                  mode="nearest")[:ny, :nx]
        c = np.clip(den[lv] / thresh, 0.0, 1.0)
        est = (c * (num[lv] / np.maximum(den[lv], 1e-9)) + (1 - c) * up).astype(np.float32)
    return est


FLAT_ELONG = 2.0        # below this a channel has no direction worth extrapolating along


def measure_aniso(v, keep, threshold=0.10):
    """How elongated a field is, and along what axis; returns (ra, rr, tilt, R_fft)."""
    from ._utils.elongation import reach, structure_function
    from ._utils.oriented import band_tilt, band_elongation

    ra = rr = np.inf
    tilt, _th_raw, R_fft = band_tilt(v, keep)
    if tilt:
        ra_b, rr_b = band_elongation(v, keep, tilt, threshold)
        el_b = (rr_b / ra_b if (np.isfinite(ra_b) and np.isfinite(rr_b) and ra_b > 0)
                else np.nan)
        if not np.isfinite(el_b) or max(el_b, 1.0 / el_b) < FLAT_ELONG:
            tilt = 0.0
        else:
            ra, rr = ra_b, rr_b
    if not tilt:
        ra = reach(structure_function(v, keep, 0), threshold)
        rr = reach(structure_function(v, keep, 1), threshold)
    return ra, rr, tilt, R_fft


def fill_priors(args, layers, keep, threshold=0.10, prior_from=None):
    """The roughness-penalty weight AND the trust radius for each layer, ONE PER CHANNEL."""
    trust = float(getattr(args, "fill_trust_km", 0.0) or 0.0)
    flat = float(getattr(args, "fill_trust_km_flat", 1.5) or 0.0)
    opt = getattr(args, "fill_aniso", 1.0)
    if not isinstance(opt, str):
        # an explicit weight is not a measurement, so there is nothing to switch on:
        # every layer keeps the single radius, exactly as before this option existed
        return {name: (float(opt), trust, 0.0) for name, _ in layers}

    want = getattr(args, "fill_aniso_direction", "auto") or "auto"
    lo, hi = {"auto": (1.0 / 64.0, 64.0),
              "range": (1.0, 64.0),
              "azimuth": (1.0 / 64.0, 1.0)}[want]
    out = {}
    for name, v in layers:
        if name == "snr":                      # not a physical offset field
            out[name] = (1.0, min(trust, flat) if flat else trust, 0.0)
            continue
        ra, rr, tilt, R_fft = measure_aniso(v, keep, threshold)
        # A field with no structure above the threshold cannot run away, so keep the full
        # trust radius instead of falling back to the short isotropic one.
        featureless = not (np.isfinite(ra) or np.isfinite(rr))
        raw = (float(rr / ra) if np.isfinite(rr) and np.isfinite(ra) and ra > 0 else 1.0)
        # Inherit the prior: a second pass sees the residual, which is isotropic, so measuring
        # there would cross the holes isotropically.  A resumed run has no earlier pass to
        # inherit from and measures on this pass's field plus the applied rubbersheet.
        cache = getattr(args, "_fill_aniso_seen", None)
        if cache is None:
            cache = {}
            setattr(args, "_fill_aniso_seen", cache)
        inherited = resumed = False
        if featureless and cache.get(name) is not None:
            (raw, tilt), inherited = cache[name], True     # the ANGLE is inherited too
        elif featureless and (prior_from or {}).get(name) is not None:
            ra2, rr2, tilt, _R = measure_aniso(prior_from[name], keep, threshold)
            if np.isfinite(ra2) or np.isfinite(rr2):
                ra, rr, resumed = ra2, rr2, True
                raw = (float(rr / ra) if np.isfinite(rr) and np.isfinite(ra) and ra > 0
                       else 1.0)
                featureless = False
                cache[name] = (raw, tilt)
        elif not featureless:
            cache[name] = (raw, tilt)
        w = float(np.clip(raw, lo, hi))
        note = ""
        if want != "auto" and not (lo <= raw <= hi):
            note = (f"  [!] measured x{raw:.3g} is {'azimuth' if raw < 1 else 'range'}"
                    f"-elongated, which contradicts --fill-aniso-direction {want}; clipped")
        elong = max(w, 1.0 / w) if w > 0 else 1.0
        t = (trust if (elong >= FLAT_ELONG or featureless)
             else (min(trust, flat) if flat else trust))
        out[name] = (w, t, tilt)
        frame = ("" if not tilt else
                 f"  [tilted: bands lie {tilt:+.1f} deg off the range axis at R {R_fft:.2f}, "
                 f"so this was measured in the BAND frame -- the axis-only estimator would "
                 f"have reported cot({abs(tilt):.0f} deg) = {1/np.tan(np.radians(abs(tilt))):.2g}]")
        print(f"[rdr-fill] --fill-aniso auto: {name} field reaches a {threshold:g} px change "
              f"in {ra:.0f} cells along track and {rr:.0f} across it -> x{w:.3g} "
              f"({'isotropic' if elong < 1.02 else ('range' if w > 1 else 'azimuth') + '-elongated'})"
              f", trusted to {t:g} km"
              f"{'  [featureless: the residual is noise, so this is inherited from the pass '
                 'that could measure it]' if inherited else
                 ('  [featureless: this pass is a RESUME and sees only the residual, so the '
                  'prior was measured on residual + the rubbersheet already applied]'
                  if resumed else
                  ('' if elong >= FLAT_ELONG else
                   ('  [featureless: never reaches the threshold, nothing to run away with]'
                    if featureless else '  [flat: no direction to extrapolate along]')))}"
              f"{frame}{note}", flush=True)
    return out


def fill_holes(azf, rgf, snrf, keep, args, smooth_out=None, cell=None, prior_from=None):
    """Interpolate across the holes on the RADAR lattice; returns the filled flag."""
    from ._utils.fill import fill as fill_method
    from ._utils.oriented import oriented_fill, to_band_frame, from_band_frame

    # cutoff wavelength -> penalty: the DCT transfer is 1/(1 + s*lambda^2), half-power
    # near a wavelength of 2*pi*s**0.25 samples.  One scale inside the trust radius;
    # --fill-trust-km covers the distance where the rim cannot be checked.
    s = args.fill_s if args.fill_s else (args.fill_cutoff / (2 * np.pi)) ** 4
    print(f"[rdr-fill] cutoff {args.fill_cutoff:g} samples -> s={s:.4g}", flush=True)

    print(f"[rdr-fill] method {args.fill_method}", flush=True)
    layers = (("azimuth", azf), ("range", rgf), ("snr", snrf))
    prior = fill_priors(args, layers, keep, prior_from=prior_from)
    out, rejected, per_channel = [], np.zeros_like(keep), {}
    for name, v in layers:
        tilt = prior[name][2]
        kw = (dict(s=s, robust=args.fill_robust, aniso=prior[name][0])
              if args.fill_method.startswith("pls") else {})
        axis = "the range axis" if not tilt else f"an axis {tilt:+.1f} deg off range"
        print(f"[rdr-fill] {name}:  penalty weighted x{prior[name][0]:g} on {axis}",
              flush=True)
        if tilt and args.fill_method.startswith("pls"):
            z, info = oriented_fill(v, keep, prior[name][0], tilt,
                                    s=s, robust=args.fill_robust)
        else:
            z, info = fill_method(v, keep, method=args.fill_method, **kw)
        if info.get("rejected") is not None and name != "snr":
            per_channel[name] = info["rejected"]
            rejected = rejected | info["rejected"]
        out.append(z)

    # Windows that failed at the search limit but cleared --snr-min are holes, not data.
    # Every channel's flags count: regional_level is a convex combination, so a surviving
    # outlier sets the level the trust fade blends toward.
    good = keep & ~(rejected if args.fill_reject else np.zeros_like(keep))
    if args.fill_reject:
        share = ", ".join(f"{n} {100 * (keep & r).mean():.2f} %"
                          for n, r in per_channel.items())
        print(f"[rdr-fill] {int((keep & rejected).sum())} measured windows rejected as "
              f"unexplainable ({100 * (keep & rejected).mean():.2f} % of the lattice, union "
              f"over the channels: {share}) and filled instead")

    # Two scales: within --fill-hole-km of a good window the surface follows the data,
    # beyond it fades into a second stiffer fit, weighted exp(-(d/km)^2) on isotropic
    # distance.  --fill-hole-cutoff auto hands the holes over only if the data surface
    # leaves the robust one by more than --fill-bulb-px deeper than BULB_DEPTH_KM into
    # a hole.  The verdict is the azimuth channel's and applies to all three.
    hole = getattr(args, "fill_hole_cutoff", None)
    auto = isinstance(hole, str) and hole.lower() == "auto"
    if auto:
        hole = BULB_HOLE_CUTOFF
    if hole and cell is not None and args.fill_method.startswith("pls"):
        from scipy.ndimage import distance_transform_edt, label
        robust_h = int(args.fill_hole_robust)
        if (auto and not args.fill_s and args.fill_cutoff >= hole
                and min(args.fill_robust, robust_h) >= 1):
            print(f"[rdr-fill] bulb check: the fill already is the robust one (cutoff "
                  f"{args.fill_cutoff:g} >= {hole:g}, bisquare on) -- nothing to hand the "
                  f"holes to", flush=True)
        else:
            s_hole = (hole / (2 * np.pi)) ** 4
            km = float(args.fill_hole_km)
            px = float(getattr(args, "fill_bulb_px", 2.0))
            d = distance_transform_edt(~good, sampling=cell) / 1000.0
            w = np.exp(-(d / km) ** 2).astype(np.float32)
            deep = ~good & (d > BULB_DEPTH_KM)
            km2 = cell[0] * cell[1] / 1e6
            blend = not auto
            for k, (name, v) in enumerate(layers):
                if k > 0 and not blend:
                    break
                aniso_k, _, tilt = prior[name]
                if tilt:
                    zh, _ = oriented_fill(v, keep, aniso_k, tilt, s=s_hole, robust=robust_h)
                else:
                    zh, _ = fill_method(v, keep, method=args.fill_method, s=s_hole,
                                        robust=robust_h, aniso=aniso_k)
                dev = np.abs(out[k] - zh)
                if k == 0 and auto:
                    worst = float(dev[deep].max()) if deep.any() else 0.0
                    over = deep & (dev > px)
                    n_reg = label(over)[1] if over.any() else 0
                    far = good & (np.abs(v - zh) > 5.0)
                    blend = worst > px
                    verdict = ("BULBS" if blend else "clean")
                    print(f"[rdr-fill] bulb check ({name}): {verdict} -- the data surface "
                          f"leaves the robust surface (cutoff {hole:g} / robust {robust_h}) "
                          f"by at most {worst:.2f} px deeper than {BULB_DEPTH_KM:g} km into "
                          f"the holes (threshold {px:g}); {over.sum() * km2:.1f} km2 in "
                          f"{n_reg} regions beyond it; {int(far.sum())} kept windows sit "
                          f"> 5 px from the robust surface"
                          + ("" if far.sum() == 0 else " (--outlier-spike would drop them)")
                          + ("; holes handed over" if blend else "; fill kept as is"),
                          flush=True)
                    if not blend:
                        break
                moved = dev[~good]
                out[k] = (w * out[k] + (1 - w) * zh).astype(np.float32)
                print(f"[rdr-fill] {name}: holes beyond {km:g} km of a good window handed to a "
                      f"cutoff {hole:g} / robust {robust_h} fit "
                      f"({100 * (w < 0.5).mean():.1f} % of the lattice past the half-weight "
                      f"point); the two surfaces differ by mean {moved.mean():.3f} px, "
                      f"p99 {np.percentile(moved, 99):.2f} px over the holes", flush=True)

    if any(p[1] > 0 for p in prior.values()) and cell is not None:
        from scipy.ndimage import distance_transform_edt

        # Distance to the nearest TRUSTED sample (`good`, not `keep`).  The ruler is stretched
        # by sqrt(aniso) PER CHANNEL, because the rim's information reaches that much further
        # along the fill's long axis; a mismatched ruler fences cells the fill could still
        # reach into regional_level(), which is isotropic.  Scale the long axis down, never
        # the short one up.
        notes, dcache = [], {}
        for k, (name, v) in enumerate(layers):
            aniso, trust, tilt = prior[name]
            if trust <= 0:
                notes.append(f"{name} (unbounded)")
                continue
            stretch = float(np.sqrt(aniso))
            key = (round(stretch, 6), round(tilt, 3))
            if key not in dcache:
                samp = ((cell[0], cell[1] / stretch) if stretch >= 1.0
                        else (cell[0] * stretch, cell[1]))
                if tilt:
                    # the ruler follows the tilt too, for the same reason
                    _, gv, solid = to_band_frame(good.astype(np.float64), good, tilt)
                    dd = distance_transform_edt(~(gv & solid), sampling=samp) / 1000.0
                    dcache[key] = from_band_frame(dd, tilt, good.shape)
                else:
                    dcache[key] = distance_transform_edt(~good, sampling=samp) / 1000.0
            d = dcache[key]
            w = np.where(d <= trust, np.float32(1.0),
                         np.exp(-((d - trust) / trust) ** 2)).astype(np.float32)
            reg = regional_level(v, good)
            faded = (w * out[k] + (1 - w) * reg).astype(np.float32)
            moved = np.abs(faded - out[k])[~good].mean()
            _ax, _f = (("range", stretch) if stretch >= 1.0 else ("azimuth", 1.0 / stretch))
            if tilt:
                _ax += f" tilted {tilt:+.1f} deg"
            notes.append(f"{name} {trust:g} km (ruler: {_ax} axis x{_f:.3g}, half-weight at "
                         f"{trust * (1 + np.sqrt(np.log(2))):.1f} km, fences "
                         f"{100 * (w < 0.5).mean():.2f} % of the lattice, mean |change| "
                         f"over the holes {moved:.3f})")
            out[k] = faded
        print("[rdr-fill] fill trusted, then faded into the regional level beyond it: "
              + "; ".join(notes), flush=True)
    elif any(p[1] > 0 for p in prior.values()):
        print("[rdr-fill] --fill-trust-km needs the lattice cell size; not applied")

    if smooth_out is not None:
        smooth_out.update({name: out[k] for k, (name, _) in enumerate(layers)})
    out = [np.where(good, v, z).astype(np.float32)
           for (_, v), z in zip(layers, out)]
    filled = ~good & np.isfinite(out[0])
    print(f"[rdr-fill] filled {int(filled.sum())} windows "
          f"({100 * filled.mean():.1f} % of the lattice); "
          f"{100 * (~keep & ~filled).mean():.1f} % left empty")
    return out[0], out[1], out[2], filled


def fill_interior(out, args, keys=("az", "rg", "snr")):
    """Close the holes the GEOCODING leaves inside the frame, in place."""
    from scipy import ndimage

    keys = [k for k in keys if k in out]
    if not keys:
        return
    nan = ~np.isfinite(out[keys[0]])
    lab, n = ndimage.label(nan)
    if n == 0:
        return
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    interior = nan & ~np.isin(lab, edge[edge > 0])
    if not interior.any():
        return
    print(f"[geo-fill] {int(interior.sum())} cells enclosed by data "
          f"({100 * interior.mean():.2f} % of the raster) left empty by the geocoding; "
          f"filling", flush=True)

    for k in keys:
        a = out[k].astype(np.float64)
        have = np.isfinite(a)
        todo = interior.copy()
        for _ in range(200):
            if not todo.any():
                break
            num = ndimage.uniform_filter(np.where(have, a, 0.0), 3, mode="nearest")
            den = ndimage.uniform_filter(have.astype(np.float64), 3, mode="nearest")
            grow = todo & (den > 0)
            a[grow] = (num[grow] / den[grow])
            have |= grow
            todo &= ~grow
        out[k] = a.astype(np.float32)
    if "filled" in out:
        out["filled"] = np.where(interior, 1.0, out["filled"]).astype(np.float32)
