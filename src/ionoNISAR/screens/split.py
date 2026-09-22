"""Split-spectrum ionospheric screen from the RSLC pair's two carriers, frequency A + B."""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np


from osgeo import gdal

from .._utils import raster as U
from .._utils import plots as PL

gdal.UseExceptions()


# =====================================================================
# frequency B: its own coregistration
# =====================================================================
def swath_axes(path, freq, pol="HH"):
    """(zeroDopplerTime, slantRange, centre frequency, orbit times/positions) of a crop."""
    import h5py

    with h5py.File(path, "r") as f:
        band = next(b for b in f["science"] if "RSLC" in f[f"science/{b}"])
        sw = f"science/{band}/RSLC/swaths"
        fg = f"{sw}/frequency{freq}"
        if fg not in f:
            raise SystemExit(f"{path} has no frequency{freq} "
                             f"(has {[k for k in f[sw] if k.startswith('frequency')]})")
        fc = (f"{fg}/processedCenterFrequency" if f"{fg}/processedCenterFrequency" in f
              else f"{fg}/acquiredCenterFrequency")
        og = f"science/{band}/RSLC/metadata/orbit"
        return dict(zt=np.asarray(f[f"{sw}/zeroDopplerTime"][:], np.float64),
                    sr=np.asarray(f[f"{fg}/slantRange"][:], np.float64),
                    fc=float(f[fc][()]),
                    shape=tuple(f[f"{fg}/{pol}"].shape),
                    orbit_t=np.asarray(f[f"{og}/time"][:], np.float64),
                    orbit_p=np.asarray(f[f"{og}/position"][:], np.float64))


def check_pair(path_a, path_b, pol="HH", strict=True):
    """A and B must share the azimuth grid exactly, and the same orbit."""
    a = swath_axes(path_a, "A", pol)
    b = swath_axes(path_b, "B", pol)
    print(f"[check] A {os.path.basename(path_a)}")
    print(f"          {a['shape'][0]} x {a['shape'][1]}  fc {a['fc'] / 1e6:9.3f} MHz  "
          f"dr {a['sr'][1] - a['sr'][0]:.4f} m  sr0 {a['sr'][0]:.3f} m")
    print(f"[check] B {os.path.basename(path_b)}")
    print(f"          {b['shape'][0]} x {b['shape'][1]}  fc {b['fc'] / 1e6:9.3f} MHz  "
          f"dr {b['sr'][1] - b['sr'][0]:.4f} m  sr0 {b['sr'][0]:.3f} m")
    lever = abs(b["fc"] - a["fc"])
    f0, f1 = a["fc"], b["fc"]
    det = (f0 - f1) / f0 - (f1 - f0) / f1
    print(f"[check] lever {lever / 1e6:.1f} MHz, determinant {det:+.4f} "
          f"-> the 2x2 solve amplifies phase noise ~{1 / abs(det):.1f}x")

    problems = []
    if lever < 1e6:
        problems.append(f"the two files carry the same carrier ({f0 / 1e6:.1f} MHz)")
    if a["zt"].shape != b["zt"].shape:
        problems.append(f"azimuth axes differ in LENGTH: A {a['zt'].size} vs B {b['zt'].size}")
    else:
        dz = np.abs(a["zt"] - b["zt"])
        if dz.max() > 1e-9:
            # in lines, at THIS product's sample rate -- zeroDopplerTimeSpacing, not a PRF
            # written down here (NISAR presums, so the two differ by a factor of ~1.26)
            prf = 1.0 / (a["zt"][1] - a["zt"][0])
            problems.append(f"azimuth times differ by up to {dz.max():.3e} s "
                            f"({dz.max() * prf:.3f} px at {prf:.2f} Hz) -- the crops are "
                            f"not row-aligned")
        else:
            print(f"[check] azimuth axes identical over all {a['zt'].size} lines "
                  f"(max |dt| {dz.max():.2e} s)  OK")
    if a["orbit_t"].shape != b["orbit_t"].shape or \
            np.abs(a["orbit_t"] - b["orbit_t"]).max() > 1e-6:
        problems.append("the two crops carry different orbit state-vector times")
    else:
        dp = np.abs(a["orbit_p"] - b["orbit_p"]).max()
        if dp > 1e-3:
            problems.append(f"orbit positions differ by up to {dp:.3e} m")
        else:
            print(f"[check] orbits identical ({a['orbit_t'].size} state vectors, "
                  f"max |dx| {dp:.2e} m)  OK")
    # range grids SHOULD differ -- that is the whole point -- but they must overlap
    lo = max(a["sr"][0], b["sr"][0])
    hi = min(a["sr"][-1], b["sr"][-1])
    if hi <= lo:
        problems.append(f"the range extents do not overlap: A {a['sr'][0]:.0f}..{a['sr'][-1]:.0f}, "
                        f"B {b['sr'][0]:.0f}..{b['sr'][-1]:.0f} m")
    else:
        fa = (hi - lo) / (a["sr"][-1] - a["sr"][0])
        print(f"[check] slant-range overlap {lo:.0f}..{hi:.0f} m "
              f"({100 * fa:.1f} % of A's swath)  OK")

    if problems:
        print("\n[check] THE PAIR IS NOT USABLE FOR SPLIT SPECTRUM:")
        for p in problems:
            print(f"  - {p}")
        if strict:
            raise SystemExit("frequency A and B must share one azimuth grid and one orbit")
        return False
    print("[check] A and B share one azimuth grid and one orbit -- usable")
    return True


def coregister_b(args, tag_dir):
    """Export and geometrically coregister the frequency B pair on ITS OWN radar grid."""
    from .. import coregister as O

    ref = O.load_slc(args.ref_b, "B", args.pol)
    sec = O.load_slc(args.sec_b, "B", args.pol)
    dem_raster, demI = U.load_dem(args.dem)

    win = (0, 0, ref["shape"][0], ref["shape"][1])
    centre = (win[0] + win[2] // 2, win[1] + win[3] // 2)
    print(f"[B] frequency B crop {ref['shape'][0]} x {ref['shape'][1]}, "
          f"range spacing {float(ref['rg'].range_pixel_spacing):.4f} m, "
          f"wavelength {float(ref['rg'].wavelength):.6f} m")

    gaz, grg = O.gross_from_geometry(ref, sec, demI, centre)
    print(f"[B] gross offset from orbit + DEM: azimuth {gaz:+d} px, range {grg:+d} px")

    a = O.parse_args([
        "--ref", args.ref_b, "--sec", args.sec_b, "--dem", args.dem,
        "--freq", "B", "--pol", args.pol,
        "--gpus", args.gpu, "--tag", args.tag,
        "--out-dir", args.out_dir, "--scratch", tag_dir,
        "--coreg", "geometric", "--keep-scratch",
    ])
    a.az_spacing = O.azimuth_ground_spacing(ref, demI, centre)
    win = O.inset_full_frame(ref, sec, win, (gaz, grg), a)
    pair0 = O.export_pair(a, ref, sec, win, (gaz, grg))
    pair = O.coregister_isce3(a, pair0, ref, sec, win, dem_raster)
    return dict(args=a, pair=pair, pair0=pair0, ref=ref, sec=sec, win=win,
                dem_raster=dem_raster, demI=demI)


# =====================================================================
# the rubbersheet transfer -- the one new piece of geometry
# =====================================================================
def transfer_rubbersheet(b, npz_a, applied_a, guard=-1e5, block=1024):
    """Add A's azimuth rubbersheet to frequency B's azimuth.off."""
    from .. import coregister as O

    a_args, win_b = b["args"], b["win"]
    rgA = b["ref_a"]["rg"]
    rgB = b["ref"]["rg"]
    winA = b["win_a"]

    # slant range of every A lattice column, in absolute metres
    d = np.load(npz_a)
    searchA = tuple(int(v) for v in d["search"])
    winszA = tuple(int(v) for v in d["winsize"])
    skipA = tuple(int(v) for v in d["skip"])
    nlat_a, nlat_r = applied_a.shape
    jA = searchA[1] + winszA[1] // 2 + skipA[1] * np.arange(nlat_r)
    R_lat = (rgA.starting_range + (winA[1] + jA) * float(rgA.range_pixel_spacing))

    path = os.path.join(b["args"].scratch, "geo2rdr", "azimuth.off")
    ds = gdal.Open(path)
    band = ds.GetRasterBand(1)
    na, nr = ds.RasterYSize, ds.RasterXSize

    # --- range: B crop column -> fractional A lattice column, via slant range
    R_b = rgB.starting_range + (win_b[1] + np.arange(nr)) * float(rgB.range_pixel_spacing)
    t = np.interp(R_b, R_lat, np.arange(nlat_r))          # clamped at both ends by np.interp
    jj = np.minimum(t.astype(np.int64), nlat_r - 2)
    wj = (t - jj).astype(np.float32)

    # --- azimuth: shared grid, so this is the plain lattice map with the two origins
    iA = searchA[0] + winszA[0] // 2 + skipA[0] * np.arange(nlat_a)   # A crop rows
    row_abs = winA[0] + iA                                            # A frame rows
    # B frame row of every B crop row.  zeroDopplerTime is shared, so frame rows correspond
    # one to one; only the crop origins differ.
    rows_b = win_b[0] + np.arange(na)
    s = np.interp(rows_b, row_abs, np.arange(nlat_a))
    ii = np.minimum(s.astype(np.int64), nlat_a - 2)
    wi = (s - ii).astype(np.float32)

    print(f"[B] rubbersheet transfer: A lattice {applied_a.shape} in A pixels -> "
          f"B crop {na} x {nr}")
    print(f"[B]   slant range {R_lat[0]:.0f}..{R_lat[-1]:.0f} m (A lattice) vs "
          f"{R_b[0]:.0f}..{R_b[-1]:.0f} m (B crop); "
          f"dr {float(rgA.range_pixel_spacing):.4f} -> {float(rgB.range_pixel_spacing):.4f} m")
    inside = (R_b >= R_lat[0]) & (R_b <= R_lat[-1])
    print(f"[B]   {100 * inside.mean():.1f} % of B columns fall inside the A lattice's "
          f"range span; outside it the edge value is held, not extrapolated")

    f = np.where(np.isfinite(applied_a), applied_a, 0.0).astype(np.float32)
    cols = f[:, jj] * (1.0 - wj) + f[:, jj + 1] * wj      # lattice rows x nr

    # copy-on-write: the raster may be a compressed GeoTIFF
    drv = ds.GetDriver().ShortName
    comp = ds.GetMetadataItem("COMPRESSION", "IMAGE_STRUCTURE")
    cow = drv == "GTiff" and bool(comp)
    if cow:
        tmp = path + ".part"
        out = gdal.GetDriverByName("GTiff").Create(
            tmp, nr, na, 1, band.DataType,
            options=["TILED=YES", "BLOCKXSIZE=512", "BLOCKYSIZE=512", "COMPRESS=DEFLATE",
                     "PREDICTOR=3", "ZLEVEL=6", "NUM_THREADS=ALL_CPUS", "BIGTIFF=YES"])
        ob = out.GetRasterBand(1)
    else:
        ds = None
        ds = gdal.Open(path, gdal.GA_Update)
        band = ds.GetRasterBand(1)
        ob = band
    mid_before = mid_after = None
    t0 = time.time()
    for i0 in range(0, na, block):
        n = min(block, na - i0)
        r, w = ii[i0:i0 + n], wi[i0:i0 + n, None]
        inc = cols[r] * (1.0 - w) + cols[r + 1] * w
        cur = band.ReadAsArray(0, i0, nr, n)
        new = np.where(cur > guard, cur + inc, cur)
        ob.WriteArray(new, 0, i0)
        if i0 <= na // 2 < i0 + n:
            k = na // 2 - i0
            mid_before, mid_after = float(cur[k, nr // 2]), float(new[k, nr // 2])
    if cow:
        out.FlushCache(); out = None; ds = None
        os.replace(tmp, path)
    else:
        ds.FlushCache(); ds = None

    # the check: what was added at the crop centre against what the A lattice says there
    got = mid_after - mid_before
    i_, u_, j_, v_ = ii[na // 2], wi[na // 2], jj[nr // 2], wj[nr // 2]
    want = ((f[i_, j_] * (1 - v_) + f[i_, j_ + 1] * v_) * (1 - u_)
            + (f[i_ + 1, j_] * (1 - v_) + f[i_ + 1, j_ + 1] * v_) * u_)
    ok = abs(got - want) < 1e-4
    print(f"[B]   check at crop centre (slant range {R_b[nr // 2]:.0f} m): added "
          f"{got:+.6f} px, the A lattice there holds {want:+.6f} px -> "
          f"{'OK' if ok else 'MISMATCH'}  ({time.time() - t0:.0f}s)")
    if not ok:
        raise SystemExit("the rubbersheet transfer did not land where the range mapping "
                         "says it should; B is NOT usable for split spectrum")
    v = applied_a[np.isfinite(applied_a)]
    print(f"[B]   transferred field: std {v.std():.4f} px, p1..p99 "
          f"{np.percentile(v, 1):+.3f} .. {np.percentile(v, 99):+.3f} px")


# =====================================================================
# one band's geocoded interferogram, on the shared lattice
# =====================================================================
def band_interferogram(args, freq, scratch, ref_h5, sec_h5, win, sec_origin, looks,
                       out_dir, tag, geogrid):
    """Flatten, multilook and geocode one band onto the shared 120 m lattice."""
    from .. import coregister as O
    from .. import interferogram as C

    ptag = f"{tag}_split{freq}"
    done = os.path.join(out_dir, f"ifg_coh_geo_{ptag}.tif")
    if os.path.exists(done):
        print(f"[{freq}] {ptag} already built")
        return ptag
    r1 = O.load_slc(ref_h5, freq, args.pol)["rg"]
    r2 = O.load_slc(sec_h5, freq, args.pol)["rg"]
    dr0 = ((r2.starting_range + sec_origin[1] * r2.range_pixel_spacing)
           - (r1.starting_range + win[1] * r1.range_pixel_spacing))
    carrier = 2 * np.pi * float(O.sampled_doppler(
        O.load_slc(sec_h5, freq, args.pol)["dop"], r2.prf).eval(
            r2.sensing_start + (sec_origin[0] + win[2] / 2) / r2.prf,
            r2.starting_range + (sec_origin[1] + win[3] / 2) * r2.range_pixel_spacing)) \
        / float(r2.prf)
    sc = os.path.join(out_dir, f"scratch_{ptag}")
    os.makedirs(sc, exist_ok=True)
    argv = [
        "--ref", os.path.join(scratch, "ref.c8"),
        "--sec", os.path.join(scratch, "sec_coreg.c8"),
        "--shape", str(win[2]), str(win[3]),
        "--looks", str(looks[0]), str(looks[1]),
        "--out-dir", out_dir, "--tag", ptag, "--scratch", sc,
        "--wavelength", repr(float(r1.wavelength)),
        "--range-spacing", repr(float(r1.range_pixel_spacing)),
        "--ref-h5", ref_h5, "--sec-h5", sec_h5, "--freq", freq, "--dem", args.dem,
        "--ref-origin", str(win[0]), str(win[1]),
        "--sec-origin", str(sec_origin[0]), str(sec_origin[1]),
        "--flatten", "--range-off", os.path.join(scratch, "geo2rdr", "range.off"),
        "--dr0", repr(float(dr0)),
        *(["--interp-bias", bias] if (bias := getattr(args, f"interp_bias_{freq.lower()}", None)) else []),
        # the resampler's own term: it re-ramps about the INPUT position, so every pixel of
        # applied shift leaves -carrier * delta in ref * conj(sec).  Both bands carry it and
        # it does NOT cancel in the difference -- their PRF is shared but their Doppler is
        # annotated per frequency.
        "--azimuth-off", os.path.join(scratch, "geo2rdr", "azimuth.off"),
        "--az-carrier", repr(float(carrier)),
        "--pol", args.pol,
        "--geocode", "--posting", "120.0", "--geogrid", *geogrid,
        "--no-complex-ifg",
    ] + (["--az-carrier-lut"] if args.az_carrier_mode == "lut" else [])
    print(f"[{freq}] dr0 {dr0:+.3f} m, carrier {carrier:+.4f} rad/px, "
          f"looks {looks[0]}x{looks[1]}")
    C.main(argv)
    return ptag


# isce3's Phass/ICU fill for "not unwrapped".  Finite, so every isfinite() guard misses it.
UNW_NODATA = -10000.0


def looks_for(rg_a, rg_b, looks_a):
    """Range looks for B that cover the same ground as A's, so one geocode cell gets both."""
    da = looks_a[1] * float(rg_a.range_pixel_spacing)
    lr = max(1, int(round(da / float(rg_b.range_pixel_spacing))))
    print(f"[looks] A {looks_a[1]} x {float(rg_a.range_pixel_spacing):.4f} m = {da:.1f} m "
          f"of slant range; B {lr} x {float(rg_b.range_pixel_spacing):.4f} m = "
          f"{lr * float(rg_b.range_pixel_spacing):.1f} m")
    return (looks_a[0], lr)


# =====================================================================
# the solve
# =====================================================================
def warp_to_grid(path, gt, epsg, shape):
    """A raster resampled onto (gt, epsg, shape) with nearest neighbour, as an array."""
    from osgeo import gdal, osr
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(int(epsg))
    ds = gdal.Warp("", path, format="MEM", outputType=gdal.GDT_Float32,
                   dstSRS=srs.ExportToWkt(), resampleAlg="near", dstNodata=float("nan"),
                   width=int(shape[1]), height=int(shape[0]),
                   outputBounds=(gt[0], gt[3] + shape[0] * gt[5],
                                 gt[0] + shape[1] * gt[1], gt[3]))
    if ds is None:
        raise SystemExit(f"could not warp {path} onto the solve grid")
    return ds.GetRasterBand(1).ReadAsArray().astype(np.float64)


def reject_outliers(v, valid, nmad, size=9):
    """Drop pixels a local median says are spikes rather than field."""
    from scipy.ndimage import median_filter

    med = median_filter(np.where(valid, v, 0.0).astype(np.float32), size=size)
    resid = np.abs(v - med)
    mad = float(np.median(resid[valid]))
    keep = valid & (resid < nmad * 1.4826 * mad)
    print(f"[outlier] local-median MAD {mad:.2f} rad; {nmad:g}-MAD keeps "
          f"{100 * keep.sum() / max(valid.sum(), 1):.1f} % of the mask "
          f"({int(valid.sum() - keep.sum())} spikes dropped)")
    return keep


def lowpass_weighted(v, w, sigma_px, floor_frac=0.05):
    """Gaussian low-pass weighted per pixel and normalised by the smoothed weight."""
    from scipy.ndimage import gaussian_filter

    w = w.astype(np.float32)
    num = gaussian_filter(np.where(w > 0, v * w, 0.0).astype(np.float32), sigma_px)
    den = gaussian_filter(w, sigma_px)
    floor = floor_frac * float(gaussian_filter((w > 0).astype(np.float32), sigma_px).max())
    return np.where(den > floor, num / np.maximum(den, 1e-9), np.nan)


def integrate_wrapped_gradient(phi, mask, w, sigma_px, floor_frac=0.05, iters=400, tol=1e-6):
    """The low-passed phase implied by the WRAPPED interferogram's own gradients -- no unwrapper."""
    from scipy.ndimage import gaussian_filter
    from scipy.fft import dctn, idctn

    m = np.asarray(mask, bool) & np.isfinite(phi)
    ph = np.where(m, phi, 0.0).astype(np.float64)
    w = np.where(m, np.nan_to_num(w), 0.0).astype(np.float32)
    grads, wts = [], []
    for ax in (1, 0):
        d = np.angle(np.exp(1j * np.diff(ph, axis=ax)))
        if ax == 1:
            mm, ww = m[:, 1:] & m[:, :-1], np.minimum(w[:, 1:], w[:, :-1])
        else:
            mm, ww = m[1:, :] & m[:-1, :], np.minimum(w[1:, :], w[:-1, :])
        ww = np.where(mm, ww, 0.0).astype(np.float32)
        g = lowpass_weighted(np.where(mm, d, 0.0), ww, sigma_px, floor_frac)
        den = gaussian_filter(ww, sigma_px)
        grads.append(g)
        wts.append(np.where(np.isfinite(g), den / max(float(den.max()), 1e-12), 0.0))
    gx, gy = grads
    Wx, Wy = wts
    Gx, Gy = np.nan_to_num(gx), np.nan_to_num(gy)
    ny, nx = ph.shape

    def A(p):
        """-(D^T W D) p: the weighted Laplacian, negative semi-definite like `lam` below."""
        dx, dy = np.diff(p, axis=1) * Wx, np.diff(p, axis=0) * Wy
        out = np.zeros_like(p)
        out[:, :-1] -= dx
        out[:, 1:] += dx
        out[:-1, :] -= dy
        out[1:, :] += dy
        return -out

    b = np.zeros((ny, nx))
    b[:, :-1] -= Wx * Gx
    b[:, 1:] += Wx * Gx
    b[:-1, :] -= Wy * Gy
    b[1:, :] += Wy * Gy
    b = -b
    ii = np.arange(ny)[:, None]
    jj = np.arange(nx)[None, :]
    lam = (2 * np.cos(np.pi * ii / ny) - 2) + (2 * np.cos(np.pi * jj / nx) - 2)
    lam[0, 0] = -1e-3

    def Minv(r):
        """DCT inverse of the unweighted Neumann Laplacian, the mean pinned to zero."""
        R = dctn(r, type=2, norm="ortho") / lam
        R[0, 0] = 0.0
        return idctn(R, type=2, norm="ortho")

    x = np.zeros((ny, nx))
    r = b - A(x)
    z = Minv(r)
    p = z.copy()
    rz = float((r * z).sum())
    bn = float(np.sqrt((b * b).sum()))
    for _ in range(iters):
        Ap = A(p)
        alpha = rz / float((p * Ap).sum())
        x += alpha * p
        r -= alpha * Ap
        if float(np.sqrt((r * r).sum())) < tol * bn:
            break
        z = Minv(r)
        rz_new = float((r * z).sum())
        p = z + (rz_new / rz) * p
        rz = rz_new
    reach = np.isfinite(lowpass_weighted(ph, w, sigma_px, floor_frac))
    x = np.where(reach, x, np.nan)
    return x - np.nanmean(x), gx, gy


def fit_trend(v, keep, order):
    """A low-order surface through `v` on `keep`, over the whole grid; 0 when order is 0."""
    out = np.zeros(v.shape, np.float64)
    if order <= 0 or keep.sum() < 100:
        return out
    ny, nx = v.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    yy, xx = yy / ny - 0.5, xx / nx - 0.5
    cols = [np.ones_like(yy), yy, xx]
    if order >= 2:
        cols += [yy * yy, xx * xx, yy * xx]
    G = np.stack([c[keep] for c in cols], 1)
    coef, _, _, _ = np.linalg.lstsq(G, v[keep].astype(np.float64), rcond=None)
    for c, col in zip(coef, cols):
        out += c * col
    print(f"[fill] continuation trend of order {order}: {coef[1]:+.1f} rad per frame down "
          f"the rows, {coef[2]:+.1f} across the columns")
    return out


def fill_gaps(v, keep, domain):
    """Continue the screen across the cells the low-pass could not reach, without a step."""
    from scipy.ndimage import distance_transform_edt, gaussian_filter, zoom

    if not keep.any():
        return np.where(domain, v, np.nan).astype(np.float32)
    src = np.where(keep, v, np.nan).astype(np.float32)
    dist, idx = distance_transform_edt(~keep, return_distances=True, return_indices=True)
    out = src[tuple(idx)]                       # nearest measured value, everywhere
    need = float(dist[domain].max()) if domain.any() else 0.0

    d = 8                                       # 960 m cells at the 120 m posting
    # sigma_eff of n relaxations at sigma 2 px is 2*sqrt(n); ask for twice the deepest gap
    n = int(min(4000, max(50, (need / d) ** 2)))
    k, s0 = keep[::d, ::d], out[::d, ::d]
    c = s0.copy()
    for _ in range(n):
        c = np.where(k, s0, gaussian_filter(c, 2.0))
    out = np.where(keep, src, zoom(c, (v.shape[0] / c.shape[0], v.shape[1] / c.shape[1]),
                                   order=1)[:v.shape[0], :v.shape[1]])
    for _ in range(40):
        out = np.where(keep, src, gaussian_filter(out, 2.0))
    print(f"[fill] continued the screen over {100 * (domain & ~keep).mean():.1f} % of the "
          f"frame; the deepest gap was {0.12 * need:.1f} km from measured ground, closed "
          f"with {n} coarse relaxations")
    return np.where(domain, out, np.nan).astype(np.float32)


def solve_split(args, pa, pb, ca, cb, f0, f1, gt, epsg, out_dir, tag):
    """phi_A and phi_B on one grid -> the dispersive screen."""
    from scipy.ndimage import uniform_filter
    from .._utils import unwrap as unwrap_backend

    det = (f0 - f1) / f0 - (f1 - f0) / f1
    print(f"\n[solve] frequency A {f0 / 1e6:.3f} MHz, B {f1 / 1e6:.3f} MHz, "
          f"lever {abs(f1 - f0) / 1e6:.1f} MHz")
    print(f"[solve] determinant {det:+.4f} -> phase noise amplified ~{1 / abs(det):.1f}x "
          f"(inherent to the lever; --iono-filter-km buys back looks instead)")

    valid = np.isfinite(pa) & np.isfinite(pb) & np.isfinite(ca) & np.isfinite(cb)
    # ONE THRESHOLD FOR TWO UNEQUAL LOOK COUNTS.  looks_for matches the GROUND cell, and B's
    # range spacing is 8x A's, so B carries an eighth of A's samples per cell.  The
    # zero-signal coherence floor is 0.886/sqrt(N), which differs between the bands, so one
    # threshold is not the same statement about both.  --coh-thresh-b exists so the question
    # can be re-asked per frame; the default is --coh-thresh for both.
    tb = args.coh_thresh if args.coh_thresh_b is None else args.coh_thresh_b
    mask = valid & (ca >= args.coh_thresh) & (cb >= tb)
    print(f"[mask] valid in both bands {100 * valid.mean():.1f} %, coherent in both "
          f"{100 * mask.mean():.1f} % (coh_A median {np.median(ca[valid]):.3f} vs threshold "
          f"{args.coh_thresh}, coh_B median {np.median(cb[valid]):.3f} vs {tb})")
    if mask.sum() < 1000:
        raise SystemExit(f"only {int(mask.sum())} pixels clear --coh-thresh "
                         f"{args.coh_thresh} / {tb} in both bands; nothing to solve")

    # ICE IS READ HERE, not at the gate below, because --diff-weight coherence needs it
    # BEFORE the band difference is averaged: dropping ice from `mask` afterwards does
    # nothing about the (dw-1)/2 px of it that the box already mixed into the surrounding
    # land.  The gate itself still happens where it did.
    ice = np.zeros(mask.shape, bool)
    if args.glacier_mask:
        gm = U.read_gtiff(args.glacier_mask)
        if gm.shape != mask.shape:
            # The solve grid moves with the pair, so the mask is warped onto it here rather
            # than required to arrive on it.  Nearest-neighbour: it is a 0/1 label, and
            # interpolating a label invents rims.
            print(f"[mask] glacier: {args.glacier_mask} is {gm.shape}, not {mask.shape}; "
                  f"warping it onto the solve grid (nearest)")
            gm = warp_to_grid(args.glacier_mask, gt, epsg, mask.shape)
        ice = np.isfinite(gm) & (gm > 0.5)

    # --- the band difference, complex-averaged BEFORE it is phased.
    # A and B are different range spectra, so their speckle realisations are largely
    # independent and the per-pixel arg(I_A conj(I_B)) is close to uniform random even where
    # each band is individually coherent.  The dispersive signal is only ~4 % of the total
    # phase, so it emerges only after the random part is averaged down.
    dw = args.diff_win
    if args.diff_weight == "coherence":
        # the self-band construction: weight each phasor by
        # min(coh) and normalise by the SMOOTHED WEIGHT, so a decorrelated cell contributes
        # in proportion to what it knows.  Ice is zeroed outright -- its coherence is 0.080
        # against land's 0.456, but 0.08 x 441 cells is still a push.
        w_d = np.where(valid & ~ice, np.minimum(np.nan_to_num(ca), np.nan_to_num(cb)), 0.0)
        z = w_d * np.exp(1j * np.where(valid, pa - pb, 0.0))
        zs = uniform_filter(z.real, dw) + 1j * uniform_filter(z.imag, dw)
        phi_diff = np.angle(zs)
        coh_diff = np.clip(np.abs(zs) / np.maximum(uniform_filter(w_d, dw), 1e-12), 0, 1)
    else:
        z = np.exp(1j * np.where(valid, pa, 0.0)) * np.conj(np.exp(1j * np.where(valid, pb, 0.0)))
        z = np.where(valid, z, 0)
        zs = uniform_filter(z.real, dw) + 1j * uniform_filter(z.imag, dw)
        phi_diff = np.angle(zs)
        coh_diff = np.clip(np.abs(zs) / np.maximum(uniform_filter(np.abs(z), dw), 1e-12), 0, 1)

    # DIAGNOSTIC (no behaviour change).  `z` above is a UNIT phasor on all of `valid`, so a
    # fjord pixel at coherence 0.02 pushes exactly as hard as a land pixel at 0.95, and
    # `valid` is only "the geocoder wrote something here" -- this route builds its band
    # interferograms with no water mask and applies the ice gate BELOW, i.e. after this
    # average.  So decorrelated ground is injected (dw-1)/2 px into the surrounding land.
    # the self-band estimator does not do this (it weights by min(coh)), and
    # splitA does not show the fringe band.  Measure the cost directly: build the weighted
    # estimate too and report the disagreement against distance from the decorrelated edge,
    # in rad of SCREEN (x 1/|det|), which is the number that matters.
    _wt = np.where(valid, np.minimum(np.nan_to_num(ca), np.nan_to_num(cb)), 0.0)
    _zw = _wt * np.exp(1j * np.where(valid, pa - pb, 0.0))
    _zws = uniform_filter(_zw.real, dw) + 1j * uniform_filter(_zw.imag, dw)
    _dphi = np.angle(np.exp(1j * (phi_diff - np.angle(_zws))))
    _coh_ok = valid & (ca >= args.coh_thresh) & (cb >= args.coh_thresh)
    from scipy.ndimage import distance_transform_edt as _edt
    _d = _edt(_coh_ok)                      # px into the coherent region from its edge
    print(f"[diff/diag] the ACTIVE band difference (--diff-weight {args.diff_weight}) minus "
          f"the coherence-weighted one, by distance from the decorrelated edge "
          f"(1/|det| = {1 / abs(det):.2f}); zero by construction under `coherence`:")
    print(f"[diff/diag] {'px':>4} {'km':>6} {'n':>9} {'coh_diff':>9} "
          f"{'|d phi_diff|':>13} {'-> rad of screen':>17}")
    for _lo, _hi in ((1, 2), (2, 4), (4, 6), (6, 8), (8, 11), (11, 15), (15, 25), (25, 10**6)):
        _z2 = _coh_ok & (_d >= _lo) & (_d < _hi)
        if _z2.sum() < 500:
            continue
        _m = float(np.median(np.abs(_dphi[_z2])))
        print(f"[diff/diag] {_lo:>4} {_lo * abs(gt[1]) / 1000:>6.2f} {int(_z2.sum()):>9} "
              f"{float(np.median(coh_diff[_z2])):>9.3f} {_m:>13.4f} {_m / abs(det):>17.2f}")

    mask = mask & (coh_diff >= args.coh_thresh)

    # ICE.  Coherence over ice is far below land in both bands, and what does clear the
    # gate is amplified by 1/|det| in the solve, so the screen there would be noise with the
    # amplitude of signal.  The azimuth routes never see this: ampcor's SNR gate and
    # --mask-glacier leave them continuing a low-pass from the surrounding land.
    #
    # Dropped from the MASK, not from the output: the weighted low-pass then carries the
    # land solution across the glacier the same way.  Blanking instead would put a step at
    # every margin -- see screen_from_offsets on why that is worse.
    if args.glacier_mask:
        was = mask.sum()
        mask = mask & ~ice
        print(f"[mask] glacier: {100 * ice.mean():.1f} % of the raster is ice; dropped "
              f"{int(was - mask.sum())} of {int(was)} measured px "
              f"({100 * (was - mask.sum()) / max(int(was), 1):.1f} %), mask now "
              f"{100 * mask.mean():.1f} % -- the low-pass carries land across it")

    px_km = abs(gt[1]) / 1000.0
    print(f"[diff] complex-averaged over {dw}x{dw} px ({dw * px_km:.1f} km); "
          f"band-difference coherence median {np.median(coh_diff[valid]):.3f}, "
          f"mask now {100 * mask.mean():.1f} %")
    pd_m = phi_diff[mask]
    print(f"[diff] |phi_diff| max {np.abs(pd_m).max():.4f} against pi = {np.pi:.4f}; "
          f"{100 * (np.abs(pd_m) > 0.9 * np.pi).mean():.2f} % past 0.9 pi")

    # THE BRANCH CUT.  phi_diff carries an arbitrary constant -- the dispersive datum is not
    # observable from the SAR data alone -- and if that constant sits near +-pi the whole
    # field is parked ON the cut, however coherent it is.  The unwrapper is then handed a
    # field that flips branch cell to cell, and every cycle it gets wrong is multiplied by
    # 1/|det| in the solve, i.e. 2 pi / |det| of screen.
    #
    # Rotating by the circular mean costs nothing: it moves disp by the constant -c/det, and
    # --level removes the mean afterwards anyway.
    if args.diff_datum == "rotate":
        c = float(np.angle(np.exp(1j * pd_m).mean()))
        phi_diff = np.angle(np.exp(1j * (phi_diff - c)))
        pd_m = phi_diff[mask]
        print(f"[diff] rotated the band difference by {-c:+.4f} rad to zero circular mean "
              f"(the dispersive datum is unobservable; --level removes it anyway)")
        print(f"[diff] after the rotation: |phi_diff| median "
              f"{np.median(np.abs(pd_m)):.3f}, {100 * (np.abs(pd_m) > 0.9 * np.pi).mean():.2f}"
              f" % past 0.9 pi -- one missed cycle here is worth "
              f"{2 * np.pi / abs(det):.0f} rad of screen")

    # not out_dir: the unwrapper's four scratch rasters are 38 MB each, they are overwritten
    # by the second unwrap anyway, and left beside the products the figure stage renders
    # them as if they were products
    unw_dir = os.path.join(out_dir, "scratch_unwrap")
    os.makedirs(unw_dir, exist_ok=True)

    def unwrap(what, phi, w, m):
        """Unwrap, and turn what the unwrapper DECLINED into NaN so the mask can drop it."""
        print(f"[unwrap] {what} ...", flush=True)
        # nlooks is snaphu's only statistical input and the two fields do NOT share one.
        # phi_A is the plain 24x16 multilook; the band difference was additionally complex-
        # averaged over diff_win x diff_win before it got here, so it carries diff_win^2
        # times as many samples.  Handing snaphu phi_A's count for both would make it treat
        # the band difference as far noisier than it is and refuse to connect regions.
        # Two thirds accounts for the oversampling that makes neighbouring SLC samples
        # correlated -- the same figure screen_sigma's retention factors assume.
        nlooks = args.looks[0] * args.looks[1] * (2.0 / 3.0)
        if what == "band difference":
            nlooks *= float(args.diff_win) ** 2
        # `m` is THIS field's domain, not the shared `mask` -- see the call sites.
        mask = m
        unw, conn = unwrap_backend.unwrap_ifg(
            np.exp(1j * np.where(mask, phi, 0.0)).astype(np.complex64),
            np.where(mask, w, 0.0).astype(np.float32), mask,
            method=args.unwrap_method, coh_thresh=args.coh_thresh,
            cache_dir=unw_dir, nlooks=nlooks,
            snaphu_cost=args.snaphu_cost, snaphu_init=args.snaphu_init,
            snaphu_ntiles=tuple(args.snaphu_ntiles),
            snaphu_nproc=args.snaphu_nproc)
        if unw is None:
            raise SystemExit(f"unwrapping {what} failed")
        unw = np.asarray(unw, np.float64)
        declined = np.zeros(unw.shape, bool) if conn is None else (conn == 0)
        declined |= (unw == UNW_NODATA)
        n = int((declined & mask).sum())
        if n:
            print(f"[unwrap] {what}: the unwrapper declined {n} px = "
                  f"{100 * n / max(int(mask.sum()), 1):.1f} % of the measured pixels "
                  f"(component 0 / sentinel {UNW_NODATA:g}); set to NaN so the mask drops "
                  f"them before the solve")

        # THE UNWRAPPER'S COMPONENT DATUM.  Every non-zero label carries its own arbitrary
        # integer cycle datum -- a landmass reachable only across a fjord is its own
        # component -- and one cycle is 0.51076*2pi of screen through phi_A and 2pi/|det|
        # through the band difference.  The low-pass then turns the step into a smooth ramp
        # that tracks the region outline: a fringe band hugging a decorrelated margin.
        #
        # Restricting to the largest component costs coverage, so the default re-references
        # instead: each component's median offset from a smooth reference built out of the
        # largest is rounded to a whole cycle and removed.  A near-integer offset is what
        # makes that safe, so the fractional part is the check, not an afterthought: a
        # component whose offset is not convincingly an integer is dropped rather than
        # guessed at, and the continuation fills it.
        good = mask & ~declined
        resid = np.angle(np.exp(1j * (unw - phi)))
        # Phass fills what it could not reach with -10000.0, a FINITE sentinel that every
        # isfinite() guard misses; wrap(unw - phi) == 0 catches that and any genuine unwrap
        # failure in one test.
        congruent = np.abs(resid) < 1e-3
        nbad = int((good & ~congruent).sum())
        print(f"[unwrap] {what}: congruence |wrap(unw - phi)| median "
              f"{float(np.median(np.abs(resid[good]))):.2e}, "
              f"{100 * nbad / max(int(good.sum()), 1):.2f} % fail the 1e-3 test")
        if nbad:
            declined = declined | (good & ~congruent)
            good = mask & ~declined

        if conn is not None and good.any() and args.unwrap_components != "all":
            lab = np.asarray(conn)
            ids, cnt = np.unique(lab[good], return_counts=True)
            tot = int(good.sum())
            big = ids[int(np.argmax(cnt))]
            print(f"[unwrap] {what}: {len(ids)} component(s); the largest holds "
                  f"{100 * cnt.max() / tot:.1f} % of the solved pixels")
            if args.unwrap_components == "largest":
                drop = good & (lab != big)
                declined = declined | drop
                print(f"[unwrap] {what}: --unwrap-components largest -- dropped "
                      f"{100 * drop.sum() / tot:.1f} % of the solved pixels")
            else:
                sig = args.iono_filter_km / (abs(gt[1]) / 1000.0)
                ref = lowpass_weighted(np.where(lab == big, unw, np.nan),
                                       np.where((lab == big) & good, w, 0.0), sig,
                                       args.iono_fill_floor)
                cyc = 2.0 * np.pi
                scale = 0.51076 if what.startswith("freq") else 1.0 / abs(det)
                moved = dropped = untouched = 0
                shift = np.zeros(unw.shape, np.float64)
                kill = np.zeros(unw.shape, bool)
                for i, c in sorted(zip(ids, cnt), key=lambda t: -t[1]):
                    if i == big:
                        continue
                    # `ref` is the largest component's field, low-passed and continued.
                    # Far from it, and over a component too small to average, the median
                    # below is dominated by how badly `ref` extrapolates rather than by any
                    # datum.  So only components large enough for the median to mean
                    # something are re-referenced; the rest are dropped, because a datum
                    # that cannot be measured cannot be trusted, and the continuation fills
                    # what they covered.
                    z2 = good & (lab == i) & np.isfinite(ref)
                    if c / tot < args.unwrap_min_component or z2.sum() < 50:
                        kill |= good & (lab == i)
                        untouched += int(c)
                        continue
                    d = float(np.median((unw - ref)[z2])) / cyc
                    k = float(np.round(d))
                    # k == 0 means there is no WHOLE-CYCLE error to remove, and the sub-cycle
                    # part is the component's real signal difference from the reference, not
                    # a datum.  Leave those alone: dropping them cost 16 % of frequency A and
                    # 14 % of the band difference -- which has no datum problem at all, its
                    # four largest components measuring +0.010, +0.036, -0.011, +0.015
                    # cycles -- for nothing.  Only a component that wants a NON-ZERO shift
                    # and does not land near a whole cycle is genuinely suspect.
                    if k == 0.0:
                        continue
                    if abs(d - k) > args.unwrap_cycle_tol:
                        kill |= good & (lab == i)
                        dropped += int((good & (lab == i)).sum())
                        continue
                    shift[good & (lab == i)] = k * cyc
                    moved += int(c)
                    if c / tot > 0.005:
                        print(f"[unwrap]   component {int(i):>4}: {100 * c / tot:>5.2f} % of "
                              f"pixels, offset {d:+7.3f} cycles -> removed {k:+.0f} "
                              f"({abs(k) * cyc * scale:.2f} rad of screen), "
                              f"residual fraction {d - k:+.3f}")
                unw = unw - shift
                declined = declined | kill
                print(f"[unwrap] {what}: re-referenced {100 * moved / tot:.1f} % of the "
                      f"solved pixels onto the largest component's datum; dropped "
                      f"{100 * dropped / tot:.1f} % whose whole-cycle offset was not within "
                      f"{args.unwrap_cycle_tol} of an integer; dropped a further "
                      f"{100 * untouched / tot:.1f} % in components too small to measure an "
                      f"offset on (their datum is unknown, so they cannot enter the solve)")
                # THE TRIPWIRE.  A whole-cycle datum in phi_A costs 0.51076*2pi of screen
                # and is a nuisance; the same thing in the BAND DIFFERENCE costs 2pi/|det|
                # and makes the screen worthless.  If it fires, the run STOPS: publishing a
                # split screen whose band difference needed re-referencing would put fringes
                # in the product that look physical.
                if moved and not what.startswith("freq") and not args.allow_diff_cycles:
                    raise SystemExit(
                        f"[unwrap] REFUSING TO CONTINUE: the band difference needed "
                        f"re-referencing on {100 * moved / tot:.1f} % of its solved pixels. "
                        f"One cycle there is {2 * np.pi / abs(det):.1f} rad of screen "
                        f"({2 * np.pi / abs(det) / (2 * np.pi):.1f} fringes), against "
                        f"{0.51076 * 2 * np.pi:.2f} rad for frequency A.  Look at why "
                        f"the band difference "
                        f"fragmented before trusting this screen; --allow-diff-cycles "
                        f"proceeds anyway once you have.")
        return np.where(declined, np.nan, unw), conn

    # UNWRAP EACH FIELD ON ITS OWN SUPPORT.  `mask` carries three coherence gates plus the
    # ice gate by this point, and frequency A needs only its own.  Handing it the others
    # cuts its domain into islands that have nothing to do with whether phi_A is
    # unwrappable, and every island is a component with its own arbitrary integer datum.
    # The solve still intersects with the full `mask` below, so no pixel enters the estimate
    # that the gates exclude.
    a_dom = mask
    if args.unwrap_support == "own":
        a_dom = valid & (ca >= args.coh_thresh)
        print(f"[unwrap] frequency A unwrapped on its OWN support: {100 * a_dom.mean():.1f} % "
              f"of the frame against the full mask's {100 * mask.mean():.1f} % -- the other "
              f"gates fragment it without saying anything about phi_A")
    # what was MEASURED, before any unwrapper declines part of it: gate 8's reference and the
    # --phia-from gradient estimator need no unwrapper and so keep this whole footprint
    mask_meas = mask.copy()
    phi_a_u, conn_a = unwrap("frequency A", pa, ca, m=a_dom)
    if args.unwrap_diff:
        phi_d_u, _ = unwrap("band difference", phi_diff, coh_diff, m=mask)
    else:
        phi_d_u = phi_diff
        print(f"[unwrap] --no-unwrap-diff: the estimate cannot leave "
              f"+-{np.pi / abs(det):.1f} rad")

    # An unwrapper does not unwrap everything it is handed, and what it declines comes back
    # as NaN.  Those pixels are not measurements and must leave the mask BEFORE the solve --
    # at 11.6x, one that gets through is worth more than the entire real screen, it survives
    # the local-median rejection wherever the declined region is bigger than the median
    # window, and the 12 km Gaussian then spreads it across a quarter of the frame.
    done = np.isfinite(phi_a_u) & np.isfinite(phi_d_u)
    lost = int((mask & ~done).sum())
    if lost:
        print(f"[unwrap] declined {lost} px = {100 * lost / mask.sum():.1f} % of the "
              f"measured pixels; dropped from the mask")
        mask = mask & done
    if args.unwrap_diff:
        added = np.round((phi_d_u - phi_diff)[mask] / (2 * np.pi))
        print(f"[unwrap] band difference gained at most {np.abs(added).max():.0f} cycle(s); "
              f"{100 * (added != 0).mean():.1f} % of measured pixels moved")

    try:
        from isce3.atmosphere.main_band_estimation import estimate_iono_main_diff
        disp, nondisp = estimate_iono_main_diff(
            f0=f0, f1=f1,
            phi0=np.where(mask, phi_a_u, 0.0).astype(np.float64),
            phi_diff_ms=np.where(mask, phi_d_u, 0.0).astype(np.float64))
    except Exception as e:
        # The 2x2 is closed form; isce3's helper is preferred only so the arithmetic is
        # theirs rather than mine, and its absence must not stop the route.
        print(f"[solve] isce3 main_band_estimation unavailable ({e}); using the "
              f"closed-form inverse of the same matrix")
        a11, a12 = 1.0, 1.0
        a21, a22 = (f1 - f0) / f1, (f0 - f1) / f0
        p0 = np.where(mask, phi_a_u, 0.0)
        pd = np.where(mask, phi_d_u, 0.0)
        disp = (a22 * p0 - a12 * pd) / det
        nondisp = (-a21 * p0 + a11 * pd) / det
    disp = np.where(mask, disp, np.nan)
    print(f"[solve] dispersive {np.nanpercentile(disp, 1):+.2f} .. "
          f"{np.nanpercentile(disp, 99):+.2f} rad (1-99 %)")
    ceiling = np.pi / abs(det)
    print(f"[solve] {100 * (np.abs(disp[mask]) > 0.95 * ceiling).mean():.2f} % of measured "
          f"pixels past 0.95 x the wrapped ceiling ({0.95 * ceiling:.1f} rad)")

    # --- reject spikes, THEN low-pass.  The other order smears each spike into the blob
    # it is meant to remove.
    good = mask & np.isfinite(disp)
    if args.outlier_mad > 0:
        good = reject_outliers(disp, good, args.outlier_mad)
    wts = np.where(good, coh_diff, 0.0)
    if args.iono_filter_km > 0:
        sig = args.iono_filter_km / px_km
        print(f"[filter] Gaussian sigma {args.iono_filter_km} km = {sig:.1f} px, weighted "
              f"by the band-difference coherence")
        disp_f = lowpass_weighted(disp, wts, sig, args.iono_fill_floor)
    else:
        sig = 0.0
        disp_f = np.where(good, disp, np.nan)

    # --- THE TWO TERMS OF THE SOLVE, KEPT APART.  disp = c_a*phi_A + c_d*phi_diff, and
    # lowpass_weighted is linear in `v` at a fixed `w`, so low-passing each term with the
    # SAME weights and the same sigma splits the filtered screen EXACTLY: term_a + term_d
    # is disp_f, before the datum and the continuation.  That is what makes the attribution
    # in check_against a measurement rather than an estimate.  The excess this route carries
    # against the routes that never touch an absolute phase is either a whole-cycle error in
    # phi_A (one cycle = c_a * 2 pi rad of screen) or a systematic in phi_diff (one rad =
    # c_d rad of screen), and the two differ by a factor c_d/c_a = 22.7 -- so the size alone
    # does not say which, but the SHAPE on phi_A's connected components does.
    coef_a, coef_d = ((f0 - f1) / f0) / det, -1.0 / det
    if sig > 0:
        term_a = coef_a * lowpass_weighted(phi_a_u, wts, sig, args.iono_fill_floor)
        term_d = coef_d * lowpass_weighted(phi_d_u, wts, sig, args.iono_fill_floor)
    else:
        term_a = np.where(good, coef_a * phi_a_u, np.nan)
        term_d = np.where(good, coef_d * phi_d_u, np.nan)
    both = np.isfinite(term_a) & np.isfinite(term_d) & np.isfinite(disp_f)
    if both.any():
        err = float(np.abs((term_a + term_d - disp_f)[both]).max())
        print(f"[terms] phi_A term {coef_a:+.5f} x phi_A, band-difference term "
              f"{coef_d:+.5f} x phi_diff; they reconstruct the screen to {err:.2e} rad")
        print(f"[terms] their sizes: phi_A term std {np.nanstd(term_a[both]):.1f} rad, "
              f"band-difference term std {np.nanstd(term_d[both]):.1f} rad, screen "
              f"{np.nanstd(disp_f[both]):.1f} rad.  If the two ADD rather than cancel, "
              f"the sensitivity is not a near-cancellation -- it is simply the "
              f"{abs(coef_d):.1f}x on the band difference.")

    # --- GATE 8: THE UNWRAPPED phi_A MUST AGREE WITH THE WRAPPED INTERFEROGRAM AT 12 km.
    # phi_A is the only absolute unwrapped phase any route consumes, and its whole-cycle
    # errors survive the low-pass as smooth humps that no downstream check can tell from
    # ionosphere.  The wrapped interferogram's own gradients, low-passed with the same weights
    # and integrated (integrate_wrapped_gradient), are the reference the unwrapper has to
    # reproduce: same field, no datum.  The run stops if it does not, because a screen built on
    # a phi_A that fails here can carry tens of radians of hump into the screen.
    phi_grad = None
    if sig > 0:
        t0 = time.time()
        phi_grad, gx_lp, gy_lp = integrate_wrapped_gradient(
            pa, mask_meas, np.where(mask_meas, coh_diff, 0.0), sig, args.iono_fill_floor)
        lp_a = term_a / coef_a
        k = np.isfinite(lp_a) & np.isfinite(phi_grad) & good
        x, y = phi_grad[k] - phi_grad[k].mean(), lp_a[k] - lp_a[k].mean()
        sxy, sxx, syy = float(x @ y), float(x @ x), float(y @ y)
        g_corr = sxy / np.sqrt(sxx * syy)
        g_lo, g_hi = sorted((sxy / sxx, syy / sxy))
        gc = []
        for g, ax in ((gx_lp, 1), (gy_lp, 0)):
            d = np.diff(lp_a, axis=ax)
            kk = np.isfinite(g) & np.isfinite(d)
            u, v = g[kk] - g[kk].mean(), d[kk] - d[kk].mean()
            gc.append(float(u @ v / np.sqrt((u @ u) * (v @ v))))
        print(f"[gate] phi_A: unwrapped-and-low-passed against the wrapped interferogram's "
              f"own low-passed gradient, integrated ({time.time() - t0:.0f} s, n={int(k.sum())}): "
              f"corr {g_corr:+.3f}, amplitude bracket [{g_lo:.3f}, {g_hi:.3f}] (1 = same "
              f"field); gradient corr along x {gc[0]:+.3f}, along y {gc[1]:+.3f}; std "
              f"unwrapped {y.std():.2f} rad, integrated {x.std():.2f} rad")
        g_min, (b_lo, b_hi) = args.gate_corr, args.gate_bracket
        passed = g_corr >= g_min and g_lo >= b_lo and g_hi <= b_hi
        lowered = "" if (g_min, b_lo, b_hi) == (0.9, 0.85, 1.15) else \
            " (LOWERED from 0.9 / [0.85, 1.15] by --gate-corr / --gate-bracket)"
        print(f"[gate] {'PASS' if passed else 'FAIL'}: needs corr >= {g_min:g} and the bracket "
              f"inside [{b_lo:g}, {b_hi:g}]{lowered}; the field that passes is what the "
              f"interferogram itself says at 12 km")
        if not passed and args.phia_from == "unwrap" and not args.allow_unwrap_mismatch:
            raise SystemExit("[gate] REFUSING TO CONTINUE: the unwrapped phi_A does not "
                             "reproduce the wrapped interferogram at the screen scale.  Use "
                             "a different unwrapper (--unwrap-method phass | icu, "
                             "--unwrap-components largest), or --phia-from gradient to build "
                             "the screen on the integrated field, or "
                             "--allow-unwrap-mismatch to publish it anyway.")
        if args.phia_from == "gradient":
            # The estimator itself: the solve's phi_A term from the integrated field.  It is
            # already at the screen scale, so it replaces the low-passed term directly;
            # the band-difference term is unchanged and the two still add to the screen.
            term_a = coef_a * phi_grad
            disp_f = np.where(np.isfinite(term_a) & np.isfinite(term_d), term_a + term_d,
                              np.nan)
            print(f"[gate] --phia-from gradient: the screen's phi_A term is the integrated "
                  f"field (std {np.nanstd(term_a):.1f} rad), not the unwrapper's")
    elif args.phia_from == "gradient":
        raise SystemExit("--phia-from gradient needs --iono-filter-km > 0: the integrated "
                         "field is defined at the screen scale")

    # --- the datum.  Both unwraps carry an arbitrary integer-cycle constant, so the
    # absolute level is NOT observable from the SAR data alone.  Removing the mean says so
    # explicitly and matches the per-column datum the other two routes carry.
    ok = np.isfinite(disp_f)
    if args.level == "zero-mean" and ok.any():
        m = float(np.mean(disp_f[ok]))
        disp_f = disp_f - m
        print(f"[level] removed the mean {m:+.2f} rad -- the product is differential in "
              f"the mean, the same caveat the offsets and MAI screens carry")

    # --- continue the screen over the gaps the low-pass could not reach.  `domain` is the
    # union of that reach and band A's geocoded footprint, i.e. exactly the ground the
    # product covers, so no step survives anywhere the interferogram has data and nothing is
    # invented beyond it.  See fill_gaps for what leaving NaN here costs.
    reached = np.isfinite(disp_f)
    if args.iono_fill != "none":
        # CONTINUE ALONG THE TREND, NOT FLAT.  fill_gaps is a pinned relaxation, so past the
        # data edge it goes flat and the ionosphere does not.  Relax the RESIDUAL from a
        # low-order surface fitted on the low-pass reach and put the surface back.  A plane
        # is the default: a quadratic follows the measured region more closely but
        # overshoots past the edge, which is where the continuation is used.
        trend = fit_trend(disp_f, reached, args.iono_fill_trend)
        disp_f = fill_gaps(disp_f - trend, reached, reached | np.isfinite(pa)) + trend

    covered = np.isfinite(disp_f)
    print(f"\n[screen] measured on {100 * mask.mean():.1f} % of the frame; the low-pass "
          f"reached {100 * reached.mean():.1f} %; the screen is defined on "
          f"{100 * covered.mean():.1f} % after the continuation, NaN over the remaining "
          f"{100 * (~covered).mean():.1f} % -- where the product has no data either")
    print("[screen] iono_screen_valid_<tag>.tif keeps the three states apart: 1 measured, "
          "0 continued, NaN not defined.  The screen is APPLIED over all of `covered`, "
          "because withholding it would put a step at the edge of the measured region")

    w = {}
    w["screen"] = U.save_gtiff(os.path.join(out_dir, f"iono_screen_{tag}.tif"),
                               disp_f.astype(np.float32), gt, epsg, nodata=np.nan)
    w["valid"] = U.save_gtiff(os.path.join(out_dir, f"iono_screen_valid_{tag}.tif"),
                              np.where(covered, mask.astype(np.float32), np.nan),
                              gt, epsg, nodata=np.nan)
    w["coh_diff"] = U.save_gtiff(os.path.join(out_dir, f"coh_diff_{tag}.tif"),
                                 np.where(valid, coh_diff, np.nan).astype(np.float32),
                                 gt, epsg, nodata=np.nan)
    w["phi_diff"] = U.save_gtiff(os.path.join(out_dir, f"phi_diff_ms_{tag}.tif"),
                                 np.where(mask, phi_diff, np.nan).astype(np.float32),
                                 gt, epsg, nodata=np.nan)
    # The attribution set.  Not rendered with the products (PL.render_tif is called only on
    # `w`), because they are diagnostics: the two terms sum to the screen, and conn_A is
    # phi_A's connected-component map, which the shared scratch_unwrap/conn_tmp.tif does NOT
    # preserve -- the band-difference unwrap overwrites it.
    diag = {}
    diag["term_a"] = U.save_gtiff(os.path.join(out_dir, f"screen_term_phiA_{tag}.tif"),
                                  term_a.astype(np.float32), gt, epsg, nodata=np.nan)
    diag["term_d"] = U.save_gtiff(os.path.join(out_dir, f"screen_term_phidiff_{tag}.tif"),
                                  term_d.astype(np.float32), gt, epsg, nodata=np.nan)
    if phi_grad is not None:
        # gate 8's reference: coef_a x the integrated wrapped-gradient field, on the same
        # footing as screen_term_phiA so the two can be differenced directly
        diag["term_a_grad"] = U.save_gtiff(
            os.path.join(out_dir, f"screen_term_phiA_grad_{tag}.tif"),
            (coef_a * phi_grad).astype(np.float32), gt, epsg, nodata=np.nan)
    if conn_a is not None:
        diag["conn_a"] = U.save_gtiff(os.path.join(out_dir, f"unw_conn_phiA_{tag}.tif"),
                                      np.where(np.isfinite(phi_a_u),
                                               np.asarray(conn_a, np.float32), np.nan),
                                      gt, epsg, nodata=np.nan)
    return disp_f, covered, mask, w, diag


# =====================================================================
# back onto the offset lattice, so the interferogram stage's --iono-screen can apply it
# =====================================================================
def lattice_geoloc(a, ref_a, demI, win_a, lat_shape, epsg):
    """Map coordinates (X, Y) of every correlation-window centre."""
    from .. import coregister as O

    d = np.load(a.offsets)
    g = argparse.Namespace(
        scratch=a.scratch, cache_dir=a.cache_dir, geoloc_cache=True,
        skip=tuple(int(v) for v in d["skip"]),
        winsize=tuple(int(v) for v in d["winsize"]),
        search=tuple(int(v) for v in d["search"]))
    return O.lattice_map_coords(g, ref_a, demI, win_a, lat_shape, epsg,
                                np.ones(lat_shape, bool))


def screen_to_lattice(screen, gt, npz_a, X, Y, out_npz, datum="as-is", measured=None):
    """Sample a map-grid screen at every correlation-window centre and save it as an npz."""
    d = np.load(npz_a)
    col = (X - gt[0]) / gt[1]
    row = (Y - gt[3]) / gt[5]
    ny, nx = screen.shape
    ok = (np.isfinite(col) & np.isfinite(row)
          & (col >= 0) & (col < nx - 1) & (row >= 0) & (row < ny - 1))
    out = np.full(X.shape, np.nan, np.float32)
    from scipy.ndimage import map_coordinates
    out[ok] = map_coordinates(np.nan_to_num(screen), [row[ok], col[ok]], order=1,
                              mode="nearest")
    src_ok = map_coordinates(np.isfinite(screen).astype(np.float32),
                             [np.where(ok, row, 0), np.where(ok, col, 0)],
                             order=1, mode="nearest") > 0.99
    valid = ok & src_ok
    out = np.where(valid, out, np.nan).astype(np.float32)
    # `valid` means MEASURED, as in every other route's npz: the map-grid solve mask pulled
    # onto the lattice, not "the screen is defined here".  The screen is defined everywhere
    # by construction, so the two are different statements.
    if measured is not None:
        meas = np.zeros(X.shape, bool)
        meas[ok] = map_coordinates(np.asarray(measured, np.float32), [row[ok], col[ok]],
                                   order=1, mode="constant", cval=0.0) > 0.5
        valid = valid & meas
    if datum == "per-column":
        # THE SAME DATUM THE AZIMUTH ROUTE CARRIES.  The offsets route integrates along
        # track and subtracts the per-column azimuth mean, so a purely range-varying field
        # cannot survive in it.  This route keeps it, which is an advantage only if what it
        # keeps is ionosphere.  It need not be: the band difference enters the solve with
        # gain 1/|det|, so a systematic phi_diff error of a fraction of a radian across the
        # swath reproduces a frame-scale range bend, far below the per-pixel noise and
        # invisible in phi_diff itself -- a non-dispersive leak that tracks the geometry
        # rather than the ionosphere.
        #
        # Removing the column mean THROWS AWAY any real range-varying ionosphere along with
        # the leak.  That is the trade, and whether it is the right one has to be measured
        # per frame rather than assumed.
        with np.errstate(invalid="ignore"):
            colmean = np.nanmean(np.where(valid, out, np.nan), axis=0, keepdims=True)
        colmean = np.where(np.isfinite(colmean), colmean, 0.0)
        out = np.where(valid, out - colmean, np.nan).astype(np.float32)
        print(f"[screen] per-column datum removed: column means spanned "
              f"{np.nanmin(colmean):+.2f} .. {np.nanmax(colmean):+.2f} rad")
    # A NaN here would go into the npz as np.nan_to_num -> 0.0, and the offsets screen's
    # to_lattice hands that straight to the interferogram stage, which applies
    # exp(-1j*screen) without looking at `valid`.  An uncovered cell would then not be an
    # absent correction but a ZERO one, and the product would step by the full local screen
    # value at every rim.  fill_gaps closes the
    # gaps on the map grid; this closes the ones the lattice sampling can still open at the
    # swath edge, and says so rather than zeroing them.  `valid` is unchanged: it still marks
    # where the screen was defined, not where it was filled in here.
    gap = ~np.isfinite(out)
    if gap.any():
        from scipy.ndimage import distance_transform_edt
        if (~gap).any():
            idx = distance_transform_edt(gap, return_distances=False, return_indices=True)
            out = out[tuple(idx)].astype(np.float32)
        else:
            out = np.zeros_like(out)
        print(f"[screen] {int(gap.sum())} lattice cells ({100 * gap.mean():.2f} %) fell "
              f"outside the map-grid screen; filled from the nearest one rather than "
              f"written as a hard 0.0, which would be APPLIED as a zero correction")
    np.savez_compressed(out_npz, screen=out.astype(np.float32), valid=valid,
                        window=d["window"], skip=d["skip"], winsize=d["winsize"],
                        search=d["search"])
    print(f"[screen] on the {X.shape[0]} x {X.shape[1]} offset lattice: measured on "
          f"{100 * valid.mean():.1f} % of it -> {out_npz}")
    return out_npz


# =====================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref", required=True, help="reference RSLC (frequency A)")
    p.add_argument("--sec", required=True, help="secondary RSLC (frequency A)")
    p.add_argument("--ref-b", default=None, help="frequency B reference crop "
                                                 "(default: found in --cache-dir)")
    p.add_argument("--sec-b", default=None)
    p.add_argument("--pol", default="HH")
    p.add_argument("--dem", required=True)
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--tag", required=True, help="names every product of this pair")
    p.add_argument("--scratch", default="offsets_scratch",
                   help="the frequency A scratch this recycles (ref.c8, sec_coreg.c8, "
                        "geo2rdr/*.off, rubbersheet_az.npy)")
    p.add_argument("--scratch-b", default=None,
                   help="frequency B's own scratch (default <scratch>/freqB)")
    p.add_argument("--offsets", default=None,
                   help="the offsets npz, for the lattice geometry "
                        "(default outputs_offsets/offsets_<tag>.npz)")
    p.add_argument("--geoloc", default=None,
                   help="cached lattice geolocation npz (default: the one in --cache-dir "
                        "for this lattice, or computed from --scratch's rdr2geo if there "
                        "is none)")
    p.add_argument("--out-dir", default="outputs_unified/split")
    p.add_argument("--gpu", default="0", metavar="LIST",
                   help="CUDA device(s) to use: a list such as 0,1 or 'all' "
                        "(default 0)")
    p.add_argument("--looks", nargs=2, type=int, default=(24, 16), metavar=("AZ", "RG"),
                   help="frequency A's multilook; B's range looks are derived so both "
                        "cover the same ground (default 24 16)")
    p.add_argument("--geogrid", nargs=7, required=True,
                   metavar=("X0", "Y0", "DX", "DY", "NX", "NY", "EPSG"),
                   help="the shared lattice everything else sits on.  Required: the split "
                        "screen is only comparable to the other routes' if it lands on the "
                        "same cells, and the route driver passes the lattice it read "
                        "from the coregistration's own product")
    p.add_argument("--coh-thresh", type=float, default=0.2,
                   help="per-band and band-difference coherence gate (default 0.2)")
    p.add_argument("--diff-win", type=int, default=21,
                   help="complex-average window for the band difference, px (default 21)")
    p.add_argument("--iono-filter-km", type=float, default=12.0,
                   help="Gaussian sigma for the coherence-weighted low-pass.  12 km, not "
                        "the 2 km the offsets screen uses: this is an 11.6x-amplified "
                        "solve and needs far more smoothing (default 12)")
    p.add_argument("--unwrap-components", default="rereference",
                   choices=["rereference", "largest", "all"],
                   help="what to do with the unwrapper's connected components, each of "
                        "which carries its OWN arbitrary integer cycle datum.  "
                        "'rereference' rounds each component's median offset from the "
                        "largest to a whole cycle and removes it, keeping the coverage; "
                        "'largest' keeps only the largest and drops the rest (what "
                        "restricting to the largest component does, at the cost of "
                        "coverage); 'all' accepts every component's own datum, which can "
                        "leave several cycles of offset between them (default rereference)")
    p.add_argument("--unwrap-support", default="mask", choices=["mask", "own"],
                   help="'own' unwraps frequency A on `valid & ca >= coh-thresh` rather "
                        "than the full mask, on the theory that band B's gate and the ice "
                        "gate fragment a domain they say nothing about.  It measures worse: "
                        "an unwrapper that applies its own coherence threshold internally "
                        "fragments the narrower support more, not less.  The default is "
                        "(default mask)")
    p.add_argument("--allow-diff-cycles", action="store_true",
                   help="proceed even if the BAND DIFFERENCE needed whole-cycle "
                        "re-referencing.  One cycle there is 2*pi/|det| of screen against "
                        "0.51076*2*pi for frequency A, so the run stops by default and this "
                        "should stay unnecessary")
    p.add_argument("--unwrap-min-component", type=float, default=0.005,
                   help="smallest component, as a fraction of the solved pixels, whose "
                        "offset from the largest is trusted.  Below it the reference is "
                        "extrapolated rather than measured and the estimate is noise "
                        "(default 0.005)")
    p.add_argument("--unwrap-cycle-tol", type=float, default=0.35,
                   help="how close to a whole cycle a component's offset must be before it "
                        "is believed and removed.  Further than this the offset is not "
                        "convincingly an unwrapping datum, so the component is dropped "
                        "rather than guessed at (default 0.35)")
    p.add_argument("--iono-fill", default="smooth", choices=["smooth", "none"],
                   help="'smooth' continues the screen across the gaps the low-pass could "
                        "not reach, pinned to the measured cells, so the applied correction "
                        "has no step.  'none' leaves NaN, which screen_to_lattice then had "
                        "to write as a hard 0.0 -- worth 1.3 to 2.7 fringes at every hole "
                        "rim (default smooth)")
    p.add_argument("--iono-fill-trend", type=int, default=1, choices=[0, 1, 2],
                   help="order of the surface the continuation follows past the data edge: "
                        "0 flat, which runs low past the data edge; 1 plane; 2 quadratic, "
                        "which follows the measured region more closely but overshoots "
                        "beyond it (default 1)")
    p.add_argument("--iono-fill-floor", type=float, default=0.05,
                   help="how much local coverage the coherence-weighted low-pass needs "
                        "before it is trusted, as a fraction of the frame's best.  0.05 "
                        "reaches ~22 km past the data edge; raise it to lean on the "
                        "continuation instead of on the Gaussian's tail (default 0.05)")
    p.add_argument("--outlier-mad", type=float, default=3.0,
                   help="reject pixels this many MADs from a local median BEFORE smoothing "
                        "(0 disables).  This is what removes the bubbles (default 3)")
    p.add_argument("--unwrap-method", default="snaphu",
                   choices=["snaphu", "phass", "icu"],
                   help="snaphu (default) is statistical-cost network-flow and weighs "
                        "coherence instead of thresholding it; phass and icu are "
                        "isce3-native and faster, but are more likely to fail the "
                        "consistency gate on a frame with patchy coherence")
    p.add_argument("--phia-from", default="unwrap", choices=["unwrap", "gradient"],
                   help="what the solve's phi_A term is built from.  'unwrap': the "
                        "unwrapper's field, low-passed -- and gate 8 must pass.  'gradient': "
                        "the wrapped interferogram's own low-passed gradient, integrated "
                        "(integrate_wrapped_gradient) -- no unwrapper in the estimate at "
                        "all.  A FALLBACK, not the product: the integration runs low "
                        "towards the frame edges, and its coverage still follows the "
                        "unwrapper's declines through the mask (default unwrap)")
    p.add_argument("--gate-corr", type=float, default=0.9, metavar="R",
                   help="gate 8: the unwrapped phi_A must correlate with the integrated "
                        "gradient field at least this well (default 0.9)")
    p.add_argument("--gate-bracket", type=float, nargs=2, default=(0.85, 1.15),
                   metavar=("LO", "HI"),
                   help="gate 8: the amplitude bracket of unwrapped against integrated must "
                        "sit inside [LO, HI] (default 0.85 1.15).  Loosening either is a "
                        "decision about how much unwrapper bias to publish; the numbers are "
                        "printed either way so the choice is on the record")
    p.add_argument("--allow-unwrap-mismatch", action="store_true",
                   help="proceed when gate 8 fails.  The screen will carry whatever the "
                        "unwrapper got wrong, smoothed into humps; for diagnostics only")
    p.add_argument("--snaphu-cost", default="smooth", choices=["smooth", "defo"],
                   help="snaphu statistical cost mode.  'smooth' is right here: the field "
                        "is propagation phase, not a deformation discontinuity")
    p.add_argument("--snaphu-init", default="mcf", choices=["mcf", "mst"])
    p.add_argument("--snaphu-ntiles", type=int, nargs=2, default=(1, 1),
                   metavar=("NROW", "NCOL"),
                   help="snaphu tiling.  KEEP 1 1: a frame-sized grid unwraps as one tile "
                        "in minutes, and tiling costs coverage and continuity -- it leaves "
                        "large parts of the frame unlabelled and collects discontinuities "
                        "over the tile overlaps (default 1 1)")
    p.add_argument("--snaphu-nproc", type=int, default=16)
    p.add_argument("--unwrap-diff", action=argparse.BooleanOptionalAction, default=True,
                   help="unwrap the band difference before the solve.  On: without it the "
                        "estimate is bounded by +-pi/|det| = 36.5 rad (default on)")
    p.add_argument("--level", default="zero-mean", choices=["zero-mean", "none"])
    p.add_argument("--diff-datum", default="rotate", choices=["rotate", "as-is"],
                   help="'rotate' turns the band difference to zero circular mean before "
                        "unwrapping, so an arbitrary constant near +-pi cannot park the "
                        "field on the branch cut.  Free -- the dispersive datum is "
                        "unobservable and --level removes the mean at the end.  'as-is' "
                        "leaves the datum where it is (default rotate)")
    p.add_argument("--glacier-mask", default=None, metavar="TIF",
                   help="raster on the solve grid, >0.5 where ice.  Those cells leave the "
                        "MASK so the low-pass carries the land solution across them, the "
                        "same treatment offsets/MAI give glaciers")
    p.add_argument("--coh-thresh-b", type=float, default=None, metavar="C",
                   help="a separate coherence gate for frequency B (default: the same as "
                        "--coh-thresh).  They are NOT comparable numbers: A is 24 x 16 = 384 "
                        "samples per multilook cell and B is 24 x 2 = 48, because looks_for "
                        "matches the ground cell and B's range spacing is 8x A's.  The "
                        "zero-signal coherence floor is 0.886/sqrt(N), so 0.045 for A and "
                        "0.128 for B at these look counts, so a single threshold is not "
                        "the same statement about both bands.  Raising B's gate costs "
                        "coverage quickly, so the flag exists to re-ask per frame, not to be "
                        "turned on by default.")
    p.add_argument("--diff-weight", default="uniform", choices=["uniform", "coherence"],
                   help="how the diff-win x diff-win band-difference average is weighted.  "
                        "`uniform` is the unit phasor, so a fjord pixel at coherence 0.02 "
                        "pushes as hard as land at 0.95 "
                        "and decorrelated ground is mixed (diff_win-1)/2 px into the land "
                        "around it.  `coherence` is splitA's construction "
                        "weight by min(coh_A, coh_B), "
                        "normalise by the smoothed weight, and zero the glacier mask BEFORE "
                        "the box rather than after.  It is worth a fraction of a radian of "
                        "screen at the ice edge: take it because it removes an argument, "
                        "not because it will move the score.")
    p.add_argument("--az-carrier-mode", default="scalar", choices=["scalar", "lut"],
                   help="how the resampler's azimuth carrier is taken out of each band's "
                        "interferogram.  `scalar` is one number at the crop centre; `lut` "
                        "evaluates the annotated Doppler per pixel.  KEEP `scalar`: `lut` "
                        "measures worse.  The residual (c(x)-c_hat)*aoff(x) really is "
                        "there, several radians peak to peak in both bands, but removing it "
                        "moves the screen AWAY from the independent estimates.  The reading "
                        "that fits is that the annotated dopplerCentroid describes the "
                        "antenna while the focused samples carry a carrier that is constant "
                        "to within what this measures, so the LUT's spatial variation is "
                        "not in the data.")
    p.add_argument("--interp-bias-a", default=None, metavar="NPZ",
                   help="the resampler bias curve for band A, handed to "
                        "the interferogram stage's --interp-bias when band A's interferogram is built")
    p.add_argument("--interp-bias-b", default=None, metavar="NPZ",
                   help="the same for band B, whose bias reaches the split screen "
                        "amplified by 1/|det|")


    p.add_argument("--datum", default="as-is", choices=["as-is", "per-column"],
                   help="'per-column' removes the per-range-column azimuth mean on the "
                        "offset lattice, the same datum offsets/MAI carry.  Use it when the "
                        "frame-scale range bend is a non-dispersive leak rather than "
                        "ionosphere -- see screen_to_lattice (default as-is)")
    p.add_argument("--test", action="store_true",
                   help="geometry, lever and coverage only -- no coregistration, no solve")
    return p.parse_args(argv)


def main(argv=None):
    from .. import coregister as O

    a = parse_args(argv)
    a.offsets = a.offsets or f"outputs_offsets/offsets_{a.tag}.npz"
    a.scratch_b = a.scratch_b or os.path.join(a.scratch, "freqB")
    os.makedirs(a.out_dir, exist_ok=True)

    for side in ("ref_b", "sec_b"):
        if getattr(a, side) is None:
            raise SystemExit(f"--{side.replace('_', '-')} is required: the frequency B "
                             f"granule matching the frequency A one given as --{side[:3]}")
    print(f"frequency A  {a.ref}\n             {a.sec}")
    print(f"frequency B  {a.ref_b}\n             {a.sec_b}\n")
    for pa_, pb_ in ((a.ref, a.ref_b), (a.sec, a.sec_b)):
        check_pair(pa_, pb_, pol=a.pol)
        print()

    d = np.load(a.offsets)
    win_a = tuple(int(v) for v in d["window"])
    so_a = tuple(int(v) for v in d["sec_origin"])
    ref_a = O.load_slc(a.ref, "A", a.pol)
    f0 = float(299_792_458.0 / ref_a["rg"].wavelength)
    ref_b_meta = O.load_slc(a.ref_b, "B", a.pol)
    f1 = float(299_792_458.0 / ref_b_meta["rg"].wavelength)
    looks_b = looks_for(ref_a["rg"], ref_b_meta["rg"], tuple(a.looks))

    if a.test:
        det = (f0 - f1) / f0 - (f1 - f0) / f1
        print(f"\n[test] lever {abs(f1 - f0) / 1e6:.1f} MHz, determinant {det:+.4f}, "
              f"noise amplification {1 / abs(det):.1f}x")
        print(f"[test] A looks {a.looks}, B looks {looks_b}")
        print("[test] stopping before the coregistration (--test)")
        return 0

    # --- frequency B: coregister, then inherit A's rubbersheet
    b = coregister_b(a, a.scratch_b)
    b["ref_a"], b["win_a"] = ref_a, win_a
    rb = os.path.join(a.scratch, "geo2rdr", "rubbersheet_az.npy")
    stamp = os.path.join(a.scratch_b, "geo2rdr", "rubbersheet_transferred")
    if os.path.exists(rb) and not os.path.exists(stamp):
        transfer_rubbersheet(b, a.offsets, np.load(rb))
        open(stamp, "w").close()
        # the resample has to happen again now that azimuth.off has changed
        O.coregister_isce3(b["args"], b["pair0"], b["ref"], b["sec"], b["win"],
                           b["dem_raster"], resample_only=True)
    elif os.path.exists(stamp):
        print("[B] azimuth.off already carries A's rubbersheet")
    else:
        print(f"[B] WARNING: {rb} is missing, so B is coregistered by GEOMETRY ONLY while "
              f"A carries a rubbersheet.  The band difference will contain that "
              f"registration difference as a fictitious dispersive signal.")

    # --- one geocoded interferogram per band, on the shared lattice
    ta = band_interferogram(a, "A", a.scratch, a.ref, a.sec, win_a, so_a,
                            tuple(a.looks), a.out_dir, a.tag, a.geogrid)
    tb = band_interferogram(a, "B", a.scratch_b, a.ref_b, a.sec_b, b["win"],
                            b["pair0"]["sec_origin"], looks_b, a.out_dir, a.tag, a.geogrid)

    pa = U.read_gtiff(os.path.join(a.out_dir, f"ifg_phase_geo_{ta}.tif"))
    ca = U.read_gtiff(os.path.join(a.out_dir, f"ifg_coh_geo_{ta}.tif"))
    pb = U.read_gtiff(os.path.join(a.out_dir, f"ifg_phase_geo_{tb}.tif"))
    cb = U.read_gtiff(os.path.join(a.out_dir, f"ifg_coh_geo_{tb}.tif"))
    gt = (float(a.geogrid[0]), float(a.geogrid[2]), 0.0,
          float(a.geogrid[1]), 0.0, float(a.geogrid[3]))
    epsg = int(a.geogrid[6])

    screen, covered, mask, written, diag = solve_split(a, pa, pb, ca, cb, f0, f1, gt,
                                                       epsg, a.out_dir, a.tag)

    lat = tuple(d["azimuth"].shape)
    geoloc = a.geoloc or os.path.join(a.cache_dir,
                                      f"geoloc_offsets_{lat[0]}x{lat[1]}_{epsg}.npz")
    if os.path.exists(geoloc):
        g = np.load(geoloc)
        X, Y = g["X"], g["Y"]
        print(f"[geoloc] {int(np.isfinite(X).sum())} window centres from {geoloc}")
    else:
        X, Y = lattice_geoloc(a, ref_a, b["demI"], win_a, lat, epsg)
    screen_to_lattice(screen, gt, a.offsets, X, Y,
                      os.path.join(a.out_dir, f"iono_screen_{a.tag}.npz"),
                      datum=a.datum, measured=mask)

    for p in written.values():
        PL.render_tif(p)
    print(f"\nproducts in {a.out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
