"""Coherence streak index: residual log coherence, the loss field, and its directionality
R_FFT and theta measured in radar coordinates."""
from __future__ import annotations

import os
import warnings

import numpy as np

from .gslc_stack import nan_gaussian

K_TEC = 40.31          # m^3/s^2


C_LIGHT = 299792458.0
# Radar constants.  The literals are the 40 MHz mode; set_radar_constants() replaces them
# from the stack's own accumulation or granule at run time, which is how the other modes are
# handled.
F_A = 1239.0e6
F_B = 1293.5e6
RHO_AZ = 5.347         # m, azimuth resolution.  IDENTICAL for A and B: the product carries
                       # azimuthBandwidth 1262.98 Hz on both, which is what makes control 2 work
RHO_RG = {"A": 3.75, "B": 29.98}    # m, SLANT range resolution c/2B from rangeBandwidth
                                    # 40.00 and 5.00 MHz.  The 8x split is the whole discriminator
AZBW_REF = 1262.98                  # Hz, the azimuthBandwidth RHO_AZ was quoted at


AB_PRED = {"az": 1.090, "rg": 8.7}  # A/B amplitude each mechanism predicts (docstring, item 2):


def set_radar_constants(fc_A, fc_B, bw_A, bw_B, azbw=None, verbose=True):
    """Point every frequency-dependent number at the stack's own mode (4005 or 2005)."""
    global F_A, F_B, RHO_AZ, RHO_RG, AB_PRED
    F_A, F_B = float(fc_A), float(fc_B)
    RHO_RG = {"A": C_LIGHT / (2 * bw_A), "B": C_LIGHT / (2 * bw_B)}
    if azbw:
        RHO_AZ = 5.347 * AZBW_REF / float(azbw)
    q = (F_B / F_A) ** 2
    AB_PRED = {"az": q, "rg": q * bw_A / bw_B}
    if verbose:
        print(f"  radar: f_A {F_A / 1e6:.1f} MHz (rho_rg {RHO_RG['A']:.2f} m), "
              f"f_B {F_B / 1e6:.1f} MHz (rho_rg {RHO_RG['B']:.2f} m), rho_az {RHO_AZ:.3f} m; "
              f"A/B predictions: azimuth {AB_PRED['az']:.3f}, range {AB_PRED['rg']:.2f}")


# --------------------------------------------------------------------------- geometry
def bearing(vx, vy):
    """Bearing of a map vector in degrees, 0 = +y (GRID north), 90 = +x (grid east)."""
    return np.degrees(np.arctan2(vx, vy))


def grid_centres(lay, rebin, shape):
    """Map coordinates of the working cell centres."""
    cell_x = abs(lay["dx"] * lay["fx"]) * rebin
    cell_y = abs(lay["dy"] * lay["fy"]) * rebin
    x_ul = lay["x0"] - lay["dx"] / 2.0
    y_ul = lay["y0"] - lay["dy"] / 2.0
    x = x_ul + (np.arange(shape[1]) + 0.5) * cell_x
    y = y_ul - (np.arange(shape[0]) + 0.5) * cell_y
    return x, y, cell_x, cell_y


def native_grid(lay, rebin, shape):
    """The working grid as delivered: the accumulation's own projection (EPSG:3413 here)."""
    x, y, cx, _ = grid_centres(lay, rebin, shape)
    return {"epsg": int(lay["epsg"]), "x": x, "y": y, "cell": float(cx), "shape": shape,
            "name": f"EPSG:{lay['epsg']}"}


def utm_grid_for(lay, rebin, shape):
    """A UTM grid covering the same ground at the same cell size."""
    from pyproj import CRS, Transformer
    x, y, cx, _ = grid_centres(lay, rebin, shape)
    src = CRS.from_epsg(int(lay["epsg"]))
    wgs = CRS.from_epsg(4326)
    to_ll = Transformer.from_crs(src, wgs, always_xy=True)
    lonc, latc = to_ll.transform(float(np.mean(x)), float(np.mean(y)))
    zone = int(np.floor((lonc + 180.0) / 6.0) + 1)
    epsg = (32600 if latc >= 0 else 32700) + zone
    dst = CRS.from_epsg(epsg)
    to_utm = Transformer.from_crs(src, dst, always_xy=True)
    gx, gy = np.meshgrid(x, y)
    ux, uy = to_utm.transform(gx, gy)
    x0, x1 = float(np.nanmin(ux)), float(np.nanmax(ux))
    y0, y1 = float(np.nanmin(uy)), float(np.nanmax(uy))
    nx = int(np.ceil((x1 - x0) / cx)) + 1
    ny = int(np.ceil((y1 - y0) / cx)) + 1
    return {"epsg": epsg, "x": x0 + np.arange(nx) * cx, "y": y1 - np.arange(ny) * cx,
            "cell": float(cx), "shape": (ny, nx),
            "name": f"EPSG:{epsg} (UTM {zone}{'N' if latc >= 0 else 'S'})"}


def resample_grid(field, src_grid, dst_grid, order=1):
    """Bilinear (order=1) or nearest (order=0) resample between two map grids."""
    from pyproj import CRS, Transformer
    from scipy.ndimage import map_coordinates
    tf = Transformer.from_crs(CRS.from_epsg(dst_grid["epsg"]), CRS.from_epsg(src_grid["epsg"]),
                              always_xy=True)
    gx, gy = np.meshgrid(dst_grid["x"], dst_grid["y"])
    sx, sy = tf.transform(gx, gy)
    sxa, sya = src_grid["x"], src_grid["y"]
    col = (sx - sxa[0]) / (sxa[1] - sxa[0])
    row = (sy - sya[0]) / (sya[1] - sya[0])
    a = np.asarray(field)
    if a.dtype == bool:
        out = map_coordinates(a.astype(np.float32), [row, col], order=0, mode="constant",
                              cval=0.0)
        return out > 0.5
    fill = np.where(np.isfinite(a), a, 0.0)
    out = map_coordinates(fill, [row, col], order=order, mode="constant", cval=np.nan)
    good = map_coordinates(np.isfinite(a).astype(np.float32), [row, col], order=order,
                           mode="constant", cval=0.0)
    return np.where(good > 0.99, out, np.nan)


def geom_from_cube(gslc_path, grid, height_m=500.0):
    """Azimuth and range axes on ANY map grid, from the cube's scalar fields."""
    import h5py
    from pyproj import CRS, Transformer
    from scipy.interpolate import RegularGridInterpolator

    with h5py.File(gslc_path, "r") as h:
        rg = h["science/LSAR/GSLC/metadata/radarGrid"]
        hgt = rg["heightAboveEllipsoid"][()]
        k = int(np.argmin(np.abs(hgt - height_m)))
        cube_epsg = int(rg["projection"][()])
        xc, yc = rg["xCoordinates"][()], rg["yCoordinates"][()]
        flds = {"az": np.asarray(rg["zeroDopplerAzimuthTime"][k], dtype=np.float64),
                "rg": np.asarray(rg["slantRange"][k], dtype=np.float64),
                "inc": np.asarray(rg["incidenceAngle"][k], dtype=np.float64)}

    tf = Transformer.from_crs(CRS.from_epsg(grid["epsg"]), CRS.from_epsg(cube_epsg),
                              always_xy=True)
    gx, gy = np.meshgrid(grid["x"], grid["y"])
    cx_, cy_ = tf.transform(gx, gy)
    order = np.argsort(yc)
    pts = np.stack([cy_.ravel(), cx_.ravel()], axis=-1)
    on = {}
    for name, v in flds.items():
        f = RegularGridInterpolator((yc[order], xc), v[order], bounds_error=False,
                                    fill_value=None)
        on[name] = f(pts).reshape(grid["shape"])

    cell = grid["cell"]
    out = {"inc": on["inc"], "slant_m": on["rg"], "height_used": float(hgt[k])}
    for tag in ("az", "rg"):
        grow, gcol = np.gradient(on[tag])
        vx, vy = gcol / cell, -grow / cell      # +x is +col; +y is -row
        n = np.hypot(vx, vy)
        n = np.where(n > 0, n, 1.0)
        out[f"{tag}_x"], out[f"{tag}_y"] = vx / n, vy / n
    return out


def check_axes(gslc_path, grid, geom, tol_deg=5.0):
    """Independent audit of the azimuth/range axes against the frame's own boundingPolygon."""
    import re

    import h5py
    from pyproj import CRS, Transformer

    with h5py.File(gslc_path, "r") as h:
        wkt = h["science/LSAR/identification/boundingPolygon"][()]
    wkt = wkt.decode() if isinstance(wkt, bytes) else str(wkt)
    vals = [tuple(float(t) for t in v.split()[:2])
            for v in re.search(r"\(\((.*)\)\)", wkt).group(1).split(",")]
    lon = np.array([v[0] for v in vals])
    lat = np.array([v[1] for v in vals])
    tf = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(grid["epsg"]), always_xy=True)
    px, py = tf.transform(lon, lat)

    dx, dy = np.diff(px), np.diff(py)
    w = np.hypot(dx, dy)
    keep = w > 0.05 * w.max()          # drop the densifying micro-segments along each side
    th = np.radians(bearing(dx[keep], dy[keep]))
    z = np.sum(w[keep] * np.exp(4j * th))
    a0 = np.degrees(np.angle(z)) / 4.0                     # in (-45, 45]
    fam = np.array([a0, a0 + 90.0])

    az_b = float(np.nanmedian(bearing(geom["az_x"], geom["az_y"])))
    rg_b = float(np.nanmedian(bearing(geom["rg_x"], geom["rg_y"])))
    i = int(np.argmin(np.abs(angdiff180(fam, az_b))))
    d_az = float(angdiff180(fam[i], az_b))
    d_rg = float(angdiff180(fam[1 - i], rg_b))
    print(f"  CONTROL boundingPolygon edges {fam[i]:+.2f} / {fam[1 - i]:+.2f} deg vs derived "
          f"axes {az_b:+.2f} / {rg_b:+.2f}   (off by {d_az:+.2f} / {d_rg:+.2f} deg)")
    if max(abs(d_az), abs(d_rg)) > tol_deg:
        raise SystemExit(
            f"AXIS CHECK FAILED: the frame polygon says {fam[i]:+.2f}/{fam[1 - i]:+.2f} deg but "
            f"radarGrid gives {az_b:+.2f}/{rg_b:+.2f} deg (tolerance {tol_deg} deg).\n"
            f"  Every directional result would be wrong.  Do not read the numbers.")
    return {"poly_az": float(fam[i]), "poly_rg": float(fam[1 - i]),
            "d_az": d_az, "d_rg": d_rg}


# --------------------------------------------------------------------------- visualisation
MIN_INTERIOR_FILL = 0.15   # floor on the rotated interior's valid fraction; see _interior()


ROT_SIGN = +1.0     # fixed empirically by selftest check 5, not guessed:


def radar_frame(img, ok, az_bearing, sign=None):
    """Rotate the frame so the AZIMUTH direction is VERTICAL in the output."""
    from scipy.ndimage import rotate
    ang = (ROT_SIGN if sign is None else sign) * az_bearing
    v = rotate(np.where(ok, img, 0.0), ang, reshape=True, order=1, cval=0.0, mode="constant")
    w = rotate(ok.astype(np.float64), ang, reshape=True, order=1, cval=0.0, mode="constant")
    good = w > 0.5
    out = np.full(v.shape, np.nan)
    out[good] = v[good] / w[good]
    return out, good


def lowpass(r, ok, cell, km):
    """Isotropic display low-pass, applied to the residual before it is drawn."""
    if not km or km <= 0:
        return r
    return np.where(ok, nan_gaussian(np.where(ok, r, np.nan), km * 1e3 / cell), np.nan)


def _interior(img):
    """Bounding-box crop of the rotated frame, with its validity weight."""
    ok = np.isfinite(img)
    if not ok.any():
        return None, None
    ys, xs = np.nonzero(ok)
    r0, r1, c0, c1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    sub = img[r0:r1, c0:c1]
    w = np.isfinite(sub).astype(np.float64)
    fill = float(w.mean())
    if sub.size < 4096 or fill < MIN_INTERIOR_FILL:
        why = "too small" if sub.size < 4096 else f"fill {fill:.3f} < {MIN_INTERIOR_FILL}"
        print(f"    texture skipped: rotated interior {sub.shape[0]}x{sub.shape[1]}, {why}"
              f" -- R_fft and theta_fft will be NaN for this pair")
        return None, None
    return np.where(np.isfinite(sub), sub, 0.0), w


def directionality(r, ok, cell, az_b, smooth_km=2.0):
    """Streak ORIENTATION and STRENGTH, measured in radar coordinates."""
    rl = lowpass(r, ok, cell, smooth_km)
    rot, _ = radar_frame(rl, ok, az_b)          # azimuth vertical, range horizontal
    a, w = _interior(rot)
    out = {"R_fft": np.nan, "theta_fft": np.nan, "D_RA": np.nan}
    if a is None:
        return out
    a = a - (a * w).sum() / w.sum()
    a = np.where(w > 0, a, 0.0)

    # ---- 2-D Fourier angular concentration.  Hann-windowed: without it the crop's own edges
    # put a cross of power on the array axes, which HERE ARE range and azimuth -- i.e. exactly
    # the directions under test.  That would not be a small bias, it would be the answer.
    ny, nx = a.shape
    win = a * w * np.hanning(ny)[:, None] * np.hanning(nx)[None, :]
    P = np.abs(np.fft.fftshift(np.fft.fft2(win))) ** 2
    ky = np.fft.fftshift(np.fft.fftfreq(ny))[:, None] + np.zeros((1, nx))
    kx = np.zeros((ny, 1)) + np.fft.fftshift(np.fft.fftfreq(nx))[None, :]
    k = np.hypot(kx, ky)
    m = (k > 3.0 / max(ny, nx)) & (k < 0.33)     # drop DC/trend and the aliased corners
    if m.sum() > 64:
        # ky indexes ROWS, which increase DOWNWARD, so it is negated to put the angle in a
        # right-handed frame with +y up.  Without that the reported orientation has the
        # wrong SIGN.
        th = np.arctan2(-ky[m], kx[m])           # wavevector angle CCW from +x, y up
        z = (P[m] * np.exp(2j * th)).sum() / P[m].sum()
        out["R_fft"] = float(abs(z))
        # power piles up PERPENDICULAR to the bands, so rotate the mean angle by 90
        out["theta_fft"] = float((np.degrees(np.angle(z)) / 2.0 + 180.0) % 180.0 - 90.0)
        out["D_RA"] = float(abs(z) * np.cos(2.0 * np.radians(out["theta_fft"])))
    return out


# --------------------------------------------------------------------------- field prep
def amp_mask(amps, mask, pct):
    """Valid-ground mask from BACKSCATTER instead of from stack coherence."""
    out = {}
    for p in amps:
        a = np.where(mask[p] & (amps[p] > 0), amps[p], np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            db = 10.0 * np.log10(a)
            t = np.nanpercentile(db, pct)
        out[p] = mask[p] & np.isfinite(db) & (db > t)
    return out


def residual_log_coh(cohs, mask, inc, min_stack_coh, inc_order=2,
                     amps=None, amp_pct=None, loo=False, med_reg=False):
    """Residual log-coherence: the per-pair field with everything STATIC removed."""
    keys = list(cohs)
    stack = np.stack([np.where(mask[p], np.log(np.maximum(cohs[p], 1e-6)), np.nan)
                      for p in keys])
    with warnings.catch_warnings():         # cells valid in NO pair are an all-NaN slice;
        warnings.simplefilter("ignore")     # they are dropped by the isfinite test below
        med = np.nanmedian(stack, axis=0)
        refs = ([np.nanmedian(np.delete(stack, i, axis=0), axis=0) for i in range(len(keys))]
                if loo and len(keys) > 1 else [med] * len(keys))
    if amps is not None and amp_pct is not None:
        gm = amp_mask(amps, mask, amp_pct)
        keep = np.all([gm[p] for p in keys], axis=0)
    else:
        keep = med > np.log(min_stack_coh)
    ok = np.all([mask[p] for p in keys], axis=0) & np.isfinite(med) & keep \
        & np.all([np.isfinite(r) for r in refs], axis=0)
    res, coef = {}, {}
    v = inc[ok]
    cols = [np.vander(v - v.mean(), inc_order + 1)]
    if med_reg:
        cols.append((med[ok] - med[ok].mean())[:, None])
    V = np.hstack(cols)
    for i, p in enumerate(keys):
        r = np.where(ok, stack[i] - refs[i], np.nan)
        c, *_ = np.linalg.lstsq(V, r[ok], rcond=None)
        r[ok] -= V @ c
        r[ok] -= r[ok].mean()
        res[p] = r
        coef[p] = float(c[-1]) if med_reg else float("nan")
    return res, ok, med, coef


def loss_field(r):
    """Decorrelation DEPTH: how far this pair fell BELOW the stack, and zero where it did not."""
    return np.where(np.isfinite(r), np.maximum(-r, 0.0), np.nan)


def angdiff180(a, b):
    """Signed difference of two ORIENTATIONS (mod 180), in degrees, in (-90, 90]."""
    return (np.asarray(a) - np.asarray(b) + 90.0) % 180.0 - 90.0


def measure_pairs(cache, track, frame, dates, pol="HH", spacing=120.0, rebin=4,
                  freqs=("A", "B"), min_valid=0.5, min_stack_coh=0.15, inc_order=2,
                  dir_smooth_km=2.0):
    """R_FFT and theta for every date pair in an accumulated GSLC stack."""
    import glob
    import itertools
    import json

    from .gslc_stack import radar_constants, rebin_split

    dates = sorted(dates)
    pairs = list(itertools.combinations(dates, 2))
    ck = os.path.join(cache, f"acc_T{track:03d}_F{frame:03d}_{pol}"
                             f"_{spacing:.0f}m_{'-'.join(dates)}.npz")
    if not os.path.exists(ck):
        raise SystemExit(f"no accumulation at {ck}; build it with the gslc-stack command")
    d = np.load(ck, allow_pickle=False)
    lay = json.loads(str(d["lay"]))

    # any full granule carries the geolocation cube: grid, projection and orbit geometry
    # are shared by every date on a track/frame
    gl = [g for g in sorted(glob.glob(os.path.join(cache, "*GSLC*.h5")))
          if os.path.getsize(os.path.realpath(g)) > 1e9]
    if not gl:
        raise SystemExit(f"no full GSLC granule in {cache} to read metadata/radarGrid from")
    set_radar_constants(**radar_constants(lay, cache, verbose=False))

    rows = []
    for freq in freqs:
        cohs, masks = {}, {}
        for p_ in pairs:
            z = rebin_split(d[f"{freq}|{p_[0]}|{p_[1]}|z"], rebin)[0]
            p1 = rebin_split(d[f"{freq}|{p_[0]}|{p_[1]}|p1"], rebin)[0]
            p2 = rebin_split(d[f"{freq}|{p_[0]}|{p_[1]}|p2"], rebin)[0]
            nn = rebin_split(d[f"{freq}|{p_[0]}|{p_[1]}|n"], rebin)[0] / (rebin ** 2)
            m = (nn > min_valid) & (p1 > 0) & (p2 > 0)
            c = np.zeros_like(p1)
            c[m] = np.abs(z[m]) / np.sqrt(p1[m] * p2[m])
            cohs[p_], masks[p_] = np.clip(c, 0, 1), m

        shape = next(iter(cohs.values())).shape
        src = native_grid(lay, rebin, shape)
        grid = utm_grid_for(lay, rebin, shape)
        cohs = {k: np.where(np.isfinite(v), v, 0.0)
                for k, v in ((k, resample_grid(v, src, grid, 1)) for k, v in cohs.items())}
        masks = {k: resample_grid(v, src, grid, 0) for k, v in masks.items()}

        geom = geom_from_cube(gl[0], grid)
        check_axes(gl[0], grid, geom)

        res, ok, _med, _a = residual_log_coh(cohs, masks, geom["inc"],
                                             min_stack_coh, inc_order)
        # the azimuth axis is taken over the cells that survived, not the whole grid
        az_b = float(np.nanmedian(bearing(geom["az_x"], geom["az_y"])[ok]))
        for p_ in pairs:
            dn = directionality(loss_field(res[p_]), ok, grid["cell"], az_b, dir_smooth_km)
            rows.append({"freq": freq, "d1": p_[0], "d2": p_[1],
                         "R_fft": dn.get("R_fft", float("nan")),
                         "theta_fft": dn.get("theta_fft", float("nan")),
                         "D_RA": dn.get("D_RA", float("nan")),
                         "coh_mean": float(np.nanmean(cohs[p_][ok])),
                         "n_cells": int(ok.sum())})
            print(f"  {freq} {p_[0]}-{p_[1]}  R_FFT {rows[-1]['R_fft']:.3f}  "
                  f"theta {rows[-1]['theta_fft']:+.1f} deg")
    return rows
