"""Accumulate a GSLC stack onto a common lattice: per date pair the interferometric sum and
the two power sums, from which coherence is formed."""
from __future__ import annotations

import time
import os

import numpy as np


GRIDS = "science/LSAR/GSLC/grids"


C = 299792458.0


# --------------------------------------------------------------------------- colour table
def phase_cmap():
    """hls from this script's own directory."""
    from matplotlib.colors import ListedColormap
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hls.cm")
    if not os.path.isfile(p):
        print("  hls.cm not found next to this script; falling back to turbo")
        return "turbo"
    return ListedColormap(np.loadtxt(p) / 255.0, name="hls")


# --------------------------------------------------------------------------- access
def local_granules(cache, dates):
    """{date: path} for GSLC granules already downloaded into the cache."""
    import glob
    out = {}
    for p in sorted(glob.glob(os.path.join(cache, "*GSLC*.h5"))):
        b = os.path.basename(p)
        for d in dates:
            if f"_{d}T" in b:
                out.setdefault(d, p)
    missing = [d for d in dates if d not in out]
    if missing:
        raise SystemExit(
            f"no local GSLC for {', '.join(missing)} in {cache}\n"
            f"  run: ionoNISAR download --level GSLC --track T --frame F "
                         f"--dates {' '.join(missing)}")
    return out


# Radar constants of the 40 MHz mode, used when the accumulation carries no carrier of its own.
DEFAULT_RADAR = {"fc_A": 1239.0e6, "fc_B": 1293.5e6, "bw_A": 40e6, "bw_B": 5e6,
                 "azbw": 1262.98}


def granule_radar(path):
    """{fc_A, fc_B, bw_A, bw_B, azbw} read from one GSLC granule."""
    import h5py
    out = {}
    with h5py.File(path, "r") as h:
        for f, tag in (("frequencyA", "A"), ("frequencyB", "B")):
            q = h[f"{GRIDS}/{f}"]
            out[f"fc_{tag}"] = float(q["centerFrequency"][()])
            out[f"bw_{tag}"] = float(q["rangeBandwidth"][()])
        out["azbw"] = float(h[f"{GRIDS}/frequencyA/azimuthBandwidth"][()])
    return out


def radar_constants(lay, cache=None, verbose=True):
    """Carrier and bandwidth per frequency, {fc_A, fc_B, bw_A, bw_B, azbw} in Hz."""
    import glob
    if lay.get("fc_A"):
        rc, src = {k: lay.get(k) for k in DEFAULT_RADAR}, "from the accumulation"
    else:
        gl = ([g for g in sorted(glob.glob(os.path.join(cache, "*GSLC*.h5")))
               if os.path.getsize(os.path.realpath(g)) > 1e9] if cache else [])
        if gl:
            rc, src = granule_radar(gl[0]), "from " + os.path.basename(gl[0])[:44]
        else:
            rc, src = dict(DEFAULT_RADAR), ("DEFAULT 40 MHz mode 4005: no carrier in the "
                                            "accumulation and no granule to read one from")
    rc = {k: (rc.get(k) if rc.get(k) else DEFAULT_RADAR[k]) for k in DEFAULT_RADAR}
    if verbose:
        r = rc["fc_B"] / rc["fc_A"]
        print(f"  radar: f_A {rc['fc_A'] / 1e6:.1f} MHz / {rc['bw_A'] / 1e6:.0f} MHz, "
              f"f_B {rc['fc_B'] / 1e6:.1f} MHz / {rc['bw_B'] / 1e6:.0f} MHz, "
              f"r = {r:.6f}, 1/(r - 1/r) = {1 / (r - 1 / r):.2f}   ({src})")
    return rc


def grid_of(h, pol):
    """Geogrid constants for both frequencies, plus the two datasets and the range calibration."""
    cal = h[GRIDS.replace("grids", "metadata") + "/calibrationInformation"]
    g = {}
    for f, tag in (("frequencyA", "A"), ("frequencyB", "B")):
        q = h[f"{GRIDS}/{f}"]
        if pol not in q:
            raise SystemExit(f"{f} has no {pol}; available {list(q['listOfPolarizations'][:])}")
        g[tag] = {"ds": q[pol],
                  "x0": float(q["xCoordinates"][0]), "dx": float(q["xCoordinateSpacing"][()]),
                  "y0": float(q["yCoordinates"][0]), "dy": float(q["yCoordinateSpacing"][()]),
                  "fc": float(q["centerFrequency"][()]),
                  # range bandwidth sets c/(2B): 40 MHz (mode 4005) or 20 MHz (mode 2005) on A
                  "bw": float(q["rangeBandwidth"][()]) if "rangeBandwidth" in q else None,
                  "azbw": (float(q["azimuthBandwidth"][()])
                           if "azimuthBandwidth" in q else None),
                  "shape": q[pol].shape,
                  "epsg": int(np.asarray(q["projection"])) if "projection" in q else None,
                  "delay": (float(cal[f + "/commonDelay"][()])
                            if f + "/commonDelay" in cal else None)}
    return g


# slant-range resolution c/(2B) per frequency: 40 MHz -> 3.75 m, 5 MHz -> 29.98 m (mode
# 2005: 20 MHz -> 7.49 m); the bandwidth comes from the granule
def check_delay(dates, grids):
    """Abort if the dates do not share a range calibration."""
    bad = []
    for tag in ("A", "B"):
        bw = grids[0][tag].get("bw") or DEFAULT_RADAR[f"bw_{tag}"]
        vals = {d: g[tag]["delay"] for d, g in zip(dates, grids)}
        if any(v is None for v in vals.values()):
            continue
        ref = vals[dates[0]]
        if max(abs(v - ref) for v in vals.values()) > 1e-6:
            res = C / (2 * bw)
            bad.append(f"  frequency {tag} (slant-range resolution {res:.2f} m):")
            for d in dates:
                dl = vals[d] - ref
                bad.append(f"    {d}  commonDelay {vals[d]:9.4f} m   delta {dl:+8.4f} m "
                           f"= {abs(dl) / res:.2f} resolution cells")
    if bad:
        raise SystemExit(
            "commonDelay mismatch -- these dates are on different range references and will\n"
            "decorrelate if interfered.  Re-geocode the odd granule onto the reference with\n"
            "  ./regeocode_delay.py --rslc <granule> --reference <granule>\n"
            + "\n".join(bad))


def align(grids, tag, spacing):
    """Integer per-date index offsets onto a common lattice, snapped to the look factor."""
    ref = grids[0][tag]
    fy, fx = int(round(spacing / abs(ref["dy"]))), int(round(spacing / ref["dx"]))
    if abs(fy * abs(ref["dy"]) - spacing) > 1e-6 or abs(fx * ref["dx"] - spacing) > 1e-6:
        raise SystemExit(f"--spacing {spacing} is not a multiple of the {tag} posting "
                         f"({ref['dx']} x {abs(ref['dy'])} m)")

    offs = []
    for g in grids:
        q = g[tag]
        if abs(q["dx"] - ref["dx"]) > 1e-9 or abs(q["dy"] - ref["dy"]) > 1e-9:
            raise SystemExit(f"{tag}: dates are on different postings")
        # index in THIS date = index in reference + off, hence ref minus date
        oy, ox = (ref["y0"] - q["y0"]) / ref["dy"], (ref["x0"] - q["x0"]) / ref["dx"]
        if max(abs(oy - round(oy)), abs(ox - round(ox))) > 0.01:
            raise SystemExit(f"{tag}: dates are not on a common lattice "
                             f"(offset {oy:.3f}, {ox:.3f} cells)")
        offs.append((int(round(oy)), int(round(ox))))

    # overlap window in REFERENCE index space, then snap to whole output cells
    y_lo = max(-o[0] for o in offs)
    x_lo = max(-o[1] for o in offs)
    y_hi = min(g[tag]["shape"][0] - o[0] for g, o in zip(grids, offs))
    x_hi = min(g[tag]["shape"][1] - o[1] for g, o in zip(grids, offs))
    y_lo, x_lo = -(-y_lo // fy) * fy, -(-x_lo // fx) * fx
    ny, nx = (y_hi - y_lo) // fy, (x_hi - x_lo) // fx
    if ny <= 0 or nx <= 0:
        raise SystemExit(f"{tag}: dates do not overlap")
    return {"fy": fy, "fx": fx, "y_lo": y_lo, "x_lo": x_lo, "ny": ny, "nx": nx, "offs": offs,
            "x0": ref["x0"] + x_lo * ref["dx"], "y0": ref["y0"] + y_lo * ref["dy"],
            "dx": ref["dx"], "dy": ref["dy"], "epsg": ref["epsg"]}


# --------------------------------------------------------------------------- looking
def bsum(a, fy, fx):
    """Block sum onto the coarse lattice.  Accumulates in 128-bit: 2304 complex64 terms of
    wildly unequal magnitude lose bits in float32."""
    ny, nx = a.shape
    dt = np.complex128 if np.iscomplexobj(a) else np.float64
    return a.astype(dt, copy=False).reshape(ny // fy, fy, nx // fx, fx).sum(axis=(1, 3))


def accumulate(paths, dates, pol, spacing, block_rows, pairs):
    """One pass over the granules producing every pair's looked ifg at both frequencies."""
    import h5py

    hs = [h5py.File(paths[d], "r") for d in dates]
    grids = [grid_of(h, pol) for h in hs]
    check_delay(dates, grids)
    print("  commonDelay  A %.4f  B %.4f m (consistent across all dates)"
          % (grids[0]["A"]["delay"] or float("nan"), grids[0]["B"]["delay"] or float("nan")))
    lay = {t: align(grids, t, spacing) for t in ("A", "B")}
    if (lay["A"]["ny"], lay["A"]["nx"]) != (lay["B"]["ny"], lay["B"]["nx"]):
        # trim to the common extent; the two frequencies share origin and spacing so this is
        # a pure crop, never a resample
        ny = min(lay["A"]["ny"], lay["B"]["ny"])
        nx = min(lay["A"]["nx"], lay["B"]["nx"])
        for t in ("A", "B"):
            lay[t]["ny"], lay[t]["nx"] = ny, nx
    ny, nx = lay["A"]["ny"], lay["A"]["nx"]
    print(f"  common grid {ny} x {nx} at {spacing:.0f} m  "
          f"(A looks {lay['A']['fy']}x{lay['A']['fx']}, B looks {lay['B']['fy']}x{lay['B']['fx']})")

    acc = {t: {p: {"z": np.zeros((ny, nx), np.complex128),
                   "p1": np.zeros((ny, nx)), "p2": np.zeros((ny, nx)),
                   "n": np.zeros((ny, nx))} for p in pairs} for t in ("A", "B")}

    t0 = time.time()
    for b0 in range(0, ny, block_rows):
        b1 = min(b0 + block_rows, ny)
        for t in ("A", "B"):
            L = lay[t]
            r0, r1 = L["y_lo"] + b0 * L["fy"], L["y_lo"] + b1 * L["fy"]
            c0, c1 = L["x_lo"], L["x_lo"] + nx * L["fx"]
            cur = {}
            for d, g, (oy, ox) in zip(dates, grids, L["offs"]):
                a = g[t]["ds"][r0 + oy:r1 + oy, c0 + ox:c1 + ox]
                bad = ~np.isfinite(a)
                if bad.any():
                    a = np.where(bad, 0, a)
                cur[d] = a
            pw = {d: bsum(np.abs(cur[d]) ** 2, L["fy"], L["fx"]) for d in dates}
            ok = {d: bsum((np.abs(cur[d]) > 0).astype(np.float32), L["fy"], L["fx"])
                  for d in dates}
            for d1, d2 in pairs:
                A = acc[t][(d1, d2)]
                A["z"][b0:b1] = bsum(cur[d1] * np.conj(cur[d2]), L["fy"], L["fx"])
                A["p1"][b0:b1], A["p2"][b0:b1] = pw[d1].real, pw[d2].real
                # stored as a FRACTION of the cell, not a count: A has 48x48 contributing
                # pixels and B only 48x6, so a shared absolute threshold would mask B out
                # entirely -- which it did, and produced a silently empty result
                A["n"][b0:b1] = np.minimum(ok[d1].real, ok[d2].real) / (L["fy"] * L["fx"])
            del cur, pw, ok
        el = time.time() - t0
        print(f"    rows {b0:5d}-{b1:5d} / {ny}   {el:6.0f}s"
              f"   eta {el * (ny - b1) / max(b1, 1) / 60:5.1f} min", flush=True)

    for h in hs:
        h.close()
    # carried in the layout so it survives into the cache and the output provenance
    lay["A"]["delay_A"] = grids[0]["A"]["delay"]
    lay["A"]["delay_B"] = grids[0]["B"]["delay"]
    # carrier and bandwidth travel with the accumulation so the ranking and the streak index
    # are mode-aware without a granule (4005: 1239.0/40 MHz on A; 2005: 1229.0/20 MHz)
    for t in ("A", "B"):
        lay["A"][f"fc_{t}"] = grids[0][t]["fc"]
        lay["A"][f"bw_{t}"] = grids[0][t]["bw"]
    lay["A"]["azbw"] = grids[0]["A"]["azbw"]
    return acc, lay["A"]


def rebin_split(a, R):
    """Coarsen by R and ALSO return two interleaved half-estimates."""
    ny, nx = (a.shape[0] // R) * R, (a.shape[1] // R) * R
    v = a[:ny, :nx].reshape(ny // R, R, nx // R, R)
    ii, jj = np.mgrid[0:R, 0:R]
    cb = ((ii + jj) % 2 == 0)
    h1 = (v * cb[None, :, None, :]).sum(axis=(1, 3))
    h2 = (v * (~cb)[None, :, None, :]).sum(axis=(1, 3))
    return h1 + h2, h1, h2


def nan_gaussian(a, sig):
    """Gaussian smoothing that ignores NaN, normalised by the valid support."""
    from scipy.ndimage import gaussian_filter
    w = np.isfinite(a)
    num = gaussian_filter(np.where(w, a, 0.0), sig, mode="nearest")
    den = gaussian_filter(w.astype(float), sig, mode="nearest")
    return np.where(den > 1e-3, num / np.maximum(den, 1e-12), np.nan)


def build_cache(cache, track, frame, dates, pol="HH", spacing=240.0, block_rows=16,
                refresh=False):
    """Accumulate the stack once and cache it; the granules can be deleted afterwards."""
    import itertools
    import json
    import os

    dates = sorted(dates)
    pairs = list(itertools.combinations(dates, 2))
    ck = os.path.join(cache, f"acc_T{track:03d}_F{frame:03d}_{pol}"
                             f"_{spacing:.0f}m_{'-'.join(dates)}.npz")
    if os.path.exists(ck) and not refresh:
        print(f"reusing accumulation {ck} ({os.path.getsize(ck) / 1e6:.0f} MB)")
        return ck
    paths = local_granules(cache, dates)
    for d in dates:
        print(f"  {d}  {os.path.getsize(paths[d]) / 1e9:5.1f} GB  "
              f"{os.path.basename(paths[d])[:60]}")
    acc, lay = accumulate(paths, dates, pol, spacing, block_rows, pairs)
    os.makedirs(cache, exist_ok=True)
    np.savez(ck, lay=np.array(json.dumps({k: v for k, v in lay.items() if k != "offs"})),
             **{f"{t}|{p[0]}|{p[1]}|{f}": acc[t][p][f]
                for t in ("A", "B") for p in pairs for f in ("z", "p1", "p2", "n")})
    print(f"cached accumulation -> {ck} ({os.path.getsize(ck) / 1e6:.0f} MB)")
    return ck
