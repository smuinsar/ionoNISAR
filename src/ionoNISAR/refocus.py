"""Doppler-dependent azimuth refocus of the coregistered secondary."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time

import numpy as np
import scipy.fft as sfft
from scipy.ndimage import map_coordinates

SIGN = -1.0          # d(f) = SIGN * A * f / B; the sign convention of the geometry.
# The sweep reaches 24 km (~800 km of ionospheric height); a pick at the top of the sweep is
# reported, because it means the range may be too narrow for the frame.
A_SWEEP_KM = (0.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 24.0)


# ------------------------------------------------------------------ lattice bookkeeping
def lattice_geom(npz_path=None, npy_path=None, search=(64, 64), winsize=(64, 64),
                 skip=(16, 16), az_spacing=None, sharp_median=(0, 0), steer_exclude_px=0.0,
                 steer_cutoff=30.0, steer_aniso=1.0):
    """The lattice fields the model reads, and where the nodes sit on the crop grid."""
    rb_meas = None
    if npz_path:
        z = np.load(npz_path)
        rb = np.asarray(z["applied"], np.float64)
        if "azimuth" in z.files:
            rb_meas = np.asarray(z["azimuth"], np.float64)
        search = tuple(int(v) for v in z["search"]); winsize = tuple(int(v) for v in z["winsize"])
        skip = tuple(int(v) for v in z["skip"])
        az_spacing = float(z["az_spacing"]) if "az_spacing" in z.files else az_spacing
    elif npy_path:
        rb = np.asarray(np.load(npy_path), np.float64)
    else:
        raise SystemExit("need --rbsheet NPZ or --rubbersheet NPY")
    origin = (search[0] + winsize[0] // 2, search[1] + winsize[1] // 2)
    rb = np.nan_to_num(rb)
    disp = rb
    if sharp_median and max(sharp_median) > 0:
        if rb_meas is None:
            raise SystemExit("--sharp-median needs the _rbsheet.npz (its measured `azimuth` field)")
        from scipy.ndimage import median_filter
        disp = median_filter(np.nan_to_num(rb_meas, nan=0.0), size=tuple(int(v) for v in sharp_median),
                             mode="nearest")
        d = disp - rb
        print(f"[refocus] steering field: measured lattice through a {sharp_median[0]} x {sharp_median[1]} "
              f"median; differs from the applied field by p50 {np.percentile(np.abs(d), 50):.3f}, "
              f"p99 {np.percentile(np.abs(d), 99):.2f}, max {np.abs(d).max():.2f} px")
    ref = rb
    if steer_exclude_px and steer_exclude_px > 0:
        # Sharp features the smoother attenuated but did not remove are NOT Doppler-dependent
        # (T087 lattice cols 2250-2500, rows 864-900: a -4.2 px plateau 2.4 km along track with
        # 300 m edges, no range signature, quarter-band shifts equal to 0.5 px, all quarter-band
        # coherences ~0.3 -- broadband decorrelation).  Steered by the applied field's smeared
        # copy of it, the refocus lowered that patch by ~0.1.  So the steering field is the
        # applied field with such cells cut out and re-interpolated by the same penalised least
        # squares; both the displaced and the reference term use it, which leaves the band-centre
        # registration of the feature exactly as the resampler applied it.
        if rb_meas is None:
            raise SystemExit("--steer-exclude-px needs the _rbsheet.npz (its measured `azimuth` field)")
        from scipy.ndimage import median_filter, binary_dilation
        from ._utils.fill import fill as pls_fill
        med = median_filter(np.nan_to_num(rb_meas, nan=0.0), size=(5, 9), mode="nearest")
        mask = binary_dilation(np.abs(med - rb) > steer_exclude_px, iterations=2)
        z, _ = pls_fill(np.where(mask, np.nan, rb), ~mask, method="pls",
                        s=(steer_cutoff / (2 * np.pi)) ** 4, robust=0, aniso=steer_aniso, verbose=False)
        z = np.asarray(z, np.float64)
        print(f"[refocus] steering field: applied field with {100 * mask.mean():.2f} % of the lattice "
              f"(|median(measured) - applied| > {steer_exclude_px:g} px, dilated 2) cut out and "
              f"re-interpolated at cutoff {steer_cutoff:g}; changed by p99 {np.percentile(np.abs(z - rb), 99):.3f}, "
              f"max {np.abs(z - rb).max():.2f} px", flush=True)
        disp = z if max(sharp_median) == 0 else disp
        ref = z
    return {"rb": disp, "rb_ref": ref, "origin": origin, "skip": tuple(skip),
            "az_spacing": az_spacing, "sharp_median": tuple(sharp_median),
            "steer_exclude_px": float(steer_exclude_px)}


def predicted_spread(geom, A_km, bw=None):
    """A |dRB/dline| on the lattice: the group-shift spread across the band, px."""
    A_lines = A_km * 1e3 / geom["az_spacing"]
    g = np.gradient(geom["rb"], axis=0) / geom["skip"][0]       # px per line
    return A_lines * np.abs(g)


# ------------------------------------------------------------------ the phase
def block_phase(fs, prf, bw, row_c, lat_cols, geom, A_lines):
    """Phi(f) for one azimuth block, on the lattice columns.  fs is SORTED frequency (Hz)."""
    rb = geom["rb"]
    rows = row_c + SIGN * A_lines * (fs / bw) / geom["skip"][0]
    rr = np.broadcast_to(rows[:, None], (len(fs), len(lat_cols)))
    cc = np.broadcast_to(lat_cols[None, :], (len(fs), len(lat_cols)))
    g = map_coordinates(rb, [rr, cc], order=1, mode="nearest")
    g0 = map_coordinates(geom.get("rb_ref", rb), [np.full(len(lat_cols), row_c), lat_cols], order=1,
                         mode="nearest")
    g = g - g0[None, :]
    df = np.diff(fs)[:, None]
    cum = np.concatenate([np.zeros((1, len(lat_cols))),
                          np.cumsum(0.5 * (g[1:] + g[:-1]) * df, axis=0)], axis=0)
    i0 = int(np.searchsorted(fs, 0.0))
    return (cum - cum[i0][None, :]) * (2.0 * np.pi / prf)


def gain_at(gain, lines, samples):
    """Bilinear lookup of a local gain grid.  gain = (w, row_centres, col_centres) on the crop grid
    (w is n_rows x n_cols; centres in crop lines / samples).  None -> 1 everywhere."""
    if gain is None:
        return None
    w, rc, cc = gain
    r = np.interp(np.atleast_1d(lines), rc, np.arange(len(rc))) if len(rc) > 1 else np.zeros(np.size(lines))
    c = np.interp(np.atleast_1d(samples), cc, np.arange(len(cc))) if len(cc) > 1 else np.zeros(np.size(samples))
    g = np.meshgrid(r, c, indexing="ij")
    return map_coordinates(w, [g[0], g[1]], order=1, mode="nearest")


def net_shift_phase(P, Y, fr, prf, j, w, step=16):
    """The linear phase that cancels the FULL-BAND shift exp(iP) would induce."""
    nf, m = P.shape
    W = np.mean(np.abs(Y) ** 2, axis=1)
    cols = np.arange(0, m, step)
    if cols[-1] != m - 1:
        cols = np.append(cols, m - 1)
    H = np.fft.ifft(W[:, None] * np.exp(1j * P[:, cols]), axis=0)
    a = np.abs(H)
    k = np.argmax(a, axis=0)
    km = (k - 1) % nf; kp = (k + 1) % nf
    y0 = a[km, np.arange(len(cols))]; y1 = a[k, np.arange(len(cols))]; y2 = a[kp, np.arange(len(cols))]
    den = y0 - 2 * y1 + y2
    frac = np.where(np.abs(den) > 1e-30, 0.5 * (y0 - y2) / np.where(np.abs(den) > 1e-30, den, 1.0), 0.0)
    d = ((k + frac + nf / 2) % nf) - nf / 2          # peak position in samples, signed
    d = np.interp(np.arange(m), cols, d)
    # an impulse response peaking at +d is a DELAY by d, i.e. exp(-2 pi i f d / PRF); the
    # correcting phase is its opposite.  (First cut had this backwards and doubled the shift:
    # quiet ground 0.019 -> 0.032 px, streak 0.082 -> 0.155 px.)
    return (2.0 * np.pi / prf) * fr[:, None] * d[None, :]


def refocus_array(y, line0, sample0, geom, A_km, prf, bw, carrier_hz, block=240, hop=120,
                  workers=1, invert=False, gain=None, zero_net_shift=True):
    """Refocus an in-memory crop.  (line0, sample0) is its origin on the crop grid."""
    n, m = y.shape
    if n < block:
        raise ValueError(f"{n} lines is fewer than one {block}-line block")
    hop = hop or block
    A_lines = A_km * 1e3 / geom["az_spacing"]
    o_r, o_c = geom["origin"]; s_r, s_c = geom["skip"]
    # lattice columns spanned by this crop, plus one node each side for the interpolation
    u = (sample0 + np.arange(m) - o_c) / s_c
    c_lo = int(np.floor(u.min())) - 1; c_hi = int(np.ceil(u.max())) + 1
    lat_cols = np.arange(c_lo, c_hi + 1, dtype=np.float64)
    j = np.clip(np.floor(u).astype(int) - c_lo, 0, len(lat_cols) - 2)
    w = (u - (j + c_lo)).astype(np.float64)
    fr = sfft.fftfreq(block, d=1.0 / prf); order = np.argsort(fr); fs = fr[order]
    t = np.arange(block) / prf
    dem = np.exp(-2j * np.pi * carrier_hz * t)[:, None]
    hann = np.hanning(block) if hop < block else np.ones(block)
    starts = list(range(0, n - block + 1, hop))
    if starts[-1] + block < n:
        starts.append(n - block)
    out = np.zeros((n, m), np.complex128); wsum = np.zeros(n)
    for b, s0 in enumerate(starts):
        # Hann overlap-add sums to one in the interior; the first and last blocks are the only
        # cover of their outer lines, so those lines take the block at full weight instead of
        # the window's zero end (A = 0 must be the identity everywhere, edges included)
        win = hann.copy()
        if b == 0 and len(starts) > 1:
            win[:starts[1] - s0] = 1.0
        if b == len(starts) - 1 and len(starts) > 1:
            win[max(starts[b - 1] + block - s0, 0):] = 1.0
        if len(starts) == 1:
            win[:] = 1.0
        row_c = (line0 + s0 + block / 2.0 - o_r) / s_r
        Phi = block_phase(fs, prf, bw, row_c, lat_cols, geom, A_lines)
        if invert:
            Phi = -Phi
        Pu = np.empty_like(Phi); Pu[order] = Phi
        P = Pu[:, j] * (1.0 - w)[None, :] + Pu[:, j + 1] * w[None, :]
        if gain is not None:
            P = P * gain_at(gain, line0 + s0 + block / 2.0, sample0 + np.arange(m))[0][None, :]
        Y = sfft.fft(y[s0:s0 + block] * dem, axis=0, workers=workers)
        if zero_net_shift:
            P = P + net_shift_phase(P, Y, fr, prf, j, w)
        yy = sfft.ifft(Y * np.exp(1j * P), axis=0, workers=workers) * np.conj(dem)
        out[s0:s0 + block] += yy * win[:, None]; wsum[s0:s0 + block] += win
    out /= np.maximum(wsum, 1e-9)[:, None]
    return out.astype(np.complex64)


# ------------------------------------------------------------------ whole file
def _shape_from_vrt(path):
    vrt = path + ".vrt"
    if os.path.exists(vrt):
        s = open(vrt).read()
        x = re.search(r'rasterXSize="(\d+)"', s); y = re.search(r'rasterYSize="(\d+)"', s)
        if x and y:
            return int(y.group(1)), int(x.group(1))
    return None


_SHARED = {}          # geom / gain handed to forked workers without pickling 90 MB per job


def _shared(obj, key):
    return _SHARED[key] if obj is None else obj


def _strip_worker(job):
    (sec_in, sec_out, shape, c0, c1, geom, A_km, prf, bw, carrier_hz, block, hop, gain) = job
    geom = _shared(geom, "geom"); gain = _shared(gain, "gain")
    src = np.memmap(sec_in, dtype=np.complex64, mode="r", shape=shape)
    dst = np.memmap(sec_out, dtype=np.complex64, mode="r+", shape=shape)
    t = time.time()
    y = np.array(src[:, c0:c1])
    dst[:, c0:c1] = refocus_array(y, 0, c0, geom, A_km, prf, bw, carrier_hz, block, hop, gain=gain)
    dst.flush()
    return c0, c1, time.time() - t


def _gain_worker(job):
    (pair_kw, L0, block, colblock, geom, A_km, prf, bw, carrier_hz, min_spread) = job
    geom = _shared(geom, "geom")
    pair = Pair(**pair_kw)
    x, y = pair.crop(L0, 0, block, pair.shape[1])
    w, rc, cc, table = local_gain(x, y, L0, 0, geom, A_km, prf, bw, carrier_hz, block, colblock, min_spread,
                                  verbose=False)
    return L0, w[0], cc, table[0]


def local_gain_file(pair_kw, shape, geom, A_km, prf, bw, carrier_hz, block=240, colblock=2048,
                    min_spread=1.0, workers=4, state_dir=None):
    """local_gain over the whole raster, one row band per job.  Returns (w, rc, cc)."""
    na, nr = shape
    starts = list(range(0, na - block + 1, block))
    _SHARED["geom"] = geom
    jobs = [(pair_kw, L0, block, colblock, None, A_km, prf, bw, carrier_hz, min_spread) for L0 in starts]
    t0 = time.time()
    print(f"[refocus] local gain search: {len(jobs)} row bands of {block} lines x {colblock}-sample blocks "
          f"on {workers} worker(s), blocks with predicted spread >= {min_spread:g} px", flush=True)
    rows = {}
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers) as ex:
            for k, (L0, wrow, cc, trow) in enumerate(ex.map(_gain_worker, jobs)):
                rows[L0] = (wrow, cc, trow)
                if (k + 1) % 40 == 0:
                    print(f"[refocus]   {k + 1}/{len(jobs)} bands ({time.time() - t0:.0f} s)", flush=True)
    else:
        for L0, wrow, cc, trow in map(_gain_worker, jobs):
            rows[L0] = (wrow, cc, trow)
    w = np.stack([rows[L0][0] for L0 in starts]); cc = rows[starts[0]][1]
    table = np.stack([rows[L0][2] for L0 in starts])
    rc = np.array(starts, np.float64) + block / 2.0
    tested = np.isfinite(table[:, :, 0])
    vals, cnt = np.unique(w[tested], return_counts=True)
    gain0 = table[:, :, 0][tested]; best = np.nanmax(table, axis=2)[tested]
    print(f"[refocus] local gain: {int(tested.sum())} of {w.size} blocks tested; gains chosen: "
          + ", ".join(f"{v:g}: {c}" for v, c in zip(vals, cnt))
          + f"; on tested blocks mean coherence {gain0.mean():.4f} (gain 0) -> {best.mean():.4f} (chosen)"
          + f" vs {table[:, :, -1][tested].mean():.4f} (gain 1 everywhere); {time.time() - t0:.0f} s", flush=True)
    if state_dir:
        os.makedirs(state_dir, exist_ok=True)
        np.savez_compressed(os.path.join(state_dir, "refocus_gain.npz"), w=w, rc=rc, cc=cc, table=table,
                            block=block, colblock=colblock, min_spread=min_spread)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 7))
            im = ax.imshow(np.where(tested, w, np.nan), vmin=0, vmax=1, cmap="viridis", aspect="auto",
                           extent=[0, nr, na, 0], interpolation="nearest")
            ax.set_title("refocus local gain (blank = untested, predicted spread < %g px)" % min_spread)
            ax.set_xlabel("sample"); ax.set_ylabel("line"); fig.colorbar(im, ax=ax, label="gain")
            fig.tight_layout(); fig.savefig(os.path.join(state_dir, "refocus_gain.png"), dpi=100); plt.close(fig)
        except Exception as exc:
            print(f"[refocus] gain quicklook skipped: {exc}")
    return w, rc, cc


def refocus_file(sec_in, sec_out, shape, geom, A_km, prf, bw, carrier_hz, block=240, hop=120,
                 strip=4096, workers=4, state_dir=None, extra_state=None, gain=None):
    """sec_in -> sec_out, whole raster, column strips in parallel (the FFT is along azimuth,
    so strips are independent).  Writes the ENVI/VRT sidecars and refocus_state.json."""
    na, nr = shape
    if os.path.abspath(sec_in) == os.path.abspath(sec_out):
        raise SystemExit("refocus cannot run in place (overlap-add needs the input intact)")
    dst = np.memmap(sec_out, dtype=np.complex64, mode="w+", shape=shape); del dst
    edges = list(range(0, nr, strip)) + [nr]
    _SHARED["geom"] = geom; _SHARED["gain"] = gain
    jobs = [(sec_in, sec_out, shape, c0, c1, None, A_km, prf, bw, carrier_hz, block, hop, None)
            for c0, c1 in zip(edges[:-1], edges[1:])]
    t0 = time.time()
    print(f"[refocus] {os.path.basename(sec_in)} -> {os.path.basename(sec_out)}: "
          f"{na} x {nr}, A {A_km:g} km ({A_km * 1e3 / geom['az_spacing']:.0f} lines), block {block} "
          f"hop {hop}, {len(jobs)} strips of {strip} samples on {workers} worker(s)", flush=True)
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers) as ex:
            for c0, c1, dt in ex.map(_strip_worker, jobs):
                print(f"[refocus]   samples {c0}..{c1} in {dt:.0f} s", flush=True)
    else:
        for job in jobs:
            c0, c1, dt = _strip_worker(job)
            print(f"[refocus]   samples {c0}..{c1} in {dt:.0f} s", flush=True)
    # sidecars: ENVI header (same grid) and a VRT pointing at the new file
    stem_in = os.path.splitext(sec_in)[0]; stem_out = os.path.splitext(sec_out)[0]
    if os.path.exists(stem_in + ".hdr"):
        shutil.copyfile(stem_in + ".hdr", stem_out + ".hdr")
    if os.path.exists(sec_in + ".vrt"):
        s = open(sec_in + ".vrt").read().replace(os.path.basename(sec_in), os.path.basename(sec_out))
        open(sec_out + ".vrt", "w").write(s)
    st = os.stat(sec_in)
    state = {"A_km": A_km, "block": block, "hop": hop, "sign": SIGN, "prf": prf, "bw": bw,
             "carrier_hz": carrier_hz, "source": os.path.abspath(sec_in),
             "source_size": st.st_size, "source_mtime": st.st_mtime,
             "output": os.path.abspath(sec_out), "seconds": time.time() - t0,
             "rb_std_px": float(np.nanstd(geom["rb_ref"])), "sharp_median": list(geom.get("sharp_median", (0, 0))),
             "steer_exclude_px": geom.get("steer_exclude_px", 0.0)}
    if extra_state:
        state.update(extra_state)
    if state_dir:
        os.makedirs(state_dir, exist_ok=True)
        with open(os.path.join(state_dir, "refocus_state.json"), "w") as f:
            json.dump(state, f, indent=1)
    print(f"[refocus] done in {time.time() - t0:.0f} s", flush=True)
    return state


def already_refocused(state_dir, sec_path):
    """True if refocus_state.json says THIS sec file is the refocused output (resume guard)."""
    p = os.path.join(state_dir or "", "refocus_state.json")
    if not (state_dir and os.path.exists(p) and os.path.exists(sec_path)):
        return False
    s = json.load(open(p))
    st = os.stat(sec_path)
    return (s.get("applied_to") == os.path.abspath(sec_path)
            and s.get("applied_size") == st.st_size and s.get("applied_mtime") == st.st_mtime)


# ------------------------------------------------------------------ coherence / calibration
def multilook_coh(x, y, looks=(24, 16)):
    la, lr = looks
    r = (x.shape[0] // la, la, x.shape[1] // lr, lr); n = r[0] * la; m = r[2] * lr
    p = (x[:n, :m] * np.conj(y[:n, :m])).reshape(r).sum(axis=(1, 3))
    a = (np.abs(x[:n, :m]) ** 2).reshape(r).sum(axis=(1, 3))
    b = (np.abs(y[:n, :m]) ** 2).reshape(r).sum(axis=(1, 3))
    return np.abs(p) / np.sqrt(np.maximum(a * b, 1e-30))


class Pair:
    """ref / sec_coreg + geo2rdr offsets, flattened and carrier-compensated exactly as
    the interferogram stage does (lines 320-346), so a coherence measured here is the pipeline's."""

    def __init__(self, scratch, shape, wavelength, range_spacing, dr0, az_carrier,
                 sec_name="sec_coreg.c8"):
        from ._utils import raster as IU
        self.kw = dict(scratch=scratch, shape=tuple(shape), wavelength=wavelength, range_spacing=range_spacing,
                       dr0=dr0, az_carrier=az_carrier, sec_name=sec_name)
        self.shape = tuple(shape)
        self.ref = np.memmap(os.path.join(scratch, "ref.c8"), dtype=np.complex64, mode="r", shape=shape)
        self.sec = np.memmap(os.path.join(scratch, sec_name), dtype=np.complex64, mode="r", shape=shape)
        self.roff = IU.open_raster(os.path.join(scratch, "geo2rdr", "range.off"), shape)
        self.aoff = IU.open_raster(os.path.join(scratch, "geo2rdr", "azimuth.off"), shape)
        self.lam, self.rsp, self.dr0, self.azc = wavelength, range_spacing, dr0, az_carrier

    def crop(self, L0, S0, nl, ns):
        x = np.array(self.ref[L0:L0 + nl, S0:S0 + ns])
        y = np.array(self.sec[L0:L0 + nl, S0:S0 + ns])
        dR = self.dr0 + np.asarray(self.roff[L0:L0 + nl, S0:S0 + ns]) * self.rsp
        ao = np.asarray(self.aoff[L0:L0 + nl, S0:S0 + ns])
        y = (y * np.exp(1j * (4.0 * np.pi / self.lam) * dR) * np.exp(-1j * self.azc * ao))
        return x, y.astype(np.complex64)


GAIN_STEPS = (0.0, 0.25, 0.5, 0.75, 1.0)


def local_gain(x, y, line0, sample0, geom, A_km, prf, bw, carrier_hz, block=240, colblock=2048,
               min_spread=1.0, looks=(24, 16), steps=GAIN_STEPS, workers=1, verbose=True):
    """Per-block gain on the model's phase, chosen by the block's own coherence."""
    n, m = x.shape
    rs = list(range(0, n - block + 1, block)); cs = list(range(0, m, colblock))
    sp = predicted_spread(geom, A_km)
    o_r, o_c = geom["origin"]; s_r, s_c = geom["skip"]
    w = np.ones((len(rs), len(cs))); table = np.full((len(rs), len(cs), len(steps)), np.nan)
    tested = 0; gained = 0.0
    for i, r0 in enumerate(rs):
        for jx, c0 in enumerate(cs):
            c1 = min(c0 + colblock, m)
            lr0 = int((line0 + r0 - o_r) / s_r); lr1 = int((line0 + r0 + block - o_r) / s_r) + 1
            lc0 = int((sample0 + c0 - o_c) / s_c); lc1 = int((sample0 + c1 - o_c) / s_c) + 1
            smax = float(np.nanmax(sp[max(lr0, 0):max(lr1, 1), max(lc0, 0):max(lc1, 1)])) if lr1 > 0 and lc1 > 0 else 0.0
            if smax < min_spread:
                continue
            xb = x[r0:r0 + block, c0:c1]; yb = y[r0:r0 + block, c0:c1]
            best = (-1.0, 1.0)
            for k, g in enumerate(steps):
                if g == 0.0:
                    yy = yb
                else:
                    unit = (np.ones((1, 1)), np.array([r0 + block / 2.0]), np.array([0.5 * (c0 + c1)]))
                    unit = (np.full((1, 1), g), unit[1] + line0, unit[2] + sample0)
                    yy = refocus_array(yb, line0 + r0, sample0 + c0, geom, A_km, prf, bw, carrier_hz,
                                       block, block, workers=workers, gain=unit)
                c = float(multilook_coh(xb, yy, looks).mean())
                table[i, jx, k] = c
                if c > best[0]:
                    best = (c, g)
            w[i, jx] = best[1]; tested += 1
            gained += best[0] - table[i, jx, 0]
    rc = np.array(rs, np.float64) + block / 2.0 + line0
    cc = np.array([0.5 * (c0 + min(c0 + colblock, m)) for c0 in cs], np.float64) + sample0
    if tested and verbose:
        vals, cnt = np.unique(w[np.isfinite(table[:, :, 0])], return_counts=True)
        print(f"[refocus] local gain: {tested} blocks of {block} x {colblock} tested (predicted spread >= "
              f"{min_spread:g} px); gains chosen: " + ", ".join(f"{v:g}: {c}" for v, c in zip(vals, cnt))
              + f"; mean coherence gain over gain 0 on tested blocks {gained / tested:+.4f}", flush=True)
    return w, rc, cc, table


def steepest_blocks(geom, nblk, block, ns, shape):
    """Top-|dRB/dline| blocks of (block lines x ns samples), well separated.  Returns crop
    (L0, S0) origins inside the raster."""
    from scipy.ndimage import uniform_filter
    o_r, o_c = geom["origin"]; s_r, s_c = geom["skip"]
    g = np.abs(np.gradient(geom["rb"], axis=0))
    br, bc = max(block // s_r, 1), max(ns // s_c, 1)
    gm = uniform_filter(g, (br, bc), mode="constant")
    picks = []
    gm = gm.copy()
    na, nr = shape
    while len(picks) < nblk and np.nanmax(gm) > 0:
        m, n = np.unravel_index(int(np.nanargmax(gm)), gm.shape)
        L0 = int(o_r + s_r * m - block // 2); S0 = int(o_c + s_c * n - ns // 2)
        if 0 <= L0 and L0 + block <= na and 0 <= S0 and S0 + ns <= nr:
            picks.append((L0, S0, float(gm[m, n])))
        gm[max(m - 2 * br, 0):m + 2 * br + 1, max(n - bc, 0):n + bc + 1] = 0.0
    return picks


def calibrate(pair, geom, prf, bw, carrier_hz, block=240, ns=2048, nblk=24,
              sweep=A_SWEEP_KM, looks=(24, 16), hop=120):
    """Sweep A on the steepest blocks; return (best_A, gain, table)."""
    picks = steepest_blocks(geom, nblk, 3 * block, ns, pair.shape)
    print(f"[refocus] calibrating A on the {len(picks)} steepest {block} x {ns} blocks "
          f"(|dRB/dcell| {picks[-1][2]:.3f} .. {picks[0][2]:.3f} px/cell), each refocused with "
          f"{block} lines of context on either side", flush=True)
    table = np.full((len(picks), len(sweep)), np.nan)
    for i, (L0, S0, _) in enumerate(picks):
        x, y = pair.crop(L0, S0, 3 * block, ns)
        for k, A in enumerate(sweep):
            yy = y if A == 0 else refocus_array(y, L0, S0, geom, A, prf, bw, carrier_hz, block, hop)
            table[i, k] = multilook_coh(x[block:2 * block], yy[block:2 * block], looks).mean()
    mean = table.mean(axis=0)
    print("[refocus]   A [km]   " + " ".join(f"{A:6.0f}" for A in sweep))
    print("[refocus]   mean coh " + " ".join(f"{v:6.3f}" for v in mean))
    print("[refocus]   blocks improved over A=0: "
          + " ".join(f"{int((table[:, k] > table[:, 0] + 0.005).sum()):6d}" for k in range(len(sweep))))
    k = int(np.argmax(mean)); best = sweep[k]; gain = float(mean[k] - mean[0])
    print(f"[refocus] best A = {best:g} km: mean coherence on the steep blocks "
          f"{mean[0]:.3f} -> {mean[k]:.3f} (+{gain:.3f})", flush=True)
    if k == len(sweep) - 1:
        print(f"[refocus] NOTE: the best A is the top of the sweep ({best:g} km); the optimum may lie "
              f"higher -- extend A_SWEEP_KM if this recurs", flush=True)
    return best, gain, {"A_km": list(sweep), "mean_coh": mean.tolist(),
                        "blocks": [(int(a), int(b), float(c)) for a, b, c in picks],
                        "table": table.tolist()}


def implied_height(A_km, wavelength, slant_range_m, bw, az_spacing, prf, sat_height_m):
    """The ionospheric height the aperture A implies: A = (h / H) * L_s, L_s = lambda R B / (2 v_g)."""
    v_g = az_spacing * prf
    L_s = wavelength * slant_range_m * bw / (2.0 * v_g)
    return A_km * 1e3 / L_s * sat_height_m, L_s


# ------------------------------------------------------------------ pipeline entry
def geom_from_arrays(applied, search, winsize, skip, az_spacing):
    """lattice_geom without the npz: the applied field the pipeline holds in memory."""
    origin = (int(search[0]) + int(winsize[0]) // 2, int(search[1]) + int(winsize[1]) // 2)
    rb = np.nan_to_num(np.asarray(applied, np.float64))
    return {"rb": rb, "rb_ref": rb, "origin": origin, "skip": (int(skip[0]), int(skip[1])),
            "az_spacing": float(az_spacing), "sharp_median": (0, 0), "steer_exclude_px": 0.0}


def fold(f, prf):
    """Frequency folded into the sampled band (-prf/2, prf/2]."""
    return (np.asarray(f, dtype=np.float64) + prf / 2.0) % prf - prf / 2.0


def rslc_band(h5, freq="A"):
    """(prf, doppler centroid, processed azimuth bandwidth) of an RSLC, in Hz."""
    import h5py
    S = "science/LSAR/RSLC/swaths/"
    with h5py.File(h5, "r") as h:
        prf = 1.0 / float(h[S + "zeroDopplerTimeSpacing"][()])
        bw = float(h[S + f"frequency{freq}/processedAzimuthBandwidth"][()])
        dc = float(np.asarray(h["science/LSAR/RSLC/metadata/processingInformation/"
                                f"parameters/frequency{freq}/dopplerCentroid"][()]).mean())
        nominal = float(h[S + f"frequency{freq}/nominalAcquisitionPRF"][()])
    # the PRF that matters is the one the SAMPLES are on, not nominalAcquisitionPRF: NISAR
    # products are presummed, and the nominal rate would put every sub-aperture in the
    # wrong place
    if abs(nominal - prf) > 1.0:
        print(f"[band] nominalAcquisitionPRF {nominal:.2f} Hz is NOT the sample rate "
              f"({prf:.2f} Hz); the product is presummed and {prf:.2f} is what counts")
    return prf, dc, bw


def run_in_pipeline(scratch, shape, applied, search, winsize, skip, az_spacing, sec_h5,
                    wavelength, range_spacing, dr0, az_carrier, A_km="auto", block=240, hop=120,
                    gain="auto", gain_min_spread=1.0, gain_colblock=2048, calib_blocks=24,
                    workers=6, strip=4096, keep_prefocus=False, slant_range_m=None,
                    sat_height_m=7.47e5):
    """What the coregistration calls right after the rubbersheet resample: calibrate A (or take
    it), search the local gain, refocus <scratch>/sec_coreg.c8 through a temporary file and put
    the result back under the same name, record refocus_state.json.  Returns the state dict, or
    None when the calibration verdict was that there is nothing to refocus."""
    sec = os.path.join(scratch, "sec_coreg.c8"); g2r = os.path.join(scratch, "geo2rdr")
    if already_refocused(g2r, sec):
        print(f"[refocus] {sec} is already the refocused secondary (refocus_state.json matches); skipped")
        return json.load(open(os.path.join(g2r, "refocus_state.json")))
    prf, dc, bw = rslc_band(sec_h5)
    carrier = az_carrier * prf / (2.0 * np.pi)
    geom = geom_from_arrays(applied, search, winsize, skip, az_spacing)
    print(f"[refocus] PRF {prf:.2f} Hz, bandwidth {bw:.1f} Hz, carrier {carrier:+.1f} Hz "
          f"(--az-carrier {az_carrier:+.4f} rad/px), lattice origin {geom['origin']} skip {geom['skip']}, "
          f"applied field std {np.nanstd(geom['rb']):.3f} px", flush=True)
    pair = Pair(scratch, shape, wavelength, range_spacing, dr0, az_carrier)
    calib = None
    if str(A_km).lower() == "auto":
        A, gained, calib = calibrate(pair, geom, prf, bw, carrier, block, 2048, calib_blocks, hop=hop)
        if gained < 0.01:
            print(f"[refocus] verdict: gain {gained:+.3f} < 0.01 on the steepest blocks -- nothing to "
                  f"refocus on this pair; sec_coreg.c8 left as resampled", flush=True)
            return None
    else:
        A = float(A_km)
    h, L_s = implied_height(A, wavelength, slant_range_m or 9.7e5, bw, az_spacing, prf, sat_height_m)
    sp = predicted_spread(geom, A)
    print(f"[refocus] A = {A:g} km (synthetic aperture ~{L_s / 1e3:.1f} km -> ionosphere ~{h / 1e3:.0f} km); "
          f"predicted group-shift spread across the band: p99 {np.nanpercentile(sp, 99):.2f} px, "
          f"{100 * (sp > 1).mean():.2f} % of the lattice above 1 px", flush=True)
    gain_grid = None
    if str(gain).lower() == "auto":
        gain_grid = local_gain_file(pair.kw, shape, geom, A, prf, bw, carrier, block, gain_colblock,
                                    gain_min_spread, workers, state_dir=g2r)
    elif float(gain) != 1.0:
        gain_grid = (np.full((1, 1), float(gain)), np.array([shape[0] / 2.0]), np.array([shape[1] / 2.0]))
    tmp = os.path.join(scratch, "sec_refocus.c8")
    state = refocus_file(sec, tmp, shape, geom, A, prf, bw, carrier, block, hop, strip, workers,
                         state_dir=g2r, gain=gain_grid,
                         extra_state={"calibration": calib, "implied_iono_height_m": h, "gain": str(gain),
                                      "gain_min_spread": gain_min_spread, "gain_colblock": gain_colblock})
    # swap: the refocused file takes the name every downstream step reads
    if keep_prefocus:
        os.replace(sec, os.path.join(scratch, "sec_coreg_prefocus.c8"))
        print(f"[refocus] the resampled-only secondary kept as sec_coreg_prefocus.c8")
    else:
        os.remove(sec)
    os.replace(tmp, sec)
    for side in (tmp + ".vrt", os.path.splitext(tmp)[0] + ".hdr"):
        if os.path.exists(side):
            os.remove(side)
    st = os.stat(sec)
    state.update({"applied_to": os.path.abspath(sec), "applied_size": st.st_size, "applied_mtime": st.st_mtime})
    with open(os.path.join(g2r, "refocus_state.json"), "w") as f:
        json.dump(state, f, indent=1)
    print(f"[refocus] sec_coreg.c8 is now the refocused secondary (A {A:g} km, gain {gain}); "
          f"state in geo2rdr/refocus_state.json", flush=True)
    return state


# ------------------------------------------------------------------ self test
def selftest():
    """Synthetic: a band-limited random field, the secondary given the FORWARD distortion
    (each Doppler component shifted by the model's g), then refocused.  Checks A = 0 is the
    identity, that the refocus restores the coherence, and that the wrong sign does not."""
    rng = np.random.default_rng(0)
    prf, bw, fc = 1520.0, 1262.5, -557.0
    n, m = 4800, 512
    rb = np.zeros((400, 64)); rows = np.arange(400)
    rb += (-3.0 / (1 + np.exp(-(rows - 150) / 6.0)) + 3.0 / (1 + np.exp(-(rows - 220) / 8.0)))[:, None]
    geom = {"rb": rb, "origin": (96, 96), "skip": (16, 16), "az_spacing": 4.44}
    t = np.arange(n) / prf
    X = rng.standard_normal((n, m)) + 1j * rng.standard_normal((n, m))
    fr = sfft.fftfreq(n, d=1 / prf)
    X[np.abs(fr) > bw / 2] = 0
    x = sfft.ifft(X, axis=0) * np.exp(2j * np.pi * fc * t)[:, None]
    # forward distortion: the model's -Phi, block for block what the refocus will undo
    A = 9.0
    x = x.astype(np.complex64)
    # the model distortion WITHOUT its net shift (the rubbersheet has already taken that out
    # of real data, and the refocus deliberately leaves the band-centre registration alone),
    # so the refocus is its exact inverse
    y = refocus_array(x, 0, 0, geom, A, prf, bw, fc, 240, 120, invert=True)
    c0 = multilook_coh(x, y).mean()
    c1 = multilook_coh(x, refocus_array(y, 0, 0, geom, A, prf, bw, fc, 240, 120)).mean()
    c2 = multilook_coh(x, refocus_array(y, 0, 0, geom, -A, prf, bw, fc, 240, 120)).mean()
    ident = np.abs(refocus_array(x, 0, 0, geom, 0.0, prf, bw, fc, 240, 120) - x).max() / np.abs(x).max()
    print(f"[selftest] distorted pair coherence {c0:.3f}; refocused {c1:.3f}; wrong sign {c2:.3f}; "
          f"A=0 identity max |diff| / max |x| {ident:.2e}")
    ok = c1 > 0.95 and c0 < 0.9 and c2 < c0 and ident < 1e-5
    print("[selftest] " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


# ------------------------------------------------------------------ CLI
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scratch", default=None,
                    help="the pair's scratch directory (required unless --selftest)")
    ap.add_argument("--sec", default="sec_coreg.c8", help="secondary inside --scratch (default sec_coreg.c8)")
    ap.add_argument("--out", default="sec_refocus.c8", help="output name inside --scratch")
    ap.add_argument("--shape", nargs=2, type=int, default=None, metavar=("LINES", "SAMPLES"),
                    help="raster shape; default from <sec>.vrt")
    ap.add_argument("--rbsheet", default=None, help="offsets_<tag>_rbsheet.npz (applied field + lattice geometry)")
    ap.add_argument("--rubbersheet", default=None, help="geo2rdr/rubbersheet_az.npy instead of --rbsheet")
    ap.add_argument("--search", nargs=2, type=int, default=(64, 64))
    ap.add_argument("--winsize", nargs=2, type=int, default=(64, 64))
    ap.add_argument("--skip", nargs=2, type=int, default=(16, 16))
    ap.add_argument("--az-spacing", type=float, default=None, help="m per azimuth line (default from --rbsheet)")
    ap.add_argument("--steer-exclude-px", type=float, default=0.0,
                    help="cut sharp Doppler-independent features (|5x9 median of the measured field - applied| "
                         "above this, px) out of the steering field and re-interpolate; 0 = off")
    ap.add_argument("--steer-cutoff", type=float, default=30.0, help="cutoff (lattice cells) of that re-interpolation")
    ap.add_argument("--steer-aniso", type=float, default=1.0, help="range-axis penalty weight of that re-interpolation")
    ap.add_argument("--sharp-median", nargs=2, type=int, default=(0, 0), metavar=("ROWS", "COLS"),
                    help="steer the sub-apertures with the MEASURED lattice field (npz `azimuth`) through "
                         "this median filter instead of the applied (smoothed) field; 0 0 = off")
    ap.add_argument("--A-km", default="auto", help="pierce aperture in km, or 'auto' to calibrate on the steepest blocks")
    ap.add_argument("--block", type=int, default=240)
    ap.add_argument("--hop", type=int, default=120)
    ap.add_argument("--prf", type=float, default=None, help="sampled PRF, Hz (default from --sec-h5)")
    ap.add_argument("--bw", type=float, default=None, help="processed azimuth bandwidth, Hz (default from --sec-h5)")
    ap.add_argument("--carrier-hz", type=float, default=None,
                    help="folded azimuth carrier, Hz; default --az-carrier * prf / 2pi, else the folded LUT mean")
    ap.add_argument("--sec-h5", default=None, help="secondary RSLC, for --prf/--bw/--carrier-hz defaults")
    g = ap.add_argument_group("flattening, for --A-km auto and --crop (the ifg step's own values)")
    g.add_argument("--wavelength", type=float, default=None,
                   help="radar wavelength in m; read from the granule when omitted")
    g.add_argument("--range-spacing", type=float, default=None)
    g.add_argument("--dr0", type=float, default=None)
    g.add_argument("--az-carrier", type=float, default=None, help="rad per px of azimuth offset")
    ap.add_argument("--gain", default="1", help="'auto' = per-block gain by coherence (local_gain), else a number")
    ap.add_argument("--gain-min-spread", type=float, default=1.0, help="test the gain only where the predicted spread exceeds this (px)")
    ap.add_argument("--gain-colblock", type=int, default=2048)
    ap.add_argument("--calib-blocks", type=int, default=24)
    ap.add_argument("--calib-samples", type=int, default=2048)
    ap.add_argument("--crop", nargs=4, type=int, default=None, metavar=("L0", "S0", "NL", "NS"),
                    help="report coherence before/after on this crop instead of writing a file")
    ap.add_argument("--crop-npz", default=None, help="save the crop's coherence before/after here")
    ap.add_argument("--strip", type=int, default=4096)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--slant-range-m", type=float, default=9.7e5)
    ap.add_argument("--sat-height-m", type=float, default=7.47e5)
    ap.add_argument("--selftest", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.scratch:
        raise SystemExit("--scratch is required unless --selftest")
    sec_in = os.path.join(a.scratch, a.sec)
    shape = tuple(a.shape) if a.shape else _shape_from_vrt(sec_in)
    if shape is None:
        raise SystemExit(f"no --shape and no {sec_in}.vrt to read it from")
    geom = lattice_geom(a.rbsheet, a.rubbersheet, a.search, a.winsize, a.skip, a.az_spacing,
                        sharp_median=tuple(a.sharp_median), steer_exclude_px=a.steer_exclude_px,
                        steer_cutoff=a.steer_cutoff, steer_aniso=a.steer_aniso)
    if geom["az_spacing"] is None:
        raise SystemExit("--az-spacing (or an --rbsheet npz carrying it) is required")
    prf, bw, carrier = a.prf, a.bw, a.carrier_hz
    if a.sec_h5 and (prf is None or bw is None or (carrier is None and a.az_carrier is None)):
        p, dc, b = rslc_band(a.sec_h5)
        prf = prf or p; bw = bw or b
        if carrier is None and a.az_carrier is None:
            carrier = float(fold(dc, prf))
    if carrier is None and a.az_carrier is not None and prf is not None:
        carrier = a.az_carrier * prf / (2.0 * np.pi)
    if prf is None or bw is None or carrier is None:
        raise SystemExit("need --prf, --bw and --carrier-hz (or --sec-h5 / --az-carrier)")
    print(f"[refocus] PRF {prf:.2f} Hz, bandwidth {bw:.1f} Hz, carrier {carrier:+.1f} Hz, "
          f"lattice origin {geom['origin']} skip {geom['skip']}, az spacing {geom['az_spacing']:.4f} m")

    need_pair = a.A_km == "auto" or a.crop is not None or a.gain == "auto"
    pair = None
    if need_pair:
        miss = [k for k in ("wavelength", "range_spacing", "dr0", "az_carrier") if getattr(a, k) is None]
        if miss:
            raise SystemExit(f"--A-km auto / --crop need {', '.join('--' + k.replace('_', '-') for k in miss)}")
        pair = Pair(a.scratch, shape, a.wavelength, a.range_spacing, a.dr0, a.az_carrier, a.sec)

    calib = None
    if a.A_km == "auto":
        A, gain, calib = calibrate(pair, geom, prf, bw, carrier, a.block, a.calib_samples,
                                   a.calib_blocks, hop=a.hop)
        if gain < 0.01:
            print(f"[refocus] verdict: gain {gain:+.3f} < 0.01 -- nothing to refocus; no output written")
            return 0
    else:
        A = float(a.A_km)
    sp = predicted_spread(geom, A)
    if a.wavelength:
        h, L_s = implied_height(A, a.wavelength, a.slant_range_m, bw,
                                geom["az_spacing"], prf, a.sat_height_m)
        print(f"[refocus] A = {A:g} km (synthetic aperture ~{L_s / 1e3:.1f} km -> "
              f"ionosphere ~{h / 1e3:.0f} km)")
    else:
        print(f"[refocus] A = {A:g} km")
    print(f"[refocus] predicted group-shift spread across the band: "
          f"p99 {np.nanpercentile(sp, 99):.2f} px, max {np.nanmax(sp):.2f} px, "
          f"{100 * (sp > 1).mean():.2f} % of the lattice above 1 px")

    if a.crop is not None:
        L0, S0, nl, ns = a.crop
        x, y = pair.crop(L0, S0, nl, ns)
        c0 = multilook_coh(x, y)
        gain = None
        if a.gain == "auto":
            wg, rc, cc, _ = local_gain(x, y, L0, S0, geom, A, prf, bw, carrier, a.block, a.gain_colblock,
                                       a.gain_min_spread, workers=a.workers)
            gain = (wg, rc, cc)
        elif float(a.gain) != 1.0:
            gain = (np.full((1, 1), float(a.gain)), np.array([L0 + nl / 2.0]), np.array([S0 + ns / 2.0]))
        yy = refocus_array(y, L0, S0, geom, A, prf, bw, carrier, a.block, a.hop, workers=a.workers, gain=gain)
        c1 = multilook_coh(x, yy)
        o_r, s_r = geom["origin"][0], geom["skip"][0]
        print(f"[refocus] crop lines {L0}..{L0 + nl} samples {S0}..{S0 + ns}: mean coherence "
              f"{c0.mean():.4f} -> {c1.mean():.4f}")
        print("  lattice row | coh before -> after (10 multilook rows each)")
        for i in range(0, c0.shape[0] - 9, 10):
            gtxt = ""
            if gain is not None:
                gv = gain_at(gain, L0 + 24 * i + 120, S0 + np.arange(0, ns, 256))[0]
                gtxt = "   gain " + " ".join(f"{v:.2f}" for v in gv)
            print(f"  {(L0 + 24 * i - o_r) / s_r:8.0f} | {c0[i:i + 10].mean():.3f} -> {c1[i:i + 10].mean():.3f}{gtxt}")
        if a.crop_npz:
            np.savez(a.crop_npz, before=c0, after=c1, crop=np.array(a.crop), A_km=A)
        return 0

    sec_out = os.path.join(a.scratch, a.out)
    gain = None
    if a.gain == "auto":
        gain = local_gain_file(pair.kw, shape, geom, A, prf, bw, carrier, a.block, a.gain_colblock,
                               a.gain_min_spread, a.workers, state_dir=os.path.join(a.scratch, "geo2rdr"))
    elif float(a.gain) != 1.0:
        gain = (np.full((1, 1), float(a.gain)), np.array([shape[0] / 2.0]), np.array([shape[1] / 2.0]))
    refocus_file(sec_in, sec_out, shape, geom, A, prf, bw, carrier, a.block, a.hop, a.strip,
                 a.workers, state_dir=os.path.join(a.scratch, "geo2rdr"),
                 extra_state={"calibration": calib, "implied_iono_height_m": h, "gain": a.gain,
                              "gain_min_spread": a.gain_min_spread, "gain_colblock": a.gain_colblock}, gain=gain)
    return 0


if __name__ == "__main__":
    sys.exit(main())
