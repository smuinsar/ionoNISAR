"""Ionospheric phase screen from a dense azimuth offset field."""

from __future__ import annotations

import os

import numpy as np


def nan_lowpass(v, sig_a, sig_r):
    """Gaussian low-pass that ignores NaN, normalised against the same filter on the weight."""
    from scipy.ndimage import gaussian_filter

    w = np.isfinite(v).astype(np.float32)
    s = gaussian_filter(np.where(w > 0, v, 0.0).astype(np.float32), (sig_a, sig_r))
    n = gaussian_filter(w, (sig_a, sig_r))
    return np.where(n > 0.05, s / np.maximum(n, 1e-6), np.nan)


def nan_lowpass_flat(v, sig_a, sig_r, passes=1):
    """Low-pass with a FLAT PASSBAND, built from the same NaN-aware Gaussian."""
    if passes <= 1:
        return nan_lowpass(v, sig_a, sig_r)
    fin = np.isfinite(v)
    y = nan_lowpass(v, sig_a, sig_r)
    reach = np.isfinite(y)                    # where the kernel had enough weight at all
    for _ in range(int(passes) - 1):
        r = np.where(fin & reach, v - np.where(reach, y, 0.0), np.nan)
        c = nan_lowpass(r, sig_a, sig_r)
        y = np.where(reach & np.isfinite(c), y + c, y)
    return np.where(reach, y, np.nan)


def integrate_alongtrack(g_ds, cut_km=0.0, row_m=None):
    """Invert the along-track difference operator, optionally with a damping term."""
    if not cut_km or cut_km <= 0 or row_m is None:
        return np.cumsum(g_ds, axis=0)
    n = g_ds.shape[0]
    ext = np.concatenate([g_ds, -g_ds[::-1]], axis=0)          # integral becomes periodic
    N = ext.shape[0]
    k = np.fft.fftfreq(N)[:, None]
    D = np.exp(2j * np.pi * k) - 1.0
    mu = 2.0 * np.pi / (cut_km * 1000.0 / row_m)               # rows -> damping constant
    H = np.conj(D) / (np.abs(D) ** 2 + mu ** 2)
    phi = np.real(np.fft.ifft(np.fft.fft(ext, axis=0) * H, axis=0))[:n]
    return phi


def carry_across_range(grad):
    """Interpolate the integrand across mask holes ALONG RANGE, before the integral."""
    out = np.array(grad, dtype=np.float64, copy=True)
    j = np.arange(out.shape[1])
    for i, row in enumerate(out):
        m = np.isfinite(row)
        if m.sum() > 1:
            out[i] = np.interp(j, j[m], row[m])
        elif m.sum() == 1:
            out[i] = row[m][0]
        else:
            out[i] = 0.0
    return out


def platform_speed(ref_meta):
    """Spacecraft speed at mid-scene, m/s, from the orbit."""
    rg = ref_meta["rg"]
    _, v = ref_meta["orbit"].interpolate(rg.sensing_start + 0.5 * rg.length / rg.prf)
    return float(np.linalg.norm(np.asarray(v)))


def pierce_sweep(R, shell_km, ref_meta):
    """kappa = dx_pierce / dx_satellite, per range column.  Dimensionless, in (0, 1)."""
    if not shell_km or shell_km <= 0:
        return None
    h = float(shell_km) * 1000.0
    orb = ref_meta["orbit"]
    rgp = ref_meta["rg"]
    pos, _ = orb.interpolate(rgp.sensing_start + 0.5 * rgp.length / rgp.prf)
    rs = float(np.linalg.norm(np.asarray(pos)))              # satellite radius
    R = np.asarray(R, np.float64)
    # Local Earth radius under the scene, from the satellite radius and the mid slant range:
    # solve for the target radius that the DEM-referenced geometry implies is unnecessary --
    # the ellipsoid value at this latitude is good to a few km and kappa is insensitive to it.
    lat = np.arcsin(np.clip(float(np.asarray(pos)[2]) / rs, -1.0, 1.0))
    a_e, b_e = 6378137.0, 6356752.314245
    rt = a_e * b_e / np.sqrt((a_e * np.sin(lat)) ** 2 + (b_e * np.cos(lat)) ** 2)
    # P(t) = S + t (T - S); |P| = rt + h.  t is the fraction from the SATELLITE.
    sdott = (rs ** 2 + rt ** 2 - R ** 2) / 2.0
    aq, bq, cq = R ** 2, sdott - rs ** 2, rs ** 2 - (rt + h) ** 2
    disc = bq * bq - aq * cq
    if np.any(disc < 0):
        raise ValueError(f"no ionospheric pierce point at {shell_km:g} km for this geometry")
    t = (-bq - np.sqrt(disc)) / aq
    return 1.0 - t                                            # dP/dS


def integration_geometry(args, ref_meta, win, nwa):
    """The lattice terms the integral needs: metres per row, per column, and slant range."""
    rg = ref_meta["rg"]
    j0 = win[1] + args.search[1] + args.winsize[1] // 2
    vs = platform_speed(ref_meta)
    R = (rg.starting_range
         + (j0 + args.skip[1] * np.arange(nwa)) * rg.range_pixel_spacing)
    shell = float(getattr(args, "iono_shell_km", 0.0) or 0.0)
    kappa = pierce_sweep(R, shell, ref_meta)
    da = args.skip[0] * args.az_spacing                       # m of ground per lattice row
    if kappa is None:
        # integrate over platform travel, i.e. assume kappa = v_g/v_s
        ds = args.skip[0] * vs / float(rg.prf)
        ds = np.full(int(nwa), ds, np.float64)
    else:
        # da / kappa, per column.  Written as an effective step so the cumsum below is
        # unchanged and screen_sigma keeps driving the identical operator.
        ds = da / kappa
    return dict(
        da=da,
        dr=args.skip[1] * float(rg.range_pixel_spacing),      # m of slant range per column
        vs=vs,
        ds=ds,                                                # m per row, PER RANGE COLUMN
        kappa=kappa,
        shell_km=shell,
        lam=float(rg.wavelength),
        R=R)


def integrate_field(args, geom, v_m, keep):
    """Azimuth field in METRES -> phase screen in radians.  LINEAR in `v_m` given `keep`."""
    sm = nan_lowpass_flat(np.where(keep, v_m, np.nan),
                          args.iono_screen_km * 1000.0 / geom["da"],
                          args.iono_screen_km * 1000.0 / geom["dr"],
                          int(getattr(args, "iono_screen_passes", 1) or 1))
    grad = carry_across_range(-(4.0 * np.pi) * sm / (geom["lam"] * geom["R"][None, :]))
    phi = integrate_alongtrack(grad * geom["ds"],
                               float(getattr(args, "iono_integrate_km", 0.0) or 0.0),
                               float(np.mean(geom["ds"])))
    phi = phi - phi.mean(axis=0, keepdims=True)
    # the gain is part of the operator, so the Monte Carlo has to carry it too
    return phi * float(getattr(args, "iono_screen_gain", 1.0) or 1.0)


def residual_retention(corr_cells, det=9, n=384, seed=0):
    """Fraction of a correlated field's std that survives a `det`-cell median detrend."""
    from scipy.ndimage import median_filter, uniform_filter

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, n)).astype(np.float32)
    if corr_cells > 1:
        x = uniform_filter(x, int(corr_cells))
    x /= x.std()
    r = x - median_filter(x, size=det)
    return float(1.4826 * np.median(np.abs(r)))


def calibrate_level(field, mask, shape, corr_cells, label=""):
    """Scale a relative noise `shape` to the field's own scatter.  Returns sigma, same units."""
    from scipy.ndimage import median_filter

    m = mask & np.isfinite(field) & np.isfinite(shape) & (shape > 0)
    if not m.any():
        return np.full(field.shape, np.nan, np.float64)
    det = max(9, int(2 * corr_cells) | 1)
    resid = field - median_filter(np.where(m, field, 0.0).astype(np.float32), size=det)
    ret = residual_retention(corr_cells, det=det)
    obs = 1.4826 * float(np.median(np.abs(resid[m]))) / max(ret, 1e-3)
    k = obs / max(float(np.median(shape[m])), 1e-12)
    print(f"[sigma] {label or 'field'}: {det}-cell residual {obs * ret:.4f}, corrected for "
          f"the {corr_cells:g}-cell correlation (retention {ret:.2f}) to {obs:.4f} px "
          f"over {100 * m.mean():.1f} % of the lattice")
    return k * shape


def screen_sigma(args, ref_meta, win, az, keep, sig_px, corr_cells, n_real=32, seed=0,
                 noise_gen=None, post=None):
    """1-sigma PRECISION of the integrated screen, in radians, by Monte Carlo."""
    from scipy.ndimage import uniform_filter

    geom = integration_geometry(args, ref_meta, win, az.shape[1])
    if not np.isfinite(sig_px).any():
        return np.full(az.shape, np.nan, np.float32)
    print(f"[sigma] per-cell azimuth noise: median "
          f"{np.nanmedian(sig_px[keep]):.4f} px = "
          f"{np.nanmedian(sig_px[keep]) * args.az_spacing:.3f} m over the integrated mask")

    rng = np.random.default_rng(seed)
    fill = float(np.nanmedian(sig_px))
    smap = np.where(np.isfinite(sig_px), sig_px, fill).astype(np.float32)

    # `corr_cells` may be one width or (rows, cols): on an interleaved sub-lattice the
    # windows no longer overlap along track but still do across it
    cw = (corr_cells, corr_cells) if np.isscalar(corr_cells) else tuple(corr_cells)
    cw = tuple(max(1, int(round(c))) for c in cw)

    def default_gen(r):
        n = r.standard_normal(az.shape).astype(np.float32)
        if max(cw) > 1:
            n = uniform_filter(n, cw)
            n /= n.std()
        return n * smap

    # `noise_gen` exists for fields whose noise was shaped by an operator no single
    # correlation width describes -- MAI standalone interpolates 79 % of its cells from the
    # 21 % it measured, and interpolation both SMOOTHS the noise and reduces its variance.
    # Generating through that operator gets both exactly; assuming a width gets neither.
    gen = noise_gen or default_gen
    acc = np.zeros(az.shape, np.float64)
    acc2 = np.zeros(az.shape, np.float64)
    pacc = np.zeros(az.shape, np.float64) if post is not None else None
    pacc2 = np.zeros(az.shape, np.float64) if post is not None else None
    for i in range(n_real):
        phi = integrate_field(args, geom, gen(rng) * args.az_spacing, keep)
        acc += phi
        acc2 += phi * phi
        if post is not None:
            q = post(phi)
            pacc += q
            pacc2 += q * q
    var = np.maximum(acc2 / n_real - (acc / n_real)**2, 0.0) * (n_real / (n_real - 1.0))
    sig = np.sqrt(var).astype(np.float32)
    print(f"[sigma] {n_real} realisations through the integrator: screen sigma median "
          f"{np.nanmedian(sig[keep]):.3f} rad, p99 {np.nanpercentile(sig[keep], 99):.3f} rad")
    if post is None:
        return np.where(keep, sig, np.nan).astype(np.float32)
    pvar = np.maximum(pacc2 / n_real - (pacc / n_real)**2, 0.0) * (n_real / (n_real - 1.0))
    psig = np.sqrt(pvar).astype(np.float32)
    print(f"[sigma] after `post`: median {np.nanmedian(psig[keep]):.4f} rad, "
          f"p99 {np.nanpercentile(psig[keep], 99):.4f} rad")
    return (np.where(keep, sig, np.nan).astype(np.float32),
            np.where(keep, psig, np.nan).astype(np.float32))


def screen_from_offsets(args, ref_meta, win, az, keep, filled=None, az_filled=None):
    """The phase screen on the offset lattice, in radians.  Returns (screen, valid)."""
    from scipy.ndimage import distance_transform_edt

    geom = integration_geometry(args, ref_meta, win, az.shape[1])
    da, dr, ds, R, vs = geom["da"], geom["dr"], geom["ds"], geom["R"], geom["vs"]
    rg = ref_meta["rg"]

    mode = str(getattr(args, "iono_screen_integrand", "raw") or "raw")
    if mode == "filled" and az_filled is not None:
        # integrate the filled surface wherever it is defined, not just the measured cells
        src = np.asarray(az_filled, np.float64)
        use = np.isfinite(src)
        v = np.where(use, src, np.nan) * args.az_spacing
        print(f"[iono] integrand: ANISOTROPICALLY FILLED field over {100*use.mean():.1f} % "
              f"of the lattice (measured {100*np.asarray(keep, bool).mean():.1f} %); "
              f"carry_across_range will have little left to do, and screen_sigma is no "
              f"longer exact -- see the docstring")
    else:
        if mode == "filled":
            print("[iono] --iono-screen-integrand filled requested but no filled field was "
                  "passed; falling back to raw")
        v = np.where(keep, az, np.nan).astype(np.float64) * args.az_spacing    # metres
    passes = int(getattr(args, "iono_screen_passes", 1) or 1)
    sm = nan_lowpass_flat(v, args.iono_screen_km * 1000.0 / da,
                          args.iono_screen_km * 1000.0 / dr, passes)
    q = sm[np.isfinite(sm)]
    print(f"[iono] azimuth field low-passed at {args.iono_screen_km:g} km"
          f"{'' if passes <= 1 else f', {passes} passes (flat passband)'}: "
          f"std {q.std():.3f} m, p1..p99 {np.percentile(q, 1):+.2f} .. "
          f"{np.percentile(q, 99):+.2f} m over {100 * np.isfinite(sm).mean():.1f} % of "
          f"the lattice")

    # The integral.  Sign is negative, per the regression against this pair's own phase --
    # and per the derivation in the module docstring, which agrees with it.  The step is
    # da/kappa, not da: the measured shift responds to the pierce point's sweep rate, not the
    # ground's.  See pierce_sweep for why, and for the measurement that recovered it.
    kap = geom.get("kappa")
    if kap is None:
        print(f"[iono] integrating over {float(np.mean(ds)):.4f} m of platform travel per row "
              f"(v_s {vs:.1f} m/s), not {da:.4f} m of ground: "
              f"x{vs / (args.az_spacing * float(rg.prf)):.3f}   "
              f"[--iono-shell-km 0 puts the screen on the ground instead]")
    else:
        print(f"[iono] ionospheric shell {geom['shell_km']:g} km -> pierce sweep kappa "
              f"{float(np.min(kap)):.4f}..{float(np.max(kap)):.4f} near-to-far; integrating "
              f"over {float(np.mean(ds)):.2f} m per row against {da:.2f} m of ground "
              f"(x{float(np.mean(ds)) / da:.3f}), where the old v_s/v_g step was "
              f"x{vs / (args.az_spacing * float(rg.prf)):.3f} -- a factor "
              f"{float(np.mean(ds)) / (args.skip[0] * vs / float(rg.prf)):.3f} more screen")
    grad = -(4.0 * np.pi) * sm / (float(rg.wavelength) * R[None, :])
    held = ~np.isfinite(grad)
    grad = carry_across_range(grad)
    print(f"[iono] integrand carried across range into {100 * held.mean():.1f} % of the "
          f"lattice the low-pass could not reach; p1..p99 there "
          f"{np.percentile(grad[held], 1) if held.any() else 0:+.2e} .. "
          f"{np.percentile(grad[held], 99) if held.any() else 0:+.2e} rad/m "
          f"(zeroing it instead is what put the constant-range seam in the product)")
    cut = float(getattr(args, "iono_integrate_km", 0.0) or 0.0)
    phi = integrate_alongtrack(grad * ds, cut, float(np.mean(ds)))
    if cut > 0:
        print(f"[iono] along-track integration damped above {cut:g} km "
              f"(Tikhonov); a plain cumsum turns the integrand's noise into a random walk "
              f"and beyond ~16 km that screen is worse than no screen")
    phi -= phi.mean(axis=0, keepdims=True)          # the per-column datum; see the docstring

    # The along-track derivative of the interferogram can be written as a linear function
    # of the azimuth-shift observable, dphi/daz = alpha * phi_shift + beta, and that is what
    # is integrated.  Everything above this line is that integrand with alpha pinned to its
    # theoretical value of 1, so `gain` is alpha and 1.0 leaves the screen unscaled.  The
    # constant beta is not carried: it is per-column, and the per-column datum on the line
    # above already removes any such constant.  The gain is an empirical calibration, so it
    # is off by default.
    gain = float(getattr(args, "iono_screen_gain", 1.0) or 1.0)
    if gain != 1.0:
        phi *= gain
        print(f"[iono] gain alpha = {gain:g} applied to the integrated screen "
              f"(1.0 = the theoretical constant alone)")

    # VALIDITY, and only validity: a gap narrower than --iono-screen-max-gap has had its
    # integrand invented by carry_across_range, so `filled` is what says where the screen
    # was measured rather than continued.  It does NOT mask the screen -- blanking at every
    # glacier rim injects a step the interferogram does not have, and masking the integrand
    # before the cumsum propagates a hole down the whole column.
    gap = distance_transform_edt(~keep, sampling=(da, dr))
    reach = np.isfinite(sm) & (gap <= args.iono_screen_max_gap * 1000.0)
    valid = reach
    if filled is not None and np.shape(filled) == np.shape(valid):
        drop = np.asarray(filled, bool)
        # BOTH numbers, always, because they answer different questions and quoting one as
        # "coverage" has caused real confusion:
        #   reach  -- where the screen is a LOW-PASS OF MEASUREMENTS rather than a flat
        #             continuation.  This is what the correction is built from.
        #   valid  -- reach minus every cell whose offset came out of the hole-fill.
        # The second reads below the first, because the 2 km low-pass
        # reaches ~28 points past `keep` and the fill flag then revokes exactly that.
        #
        # It also makes offsets and MAI comparable at last: MAI calls this function with no
        # `filled` at all, so it reports the `reach` number
        # -- 89 % against offsets' 61 % on the same field.  That gap was a reporting policy,
        # not a difference in the measurement, and it was read as one more than once.
        if getattr(args, "iono_screen_count_filled", False):
            print(f"[iono] validity: {100 * reach.mean():.1f} % reached by the "
                  f"{args.iono_screen_max_gap:g} km gap test; counting filled cells as valid "
                  f"(--iono-screen-count-filled), so valid = reach.  Without it this would "
                  f"read {100 * (reach & ~drop).mean():.1f} %")
        else:
            valid = reach & ~drop
            print(f"[iono] validity: {100 * reach.mean():.1f} % passed the "
                  f"{args.iono_screen_max_gap:g} km gap test, {100 * valid.mean():.1f} % "
                  f"after dropping cells whose azimuth offset came out of the fill "
                  f"(the screen itself is unchanged; --iono-screen-count-filled keeps them)")
    phi = phi.astype(np.float32)
    lo, hi = np.percentile(phi[valid], 1), np.percentile(phi[valid], 99)
    print(f"[iono] screen: p1..p99 {lo:+.1f} .. {hi:+.1f} rad "
          f"({(hi - lo) / (2 * np.pi):.1f} fringes), measured on {100 * valid.mean():.1f} % "
          f"of the lattice (gap <= {args.iono_screen_max_gap:g} km); outside that it is the "
          f"flat continuation and the validity layer says so")
    return phi, valid


def fit_screen_gain(unc, coh, screen, block=16, coh_min=0.3, w_min=0.2):
    """Fit the factor the applied screen is short by, against the pair's own phase."""
    unc = np.asarray(unc, np.float64)
    coh = np.asarray(coh, np.float64)
    scr = np.asarray(screen, np.float64)
    B = int(block)
    ny, nx = (unc.shape[0] // B) * B, (unc.shape[1] // B) * B
    blk = lambda v: v[:ny, :nx].reshape(ny // B, B, nx // B, B).mean(axis=(1, 3))

    w = np.where(np.isfinite(coh), np.clip(coh, 0.0, 1.0), 0.0)
    cB = blk(w * np.exp(1j * np.where(np.isfinite(unc), unc, 0.0)))
    wB = blk(w)
    ph, cohB = np.angle(cB), np.abs(cB) / np.maximum(wB, 1e-9)
    sB = blk(np.where(np.isfinite(scr), scr, np.nan))

    dI = np.angle(np.exp(1j * (ph[1:, :] - ph[:-1, :])))
    dS = sB[1:, :] - sB[:-1, :]
    g = ((cohB[1:, :] > coh_min) & (cohB[:-1, :] > coh_min)
         & (wB[1:, :] > w_min) & (wB[:-1, :] > w_min)
         & np.isfinite(dI) & np.isfinite(dS))
    n = int(g.sum())
    if n < 100 or not np.isfinite(dS[g]).any() or dS[g].std() == 0:
        return float("nan"), float("nan"), n
    x, y = dS[g], dI[g]
    (alpha, _beta), *_ = np.linalg.lstsq(np.vstack([x, np.ones_like(x)]).T, y, rcond=None)
    return float(alpha), float(np.corrcoef(x, y)[0, 1]), n


# The settings that DEFINE a screen, stored with it so a reuse gate can tell whether the
# screen on disk is the one being asked for.
SCREEN_PARAMS = ("iono_screen_km", "iono_screen_passes", "iono_screen_gain",
                 "iono_shell_km", "iono_screen_max_gap")


def screen_params(args):
    """The defining settings, as a dict of 0-d arrays ready for savez."""
    out = {}
    for k in SCREEN_PARAMS:
        v = getattr(args, k, None)
        if v is not None:
            out["p_" + k] = np.array(float(v))
    return out


def screen_mismatch(path, args):
    """Which stored settings differ from the requested ones.  [] means safe to reuse."""
    try:
        z = np.load(path)
    except Exception as e:
        return [f"unreadable ({e})"]
    diffs, unknown = [], []
    for k in SCREEN_PARAMS:
        want = getattr(args, k, None)
        if want is None:
            continue
        key = "p_" + k
        if key not in z.files:
            if k == "iono_shell_km":
                # a screen with no p_iono_shell_km carries no pierce-point term, so its
                # shell is 0 rather than unknown; the other three have stable defaults
                if not np.isclose(0.0, float(want), rtol=0, atol=1e-12):
                    diffs.append(f"{k}: on disk 0 (predates the term), requested {float(want):g}")
            else:
                unknown.append(k)
        elif not np.isclose(float(z[key]), float(want), rtol=1e-9, atol=1e-12):
            diffs.append(f"{k}: on disk {float(z[key]):g}, requested {float(want):g}")
    if unknown and not diffs:
        print(f"[iono] NOTE: {os.path.basename(path)} predates the provenance keys "
              f"({', '.join(unknown)} unknown); reusing it on trust")
    return diffs


def save_screen(args, phi, valid, win):
    """Write the screen on its own lattice, for the interferogram stage to pick up."""
    os.makedirs(args.scratch, exist_ok=True)
    path = os.path.join(args.scratch, f"iono_screen_{args.tag}.npz")
    np.savez_compressed(path, screen=phi, valid=valid,
                        window=np.array(win), skip=np.array(args.skip),
                        winsize=np.array(args.winsize), search=np.array(args.search),
                        **screen_params(args))
    print(f"[iono] wrote {path}")
    return path


def to_lattice(path, ref_origin, looks, shape):
    """Resample a saved screen onto an interferogram's multilook lattice.  Returns radians."""
    from scipy.ndimage import map_coordinates

    d = np.load(path)
    phi, valid = d["screen"], d["valid"]
    win, skip, winsize, search = d["window"], d["skip"], d["winsize"], d["search"]
    la, lr = looks
    a0, r0 = ref_origin

    az_ml = a0 + (la - 1) / 2.0 + la * np.arange(shape[0])
    rg_ml = r0 + (lr - 1) / 2.0 + lr * np.arange(shape[1])
    az_off = win[0] + search[0] + winsize[0] // 2 + skip[0] * np.arange(phi.shape[0])
    rg_off = win[1] + search[1] + winsize[1] // 2 + skip[1] * np.arange(phi.shape[1])

    J, I = np.meshgrid(np.interp(rg_ml, rg_off, np.arange(phi.shape[1])),
                       np.interp(az_ml, az_off, np.arange(phi.shape[0])))
    out = map_coordinates(np.nan_to_num(phi), [I, J], order=1, mode="nearest").astype(np.float32)
    ok = map_coordinates(valid.astype(np.float32), [I, J], order=1, mode="nearest") > 0.99
    if ok.any():
        print(f"[iono] screen on the {shape[0]} x {shape[1]} multilook lattice: p1..p99 "
              f"{np.percentile(out[ok], 1):+.1f} .. {np.percentile(out[ok], 99):+.1f} rad, "
              f"measured on {100 * ok.mean():.1f} % of it")
    else:
        print("[iono] screen was measured nowhere on the multilook lattice")
    return out, ok
