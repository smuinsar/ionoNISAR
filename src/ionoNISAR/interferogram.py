"""Interferogram from the coregistered SLC pair: looks, flattening, screen, unwrap, geocode."""
from __future__ import annotations

import argparse
import os

import numpy as np

from ._utils import raster as IU


def carrier_from_lut(a, nr):
    """Per-pixel 2*pi*fd/PRF for the secondary, in place of one number at the crop centre."""
    from . import coregister as O
    from scipy.interpolate import RegularGridInterpolator

    sec = O.load_slc(a.sec_h5, a.freq, a.pol)
    rg = sec["rg"]
    lut = O.sampled_doppler(sec["dop"], rg.prf)
    ax_t, ax_r = np.asarray(lut.y_axis), np.asarray(lut.x_axis)
    itp = RegularGridInterpolator((ax_t, ax_r), np.asarray(lut.data),
                                  bounds_error=False, fill_value=None)
    prf, t0 = float(rg.prf), float(rg.sensing_start)
    R = (float(rg.starting_range)
         + (a.sec_origin[1] + np.arange(nr)) * float(rg.range_pixel_spacing))
    fd = itp(np.stack([np.full(nr, t0 + a.sec_origin[0] / prf), R], -1))
    print(f"azimuth carrier from the LUT: folded Doppler {np.asarray(lut.data).min():+.2f} "
          f".. {np.asarray(lut.data).max():+.2f} Hz over the annotation, "
          f"{fd.min():+.2f} .. {fd.max():+.2f} Hz across this crop's near line "
          f"-> {2 * np.pi * (fd.max() - fd.min()) / prf:.4f} rad/px of spread in range alone")

    def carrier(i0, n):
        t = t0 + (a.sec_origin[0] + i0 + np.arange(n)) / prf
        TT, RR = np.meshgrid(t, R, indexing="ij")
        v = itp(np.stack([TT.ravel(), RR.ravel()], -1)).reshape(n, nr)
        return (2.0 * np.pi / prf) * v

    return carrier


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default="offsets_scratch/ref.c8")
    ap.add_argument("--sec", default="offsets_scratch/sec_coreg.c8",
                    help="the COREGISTERED secondary (same grid as --ref)")
    ap.add_argument("--shape", nargs=2, type=int, required=True,
                    metavar=("LINES", "SAMPLES"),
                    help="rows and columns of the flat binary pair")
    ap.add_argument("--looks", nargs=2, type=int, default=(24, 16), metavar=("AZ", "RG"),
                    help="multilook factors (default 24 16: 107 m along track x 50 m slant range, 68-92 m "
                         "on the ground, on the 40 MHz modes; the 20 MHz mode 2005 uses 24 8 for the same cell)")
    ap.add_argument("--block", type=int, default=2400, help="lines read per block")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--tag", required=True, help="the pair's product tag")
    ap.add_argument("--coh-min", type=float, default=0.2,
                    help="blank phase below this coherence in the quicklook (default 0.2)")

    g = ap.add_argument_group("flattening (flat earth + topography)")
    g.add_argument("--flatten", action=argparse.BooleanOptionalAction, default=True,
                   help="remove the geometric phase 4*pi/lambda * (R_sec - R_ref) using the "
                        "SAME geo2rdr offsets the coregistration used (default on).  Those "
                        "offsets already carry orbit and DEM, so one subtraction takes out "
                        "the flat-earth ramp and the topographic fringes together -- this "
                        "is what resamp_slc's flatten=True does, minus a 45 GB rewrite")
    g.add_argument("--range-off", default="offsets_scratch/geo2rdr/range.off")
    g.add_argument("--interp-bias", default=None, metavar="NPZ",
                   help="the resampler bias curve for this band: the range "
                        "resampler's FULL-BAND phase bias as a function of the fractional "
                        "range shift, taken out of the secondary at frac(range.off).  Needs "
                        "--flatten's range.off.  It matters most for band B, whose bias "
                        "reaches the split screen amplified by 1/|det|")
    g.add_argument("--flatten-sign", type=float, default=1.0, choices=[1.0, -1.0],
                   help="sign of the model phase; +1 for the usual exp(-j4piR/lambda) SLC "
                        "convention (verified against the data, see below)")
    # Both come off the RSLC the caller already has open; a default here would be a second,
    # silent source of truth for the flattening.
    g.add_argument("--wavelength", type=float, required=True)
    g.add_argument("--range-spacing", type=float, required=True)
    g.add_argument("--dr0", type=float, default=None, metavar="M",
                   help="starting-range difference (secondary crop minus reference crop) in "
                        "metres; default reads it from the SLC pair via --ref-h5/--sec-h5")
    g.add_argument("--ref-h5", default=None, help="reference focused SLC, for --dr0")
    g.add_argument("--sec-h5", default=None, help="secondary focused SLC, for --dr0")
    g.add_argument("--freq", default="A", choices=["A", "B"],
                   help="which frequency --ref-h5 holds.  Geocoding reads the reference's "
                        "radar grid back out of it, and a crop written by this chain "
                        "keeps ONE frequency, so asking a B crop for A raises map::at.  B "
                        "also carries its own wavelength and range spacing, so this is not "
                        "only about the group existing (default A)")
    g.add_argument("--ref-origin", nargs=2, type=int, required=True,
                   metavar=("AZ0", "RG0"), help="origin of ref.c8 in the reference frame")
    g.add_argument("--sec-origin", nargs=2, type=int, default=(0, 0), metavar=("AZ0", "RG0"),
                   help="origin of the secondary crop in the secondary frame")
    g.add_argument("--az-carrier", type=float, default=None, metavar="RAD_PER_PX",
                   help="2*pi*fd/PRF for the SECONDARY, with fd folded into the sampled "
                        "band.  These SLCs are not at zero Doppler -- a few hundred Hz "
                        "against a PRF of order a kHz is a couple of radians per azimuth "
                        "pixel -- so a target that resampling moved by delta comes back "
                        "carrying the "
                        "carrier at its NEW position and the interferogram keeps -psi'*delta "
                        "on top of the propagation phase.  That is geometry, not ground: it "
                        "is proportional to the azimuth offset field, which here is "
                        "ionospheric, and it is 2.3 rad for every pixel of it.  Given with "
                        "--azimuth-off it is taken out, which is what re-ramping at the "
                        "OUTPUT position would have done.  It takes a further several "
                        "per cent off the along-track fringe rate.  NOTE it also removes "
                        "the (much smaller) real "
                        "sensitivity to along-track ground motion, so leave it off if that "
                        "is the signal being measured")
    g.add_argument("--azimuth-off", default=None,
                   help="geo2rdr azimuth.off, the field --az-carrier multiplies")
    g.add_argument("--az-carrier-lut", action="store_true",
                   help="evaluate the carrier PER PIXEL from the secondary's Doppler LUT "
                        "instead of using the single --az-carrier number.  resamp_slc "
                        "re-ramps at the input position with the LOCAL Doppler, so what is "
                        "left in ref*conj(sec) is c(x)*aoff(x); one number at the crop "
                        "centre removes only c_hat*aoff(x) and leaves (c(x)-c_hat)*aoff(x). "
                        "The annotated Doppler varies by a few Hz over a frame, and aoff "
                        "is of order a hundred px at the crop centre, so several radians -- "
                        "a whole fringe -- can stay in the "
                        "interferogram.  It matters most for split spectrum: band B's aoff "
                        "is +96.02 px (the ampcor pad is 128 for A and 96 for B, and B is "
                        "coregistered geometrically so it never needed one), and the Doppler "
                        "ratio B/A is exactly 1.043987 pointwise, so the residual would be "
                        "absorbed as non-dispersive IF AND ONLY IF the two aoff fields were "
                        "equal.  They are not, and 0.51076*r_A + 11.61159*(r_A - r_B) is "
                        "~19 rad of screen.  Needs --sec-h5, --freq and --azimuth-off.  "
                        "--az-carrier is still required: it is the reference the residual "
                        "is reported against, and the fallback if the LUT cannot be read")
    g.add_argument("--pol", default="HH",
                   help="polarisation to open --sec-h5 with, for --az-carrier-lut")

    g = ap.add_argument_group("filtering and unwrapping (radar coordinates)")
    g.add_argument("--filter", action="store_true",
                   help="Goldstein adaptive spectral filter before unwrapping "
                        "(the raster helpers's goldstein_filter_fast)")
    g.add_argument("--filter-alpha", type=float, default=0.5,
                   help="filter strength, 0 = none, ~1 = strong (default 0.5)")
    g.add_argument("--filter-win", type=int, default=32, help="filter FFT size (default 32)")
    g.add_argument("--filter-coh-win", type=int, default=7,
                   help="coherence window inside the filter (default 7)")
    g.add_argument("--topo-phase", action="store_true",
                   help="estimate and remove a phase term proportional to ELEVATION -- "
                        "tropospheric vertical stratification.  Default off: it also "
                        "removes any real deformation that correlates with topography.  On "
                        "the term is of order a radian per km of height, i.e. about a "
                        "centimetre of differential delay per km of relief.  It is not a "
                        "geometric "
                        "residual: B_perp is -19.6 m here, so the height of ambiguity is "
                        "3150-3930 m and a DEM or baseline error large enough to explain "
                        "-0.64 rad/km would have to be 35 %% of the DEM or 7 m of baseline")
    g.add_argument("--topo-z", default="offsets_scratch/rdr2geo/z.rdr",
                   help="rdr2geo heights on the REFERENCE crop grid, sampled at the centre "
                        "of every look box (--coreg geometric writes it; --keep-scratch "
                        "keeps it)")
    g.add_argument("--iono-screen", default=None, metavar="NPZ",
                   help="ionospheric phase screen on the offset lattice, subtracted before "
                        "the filter and the unwrapping.  The uncorrected products are "
                        "written too, so the two can be compared without a re-run")
    g.add_argument("--iono-screen-calibrate", action="store_true",
                   help="fit the screen gain against this pair's own phase "
                        "and PRINT the gain the applied screen is short by.  Purely "
                        "diagnostic: nothing subtracted changes.  Feed the number "
                        "back through --iono-screen-gain when you want it applied")
    g.add_argument("--iono-calib-block", type=int, default=16, metavar="N",
                   help="multilook cells per side in that fit (default 16, ~1.7 km "
                        "along track here).  alpha is stable from 4 to 32")
    g.add_argument("--unwrap", action="store_true",
                   help="unwrap the (filtered) interferogram in radar coordinates")
    g.add_argument("--unwrap-method", default="snaphu",
                   choices=["snaphu", "phass", "icu"],
                   help="snaphu (default) via snaphu-py; phass and icu are isce3-native")
    g.add_argument("--unwrap-coh-thresh", type=float, default=0.2,
                   help="coherence below this is not unwrapped (default 0.2)")

    g = ap.add_argument_group("geocoding")
    g.add_argument("--geocode", action="store_true",
                   help="also geocode the interferogram with isce3 and plot it on the map "
                        "grid (needs --ref-h5 and --dem)")
    g.add_argument("--dem", default=None)
    g.add_argument("--posting", type=float, default=90.0, help="output posting in m")
    g.add_argument("--epsg", type=int, default=None, help="default: UTM of the scene centre")
    g.add_argument("--geogrid", nargs=7, default=None,
                   metavar=("X0", "Y0", "DX", "DY", "NX", "NY", "EPSG"),
                   help="geocode onto exactly this lattice, overriding --posting/--epsg.  "
                        "the coregistration passes the grid its offsets landed on, so "
                        "the interferogram and the offsets come out as the same raster and "
                        "differencing them needs no resampling.  Left to itself this script "
                        "fits a bbox to the multilook grid, which lands on a lattice of its "
                        "own -- half a cell off the 120 m stack here, so not even an integer "
                        "crop away")
    g.add_argument("--interp", default="BILINEAR",
                   choices=["SINC", "BILINEAR", "BICUBIC", "NEAREST", "BIQUINTIC"])
    g.add_argument("--scratch", default="offsets_scratch")
    g.add_argument("--mask-water", action="store_true",
                   help="blank ESA WorldCover permanent water in the geocoded products.  "
                        "Water decorrelates, so what is there is noise that looks like "
                        "fringes.  Ice is deliberately NOT masked: it is kept out of the "
                        "offset field because it moves, which says nothing about the "
                        "interferogram over it")
    g.add_argument("--water-year", type=int, default=2021, choices=[2020, 2021])
    g.add_argument("--cache-dir", default="cache",
                   help="where the WorldCover tiles --mask-water needs are cached")

    g = ap.add_argument_group("output size")
    g.add_argument("--keep-rdr-npz", action="store_true",
                   help="also write the radar-domain phase/coherence as ifg_rdr_<tag>.npz "
                        "(55-82 MB a rung).  Off by default: it is the same measurement as "
                        "the geocoded products, which is what everything downstream reads")
    g.add_argument("--no-complex-ifg", action="store_true",
                   help="skip ifg_geo_<tag>.tif, the geocoded COMPLEX interferogram "
                        "(35 MB a rung).  Its phase and coherence are written separately "
                        "and nothing here reads it back")
    a = ap.parse_args(argv)

    if a.flatten and a.dr0 is None:
        if not (a.ref_h5 and a.sec_h5):
            raise SystemExit("--flatten needs --dr0 or both --ref-h5 and --sec-h5")
        import h5py
        def start_range(p):
            with h5py.File(p, "r") as h:
                return float(h["radargrid"].attrs["starting_range"])
        a.dr0 = ((start_range(a.sec_h5) + a.sec_origin[1] * a.range_spacing)
                 - (start_range(a.ref_h5) + a.ref_origin[1] * a.range_spacing))
        print(f"starting-range difference (sec crop - ref crop): {a.dr0:+.4f} m")

    NA, NR = a.shape
    la, lr = a.looks
    ml_a, ml_r = NA // la, NR // lr
    print(f"{NA} x {NR} -> {ml_a} x {ml_r} at {la} x {lr} looks", flush=True)

    ref = np.memmap(a.ref, dtype=np.complex64, mode="r", shape=(NA, NR))
    sec = np.memmap(a.sec, dtype=np.complex64, mode="r", shape=(NA, NR))
    ifg = np.zeros((ml_a, ml_r), np.complex64)
    p1 = np.zeros((ml_a, ml_r), np.float32)
    p2 = np.zeros((ml_a, ml_r), np.float32)

    roff = None
    if a.flatten:
        # raw ENVI or compressed GeoTIFF -- see the raster helpers's open_raster
        roff = IU.open_raster(a.range_off, (NA, NR))
        j = np.arange(ml_r * lr, dtype=np.float64)
        print(f"flattening with {a.range_off} (lambda {a.wavelength:.6f} m)", flush=True)
    ib = None
    if a.interp_bias:
        if roff is None:
            raise SystemExit("--interp-bias needs --flatten's range.off: the bias is a "
                             "function of the fractional range offset")
        cb = np.load(a.interp_bias)
        ib = (np.asarray(cb["delta"], np.float64), np.asarray(cb["full_bias"], np.float64))
        print(f"resampler full-band bias from {a.interp_bias}: {np.ptp(ib[1]):.4f} rad "
              f"peak-to-peak over the fractional shift, taken out of the secondary", flush=True)

    aoff = None
    az_c = None                       # None -> the scalar a.az_carrier
    az_res = [np.inf, -np.inf]        # min/max of what the scalar would have got wrong
    if a.az_carrier and a.azimuth_off:
        aoff = IU.open_raster(a.azimuth_off, (NA, NR))
        print(f"azimuth carrier {a.az_carrier:+.4f} rad/px x {a.azimuth_off}", flush=True)
        if a.az_carrier_lut:
            az_c = carrier_from_lut(a, ml_r * lr)

    step = (a.block // la) * la                     # whole multilook rows per block
    for i0 in range(0, ml_a * la, step):
        n = min(step, ml_a * la - i0)
        x = np.asarray(ref[i0:i0 + n, :ml_r * lr])
        y = np.asarray(sec[i0:i0 + n, :ml_r * lr])
        if roff is not None:
            # R_sec - R_ref for the ground point seen at this reference pixel: the geo2rdr
            # offset is in secondary pixels, so it becomes metres through the range spacing,
            # plus the constant difference between the two crops' starting ranges.
            dR = a.dr0 + np.asarray(roff[i0:i0 + n, :ml_r * lr]) * a.range_spacing
            # ifg = ref * conj(sec) carries +4*pi/lambda * dR; take it back out
            y = y * np.exp(1j * a.flatten_sign * (4.0 * np.pi / a.wavelength)
                           * dR).astype(np.complex64)
            if ib is not None:
                # ifg = ref * conj(sec) carries the measured +bias(frac); rotating the
                # secondary by +bias takes it out of the product
                fr = np.asarray(roff[i0:i0 + n, :ml_r * lr]) % 1.0
                y = y * np.exp(1j * np.interp(fr, ib[0], ib[1])).astype(np.complex64)
        if aoff is not None:
            # The secondary sample was taken at azimuth position a + delta, so it carries
            # the SLC's own azimuth carrier evaluated there and ref * conj(sec) keeps
            # -psi'*delta.  Rotating the secondary by exp(-j psi' delta) puts the carrier
            # back at the output position, which is what makes resampling phase-neutral.
            ao = np.asarray(aoff[i0:i0 + n, :ml_r * lr])
            if az_c is None:
                psi = a.az_carrier
            else:
                psi = az_c(i0, n)
                r = (psi - a.az_carrier) * ao
                az_res[0] = min(az_res[0], float(r.min()))
                az_res[1] = max(az_res[1], float(r.max()))
            y = y * np.exp(-1j * psi * ao).astype(np.complex64)
        r = n // la
        sl = (r, la, ml_r, lr)
        k = i0 // la
        ifg[k:k + r] = (x * np.conj(y)).reshape(sl).sum(axis=(1, 3))
        p1[k:k + r] = (x.real**2 + x.imag**2).reshape(sl).sum(axis=(1, 3))
        p2[k:k + r] = (y.real**2 + y.imag**2).reshape(sl).sum(axis=(1, 3))
        print(f"  {i0 + n}/{ml_a * la} lines", flush=True)

    if az_c is not None and np.isfinite(az_res[0]):
        print(f"azimuth carrier per pixel: (c(x) - {a.az_carrier:+.4f}) x aoff(x) spanned "
              f"{az_res[0]:+.2f} .. {az_res[1]:+.2f} rad = {az_res[1] - az_res[0]:.2f} rad "
              f"peak to peak, which one number at the crop centre would have left in")

    coh = np.abs(ifg) / np.sqrt(np.maximum(p1 * p2, 1e-30))
    disp = ifg                                   # what gets plotted, unwrapped and geocoded

    screen = None
    if a.iono_screen:
        from .screens import offsets as ION
        # Subtracted BEFORE the filter and the unwrapping: Goldstein on a field that is
        # still ramping smears the very structure being taken out, and an unwrapper handed
        # a frame-spanning ramp has to carry it through every fringe.  A pure rotation, so
        # coh / p1 / p2 stay exactly as measured.
        screen, _ = ION.to_lattice(a.iono_screen, a.ref_origin, a.looks, ifg.shape)
        if a.iono_screen_calibrate:
            # the gain fit on THIS pair, measured before the screen is removed:
            # what the applied screen is short by.  A diagnostic, not an action -- it never
            # changes what is subtracted, because a gain fitted to the same interferogram it
            # then corrects is circular unless a human looks at it and sets it deliberately.
            al, r, n = ION.fit_screen_gain(np.angle(ifg), coh, screen,
                                           block=a.iono_calib_block)
            print(f"[iono/gain] fit on {n} block pairs "
                  f"({a.iono_calib_block}x{a.iono_calib_block} looks): alpha = {al:.4f}, "
                  f"r = {r:.3f}", flush=True)
            print(f"[iono/gain] alpha 1.0 would mean the applied screen already has the "
                  f"right amplitude; the calibrated setting is the --iono-screen-gain used "
                  f"to BUILD this screen times {al:.4f}", flush=True)
        disp = (ifg * np.exp(-1j * screen)).astype(np.complex64)

    topo = None
    if a.topo_phase:
        # After the ionospheric screen, so the slope is fitted to what is actually left,
        # and before the filter, for the same reason the screen is: Goldstein on a field
        # that is still ramping smears the structure being removed.  Also a pure rotation.
        h = heights_on_lattice(a.topo_z, a.looks, ifg.shape)
        w = np.where(np.isfinite(coh) & (coh >= 0.25) & np.isfinite(h),
                     np.nan_to_num(coh), 0.0)
        topo = (topo_slope(np.angle(disp), h, w, a.wavelength)
                * np.nan_to_num(h)).astype(np.float32)
        disp = (disp * np.exp(-1j * topo)).astype(np.complex64)

    if a.filter:
        from ._utils import raster as B
        print(f"Goldstein filter: alpha {a.filter_alpha}, fft {a.filter_win}", flush=True)
        # returns its own coherence estimate from the filtered spectrum; keep the
        # multilook coherence as the product, it is the one with a known number of looks.
        # Filters `disp`, not `ifg`: with an ionospheric screen subtracted the two differ,
        # and filtering `ifg` would silently throw the correction away.
        disp, _ = B.goldstein_filter_fast(disp, alpha=a.filter_alpha, nfft=a.filter_win,
                                          cc_win=a.filter_coh_win)

    unw = None
    if a.unwrap:
        from ._utils import unwrap
        mask = (np.abs(disp) > 0) & np.isfinite(coh)
        print(f"unwrapping ({a.unwrap_method}) over {100 * mask.mean():.1f} % of the grid, "
              f"coherence >= {a.unwrap_coh_thresh}", flush=True)
        unw, conn = unwrap.unwrap_ifg(
            disp, coh, mask, method=a.unwrap_method, coh_thresh=a.unwrap_coh_thresh,
            cache_dir=a.scratch)
        unw = np.where(mask, unw, np.nan).astype(np.float32)
        v = unw[np.isfinite(unw)]
        if v.size:
            print(f"unwrapped phase: {v.min():+.2f} .. {v.max():+.2f} rad "
                  f"({(v.max() - v.min()) / (2 * np.pi):.1f} fringes, "
                  f"{(v.max() - v.min()) * a.wavelength / (4 * np.pi):.3f} m of LOS)")

    phase = np.angle(disp)
    os.makedirs(a.out_dir, exist_ok=True)
    v = coh[np.isfinite(coh)]
    # The radar-domain npz is 55-82 MB a rung and only --keep-rdr-npz still writes it.
    # Everything downstream reads the GEOCODED products, and the radar field is what the
    # geocoding is derived from, so keeping both was storing the same measurement twice.
    if a.keep_rdr_npz:
        npz = os.path.join(a.out_dir, f"ifg_rdr_{a.tag}.npz")
        extra = {} if unw is None else dict(unwrapped=unw)
        if screen is not None:
            # `phase` is the CORRECTED field; carrying the screen makes the uncorrected one
            # exactly recoverable as phase + iono_screen, with no second run
            extra["iono_screen"] = screen
        if topo is not None:
            extra["topo_phase"] = topo
        np.savez_compressed(npz, phase=phase.astype(np.float32),
                            coherence=coh.astype(np.float32),
                            looks=np.array(a.looks), shape=np.array(a.shape), **extra)
        print(f"wrote {npz}")
    print(f"coherence: median {np.median(v):.3f}  "
          f"p10..p90 {np.percentile(v, 10):.3f} .. {np.percentile(v, 90):.3f}  "
          f"fraction > 0.3: {100 * np.mean(v > 0.3):.1f} %")
    plot(phase, coh, a)
    if a.geocode:
        geocode(disp, coh, a, unw=unw, screen=screen, topo=topo)
    return 0


def heights_on_lattice(path, looks, shape, cache=None):
    """rdr2geo's z.rdr at the centre of every look box, in km.  NaN where it has no height."""
    from osgeo import gdal

    la, lr = looks
    cache = cache or os.path.join(os.path.dirname(os.path.abspath(path)) or ".", "..",
                                  f"heights_{la}x{lr}.npy")
    cache = os.path.normpath(cache)
    if os.path.exists(cache):
        h = np.load(cache)
        if h.shape == tuple(shape):
            print(f"[topo] heights from {cache} ({h.shape[0]} x {h.shape[1]}, "
                  f"{100 * np.isfinite(h).mean():.1f} % valid)")
            return h.astype(np.float32)
        print(f"[topo] {cache} is {h.shape}, not the {tuple(shape)} this run needs; "
              f"falling back to {path}")
    if not os.path.exists(path):
        raise SystemExit(
            f"--topo-phase needs heights, but neither {cache} nor {path} exists.\n"
            f"  It is built from rdr2geo's z.rdr, so if that is gone too, re-run "
            f"the coregistration")

    ds = gdal.Open(path)
    b = ds.GetRasterBand(1)
    la, lr = looks
    out = np.empty(shape, np.float32)
    for k in range(shape[0]):
        row = b.ReadAsArray(0, k * la + la // 2, shape[1] * lr, 1)[0]
        out[k] = row[lr // 2::lr][:shape[1]]
    return np.where(np.isfinite(out) & (out > -500.0), out / 1000.0, np.nan)


def topo_slope(phase, h, w, lam, patch=128, kmax=6.0, dk=0.01, nbin=256):
    """Phase per km of height, estimated on WRAPPED phase.  Returns rad/km."""
    ks = np.arange(-kmax, kmax + dk / 2, dk)
    tot = np.zeros(ks.size)
    used = 0
    for i in range(0, phase.shape[0] - patch + 1, patch):
        for j in range(0, phase.shape[1] - patch + 1, patch):
            sl = (slice(i, i + patch), slice(j, j + patch))
            ww, hh, pp = w[sl], h[sl], phase[sl]
            m = (ww > 0) & np.isfinite(hh)
            if m.sum() < 200 or np.ptp(hh[m]) < 0.5:
                continue
            e = np.linspace(hh[m].min(), hh[m].max(), nbin + 1)
            idx = np.clip(np.searchsorted(e, hh[m]) - 1, 0, nbin - 1)
            zc = np.bincount(idx, weights=ww[m] * np.cos(pp[m]), minlength=nbin)
            zs = np.bincount(idx, weights=ww[m] * np.sin(pp[m]), minlength=nbin)
            hc = 0.5 * (e[1:] + e[:-1])
            tot += np.abs(((zc + 1j * zs)[None] * np.exp(-1j * ks[:, None] * hc[None])).sum(1))
            used += 1
    if not used:
        print("[topo] no patch has 500 m of relief under coherent ground; slope left at 0")
        return 0.0
    k = float(ks[int(np.argmax(tot))])
    print(f"[topo] phase-elevation slope {k:+.2f} rad per km of height over {used} patches "
          f"({k / (4 * np.pi / lam) * 100:+.2f} cm of delay per km of relief"
          f"), peak {tot.max() / tot.mean():.2f}x the mean of the periodogram")
    return k


def geocode(ifg, coh, a, unw=None, screen=None, topo=None):
    """Geocode the multilooked interferogram with isce3, then plot it on the map grid."""
    import isce3
    from osgeo import gdal
    from . import coregister as N
    from ._utils import raster as B
    from ._utils import plots as PL

    if not a.dem:
        raise SystemExit("--geocode needs --dem")
    ref = N.load_slc(a.ref_h5, a.freq)
    dem_raster, demI = B.load_dem(a.dem)
    rg = ref["rg"]
    la, lr = a.looks
    a0, r0 = a.ref_origin
    ml_a, ml_r = ifg.shape
    # centre of the first look box, then one sample per box
    ml_grid = isce3.product.RadarGridParameters(
        rg.sensing_start + (a0 + (la - 1) / 2.0) / rg.prf, rg.wavelength, rg.prf / la,
        rg.starting_range + (r0 + (lr - 1) / 2.0) * rg.range_pixel_spacing,
        rg.range_pixel_spacing * lr, rg.lookside, ml_a, ml_r, rg.ref_epoch)

    if a.geogrid:
        # exactly the lattice the caller names -- see --geogrid
        x0, y0, dx, dy, nx, ny, code = a.geogrid
        gg = isce3.product.GeoGridParameters(float(x0), float(y0), float(dx), float(dy),
                                             int(float(nx)), int(float(ny)),
                                             int(float(code)))
    else:
        epsg = a.epsg
        if epsg is None:
            from nisar.workflows.dumpconfig import point_to_epsg
            c = isce3.geometry.rdr2geo_bracket(
                ml_grid.sensing_mid, 0.5 * (ml_grid.starting_range + ml_grid.end_range),
                ref["orbit"], ml_grid.lookside, 0.0, ml_grid.wavelength, dem=demI)
            lon, lat = np.degrees(N.ellip.xyz_to_lon_lat(c)[:2])
            epsg = int(point_to_epsg(float(lon), float(lat)))
        px = py = float(a.posting)
        gg = isce3.product.bbox_to_geogrid(ml_grid, ref["orbit"], N.zero_lut, px, -py, epsg,
                                           min_height=demI.min_height,
                                           max_height=demI.max_height)
    print(f"geocoding onto {gg.length} x {gg.width} @ {gg.spacing_x:g} m, "
          f"EPSG {gg.epsg}", flush=True)

    os.makedirs(a.scratch, exist_ok=True)
    z = (coh * np.exp(1j * np.angle(ifg))).astype(np.complex64)
    layers = [("ifg", z, gdal.GDT_CFloat32, isce3.geocode.GeocodeCFloat32),
              ("coh", coh.astype(np.float32), gdal.GDT_Float32, isce3.geocode.GeocodeFloat32)]
    if unw is not None:
        # unwrapped phase is continuous, so it geocodes like any other real field -- unlike
        # the wrapped phase, which must go through the complex form
        layers.append(("unw", np.nan_to_num(unw).astype(np.float32), gdal.GDT_Float32,
                       isce3.geocode.GeocodeFloat32))
    if screen is not None:
        # carried through the same geocoding so the uncorrected interferogram can be
        # written on this grid too -- the screen is smooth and continuous, so like the
        # unwrapped phase it needs no complex detour
        layers.append(("screen", screen.astype(np.float32), gdal.GDT_Float32,
                       isce3.geocode.GeocodeFloat32))
    if topo is not None:
        layers.append(("topo", topo.astype(np.float32), gdal.GDT_Float32,
                       isce3.geocode.GeocodeFloat32))
    out = {}
    for name, arr, gdt, cls in layers:
        src = os.path.join(a.scratch, f"ml_{name}.tif")
        dst = os.path.join(a.scratch, f"geo_{name}.tif")
        for p, v, t in ((src, arr, gdt),
                        (dst, np.zeros((gg.length, gg.width), arr.dtype), gdt)):
            ds = gdal.GetDriverByName("GTiff").Create(p, int(v.shape[1]), int(v.shape[0]),
                                                     1, t)
            ds.GetRasterBand(1).WriteArray(v)
            ds.FlushCache()
            ds = None
        g = cls()
        g.orbit = ref["orbit"]
        g.ellipsoid = N.ellip
        g.doppler = N.zero_lut
        g.threshold_geo2rdr = 1.0e-8
        g.numiter_geo2rdr = 25
        g.data_interpolator = a.interp
        g.geogrid(gg.start_x, gg.start_y, gg.spacing_x, gg.spacing_y,
                  gg.width, gg.length, gg.epsg)
        g.geocode(radar_grid=ml_grid, input_raster=isce3.io.Raster(src),
                  output_raster=isce3.io.Raster(dst, update=True), dem_raster=dem_raster,
                  output_mode=isce3.geocode.GeocodeOutputMode.INTERP)
        out[name] = gdal.Open(dst).ReadAsArray()
        print(f"[geocode] {name}: {100 * np.mean(np.abs(out[name]) > 0):.1f} % of cells filled")

    gc, gcoh = out["ifg"], out["coh"]
    valid = np.abs(gc) > 0
    if a.mask_water:
        from . import masks as M
        gt = (gg.start_x, gg.spacing_x, 0.0, gg.start_y, 0.0, gg.spacing_y)
        tr, crs, corners, _ = M.grid_footprint(gt, gc.shape, gg.epsg)
        valid &= ~M.water_raster(tr, gc.shape, crs, corners, a)
    phase = np.where(valid, np.angle(gc), np.nan).astype(np.float32)
    gcoh = np.where(valid, gcoh, np.nan).astype(np.float32)
    saved = [("ifg_phase", phase), ("ifg_coh", gcoh)]
    gunw = None
    if "unw" in out:
        gunw = np.where(valid & (out["unw"] != 0), out["unw"], np.nan).astype(np.float32)
        saved.append(("ifg_unw", gunw))
    if "topo" in out:
        saved.append(("ifg_topo_phase", np.where(valid, out["topo"], np.nan).astype(np.float32)))
    if "screen" in out or "topo" in out:
        # The interferogram as it stood BEFORE the corrections came out, on this same grid.
        # They are rotations, so putting them back is one multiply and the corrected and
        # uncorrected products sit side by side without geocoding anything twice.
        back = sum(out[k] for k in ("screen", "topo") if k in out)
        saved.append(("ifg_phase_uncorrected",
                      np.where(valid, np.angle(gc * np.exp(1j * back)),
                               np.nan).astype(np.float32)))
    # one picture per GeoTIFF, so the directory can be read without a GIS.  Colour and
    # stretch come from the product NAME via the plotting helpers, so a wrapped phase written here
    # and one written by the driver are the same picture.
    for nm, arr in saved:
        p = os.path.join(a.out_dir, f"{nm}_geo_{a.tag}.tif")
        B.save_gtiff(p, arr, gg, nodata=np.nan)
        PL.quicklook(p, arr, stem=nm)
    if not a.no_complex_ifg:
        # The complex interferogram is recoverable from phase + coherence, so it is only
        # written on request -- it is 35 MB a rung and nothing downstream reads it.
        p = os.path.join(a.out_dir, f"ifg_geo_{a.tag}.tif")
        B.save_gtiff(p, np.where(valid, gc, 0).astype(np.complex64), gg)
    plot_geo(phase, gcoh, gg, a, gunw)
    return phase, gcoh, gg


def plot_geo(phase, coh, gg, a, unw=None):
    """Geocoded quicklook: wrapped phase (rmg), coherence (grey), unwrapped (hls)."""
    from ._utils import plots as PL

    ext = [gg.start_x, gg.start_x + gg.width * gg.spacing_x,
           gg.start_y + gg.length * gg.spacing_y, gg.start_y]
    # on its own line: with two corrections named it runs off the end of the axes
    corr = [nm for nm, on in (("ionosphere", a.iono_screen),
                              ("topographic phase", a.topo_phase)) if on]
    sub = f"\n{' and '.join(corr)} removed" if corr else ""
    panels = [(np.where(coh >= a.coh_min, phase, np.nan),
               f"geocoded flattened interferogram (coherence >= {a.coh_min}){sub}",
               "ifg_phase"),
              (coh, "coherence", "ifg_coh")]
    if unw is not None:
        panels.append((unw, f"unwrapped phase ({a.unwrap_method})", "ifg_unw"))
    png = os.path.join(a.out_dir, f"ifg_geo_{a.tag}.png")
    return PL.panel_grid(panels, f"{a.tag}", png, ncols=len(panels), extent=ext)


def plot(phase, coh, a):
    """The same pair in radar coordinates, decimated for display only."""
    from ._utils import plots as PL

    s = max(1, max(phase.shape) // 2200)      # the arrays are ~4400 x 3300
    ph, co = phase[::s, ::s], coh[::s, ::s]
    panels = [(np.where(co >= a.coh_min, ph, np.nan),
               f"interferometric phase (radar coords, {a.looks[0]}x{a.looks[1]} looks"
               f"{', ionosphere removed' if a.iono_screen else ''}, "
               f"coherence >= {a.coh_min})", "ifg_phase"),
              (co, "coherence", "ifg_coh")]
    png = os.path.join(a.out_dir, f"ifg_rdr_{a.tag}.png")
    return PL.panel_grid(panels, f"{a.tag} (radar)", png, ncols=2)


if __name__ == "__main__":
    raise SystemExit(main())
