"""One ionospheric-correction pipeline for a NISAR pair, three ways to the screen."""
from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import sys
import time

import numpy as np


from osgeo import gdal

from ._utils import raster as U
from ._utils import plots as PL

gdal.UseExceptions()

ROUTES = ("offsets", "split", "hybrid")
RUNGS = ("prerb", "rb", "rbc", "rbcs")
RUNG_LABEL = {"prerb": "coregistered only",
              "rb": "+ azimuth rubbersheet",
              "rbc": "+ carrier compensation",
              "rbcs": "+ ionospheric screen"}
SHARED_RUNGS = ("prerb", "rb", "rbc")     # screen-independent: built once, in common/

# The map lattice every product lands on.  It is READ, not restated: the coregistration
# already chose a grid for this pair (from --stack, --grid-like, or the scene centre) and
# geocoded its offsets onto it, so that product is the lattice.  A second copy here would
# have to be kept in step with it by hand, and nothing would notice when it was not.
# Finer postings are DERIVED from it rather than refitted: 120 divides by 8 to 15 m and the
# origin is a whole multiple of 15 m, so a 15 m grid nests EXACTLY inside a 120 m one --
# every coarse cell is 8 x 8 fine cells and the two compare with no interpolation anywhere.
BASE = {}


def load_base(path):
    """Fill BASE from a geocoded product of the coregistration run."""
    if not os.path.exists(path):
        raise SystemExit(
            f"missing {path}\n"
            f"  the output lattice is read from this, not hard-coded, so it has to exist\n"
            f"  before anything is geocoded.  It comes from the coregistration:\n"
            f"    ionoNISAR coregister\n"
            f"  or point --grid-like at any north-up geocoded product of this pair.")
    d = gdal.Open(path)
    gt = d.GetGeoTransform()
    sr = d.GetSpatialRef()
    code = sr.GetAuthorityCode(None) if sr is not None else None
    if code is None:
        raise SystemExit(f"{path} carries no EPSG code, so it cannot define the lattice")
    if gt[2] or gt[4]:
        raise SystemExit(f"{path} is rotated; the lattice has to be north-up")
    if abs(gt[1] + gt[5]) > 1e-9:
        raise SystemExit(f"{path} has non-square cells ({gt[1]:g} x {-gt[5]:g} m)")
    BASE.update(x0=gt[0], y0=gt[3], posting=float(gt[1]),
                nx=d.RasterXSize, ny=d.RasterYSize, epsg=int(code))
    print(f"[grid] {path}\n"
          f"[grid] {BASE['nx']} x {BASE['ny']} at {BASE['posting']:g} m, EPSG {BASE['epsg']},"
          f" origin ({BASE['x0']:g}, {BASE['y0']:g})")


def geogrid(posting=None):
    posting = BASE["posting"] if posting is None else posting
    k = BASE["posting"] / posting
    if abs(k - round(k)) > 1e-9:
        raise SystemExit(f"{posting} m does not divide the {BASE['posting']} m lattice")
    k = int(round(k))
    if BASE["x0"] % posting or BASE["y0"] % posting:
        raise SystemExit(f"the lattice origin is not a whole multiple of {posting} m")
    return ([repr(BASE["x0"]), repr(BASE["y0"]), repr(float(posting)), repr(-float(posting)),
             str(BASE["nx"] * k), str(BASE["ny"] * k), str(BASE["epsg"])], k)


def gt_of(posting=None):
    posting = BASE["posting"] if posting is None else posting
    return (BASE["x0"], posting, 0.0, BASE["y0"], 0.0, -posting)


def _t(msg, t0):
    print(f"    ({time.time() - t0:.0f}s) {msg}", flush=True)


# =====================================================================
# stage 1-2: what is recycled
# =====================================================================
def check_inputs(a, routes):
    """Fail before spending time, and say exactly which command produces what is missing."""
    need = [(os.path.join(a.scratch, "ref.c8"), "the exported reference crop"),
            (os.path.join(a.scratch, "sec.c8"), "the RAW secondary crop (every resample "
                                                "starts here, never from its own output)"),
            (os.path.join(a.scratch, "sec_coreg.c8"), "the coregistered + rubbersheeted "
                                                      "secondary"),
            (os.path.join(a.scratch, "geo2rdr", "range.off"), "geo2rdr range offsets "
                                                              "(the flattening)"),
            (os.path.join(a.scratch, "geo2rdr", "azimuth.off"), "geo2rdr azimuth offsets"),
            (a.offsets, "the ampcor offset field and its lattice geometry"),
            (a.dem, "the DEM")]
    if {"offsets", "hybrid"} & set(routes):
        need.append((os.path.join(a.scratch, "geo2rdr", "rubbersheet_az.npy"),
                     "the azimuth field --rubbersheet-az folded in"))
    missing = [(p, w) for p, w in need if not os.path.exists(p)]
    if missing:
        print("missing inputs:", file=sys.stderr)
        for p, w in missing:
            print(f"  {p}\n      {w}", file=sys.stderr)
        print("\nthese come from the coregistration, which has to have kept its scratch:\n"
              "  ionoNISAR coregister --ref <A ref> --sec <A sec> --dem <dem> \\\n"
              "      --gross geom geom --coreg geometric --rubbersheet-az 1 "
              "--keep-scratch ...\n", file=sys.stderr)
        raise SystemExit(1)
    d = np.load(a.offsets)
    win = tuple(int(v) for v in d["window"])
    print(f"[recycle] {a.scratch}: ref.c8 / sec.c8 / sec_coreg.c8 + geo2rdr offsets")
    print(f"[recycle] {a.offsets}: {d['azimuth'].shape[0]} x {d['azimuth'].shape[1]} "
          f"windows on crop {win}")
    tot = sum(os.path.getsize(p) for p, _ in need if os.path.isfile(p))
    print(f"[recycle] {tot / 1e9:.1f} GB reused rather than recomputed "
          f"(~30 min of GPU time)")
    return d, win, tuple(int(v) for v in d["sec_origin"]), \
        tuple(int(v) for v in d["sec_shape"])


def build_args(a, win_d, scratch):
    """An argparse namespace carrying the lattice the offsets were measured on."""
    from . import coregister as O
    argv = [
        "--ref", a.ref, "--sec", a.sec, "--dem", a.dem,
        "--search", *[str(v) for v in win_d["search"]],
        "--skip", *[str(v) for v in win_d["skip"]],
        "--winsize", *[str(v) for v in win_d["winsize"]],
        "--gpus", a.gpu, "--tag", a.tag, "--out-dir", a.out, "--scratch", scratch,
        "--snr-min", str(a.offsets_snr_min),
        "--mask-where", a.offsets_mask_where,
        "--mask-buffer", str(a.offsets_mask_buffer),
        "--outlier-mad", str(a.offsets_outlier_mad),
        # The FILL parameters matter too, and leaving them at parse_args' defaults
        # (--fill-aniso 1.0, --fill-trust-km 1.5) against the coregistration's (auto, 8) left the
        # rebuilt screen 2.5 % off the delivered one even after the masking was forwarded.
        "--fill-aniso", str(a.offsets_fill_aniso),
        "--fill-trust-km", str(a.offsets_fill_trust_km),
        "--fill-trust-km-flat", str(a.offsets_fill_trust_km_flat),
    ]
    if a.offsets_fill_aniso_direction:
        argv += ["--fill-aniso-direction", a.offsets_fill_aniso_direction]
    argv += ["--mask-water"] if a.offsets_mask_water else ["--no-mask-water"]
    argv += ["--mask-glacier"] if a.offsets_mask_glacier else ["--no-mask-glacier"]
    argv += ["--rdr-fill"] if a.offsets_rdr_fill else ["--no-rdr-fill"]
    return O.parse_args(argv)


# =====================================================================
# stage 4: the pre-rubbersheet secondary
# =====================================================================
def stage_prerb(a, args, win, sec_origin, sec_shape):
    """A second scratch whose azimuth.off has the rubbersheet SUBTRACTED, and its resample."""
    from . import coregister as O

    src, dst = a.scratch, a.scratch_prerb
    os.makedirs(os.path.join(dst, "geo2rdr"), exist_ok=True)

    def link(s, d):
        if not os.path.islink(d) and not os.path.exists(d):
            os.symlink(os.path.abspath(s), d)

    for f in ("ref.c8", "ref.c8.vrt", "sec.c8", "sec.c8.vrt"):
        if os.path.exists(os.path.join(src, f)):
            link(os.path.join(src, f), os.path.join(dst, f))

    def unapply(name, applied):
        """Private copy of <name>.off with `applied` subtracted, verified at the centre."""
        src_p = os.path.join(src, "geo2rdr", f"{name}.off")
        dst_p = os.path.join(dst, "geo2rdr", f"{name}.off")
        done = dst_p + ".rubbersheet_removed"
        if os.path.exists(done) and os.path.exists(dst_p):
            print(f"[prerb] {dst_p} already carries the pre-rubbersheet offsets")
            return
        print(f"[prerb] {name} rubbersheet to remove: {applied.shape}, "
              f"std {np.nanstd(applied):.4f} px, p1..p99 "
              f"{np.nanpercentile(applied, 1):+.3f} .. "
              f"{np.nanpercentile(applied, 99):+.3f} px")
        if not os.path.exists(done):
            print(f"[prerb] copying {name}.off "
                  f"({os.path.getsize(src_p) / 1e9:.1f} GB) ...", flush=True)
            shutil.copyfile(src_p, dst_p)
            # .hdr as well as .xml: geo2rdr writes ENVI, and the range raster's header is
            # <name>.hdr rather than <name>.off.hdr
            for a_, b_ in ((src_p + ".xml", dst_p + ".xml"),
                           (os.path.join(src, "geo2rdr", f"{name}.hdr"),
                            os.path.join(dst, "geo2rdr", f"{name}.hdr"))):
                if os.path.exists(a_):
                    shutil.copyfile(a_, b_)
        na, nr = win[2], win[3]
        before = gdal.Open(dst_p).ReadAsArray(nr // 2, na // 2, 1, 1)[0, 0]
        O.upsample_add(dst_p, -applied, args)
        after = gdal.Open(dst_p).ReadAsArray(nr // 2, na // 2, 1, 1)[0, 0]
        ii, wi = O._lattice_weights(na, applied.shape[0],
                                    args.search[0] + args.winsize[0] // 2, args.skip[0])
        jj, wj = O._lattice_weights(nr, applied.shape[1],
                                    args.search[1] + args.winsize[1] // 2, args.skip[1])
        i, u, j, v = ii[na // 2], wi[na // 2], jj[nr // 2], wj[nr // 2]
        f = np.nan_to_num(applied)
        want = ((f[i, j] * (1 - v) + f[i, j + 1] * v) * (1 - u)
                + (f[i + 1, j] * (1 - v) + f[i + 1, j + 1] * v) * u)
        got = before - after
        ok = abs(got - want) < 1e-4
        print(f"[prerb] {name} check at crop centre: removed {got:+.6f} px, the "
              f"rubbersheet there is {want:+.6f} px -> {'OK' if ok else 'MISMATCH'}")
        if not ok:
            raise SystemExit("the subtraction did not land where the lattice says it "
                             "should; prerb would not be the pre-rubbersheet product")
        open(done, "w").close()

    # range.off carries no rubbersheet, so prerb and rb are flattened by the same raster
    # and their difference is purely the azimuth field: share it by symlink.
    for f in ("range.off", "range.off.xml", "range.hdr"):
        if os.path.exists(os.path.join(src, "geo2rdr", f)):
            link(os.path.join(src, "geo2rdr", f), os.path.join(dst, "geo2rdr", f))

    applied_az = np.load(os.path.join(src, "geo2rdr", "rubbersheet_az.npy"))
    if np.any(applied_az):
        unapply("azimuth", applied_az)
    else:
        # --rubbersheet-az was off: azimuth.off is already the pre-rubbersheet field
        for f in ("azimuth.off", "azimuth.off.xml", "azimuth.hdr"):
            if os.path.exists(os.path.join(src, "geo2rdr", f)):
                link(os.path.join(src, "geo2rdr", f), os.path.join(dst, "geo2rdr", f))
        print("[prerb] rubbersheet_az.npy is all zeros; azimuth.off shared unchanged")
    out = os.path.join(dst, "sec_coreg.c8")
    na, nr = win[2], win[3]
    if os.path.exists(out) and os.path.getsize(out) == na * nr * 8:
        print(f"[prerb] reusing {out}")
        O.write_vrt(out, na, nr)
        return out
    ref = O.load_slc(a.ref, "A", a.pol)
    sec = O.load_slc(a.sec, "A", a.pol)
    dem_raster, _ = U.load_dem(a.dem)
    pair0 = dict(fref=os.path.join(dst, "ref.c8"), ref_shape=(na, nr),
                 fsec=os.path.join(dst, "sec.c8"),
                 sec_shape=sec_shape, sec_origin=sec_origin)
    # coregister_isce3 takes BOTH the offsets it reads (args.scratch/geo2rdr) and the file
    # it writes (args.scratch/sec_coreg.c8) from args.scratch, so it has to be handed THIS
    # scratch.  With the caller's args it resamples through the rubbersheeted azimuth.off --
    # not the private copy this stage exists to build -- and writes the result over
    # --scratch/sec_coreg.c8, the one input every other rung depends on.
    prerb_args = argparse.Namespace(**vars(args))
    prerb_args.scratch = dst
    # "--scratch IS NEVER MODIFIED" above is an invariant whose violation is SILENT: a
    # secondary resampled through the wrong azimuth.off lands at exactly the right size, and
    # every rung after it is quietly wrong with nothing raised.  Cheap enough to check.
    watch = {p: (os.stat(p).st_size, os.stat(p).st_mtime)
             for p in (os.path.join(src, "sec_coreg.c8"),
                       os.path.join(src, "geo2rdr", "azimuth.off"))
             if os.path.exists(p)}
    O.coregister_isce3(prerb_args, pair0, ref, sec, win, dem_raster, resample_only=True)
    for p, was in watch.items():
        if (os.stat(p).st_size, os.stat(p).st_mtime) != was:
            raise SystemExit(f"[prerb] {p} was written to; --scratch is read-only here and "
                             f"every other rung depends on it")
    if not (os.path.exists(out) and os.path.getsize(out) == na * nr * 8):
        raise SystemExit(f"[prerb] the resample did not produce {out}; the prerb rung "
                         f"cannot be built and nothing downstream should try")
    return out


# =====================================================================
# stage 4-5: the interferogram rungs
# =====================================================================
def az_carrier(a, win, sec_origin):
    """The secondary's azimuth carrier, rad per pixel of azimuth offset."""
    from . import coregister as O
    r2 = O.load_slc(a.sec, "A", a.pol)["rg"]
    na, nr = win[2], win[3]
    fd = float(O.sampled_doppler(O.load_slc(a.sec, "A", a.pol)["dop"], r2.prf).eval(
        r2.sensing_start + (sec_origin[0] + na / 2) / r2.prf,
        r2.starting_range + (sec_origin[1] + nr / 2) * r2.range_pixel_spacing))
    return 2 * np.pi * fd / float(r2.prf)


def interferogram(a, sec_path, ptag, win, sec_origin, out_dir, carrier=None, screen=None,
                  scratch=None):
    """ref x conj(sec), flattened from the DEM + orbits, multilooked and geocoded."""
    from . import coregister as O
    from . import interferogram as C

    scratch = scratch or a.scratch
    # The rbcs rung IS a function of the screen, so a screen newer than the product means
    # the product is stale.  Without this test a rebuilt screen is silently published under
    # the old interferogram, and the run reports the old numbers under the new screen's name.
    done = os.path.join(out_dir, f"ifg_coh_geo_{ptag}.tif")
    stale = (screen is not None and os.path.exists(done) and os.path.exists(screen)
             and os.path.getmtime(screen) > os.path.getmtime(done))
    if os.path.exists(done) and not a.force_ifg and not stale:
        print(f"[ifg] {ptag} already built")
        return ptag
    if stale:
        print(f"[ifg] {ptag} exists but {os.path.basename(screen)} is newer -- rebuilding, "
              f"because this rung is the screen applied")
    r1 = O.load_slc(a.ref, "A", a.pol)["rg"]
    r2 = O.load_slc(a.sec, "A", a.pol)["rg"]
    grid, _ = geogrid(a.posting)
    sc = os.path.join(out_dir, f"scratch_{ptag}")
    os.makedirs(sc, exist_ok=True)
    # rdr2geo(DEM, reference orbit) -> geo2rdr(secondary orbit) wrote range.off; the
    # flat-earth and topographic phase is 4pi/lambda * (dr0 + range.off * spacing).  dr0 is
    # the constant between the two crops' starting ranges, and sec_origin is the RAW crop
    # origin, before resampling.
    dr0 = ((r2.starting_range + sec_origin[1] * r2.range_pixel_spacing)
           - (r1.starting_range + win[1] * r1.range_pixel_spacing))
    argv = [
        "--ref", os.path.join(scratch, "ref.c8"), "--sec", sec_path,
        "--shape", str(win[2]), str(win[3]),
        "--looks", str(a.looks[0]), str(a.looks[1]),
        "--out-dir", out_dir, "--tag", ptag, "--scratch", sc,
        "--wavelength", repr(float(r1.wavelength)),
        "--range-spacing", repr(float(r1.range_pixel_spacing)),
        "--ref-h5", a.ref, "--sec-h5", a.sec, "--dem", a.dem,
        "--ref-origin", str(win[0]), str(win[1]),
        "--sec-origin", str(sec_origin[0]), str(sec_origin[1]),
        # `scratch` is the rung's OWN scratch.  With --rubbersheet-rg off every rung shares
        # one range.off by symlink, so the flattening is identical and the rung-to-rung
        # difference stays purely the azimuth rubbersheet and the phase terms.  With it on,
        # stage_prerb gives prerb a private range.off with the field subtracted, and the
        # flattening then differs between rungs BY DESIGN -- that difference IS the range
        # correction, and is why the range rung is scored on fringe rate, not coherence.
        "--flatten", "--range-off", os.path.join(scratch, "geo2rdr", "range.off"),
        "--dr0", repr(float(dr0)),
        "--geocode", "--posting", repr(float(a.posting)), "--geogrid", *grid,
        "--no-complex-ifg",
    ]
    if a.filter:
        argv += ["--filter", "--filter-alpha", str(a.filter_alpha),
                 "--filter-win", str(a.filter_win)]
    if a.mask_water:
        argv += ["--mask-water", "--cache-dir", a.cache_dir]
    if carrier is not None:
        # azimuth.off in `scratch` carries geometry PLUS that route's rubbersheet, i.e. the
        # total shift the resampler applied -- exactly the delta the carrier term scales.
        argv += ["--azimuth-off", os.path.join(scratch, "geo2rdr", "azimuth.off"),
                 "--az-carrier", repr(float(carrier))]
    if screen:
        argv += ["--iono-screen", screen]
        if a.iono_screen_calibrate:
            argv += ["--iono-screen-calibrate"]
    if a.unwrap:
        argv += ["--unwrap", "--unwrap-method", a.unwrap_method]
    if a.keep_rdr_npz:
        argv.append("--keep-rdr-npz")
    print(f"[ifg] {ptag}", flush=True)
    C.main(argv)
    if not a.keep_scratch:
        n = sum(os.path.getsize(os.path.join(sc, f)) for f in os.listdir(sc))
        shutil.rmtree(sc, ignore_errors=True)
        print(f"[ifg] removed {sc} ({n / 1e9:.2f} GB; --keep-scratch keeps it)")
    return ptag


# =====================================================================
# stage 3: the screens
# =====================================================================
def _screen_request(a):
    """Just the screen-defining settings, for comparing against what is on disk."""
    import types
    return types.SimpleNamespace(
        iono_screen_km=a.iono_screen_km, iono_screen_passes=a.iono_screen_passes,
        iono_screen_gain=a.iono_screen_gain, iono_shell_km=a.iono_shell_km,
        iono_screen_max_gap=a.iono_screen_max_gap)


def screen_offsets(a, args, win_d, win):
    """The delivered screen: integrate the cleaned ampcor azimuth field along track."""
    from . import coregister as O
    from .screens import offsets as ION

    # THE REBUILD IS NOT BIT-EQUAL TO THE COREGISTRATION'S SCREEN, and cannot be made so
    # from here.  The coregistration builds its screen AFTER the rubbersheet loop, and
    # --rubbersheet-az-redense defaults to TRUE, so the field it integrates is a SECOND
    # ampcor pass measured against the already-rubbersheeted secondary with the applied
    # field added back.  What lands in the offsets npz -- the only field this route has --
    # is the FIRST pass.  The two agree closely but not exactly, so prefer the
    # coregistration's screen, which the second gate below does; the redensed field is in
    # offsets_<tag>_rbsheet.npz.
    #
    # Both reuse gates check provenance, so changing a screen setting on an existing product
    # REBUILDS rather than silently reusing the old screen under the new settings' name.
    os.makedirs(a.dir("offsets"), exist_ok=True)
    out_npz = os.path.join(a.dir("offsets"), f"iono_screen_{a.tag}.npz")
    src = os.path.join(a.scratch, f"iono_screen_{a.tag}.npz")
    want = _screen_request(a)
    for path, why in ((out_npz, f"reusing {out_npz}"),
                      (src, f"reusing the screen the coregistration wrote: {src}")):
        if not os.path.exists(path) or a.force_screen:
            continue
        diffs = ION.screen_mismatch(path, want)
        if diffs:
            print(f"[offsets] NOT reusing {path} -- it was built with different settings:")
            for d in diffs:
                print(f"            {d}")
            print("[offsets] rebuilding the screen for the requested settings")
            break
        print(f"[offsets] {why}")
        if path is src:
            shutil.copyfile(src, out_npz)
        return out_npz

    ref = O.load_slc(a.ref, "A", a.pol)
    dem_raster, demI = U.load_dem(a.dem)
    centre = (win[0] + win[2] // 2, win[1] + win[3] // 2)
    args.az_spacing = O.azimuth_ground_spacing(ref, demI, centre)
    args.iono_screen_km = a.iono_screen_km
    args.iono_shell_km = a.iono_shell_km
    args.iono_screen_count_filled = a.iono_screen_count_filled
    args.iono_screen_integrand = a.iono_screen_integrand
    args.iono_screen_passes = a.iono_screen_passes
    args.iono_screen_gain = a.iono_screen_gain
    args.iono_screen_calibrate = a.iono_screen_calibrate
    args.iono_screen_max_gap = a.iono_screen_max_gap
    az, rg, snr = win_d["azimuth"], win_d["range"], win_d["snr"]
    off_rg = O.offset_radar_grid(ref["rg"], win, args, az.shape)
    gg = O.target_geogrid(args, off_rg, ref["orbit"], demI)
    gt = (gg.start_x, gg.spacing_x, 0.0, gg.start_y, 0.0, gg.spacing_y)
    keep, azf, rgf, snrf, filled = O.clean_lattice(args, ref, demI, win, a.offsets,
                                                   gg, gt, az, rg, snr)
    # azf is the anisotropically filled field, reachable via --iono-screen-integrand filled;
    # `raw` integrates the measured field instead.
    phi, valid = ION.screen_from_offsets(args, ref, win, az, keep, filled, az_filled=azf)
    # PRECISION.  ampcor's own SNR gives the per-cell shape; the level and the correlation
    # come from the field itself.  Windows overlap winsize/skip cells, so that many
    # neighbouring measurements share most of their pixels and do not average down.
    cc = args.winsize[0] / args.skip[0]
    sig_px = ION.calibrate_level(az, keep, 1.0 / np.sqrt(np.maximum(snr, 1e-6)), cc,
                                 label="ampcor azimuth")
    sigma = ION.screen_sigma(args, ref, win, az, keep, sig_px, cc)
    if a.iono_screen_integrand == "filled":
        print("[offsets] NOTE: sigma above was computed on the RAW field's noise model.  "
              "The screen integrated the filled surface, whose interpolated cells carry no "
              "such noise, so this sigma is not the precision of THIS screen -- it is a "
              "lower bound and should not be published beside it")
    np.savez_compressed(out_npz, screen=phi, valid=valid, sigma=sigma,
                        window=np.array(win),
                        skip=np.array(args.skip), winsize=np.array(args.winsize),
                        search=np.array(args.search), **ION.screen_params(args))
    print(f"[offsets] wrote {out_npz}")
    return out_npz


def screen_hybrid(a, args=None, d=None, win=None):
    """Hybrid: split's long along-track wavelengths over the azimuth offsets' short ones."""
    from .screens import hybrid as H

    out_dir = a.dir("hybrid")
    out_npz = os.path.join(out_dir, f"iono_screen_{a.tag}.npz")
    if os.path.exists(out_npz) and not a.force_screen:
        print(f"[hybrid] reusing {out_npz}")
        return out_npz
    # both parents first; each returns early if its screen is already on disk
    off_npz = screen_offsets(a, args, d, win)
    split_npz = screen_split(a)
    os.makedirs(out_dir, exist_ok=True)
    return H.build(off_npz, split_npz, a.offsets, out_npz, a.hybrid_cut_km)


def carries_freq_b(path, pol="HH"):
    """True when this RSLC holds frequency B itself, so no sidecar crop is needed."""
    import h5py
    try:
        with h5py.File(path, "r") as f:
            return f"/science/LSAR/RSLC/swaths/frequencyB/{pol}" in f
    except (OSError, KeyError):
        return False


def screen_split(a, args=None, d=None, win=None):
    """Split spectrum: frequency A + B, no azimuth information at all."""
    from .screens import split as S

    d = a.dir("split")
    os.makedirs(d, exist_ok=True)
    out_npz = os.path.join(d, f"iono_screen_{a.tag}.npz")
    if os.path.exists(out_npz) and not a.force_screen:
        print(f"[split] reusing {out_npz}")
        return out_npz
    grid, _ = geogrid(a.posting)
    band_b = []
    if carries_freq_b(a.ref, a.pol) and carries_freq_b(a.sec, a.pol):
        print(f"[split] frequency B is inside the RSLCs themselves; no sidecar crop needed")
        band_b = ["--ref-b", a.ref, "--sec-b", a.sec]
    S.main(["--ref", a.ref, "--sec", a.sec, "--dem", a.dem,
            *band_b,
            "--cache-dir", a.cache_dir, "--tag", a.tag,
            "--scratch", a.scratch, "--offsets", a.offsets,
            "--out-dir", d, "--gpu", a.gpu,
            "--looks", str(a.looks[0]), str(a.looks[1]),
            "--geogrid", *grid,
            "--coh-thresh", str(a.split_coh_thresh),
            "--diff-weight", a.split_diff_weight,
            "--az-carrier-mode", a.split_az_carrier,
            "--iono-filter-km", str(a.split_filter_km),
            "--unwrap-components", a.split_unwrap_components,
            "--unwrap-cycle-tol", str(a.split_unwrap_cycle_tol),
            "--iono-fill", a.split_fill,
            "--iono-fill-floor", str(a.split_fill_floor),
            "--iono-fill-trend", str(a.split_fill_trend),
            "--outlier-mad", str(a.split_outlier_mad),
            "--unwrap-method", a.split_unwrap_method,
            "--phia-from", a.split_phia_from,
            "--gate-corr", str(a.split_gate_corr),
            "--gate-bracket", str(a.split_gate_bracket[0]), str(a.split_gate_bracket[1]),
            "--snaphu-ntiles", str(a.snaphu_ntiles[0]), str(a.snaphu_ntiles[1]),
            "--snaphu-nproc", str(a.snaphu_nproc),
            "--datum", a.split_datum]
           + (["--interp-bias-b", a.split_interp_bias_b] if a.split_interp_bias_b else [])
           + (["--interp-bias-a", a.split_interp_bias_a] if a.split_interp_bias_a else [])
           + (["--coh-thresh-b", str(a.split_coh_thresh_b)]
              if a.split_coh_thresh_b is not None else [])
           + (["--glacier-mask", a.split_glacier_mask] if a.split_glacier_mask else [])
           + (["--scratch-b", a.split_scratch_b] if a.split_scratch_b else [])
           + ([] if a.split_unwrap_diff else ["--no-unwrap-diff"]))
    if os.path.exists(out_npz):
        return out_npz
    raise SystemExit(f"[split] finished but produced no screen npz in {d}")


SCREEN_FN = {"offsets": screen_offsets, "split": screen_split,
             "hybrid": screen_hybrid}


def check_screen(route, path):
    """Reject an unapplyable screen now rather than after the rbcs multilook pass."""
    need = ("screen", "valid", "window", "skip", "winsize", "search")
    try:
        with np.load(path) as d:
            missing = [k for k in need if k not in d.files]
            shape = d["screen"].shape
    except Exception as e:
        raise SystemExit(f"[{route}] {path} is not a screen npz ({type(e).__name__}: {e}); "
                         f"--iono-screen needs the offsets screen's own npz format, on the "
                         f"offset lattice, not a geocoded raster")
    if missing:
        raise SystemExit(f"[{route}] {path} is missing {', '.join(missing)}; "
                         f"--iono-screen cannot place it on the multilook lattice")
    print(f"[{route}] screen {path}: {shape[0]} x {shape[1]} on the offset lattice")
    return path


# =====================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("inputs (recycled unless missing)")
    # No defaults for the pair: a stale default is the one failure mode that produces a
    # complete, plausible set of products for the wrong acquisition.
    g.add_argument("--ref", required=True, help="reference RSLC (frequency A)")
    g.add_argument("--sec", required=True, help="secondary RSLC (frequency A)")
    g.add_argument("--dem", required=True)
    g.add_argument("--pol", default="HH")
    g.add_argument("--grid-like", default=None, metavar="TIF",
                   help="the geocoded product whose lattice every output lands on "
                        "(default: the offsets run's own "
                        "offset_azimuth_<tag>_px.tif, beside --offsets)")
    g.add_argument("--cache-dir", default="cache")
    g.add_argument("--scratch", default="offsets_scratch",
                   help="the coregistration scratch this recycles.  NEVER modified")
    # DERIVED from --scratch, not fixed: a literal default would let two pairs processed
    # into separate scratches collide here, and silently, because the reuse gates would find
    # a finished rung belonging to the OTHER pair and skip the rebuild.
    # None = "<scratch>_prerb", resolved after parsing.
    g.add_argument("--scratch-prerb", default=None,
                   help="this pipeline's own scratch: the pre-rubbersheet secondary and "
                        "the corrected secondaries (default <scratch>_prerb)")
    g.add_argument("--offsets", default=None,
                   help="default outputs_offsets/offsets_<tag>.npz")

    g = p.add_argument_group("what to run")
    g.add_argument("--routes", nargs="+", default=["offsets"], choices=list(ROUTES),
                   help="which screen estimators to run (default offsets)")
    g.add_argument("--tag", required=True,
                   help="names every product of this pair, e.g. t<track>_<date1>_<date2>")
    g.add_argument("--out", default="outputs_unified")
    g.add_argument("--gpu", default="0", metavar="LIST",
                   help="CUDA device(s) to use: a list such as 0,1 or 'all' "
                        "(default 0)")

    g = p.add_argument_group("interferogram")
    g.add_argument("--looks", nargs=2, type=int, default=(24, 16), metavar=("AZ", "RG"))
    g.add_argument("--posting", type=float, default=None,
                   help="output posting in m; must divide the lattice --grid-like sets "
                        "(default: that lattice's own posting)")
    g.add_argument("--filter", action="store_true", help="Goldstein filter the phase")
    g.add_argument("--filter-alpha", type=float, default=0.5)
    g.add_argument("--filter-win", type=int, default=32)
    g.add_argument("--mask-water", action="store_true",
                   help="blank ESA WorldCover permanent water in the products.  Ice is "
                        "deliberately NOT blanked: it is out of the MEASUREMENT because it "
                        "moves, and the product can honestly report that it got its "
                        "neighbours' correction")
    g.add_argument("--unwrap", action="store_true")
    g.add_argument("--unwrap-method", default="phass", choices=["phass", "icu"])

    g = p.add_argument_group("the offsets lattice, when the screen is REBUILT")
    # These reach clean_lattice() through build_args() and ONLY matter under --force-screen;
    # a normal run copies the screen the coregistration already built.  Defaults are
    # the coregistration's own defaults, so the two paths agree.  See build_args() for what leaving them
    # out cost.
    g.add_argument("--offsets-snr-min", type=float, default=8.0, metavar="S",
                   help="ampcor SNR floor for the rebuilt field (default 8, the coregistration's)")
    g.add_argument("--offsets-mask-water", action=argparse.BooleanOptionalAction, default=True,
                   help="ESA WorldCover permanent water out of the MEASUREMENT (default on)")
    g.add_argument("--offsets-mask-glacier", action=argparse.BooleanOptionalAction, default=True,
                   help="RGI 7.0 ice out of the measurement -- it moves, and azimuth ice "
                        "motion integrated as ionosphere is the failure this prevents "
                        "(default on)")
    g.add_argument("--offsets-mask-buffer", type=float, default=160.0, metavar="M",
                   help="dilate those masks by this many metres (default 160)")
    g.add_argument("--offsets-mask-where", choices=["geo", "rdr", "both"], default="both",
                   help="apply the masks in radar geometry as well as geocoded (default "
                        "both; 'geo' alone leaves the LATTICE unmasked, which is the bug)")
    g.add_argument("--offsets-rdr-fill", action=argparse.BooleanOptionalAction, default=True,
                   help="fill mask holes on the lattice before integrating (default on; "
                        "the coregistration implies this from --iono-screen anyway)")
    g.add_argument("--offsets-outlier-mad", type=float, default=5.0, metavar="K",
                   help="local median/MAD rejection on the lattice (default 5)")
    g.add_argument("--offsets-fill-aniso", default="auto", metavar="W",
                   help="hole-fill anisotropy (default auto, the coregistration's; 1.0 is the "
                        "isotropic setting)")
    g.add_argument("--offsets-fill-aniso-direction", default="", metavar="DIR",
                   help="force the fill's elongation direction (default empty = auto)")
    g.add_argument("--offsets-fill-trust-km", type=float, default=8.0, metavar="KM",
                   help="fill trust radius for the anisotropic channels (default 8)")
    g.add_argument("--offsets-fill-trust-km-flat", type=float, default=1.5, metavar="KM",
                   help="fill trust radius for a channel measuring ~1:1 (default 1.5)")

    g = p.add_argument_group("screens")
    g.add_argument("--iono-shell-km", type=float, default=350.0, metavar="H",
                   help="effective ionospheric shell height, km (default 350).  This is the "
                        "GEOMETRY of the azimuth-offset integrator and it is not optional: "
                        "the along-track shift is driven by the TEC gradient at the "
                        "ionospheric PIERCE POINT, which sweeps at only kappa = "
                        "dx_pierce/dx_sat of the platform rate, so the phase recovered from "
                        "a measured shift carries 1/kappa.  Set 0 to reproduce the "
                        "screen on the ground, i.e. kappa = v_g/v_s, which comes out "
                        "about half the amplitude.  See "
                        "the offsets screen's integration_geometry")
    g.add_argument("--iono-screen-km", type=float, default=2.0,
                   help="low-pass sigma for the offsets and MAI screens, km.  Set from the "
                        "data: 2 km takes 23.1 %% off the along-track fringe rate and "
                        "leaves 4 km spatial coherence at 0.903; below ~1 km offset noise "
                        "wins it back (default 2)")
    g.add_argument("--iono-screen-passes", type=int, default=1, metavar="N",
                   help="residual-correction passes for that low-pass (default 1, the "
                        "plain Gaussian).  A Gaussian has no flat passband, so it does "
                        "attenuate the ionospheric wavelengths; iterating flattens it.  "
                        "Leave this at 1 unless there is a measured reason not to")
    g.add_argument("--iono-screen-gain", type=float, default=1.0, metavar="ALPHA",
                   help="Liu et al. 2014 Eq. (2) gain for the along-track integration: "
                        "alpha is FITTED against the pair's own phase rather than taken as "
                        "the theoretical constant.  Default 1.0 is the theoretical constant "
                        "alone.  Frame dependent -- measure it, do not assume it")
    g.add_argument("--iono-screen-calibrate", action="store_true",
                   help="fit and print Liu et al. 2014 Eq. (2) for every route that "
                        "applies a screen -- the gain that screen is still short by. "
                        " Diagnostic only; --iono-screen-gain is what applies it")
    g.add_argument("--iono-screen-integrand", choices=["raw", "filled"], default="raw",
                   help="which azimuth field the screen integrates.  'raw' (default) is the "
                        "measured field masked by `keep`, with holes bridged by the 2 km "
                        "low-pass and then by carry_across_range along RANGE.  'filled' "
                        "integrates the anisotropically filled surface (--fill-aniso, 13:1 "
                        "range-elongated here) instead -- which until now was computed and "
                        "then discarded.  UNTESTED against the incumbent: carry_across_range "
                        "was measured better than zeroing and than along-track trapezoid, "
                        "but never against this.  It extrapolates 35-52 km into the largest "
                        "glaciers against an 8 km validated depth, where what it carries is "
                        "ICE MOTION, and it invalidates screen_sigma.  Score it on the "
                        "glacier offset field, not on fringes alone")
    g.add_argument("--iono-screen-count-filled", action="store_true",
                   help="count cells whose azimuth offset came out of the hole-fill as "
                        "VALID.  Changes no pixel of any screen -- only the flag layer.  Off "
                        "by default (the conservative reading), but note MAI has always "
                        "behaved as if this were ON, which is the whole of why it reports "
                        "89 %% coverage against the offsets route's 61 %% on the same field")
    g.add_argument("--iono-screen-max-gap", type=float, default=8.0,
                   help="mask gaps wider than this are FLAGGED as continued, not measured "
                        "-- and still corrected, because switching the screen off at every "
                        "glacier edge put a 4.8 rad step in (default 8 km)")
    g.add_argument("--hybrid-cut-km", type=float, default=32.0,
                   help="along-track wavelength at which the hybrid screen changes over "
                        "from split to the azimuth offsets (default 32)")
    g.add_argument("--split-coh-thresh", type=float, default=0.2)
    g.add_argument("--split-coh-thresh-b", type=float, default=None,
                   help="a separate gate for band B, which has 48 samples per multilook "
                        "cell against band A's 384 -- see the split screen's --coh-thresh-b")
    g.add_argument("--split-diff-weight", default="uniform",
                   choices=["uniform", "coherence"],
                   help="weighting of the band-difference box average; `coherence` is "
                        "splitA's construction (the split screen's --diff-weight)")
    g.add_argument("--split-az-carrier", default="scalar", choices=["scalar", "lut"],
                   help="scalar Doppler at the crop centre, or the per-pixel LUT the "
                        "resampler actually re-ramped with (the split screen "
                        "--az-carrier-mode).  Changing it rebuilds both band "
                        "interferograms.")
    g.add_argument("--split-filter-km", type=float, default=12.0,
                   help="12 km, not the offsets screen's 2: this is an 11.6x-amplified "
                        "solve and needs far more smoothing (default 12)")
    g.add_argument("--split-outlier-mad", type=float, default=3.0)
    g.add_argument("--split-unwrap-components", default="rereference",
                   choices=["rereference", "largest", "all"],
                   help="each unwrapped connected component carries its own arbitrary "
                        "integer cycle datum, worth 3.21 rad of screen per cycle through "
                        "phi_A.  'rereference' rounds each component's offset from the "
                        "largest to a whole cycle and removes it; 'all' accepts every "
                        "component's own datum (default rereference)")
    g.add_argument("--split-unwrap-cycle-tol", type=float, default=0.35,
                   help="how close to a whole cycle an offset must be before it is believed "
                        "rather than dropped (default 0.35)")
    g.add_argument("--split-fill", default="smooth", choices=["smooth", "none"],
                   help="continue the split screen across the gaps its low-pass could not "
                        "reach.  'none' leaves NaN, which the npz writes as a hard 0.0 and "
                        "the product then APPLIES as a zero correction -- a step of the full "
                        "local screen value at every rim (default smooth)")
    g.add_argument("--split-fill-floor", type=float, default=0.05,
                   help="local coverage the split low-pass needs before it is trusted, as a "
                        "fraction of the frame's best.  0.05 reaches ~22 km past the data "
                        "edge; raise it to lean on the continuation instead (default 0.05)")
    # UNWRAPPING A FIELD THAT IS NOT WRAPPED can only invent cycles, and each invented
    # cycle is 2*pi/|det| of dispersive phase.  The local-median rejection cannot catch it:
    # the error is a CONSTANT over each connected component the unwrapper mis-labelled, so
    # inside one the local median is offset too.  Read [diff] |phi_diff| max before
    # deciding; --no-split-unwrap-diff bounds the estimate at +-pi/|det| instead.
    g.add_argument("--split-unwrap-diff", action=argparse.BooleanOptionalAction, default=True,
                   help="unwrap the band difference before the 2x2 solve (default on)")
    g.add_argument("--split-unwrap-method", default="snaphu",
                   choices=["snaphu", "phass", "icu"],
                   help="which unwrapper the split route uses.  snaphu (default) weighs "
                        "coherence through its statistical cost; phass and icu are faster "
                        "but more likely to fail gate 8 on a frame with patchy coherence")
    g.add_argument("--split-scratch-b", default=None, metavar="DIR",
                   help="frequency B scratch for the split (default <scratch>/freqB)")
    g.add_argument("--split-phia-from", default="unwrap", choices=["unwrap", "gradient"],
                   help="which phi_A the split screen's phi_A term is built on.  'unwrap' is "
                        "the unwrapper's field and is gated (the split screen gate 8); "
                        "'gradient' uses the integrated low-passed gradient instead -- the "
                        "field the gate itself compares against -- which is what to reach for "
                        "when the gate reports a corr/amplitude mismatch.  The band-difference "
                        "term (the 11.6x one) is unchanged either way.  Needs "
                        "--split-filter-km > 0")
    g.add_argument("--split-gate-corr", type=float, default=0.9, metavar="R",
                   help="split gate 8 correlation threshold (the split screen's --gate-corr; "
                        "default 0.9)")
    g.add_argument("--split-gate-bracket", type=float, nargs=2, default=(0.85, 1.15),
                   metavar=("LO", "HI"),
                   help="split gate 8 amplitude bracket (the split screen's --gate-bracket; "
                        "default 0.85 1.15)")
    g.add_argument("--split-fill-trend", type=int, default=1, choices=[0, 1, 2],
                   help="order of the surface split's continuation follows past the data edge "
                        "(the split screen's --iono-fill-trend): 0 flat, 1 plane, 2 quadratic "
                        "(default 1)")
    g.add_argument("--snaphu-ntiles", type=int, nargs=2, default=(1, 1),
                   metavar=("NROW", "NCOL"),
                   help="single tile by default: tiles cost coverage and put discontinuities "
                        "over the overlaps (see the split screen's --snaphu-ntiles)")
    g.add_argument("--snaphu-nproc", type=int, default=16)
    g.add_argument("--split-glacier-mask", default=None, metavar="TIF",
                   help="ice raster on the solve grid for the split route (see "
                        "the split screen's --glacier-mask)")
    g.add_argument("--split-interp-bias-b", default=None, metavar="NPZ",
                   help="band B's resampler bias curve, see "
                        "the split screen's --interp-bias-b")
    g.add_argument("--split-interp-bias-a", default=None, metavar="NPZ")
    g.add_argument("--split-datum", default="as-is", choices=["as-is", "per-column"],
                   help="'per-column' gives the split screen the same datum offsets/MAI "
                        "carry, which removes the frame-scale range bend (default as-is)")

    g = p.add_argument_group("GSLC / output size")
    g.add_argument("--keep-scratch", action="store_true",
                   help="keep the per-rung scratch dirs and the corrected secondaries "
                        "(~270 MB per rung, 23.6 GB per route)")
    g.add_argument("--keep-rdr-npz", action="store_true",
                   help="also write the radar-domain ifg_rdr_<tag>.npz per rung (55-82 MB)")

    g = p.add_argument_group("force")
    for nm in ("screen", "ifg", "gslc"):
        g.add_argument(f"--force-{nm}", action="store_true",
                       help=f"rebuild the {nm} stage even if its products exist")
    a = p.parse_args(argv)
    # Derive the two extra scratches from --scratch so a second pair processed into its own
    # scratch is fully isolated.  A fixed name here would let pair B's reuse gate find pair
    # A's finished rung and skip the rebuild, which is silent and wrong.
    a.scratch_prerb = a.scratch_prerb or a.scratch.rstrip("/") + "_prerb"
    a.offsets = a.offsets or f"outputs_offsets/offsets_{a.tag}.npz"
    a.grid_like = a.grid_like or os.path.join(os.path.dirname(a.offsets) or ".",
                                              f"offset_azimuth_{a.tag}_px.tif")
    a.looks = tuple(a.looks)
    a.dir = lambda r: os.path.join(a.out, r)
    return a


def main(argv=None):
    t_start = time.time()
    a = parse_args(argv)
    routes = list(dict.fromkeys(a.routes))
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(os.path.join(a.out, "common"), exist_ok=True)
    for r in routes:
        os.makedirs(a.dir(r), exist_ok=True)

    print("=" * 78)
    print(f"ionosphere pipeline   tag {a.tag}   routes: {', '.join(routes)}")
    print("=" * 78)

    # Before anything geocodes: the lattice comes from the coregistration's own product, so
    # --compare-only needs it too (it reads the products back on that grid).
    load_base(a.grid_like)
    if a.posting is None:
        a.posting = BASE["posting"]

    d, win, sec_origin, sec_shape = check_inputs(a, routes)
    args = build_args(a, d, a.scratch)
    common = os.path.join(a.out, "common")

    # --- stage 3a: the pre-rubbersheet secondary
    # stage_prerb costs 47 GB of scratch and a resamp_slc pass.  Normally its ONLY consumer
    # is the prerb rung, so it is skipped outright once that rung is on disk -- otherwise a
    # second route would rebuild the pre-rubbersheet secondary just to skip the interferogram
    # that needed it, which is what makes offsets_scratch_prerb/ look permanently required.
    #
    # maialone changes that: it MEASURES on the geometry-only pair, so it needs both the
    # secondary and the pre-rubbersheet azimuth.off, and it needs them before the screens.
    prerb_done = os.path.exists(os.path.join(common, f"ifg_coh_geo_{a.tag}_prerb.tif"))
    if prerb_done and not a.force_ifg:
        print(f"[prerb] the prerb rung is already built; skipping the pre-rubbersheet "
              f"secondary entirely ({a.scratch_prerb} is not needed and can be deleted)")
        prerb = None
    else:
        prerb = stage_prerb(a, args, win, sec_origin, sec_shape)

    # --- stage 3: the screens
    screens = {}
    for r in routes:
        print(f"\n--- screen: {r} " + "-" * (60 - len(r)))
        screens[r] = check_screen(r, SCREEN_FN[r](a, args, d, win))

    # --- stage 4: the shared rungs
    print("\n--- rungs: prerb / rb / rbc (screen-independent, built once) " + "-" * 16)
    rb = os.path.join(a.scratch, "sec_coreg.c8")
    carrier = az_carrier(a, win, sec_origin)
    print(f"[ifg] secondary azimuth carrier {carrier:+.4f} rad per pixel of azimuth offset")
    for rung, sec, kw in (("prerb", prerb, {}),
                          ("rb", rb, {}),
                          ("rbc", rb, dict(carrier=carrier))):
        if sec is None:
            continue
        interferogram(a, sec, f"{a.tag}_{rung}", win, sec_origin, common, **kw)

    # --- stage 5: one rbcs per route
    for r in routes:
        print(f"\n--- rbcs: {r} " + "-" * (62 - len(r)))
        interferogram(a, rb, f"{a.tag}_rbcs", win, sec_origin, a.dir(r),
                      carrier=carrier, screen=screens[r], scratch=a.scratch)

    # --- stage 8: a picture for every raster
    print("\n--- figures " + "-" * 64)
    for dd in [common] + [a.dir(r) for r in routes]:
        PL.render_dir(dd)

    print(f"\ndone in {(time.time() - t_start) / 60:.1f} min.  products under {a.out}/")
    for dd in ["common"] + routes:
        p = os.path.join(a.out, dd)
        if os.path.isdir(p):
            n = sum(os.path.getsize(os.path.join(p, f)) for f in os.listdir(p)
                    if os.path.isfile(os.path.join(p, f)))
            print(f"  {dd + '/':<12} {n / 1e9:7.2f} GB  "
                  f"{len(glob.glob(os.path.join(p, '*')))} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
