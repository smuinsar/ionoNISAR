"""Pixel-offset tracking on focused NISAR SLCs, and the interferometry it makes possible."""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from contextlib import contextmanager

import numpy as np


import isce3
import h5py
from osgeo import gdal

from ._utils import raster as B                     # save_gtiff / load_dem / zero_lut / ellip
from ._utils import plots as PL                     # the one colour + colourbar policy
from . import masks as M                    # water/glacier masks, outliers, hole filling
from .screens import offsets as ION                   # ionospheric phase screen from the azimuth field

zero_lut, ellip = B.zero_lut, B.ellip


# ---------------------------------------------------------------- timing

_TIMINGS = []           # (stage name, seconds), in the order they ran
_T0 = None              # wall clock at the start of main()


def _hms(s):
    if s < 60:
        return f"{s:.1f} s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{int(m)}m {s:04.1f}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h {int(m):02d}m {s:04.1f}s"


@contextmanager
def timed(name):
    """Wall-clock a named stage.  Every stage is reported as it finishes and again in the
    summary at the end of the run, so a slow step is obvious without re-instrumenting."""
    t = time.time()
    try:
        yield
    finally:
        dt = time.time() - t
        _TIMINGS.append((name, dt))
        print(f"[time] {name}: {_hms(dt)}", flush=True)


def print_timings():
    if not _TIMINGS:
        return
    w = max(len(n) for n, _ in _TIMINGS)
    tot = sum(d for _, d in _TIMINGS)
    wall = time.time() - _T0 if _T0 else tot
    print("\nprocessing time")
    for n, d in _TIMINGS:
        print(f"  {n:<{w}s}  {_hms(d):>12s}  {100 * d / max(tot, 1e-9):5.1f} %")
    print(f"  {'-' * w}  {'-' * 12}")
    print(f"  {'timed stages':<{w}s}  {_hms(tot):>12s}")
    print(f"  {'wall clock':<{w}s}  {_hms(wall):>12s}")


# ---------------------------------------------------------------- inputs

def load_slc(path, freq="A", pol="HH"):
    """Geometry + image location of a focused SLC."""
    with h5py.File(path, "r") as h:
        focused = "radargrid" in h
    if not focused:
        # The nisar reader opens the file itself, with swmr=True, and h5py refuses to hand
        # out a second handle whose SWMR flag disagrees with the first -- so this must
        # happen with our own handle CLOSED, not nested inside it.
        from nisar.products.readers import SLC
        sl = SLC(hdf5file=path)
        rg, orbit = sl.getRadarGrid(freq), sl.getOrbit()
        dop = sl.getDopplerCentroid(frequency=freq)
        dset = f"science/LSAR/RSLC/swaths/frequency{freq}/{pol}"
        kind = f"NISAR RSLC {freq}/{pol}"
    with h5py.File(path, "r") as h:
        if focused:                                             # a focused SLC
            g = h["radargrid"].attrs
            side = (isce3.core.LookSide.Right if str(g["lookside"]).lower().startswith("r")
                    else isce3.core.LookSide.Left)
            rg = isce3.product.RadarGridParameters(
                float(g["sensing_start"]), float(g["wavelength"]), float(g["prf"]),
                float(g["starting_range"]), float(g["range_pixel_spacing"]), side,
                int(g["length"]), int(g["width"]), isce3.core.DateTime(str(g["ref_epoch"])))
            orbit = isce3.core.Orbit.load_from_h5(h["orbit"])
            # Native Doppler centroid of the DATA.  The image grid is zero-Doppler, so the
            # geometry (rdr2geo / geo2rdr) uses a zero LUT -- but the samples still carry an
            # azimuth carrier at fd, and resampling has to strip it before interpolating and
            # put it back after.  On this pair fd is 983 Hz against a PRF of 1910, i.e. 0.51
            # of a pixel: hand the resampler a zero Doppler instead and the resampled image
            # comes out with an azimuth phase error that walks the correlation peak.
            dop = isce3.core.LUT2d(h["dop/x"][()], h["dop/y"][()], h["dop/data"][()])
            dop.bounds_error = False
            dset = "slc"
            kind = "focused SLC"
        shape, dtype = h[dset].shape, str(h[dset].dtype)
    print(f"{kind}: {os.path.basename(path)}  {shape} {dtype}  "
          f"dr={rg.range_pixel_spacing:.4f} m  prf={rg.prf:.4f} Hz")
    return dict(path=path, dset=dset, shape=shape, rg=rg, orbit=orbit, dop=dop)


def write_vrt(binary, na, nr):
    """A VRT header for a flat complex64 file."""
    vrt = binary + ".vrt"
    with open(vrt, "w") as f:
        f.write(f'<VRTDataset rasterXSize="{nr}" rasterYSize="{na}">\n'
                f'  <VRTRasterBand dataType="CFloat32" band="1" '
                f'subClass="VRTRawRasterBand">\n'
                f'    <SourceFilename relativeToVRT="1">'
                f'{os.path.basename(binary)}</SourceFilename>\n'
                f'    <ImageOffset>0</ImageOffset>\n'
                f'    <PixelOffset>8</PixelOffset>\n'
                f'    <LineOffset>{nr * 8}</LineOffset>\n'
                f'    <ByteOrder>LSB</ByteOrder>\n'
                f'  </VRTRasterBand>\n'
                f'</VRTDataset>\n')
    return vrt


def export_window(meta, win, out, chunk=1024):
    """Write a window of an SLC to a flat complex64 binary for PyCuAmpcor (it is file-based)."""
    a0, r0, na, nr = win
    if os.path.exists(out) and os.path.getsize(out) == na * nr * 8:
        print(f"  reuse {os.path.basename(out)}")
        write_vrt(out, na, nr)
        return
    print(f"  export {os.path.basename(out)}  {na} x {nr} ({na*nr*8/1e9:.2f} GB)", flush=True)
    with h5py.File(meta["path"], "r") as h, open(out, "wb") as f:
        d = h[meta["dset"]]
        for i in range(a0, a0 + na, chunk):
            n = min(chunk, a0 + na - i)
            blk = d[i:i + n, r0:r0 + nr]
            if blk.dtype.names:                     # NISAR complex32 is stored as an (r,i) pair
                blk = blk["r"].astype(np.float32) + 1j * blk["i"].astype(np.float32)
            np.ascontiguousarray(blk, dtype=np.complex64).tofile(f)
    write_vrt(out, na, nr)


# ---------------------------------------------------------------- geometry

def gross_from_geometry(ref, sec, demI, at):
    """Bulk (azimuth, range) shift predicted from orbit + DEM at reference pixel `at`."""
    r1, r2 = ref["rg"], sec["rg"]
    i0, j0 = at
    t1 = r1.sensing_start + i0 / r1.prf
    rr = r1.starting_range + j0 * r1.range_pixel_spacing
    xyz = isce3.geometry.rdr2geo_bracket(t1, rr, ref["orbit"], r1.lookside, 0.0,
                                         r1.wavelength, dem=demI)
    t2, rr2 = isce3.geometry.geo2rdr_bracket(xyz, sec["orbit"], zero_lut,
                                             r2.wavelength, r2.lookside)
    daz = (t2 - r2.sensing_start) * r2.prf - i0
    drg = (rr2 - r2.starting_range) / r2.range_pixel_spacing - j0
    print(f"  geometry at reference pixel ({i0}, {j0}): "
          f"azimuth {daz:+.2f} px, range {drg:+.2f} px")
    return int(round(daz)), int(round(drg))


def gross_from_annotation(ref, sec):
    """The bulk shift the ANNOTATIONS alone imply.  Printed for comparison, never used."""
    r1, r2 = ref["rg"], sec["rg"]
    daz = int(round((r1.sensing_start - r2.sensing_start) * r2.prf))
    drg = int(round((r1.starting_range - r2.starting_range) / r2.range_pixel_spacing))
    return daz, drg


def azimuth_ground_spacing(ref, demI, at, n=None, ranges=3):
    """Metres on the ground per azimuth pixel: median of long-baseline rdr2geo separations."""
    rg = ref["rg"]
    n = int(n or min(10000, max(1000, rg.length // 4)))
    i0 = int(min(max(at[0], 0), max(rg.length - n - 1, 0)))
    js = np.linspace(0.1, 0.9, ranges) * rg.width
    d = []
    for j0 in js:
        rr = rg.starting_range + float(j0) * rg.range_pixel_spacing
        try:
            p = [isce3.geometry.rdr2geo_bracket(rg.sensing_start + (i0 + k) / rg.prf, rr,
                                                ref["orbit"], rg.lookside, 0.0,
                                                rg.wavelength, dem=demI) for k in (0, n)]
        except Exception:
            continue
        d.append(float(np.linalg.norm(np.asarray(p[1]) - np.asarray(p[0])) / n))
    if not d:
        raise SystemExit("could not measure the azimuth ground spacing")
    v = float(np.median(d))
    print(f"  azimuth ground spacing {v:.4f} m/px over {n} lines at {len(d)} ranges "
          f"({', '.join(f'{x:.4f}' for x in d)}); range {rg.range_pixel_spacing:.4f} m/px, slant")
    return v


def inset_full_frame(ref, sec, win, gross, args):
    """Shrink the implicit full-frame window until its secondary counterpart exists."""
    gaz, grg = gross
    pad = (args.search[0] + args.winsize[0], args.search[1] + args.winsize[1])
    lo_a, lo_r = max(0, pad[0] - gaz), max(0, pad[1] - grg)
    hi_a = min(ref["shape"][0], sec["shape"][0] - gaz - pad[0])
    hi_r = min(ref["shape"][1], sec["shape"][1] - grg - pad[1])
    if hi_a - lo_a < 1 or hi_r - lo_r < 1:
        raise SystemExit("the gross offset leaves no overlap between the two frames")
    out = (lo_a, lo_r, hi_a - lo_a, hi_r - lo_r)
    if out != win:
        print(f"full frame inset to fit the secondary: azimuth {lo_a}..{hi_a}, "
              f"range {lo_r}..{hi_r} of {ref['shape'][0]} x {ref['shape'][1]}")
    return out


# ---------------------------------------------------------------- ampcor

def resolve_gpus(spec):
    """The CUDA device ids to shard the correlation over, from an explicit spec or 'all'."""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if vis:
        ids = list(range(len([v for v in vis.split(",") if v.strip()])))
        print(f"CUDA_VISIBLE_DEVICES={vis} -> device(s) "
              f"{','.join(str(i) for i in ids)} (renumbered by CUDA)")
        return ids
    if spec == "all":
        import subprocess
        try:
            q = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                               capture_output=True, text=True, check=True, timeout=60).stdout
            ids = [int(ln) for ln in q.splitlines() if ln.strip()]
        except Exception as e:
            raise SystemExit(f"--gpus all: cannot list the CUDA devices ({e}); "
                             f"name them explicitly, e.g. --gpus 0,1")
        if not ids:
            raise SystemExit("--gpus all: no CUDA device found")
    else:
        ids = [int(g) for g in str(spec).replace(",", " ").split()]
    if not ids:
        raise SystemExit("--gpus is empty")
    print(f"using CUDA device(s) {','.join(str(i) for i in ids)}")
    return ids


def _ampcor_band(job):
    """One PyCuAmpcor run over a horizontal band of the offset lattice, on one GPU."""
    from isce3.cuda.matchtemplate import PyCuAmpcor

    o = PyCuAmpcor()
    o.algorithm = 0                       # frequency-domain cross correlation
    o.deviceID = job["gpu"]
    # file reading: without these the reader defaults to a zero-size mmap and returns
    # garbage, which shows up as offsets uniformly spread over the search range regardless
    # of SNR -- not as an error
    o.useMmap = 1
    o.mmapSize = 8
    o.nStreams = job["streams"]
    o.derampMethod = job["deramp"]
    o.referenceImageName = job["fref"] + ".vrt"    # the GDAL wrapper, not the raw binary
    o.referenceImageHeight, o.referenceImageWidth = job["ref_shape"]
    o.secondaryImageName = job["fsec"] + ".vrt"
    o.secondaryImageHeight, o.secondaryImageWidth = job["sec_shape"]
    o.windowSizeHeight, o.windowSizeWidth = job["winsize"]
    o.halfSearchRangeDown, o.halfSearchRangeAcross = job["search"]
    o.skipSampleDown, o.skipSampleAcross = job["skip"]
    o.numberWindowDown, o.numberWindowAcross = job["nwd"], job["nwa"]
    o.numberWindowDownInChunk, o.numberWindowAcrossInChunk = job["chunk"]
    o.corrSurfaceOverSamplingFactor = job["oversample"]
    o.corrSurfaceOverSamplingMethod = 1
    o.corrSurfaceZoomInWindow = 16
    o.corrStatWindowSize = 21
    o.mergeGrossOffset = 0                # write the RESIDUAL around the gross offset
    o.referenceStartPixelDownStatic = job["search"][0] + job["m0"] * job["skip"][0]
    o.referenceStartPixelAcrossStatic = job["search"][1]
    o.offsetImageName = job["tag"] + ".bin"
    o.snrImageName = job["tag"] + "_snr.bin"
    o.covImageName = job["tag"] + "_cov.bin"
    o.corrImageName = job["tag"] + "_corr.bin"
    o.setupParams()
    # inside the two crops the residual alignment is exactly +pad, because the secondary
    # crop was cut at a0 + gross - pad while the reference crop starts at a0
    o.setConstantGrossOffset(*job["pad"])
    o.checkPixelInImageRange()
    o.runAmpcor()


def _ampcor_worker(gpu, jobs, done, total):
    """Run this GPU's bands in sequence, reporting each one (the only progress signal there
    is: PyCuAmpcor writes its output file at the END of a run, so a single whole-frame run
    is silent from start to finish)."""
    import time
    for job in jobs:
        t = time.time()
        _ampcor_band(job)
        with done.get_lock():
            done.value += job["nwd"]
            frac = 100.0 * done.value / total
        print(f"[gpu {gpu}] band {job['band']}: window rows {job['m0']}.."
              f"{job['m0'] + job['nwd']} in {time.time() - t:.0f} s "
              f"({frac:.0f} % of the lattice done)", flush=True)


def export_pair(args, ref, sec, win, gross):
    """Flat complex64 binaries for both dates, plus the geometry PyCuAmpcor needs."""
    a0, r0, na, nr = win
    gaz, grg = gross
    pad = (args.search[0] + args.winsize[0], args.search[1] + args.winsize[1])
    sa0, sr0 = a0 + gaz - pad[0], r0 + grg - pad[1]
    sna, snr_ = na + 2 * pad[0], nr + 2 * pad[1]
    if sa0 < 0 or sr0 < 0 or sa0 + sna > sec["shape"][0] or sr0 + snr_ > sec["shape"][1]:
        raise SystemExit(f"the secondary window {(sa0, sr0, sna, snr_)} falls outside "
                         f"{sec['shape']}; move the window inward or reduce --search")

    os.makedirs(args.scratch, exist_ok=True)
    print("exporting flat binaries for PyCuAmpcor (it is file-based):")
    fref = os.path.join(args.scratch, "ref.c8")
    fsec = os.path.join(args.scratch, "sec.c8")
    export_window(ref, (a0, r0, na, nr), fref)
    export_window(sec, (sa0, sr0, sna, snr_), fsec)
    return dict(fref=fref, ref_shape=(na, nr), fsec=fsec, sec_shape=(sna, snr_),
                sec_origin=(sa0, sr0), pad=pad)


def lattice_shape(shape, args, winsize=None, search=None, skip=None):
    """(nwd, nwa): how many correlation windows fit in a crop of this shape."""
    na, nr = shape
    ws = winsize or args.winsize
    se = search or args.search
    sk = skip or args.skip
    nwd = (na - ws[0] - 2 * se[0]) // sk[0]
    nwa = (nr - ws[1] - 2 * se[1]) // sk[1]
    if nwd < 1 or nwa < 1:
        raise SystemExit("the window is too small for this --winsize/--search/--skip")
    return nwd, nwa


def run_ampcor(args, pair, tag, winsize=None, search=None, skip=None, gross_in_crop=None,
               lattice=None):
    """PyCuAmpcor over an exported pair; returns (azimuth, range, snr) on the subgrid."""
    fref, fsec = pair["fref"], pair["fsec"]
    na, nr = pair["ref_shape"]
    winsize = tuple(winsize or args.winsize)
    search = tuple(search or args.search)
    skip = tuple(skip or args.skip)
    pad = tuple(gross_in_crop if gross_in_crop is not None else pair["pad"])
    # `lattice` trims what lattice_shape offers.  PyCuAmpcor's own
    # checkPixelInImageRange is stricter than the formula for some window/skip
    # combinations (it fails, it does not clamp), so a caller that has probed the
    # correlator can pass the count it will actually accept.
    nwd, nwa = lattice or lattice_shape(pair["ref_shape"], args, winsize, search, skip)
    print(f"offset grid: {nwd} x {nwa} windows (window {winsize[0]}x{winsize[1]}, "
          f"step {skip[0]}x{skip[1]} px, search {search[0]}x{search[1]})", flush=True)

    # Split the lattice into bands: one process per GPU, --bands-per-gpu bands each, dealt
    # out interleaved so every card gets a spread of the frame rather than one contiguous
    # slab (correlation cost varies with content -- sea windows are not land windows).
    # More bands than GPUs also buys a progress report per band.
    common = dict(streams=args.streams, deramp=args.deramp, fref=fref, fsec=fsec,
                  ref_shape=pair["ref_shape"], sec_shape=pair["sec_shape"],
                  winsize=winsize, search=search, skip=skip,
                  chunk=tuple(args.chunk), nwa=nwa,
                  oversample=args.oversample, pad=pad)
    nband = max(1, len(args.gpus) * args.bands_per_gpu)
    edges = np.linspace(0, nwd, nband + 1).round().astype(int)
    bands = [(int(edges[b]), int(edges[b + 1] - edges[b])) for b in range(nband)
             if edges[b + 1] > edges[b]]
    # A GPU chunk larger than the lattice it covers makes PyCuAmpcor read past its own
    # allocation -- "CUDA error ... code=700" from cuArrays.cu, not a Python-level error.
    # Only a coarse pass is small enough to hit it, so it is clamped per band.
    jobs = [dict(common, band=b, m0=m0, nwd=n, gpu=args.gpus[b % len(args.gpus)],
                 chunk=(min(args.chunk[0], n), min(args.chunk[1], nwa)),
                 tag=f"{tag}_b{b:03d}") for b, (m0, n) in enumerate(bands)]
    print(f"running PyCuAmpcor on GPU(s) {','.join(str(g) for g in args.gpus)}: "
          f"{len(jobs)} band(s) of ~{bands[0][1]} window rows", flush=True)

    if len(args.gpus) == 1 and len(jobs) == 1:
        _ampcor_band(jobs[0])                       # single band: no child process needed
    else:
        # spawn, not fork: a CUDA context cannot be inherited across fork, and the child
        # would fail the moment it touched the device
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        done = ctx.Value("q", 0)
        procs = [ctx.Process(target=_ampcor_worker,
                             args=(g, [j for j in jobs if j["gpu"] == g], done, nwd))
                 for g in args.gpus]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        bad = [p.exitcode for p in procs if p.exitcode != 0]
        if bad:
            raise SystemExit(f"a PyCuAmpcor worker failed (exit {bad}); see the log above")

    # Reassemble the bands in lattice order.  Each band wrote its own rows, so this is a
    # concatenation, not a merge -- nothing overlaps and nothing is interpolated.
    off = np.concatenate([np.fromfile(f"{j['tag']}.bin", dtype=np.float32)
                          .reshape(j["nwd"], nwa, 2) for j in jobs])
    snr = np.concatenate([np.fromfile(f"{j['tag']}_snr.bin", dtype=np.float32)
                          .reshape(j["nwd"], nwa) for j in jobs])
    if off.shape[0] != nwd:
        raise SystemExit(f"reassembled {off.shape[0]} window rows, expected {nwd}")
    for j in jobs:
        for suffix in (".bin", "_snr.bin", "_cov.bin", "_corr.bin"):
            if os.path.exists(j["tag"] + suffix):
                os.remove(j["tag"] + suffix)
    return off[..., 0], off[..., 1], snr


# ---------------------------------------------------------------- coregistration

def _sliced_grid(rg, a0, r0, na, nr):
    """RadarGridParameters of a crop: same sampling, origin moved to (a0, r0)."""
    return isce3.product.RadarGridParameters(
        rg.sensing_start + a0 / rg.prf, rg.wavelength, rg.prf,
        rg.starting_range + r0 * rg.range_pixel_spacing, rg.range_pixel_spacing,
        rg.lookside, int(na), int(nr), rg.ref_epoch)


def sampled_doppler(dop, prf):
    """The Doppler centroid folded into the SAMPLED band, which is what resamp_slc needs."""
    d = np.asarray(dop.data)
    w = (d + prf / 2.0) % prf - prf / 2.0
    n = int((np.abs(w - d) > 1e-6).sum())
    if n:
        print(f"[coreg] Doppler LUT folded into (-PRF/2, PRF/2]: {n} of {d.size} nodes moved, "
              f"mean {d.mean():+.1f} -> {w.mean():+.1f} Hz (PRF {prf:.1f})")
    lut = isce3.core.LUT2d(np.asarray(dop.x_axis), np.asarray(dop.y_axis), w)
    lut.bounds_error = dop.bounds_error
    return lut


def coregister_isce3(args, pair, ref_meta, sec_meta, win, dem_raster,
                     resample_only=False):
    """Geometric coregistration with ISCE3's own chain -- no offsets computed here."""
    a0, r0, na, nr = win
    sa0, sr0 = pair["sec_origin"]
    sna, snr_ = pair["sec_shape"]
    ref_grid = _sliced_grid(ref_meta["rg"], a0, r0, na, nr)
    sec_grid = _sliced_grid(sec_meta["rg"], sa0, sr0, sna, snr_)

    topo_dir = os.path.join(args.scratch, "rdr2geo")
    g2r_dir = os.path.join(args.scratch, "geo2rdr")
    os.makedirs(topo_dir, exist_ok=True)
    os.makedirs(g2r_dir, exist_ok=True)
    out = os.path.join(args.scratch, "sec_coreg.c8")
    done = all(os.path.exists(os.path.join(g2r_dir, f)) for f in ("range.off", "azimuth.off"))

    if (not done or args.force_coreg) and not resample_only:
        Rdr2Geo = isce3.cuda.geometry.Rdr2Geo if args.gpus else isce3.geometry.Rdr2Geo
        Geo2Rdr = isce3.cuda.geometry.Geo2Rdr if args.gpus else isce3.geometry.Geo2Rdr
        print(f"[coreg] rdr2geo on the reference grid ({na} x {nr}) ...", flush=True)
        r2g = Rdr2Geo(ref_grid, ref_meta["orbit"], ellip, zero_lut,
                      lines_per_block=args.coreg_lines_per_tile)
        # The rasters are created here, exactly as nisar.workflows.rdr2geo does, and NOT
        # left to topo(dem, outdir): that convenience overload builds them transposed in
        # this isce3 build -- give it a grid of length 300, width 100 and it makes a
        # 300x100 raster, then reports "(0,0) of size 100x300 on raster of 300x100" for
        # every block and writes a transposed, mostly-invalid product.
        layers = [("x", gdal.GDT_Float64), ("y", gdal.GDT_Float64), ("z", gdal.GDT_Float64),
                  ("incidence", gdal.GDT_Float32), ("heading", gdal.GDT_Float32),
                  ("localIncidence", gdal.GDT_Float32), ("localPsi", gdal.GDT_Float32),
                  ("simamp", gdal.GDT_Float32), ("layoverShadowMask", gdal.GDT_Byte)]
        rasters = [isce3.io.Raster(os.path.join(topo_dir, f"{n}.rdr"),
                                   ref_grid.width, ref_grid.length, 1, t, "ENVI")
                   for n, t in layers]
        r2g.topo(dem_raster, *rasters, None, None)
        # The VRT carries x, y, z ONLY.  geo2rdr reads the first three bands and nothing in
        # this pipeline ever opens the other six again -- topo() just insists on writing
        # them.  On this frame they are 62 GB of write-only intermediate, so they go as soon
        # as topo() returns rather than sitting in the scratch until cleanup.
        vrt = isce3.io.Raster(os.path.join(topo_dir, "topo.vrt"), rasters[:3])
        vrt.set_epsg(r2g.epsg_out)
        del vrt, rasters                      # flush before geo2rdr reads them back
        junk = 0
        for n, _ in layers[3:]:
            for ext in (".rdr", ".hdr", ".rdr.aux.xml"):
                p = os.path.join(topo_dir, n + ext)
                if os.path.exists(p):
                    junk += os.path.getsize(p)
                    os.remove(p)
        if junk:
            print(f"[coreg] dropped {len(layers) - 3} unused rdr2geo layers "
                  f"({junk / 1e9:.1f} GB); x/y/z kept")

        print(f"[coreg] geo2rdr into the secondary grid ...", flush=True)
        Geo2Rdr(sec_grid, sec_meta["orbit"], ellip, zero_lut,
                1.0e-8, 50, args.coreg_lines_per_tile).geo2rdr(
                    isce3.io.Raster(os.path.join(topo_dir, "topo.vrt")), g2r_dir)
    else:
        print(f"[coreg] reusing the offsets in {g2r_dir} (--force-coreg to redo)")

    rg_off = isce3.io.Raster(os.path.join(g2r_dir, "range.off"))
    az_off = isce3.io.Raster(os.path.join(g2r_dir, "azimuth.off"))
    for nm, ras in (("azimuth", az_off), ("range", rg_off)):
        a = gdal.Open(os.path.join(g2r_dir, f"{nm}.off")).ReadAsArray(nr // 2, na // 2, 3, 3)
        print(f"[coreg] geo2rdr {nm}.off at the crop centre: {a[1, 1]:+.4f} px")

    if (resample_only or not os.path.exists(out)
            or os.path.getsize(out) != na * nr * 8 or args.force_coreg):
        Resamp = isce3.cuda.image.ResampSlc if args.gpus else isce3.image.ResampSlc
        # the SECONDARY's native Doppler, not zero (see load_slc), and folded into the
        # sampled band or every fractional pixel of offset costs a fringe (sampled_doppler)
        resamp = Resamp(sampled_doppler(sec_meta["dop"], sec_grid.prf),
                        sec_grid.starting_range,
                        sec_grid.range_pixel_spacing,
                        sec_grid.sensing_start, sec_grid.prf, sec_grid.wavelength)
        # Tile height from the WIDTH, not a constant: the CUDA resampler holds roughly
        # chip_size^2 (9x9) complex values per output pixel, so memory scales with
        # tile_lines * width.  4096 lines is fine at the 7438-sample width of an AOI (~20 GB)
        # and asks for ~140 GB on the 52664-sample full frame -- which is exactly how this
        # first failed: "cudaErrorMemoryAllocation: out of memory".
        lpt = args.resamp_lines_per_tile or max(64, int(2 ** 24 / max(nr, 1)))
        resamp.lines_per_tile = lpt
        print(f"[coreg] resamp tile height {lpt} lines ({lpt * nr / 1e6:.1f} Mpx per tile)")
        out_raster = isce3.io.Raster(out, nr, na, 1, gdal.GDT_CFloat32, "ENVI")
        print(f"[coreg] resamp_slc onto the reference grid ({na} x {nr}) ...", flush=True)
        resamp.resamp(isce3.io.Raster(pair["fsec"] + ".vrt"), out_raster, rg_off, az_off, 1, False)
        del out_raster
    else:
        print(f"[coreg] reusing {out}")
    write_vrt(out, na, nr)
    return dict(pair, fsec=out, sec_shape=(na, nr), sec_origin=(a0, r0),
                sec_origin0=pair["sec_origin"])


def _add_constant(path, value, block=4096):
    """Add a constant to an offset raster in place (blockwise: these are ~22 GB)."""
    ds = gdal.Open(path, gdal.GA_Update)
    b = ds.GetRasterBand(1)
    for i0 in range(0, ds.RasterYSize, block):
        n = min(block, ds.RasterYSize - i0)
        a = b.ReadAsArray(0, i0, ds.RasterXSize, n)
        b.WriteArray(a + value, 0, i0)
    ds.FlushCache()
    ds = None


def _lattice_weights(n_out, n_lat, origin, skip):
    """(index, weight) for bilinear interpolation of a `skip`-spaced lattice onto n_out px."""
    t = np.clip((np.arange(n_out, dtype=np.float64) - origin) / skip, 0.0, n_lat - 1.0)
    i0 = np.minimum(t.astype(np.int64), n_lat - 2)
    return i0, t - i0


def upsample_add(path, field, args, block=1024, guard=-1e5):
    """Add a lattice-sampled offset field to a full-resolution .off raster, in place."""
    a = np.where(np.isfinite(field), field, 0.0).astype(np.float32)
    ds = gdal.Open(path)
    b = ds.GetRasterBand(1)
    na, nr = ds.RasterYSize, ds.RasterXSize
    # these rasters may have been rewritten as compressed GeoTIFFs to save disk.
    # A compressed tile whose new bytes do not fit cannot be overwritten in place -- GDAL
    # appends it and leaks the old one -- so on a compressed target write a fresh file and
    # rename.  Raw ENVI keeps the in-place path, which needs no second copy on disk.
    drv = ds.GetDriver().ShortName
    comp = ds.GetMetadataItem("COMPRESSION", "IMAGE_STRUCTURE")
    cow = drv == "GTiff" and bool(comp)
    if cow:
        tmp = path + ".part"
        opts = ["TILED=YES", "BLOCKXSIZE=512", "BLOCKYSIZE=512", "COMPRESS=DEFLATE",
                "PREDICTOR=3", "ZLEVEL=6", "NUM_THREADS=ALL_CPUS", "BIGTIFF=YES"]
        out_ds = gdal.GetDriverByName("GTiff").Create(tmp, nr, na, 1, b.DataType,
                                                      options=opts)
        ob = out_ds.GetRasterBand(1)
    else:
        ds = None
        ds = gdal.Open(path, gdal.GA_Update)
        b = ds.GetRasterBand(1)
        ob = b
    ii, wi = _lattice_weights(na, a.shape[0], args.search[0] + args.winsize[0] // 2,
                              args.skip[0])
    jj, wj = _lattice_weights(nr, a.shape[1], args.search[1] + args.winsize[1] // 2,
                              args.skip[1])
    # The range pass runs once over the lattice rows -- 3503 x 52520 float32, 735 MB here.
    # Doing both passes inside the block loop instead would gather the full-resolution
    # grid twice per block for no saving.
    wj = wj.astype(np.float32)
    cols = a[:, jj] * (1.0 - wj) + a[:, jj + 1] * wj
    del a
    for i0 in range(0, na, block):
        n = min(block, na - i0)
        r, w = ii[i0:i0 + n], wi[i0:i0 + n, None]
        d = cols[r] * (1.0 - w) + cols[r + 1] * w
        cur = b.ReadAsArray(0, i0, nr, n)
        ob.WriteArray(np.where(cur > guard, cur + d, cur), 0, i0)
    if cow:
        out_ds.FlushCache(); out_ds = None; ds = None
        os.replace(tmp, path)
    else:
        ds.FlushCache()
        ds = None
    v = field[np.isfinite(field)]
    print(f"[rbsheet] {os.path.basename(path)}: added a {field.shape[0]} x {field.shape[1]} "
          f"field, median {np.median(v):+.4f} px, p1..p99 {np.percentile(v, 1):+.3f} .. "
          f"{np.percentile(v, 99):+.3f} px")


def rubbersheet_state(g2r_dir, shape, axis="az", why="adding the increment only"):
    """The field already folded into <axis>.off by an earlier pass, or zeros."""
    name = {"az": "azimuth", "rg": "range"}[axis]
    p = os.path.join(g2r_dir, f"rubbersheet_{axis}.npy")
    if os.path.exists(p):
        a = np.load(p)
        if a.shape == shape:
            print(f"[rbsheet] {name}.off already carries a rubbersheet "
                  f"(std {np.nanstd(a):.3f} px); {why}")
            return a
        print(f"[rbsheet] {p} is {a.shape}, not {shape} -- ignoring it")
    return np.zeros(shape, np.float32)


def save_rubbersheet_state(g2r_dir, applied, axis="az"):
    """Persist the cumulative applied field, and always draw it."""
    np.save(os.path.join(g2r_dir, f"rubbersheet_{axis}.npy"), applied)
    ttl = f"applied {'range' if axis == 'rg' else 'azimuth'} rubbersheet"
    try:
        PL.plot_panels([(np.asarray(applied, np.float32), ttl, "px", False)], None,
                       "range [offset cells]", "azimuth [offset cells]",
                       os.path.join(g2r_dir, f"rubbersheet_{axis}.png"), aspect="auto")
    except Exception as exc:                        # a picture must never fail a run
        print(f"[rbsheet] quicklook for rubbersheet_{axis} skipped: {exc}")


HOLD_LP_KM = 8.0            # the wavelength a hole's rim can honestly carry across it


def rubbersheet(args, pair, sec_meta, g2r_dir, coarse, it):
    """Fold the measured residual back into the geometric offsets and resample again."""
    cws, cse, csk = coarse
    caz, crg, csnr = run_ampcor(args, pair, os.path.join(args.scratch, f"check{it}"),
                                winsize=cws, search=cse, skip=csk, gross_in_crop=(0, 0))
    d_az, d_rg, n = residual_constant(caz, crg, csnr, cse, args)
    print(f"[coreg] check {it}: residual over {n} windows  azimuth {d_az:+.4f} px "
          f"({d_az * args.az_spacing:+.3f} m), range {d_rg:+.4f} px "
          f"({d_rg * sec_meta['rg'].range_pixel_spacing:+.3f} m)", flush=True)
    return d_az, d_rg


def residual_constant(az, rg, snr, search, args):
    """Robust median residual of a coarse pass, in pixels: the leftover constant shift."""
    good = (np.isfinite(az) & np.isfinite(rg) & (snr >= args.coreg_snr_min)
            & (np.abs(az) < 0.9 * search[0]) & (np.abs(rg) < 0.9 * search[1]))
    if good.sum() < 20:
        raise SystemExit("coregistration check: too few good windows to verify")
    out = []
    for v in (az[good], rg[good]):
        m = np.median(v)
        s = 1.4826 * np.median(np.abs(v - m)) or 1e-6
        k = np.abs(v - m) < 3 * s
        out.append(float(np.median(v[k])))
    return out[0], out[1], int(good.sum())


def scene_epsg(ref_meta, demI, at):
    """UTM/polar EPSG of the scene centre -- decided ONCE per run."""
    from nisar.workflows.dumpconfig import point_to_epsg

    rg = ref_meta["rg"]
    i0, j0 = at
    xyz = isce3.geometry.rdr2geo_bracket(
        rg.sensing_start + i0 / rg.prf,
        rg.starting_range + j0 * rg.range_pixel_spacing,
        ref_meta["orbit"], rg.lookside, 0.0, rg.wavelength, dem=demI)
    lon, lat = np.degrees(ellip.xyz_to_lon_lat(xyz)[:2])
    epsg = int(point_to_epsg(float(lon), float(lat)))
    print(f"scene centre ({lat:.4f}, {lon:.4f}) -> EPSG {epsg} for every geocoded product")
    return epsg


def refocus_secondary(args, pair, ref_meta, sec_meta, win, applied):
    """--refocus-km: hand sec_coreg.c8 and the applied lattice to the refocus stage."""
    from . import refocus as RF
    na, nr = pair["ref_shape"]
    r1, r2 = ref_meta["rg"], sec_meta["rg"]
    sa0, sr0 = pair.get("sec_origin0", pair["sec_origin"])
    dr0 = ((r2.starting_range + sr0 * r2.range_pixel_spacing)
           - (r1.starting_range + win[1] * r1.range_pixel_spacing))
    d2 = sampled_doppler(sec_meta["dop"], r2.prf)
    fd = float(d2.eval(r2.sensing_start + (sa0 + na / 2) / r2.prf,
                       r2.starting_range + (sr0 + nr / 2) * r2.range_pixel_spacing))
    az_carrier = 2 * np.pi * fd / float(r2.prf)
    slant_mid = float(r1.starting_range + (win[1] + nr / 2) * r1.range_pixel_spacing)
    return RF.run_in_pipeline(args.scratch, (na, nr), applied, args.search, args.winsize, args.skip,
                              args.az_spacing, args.sec, float(r1.wavelength),
                              float(r1.range_pixel_spacing), dr0, az_carrier,
                              A_km=args.refocus_km, block=args.refocus_block, hop=args.refocus_hop,
                              gain=args.refocus_gain, gain_min_spread=args.refocus_gain_min_spread,
                              calib_blocks=args.refocus_calib_blocks, workers=args.refocus_workers,
                              keep_prefocus=args.refocus_keep_prefocus, slant_range_m=slant_mid)


def make_interferogram(args, pair, ref_meta, sec_meta, win, gg=None, screen=None):
    """Interferogram from the coregistered pair, via the interferogram stage."""
    from . import interferogram as C

    na, nr = pair["ref_shape"]
    r1, r2 = ref_meta["rg"], sec_meta["rg"]
    sr0 = pair.get("sec_origin0", pair["sec_origin"])[1]     # BEFORE resampling
    dr0 = ((r2.starting_range + sr0 * r2.range_pixel_spacing)
           - (r1.starting_range + win[1] * r1.range_pixel_spacing))
    roff = os.path.join(args.scratch, "geo2rdr", "range.off")
    flat = args.coreg == "geometric" and os.path.exists(roff)
    if args.coreg != "none" and not flat:
        print("[ifg] no geo2rdr offsets to flatten with (--coreg geometric writes them); "
              "the interferogram will keep its flat-earth and topographic fringes")

    argv = ["--ref", pair["fref"], "--sec", pair["fsec"],
            "--shape", str(na), str(nr),
            "--looks", str(args.ifg_looks[0]), str(args.ifg_looks[1]),
            "--out-dir", args.out_dir, "--tag", args.tag, "--scratch", args.scratch,
            "--wavelength", repr(float(r1.wavelength)),
            "--range-spacing", repr(float(r1.range_pixel_spacing)),
            "--coh-min", str(args.ifg_coh_min),
            "--ref-h5", args.ref, "--sec-h5", args.sec, "--dem", args.dem,
            # Without these the geocoding uses the interferogram stage's own defaults, which belong
            # to a different scene: the multilook radar grid then starts 96 lines and 33
            # samples away from where the crop actually does, and every geocoded product
            # from this path is mislocated by ~430 m along track.
            "--ref-origin", str(win[0]), str(win[1]),
            "--sec-origin", str(pair.get("sec_origin0", pair["sec_origin"])[0]),
            str(pair.get("sec_origin0", pair["sec_origin"])[1])]
    argv += (["--flatten", "--range-off", roff, "--dr0", repr(float(dr0))] if flat
             else ["--no-flatten"])

    # The other half of sampled_doppler's problem.  With the LUT folded the resampler now
    # reproduces s(a + delta) faithfully -- which means the secondary sample still carries
    # the SLC's azimuth carrier evaluated at a + delta, and ref * conj(sec) keeps
    # -psi' * delta.  psi' is -2.23 rad/px here, delta is the ionospheric misregistration,
    # so that is 2.3 rad for every pixel of it and it is not propagation phase.  Hand the
    # interferogram the field and the carrier and it comes out; see --az-carrier.
    aoff = os.path.join(args.scratch, "geo2rdr", "azimuth.off")
    if os.path.exists(aoff):
        d2 = sampled_doppler(sec_meta["dop"], r2.prf)
        sa0, sr1 = pair.get("sec_origin0", pair["sec_origin"])
        fd = float(d2.eval(r2.sensing_start + (sa0 + na / 2) / r2.prf,
                           r2.starting_range + (sr1 + nr / 2) * r2.range_pixel_spacing))
        print(f"[ifg] secondary azimuth carrier {fd:+.1f} Hz / {r2.prf:.1f} Hz PRF = "
              f"{2 * np.pi * fd / r2.prf:+.4f} rad per azimuth pixel of offset")
        argv += ["--azimuth-off", aoff, "--az-carrier", repr(2 * np.pi * fd / float(r2.prf))]
    if args.ifg_filter:
        argv += ["--filter"]
    zrdr = os.path.join(args.scratch, "rdr2geo", "z.rdr")
    if args.ifg_topo_phase and os.path.exists(zrdr):
        argv += ["--topo-phase", "--topo-z", zrdr]
    elif args.ifg_topo_phase:
        print(f"[ifg] --ifg-topo-phase needs {zrdr} (--coreg geometric --keep-scratch); "
              f"skipped")
    if screen:
        argv += ["--iono-screen", screen]
        if getattr(args, "iono_screen_calibrate", False):
            argv += ["--iono-screen-calibrate"]
    if args.ifg_unwrap:
        argv += ["--unwrap", "--unwrap-method", args.ifg_unwrap_method,
                 "--unwrap-coh-thresh", str(args.ifg_unwrap_coh_thresh)]
    if args.ifg_keep_rdr_npz:
        argv += ["--keep-rdr-npz"]
    if args.ifg_geocode:
        argv += ["--geocode", "--posting", str(args.ifg_posting)]
        if args.mask_water:
            # Water only, and only here: the ice is masked out of the MEASUREMENT because it
            # moves, not because the interferogram has nothing to say over it.
            argv += ["--mask-water", "--water-year", str(args.water_year),
                     "--cache-dir", args.cache_dir]
        if gg is not None:
            # The offsets' own lattice, so the two products are the same raster and
            # differencing them needs no resampling.  Left to itself the interferogram stage fits a
            # bbox to the multilook grid and lands half a cell off the stack lattice.
            argv += ["--geogrid", repr(float(gg.start_x)), repr(float(gg.start_y)),
                     repr(float(gg.spacing_x)), repr(float(gg.spacing_y)),
                     str(int(gg.width)), str(int(gg.length)), str(int(gg.epsg))]
        elif args.epsg:
            argv += ["--epsg", str(args.epsg)]
    print(f"[ifg] the interferogram stage {' '.join(argv)}", flush=True)
    C.main(argv)


# ---------------------------------------------------------------- geocoding

def offset_radar_grid(rg, win, args, shape):
    """Radar grid of the offset field: one sample per correlation window, at its CENTRE."""
    a0, r0 = win[0], win[1]
    nwd, nwa = shape
    return isce3.product.RadarGridParameters(
        rg.sensing_start + (a0 + args.search[0] + args.winsize[0] // 2) / rg.prf,
        rg.wavelength, rg.prf / args.skip[0],
        rg.starting_range + (r0 + args.search[1] + args.winsize[1] // 2)
        * rg.range_pixel_spacing,
        rg.range_pixel_spacing * args.skip[1], rg.lookside,
        nwd, nwa, rg.ref_epoch)


def external_grid(args):
    """The map lattice to geocode onto, as (geotransform, width, length, epsg), or None."""
    if args.grid_like:
        from osgeo import osr
        ds = gdal.Open(args.grid_like)
        if ds is None:
            raise SystemExit(f"cannot open --grid-like {args.grid_like}")
        srs = osr.SpatialReference(wkt=ds.GetProjection())
        code = srs.GetAuthorityCode(None)
        if not code:
            raise SystemExit(f"--grid-like {args.grid_like} has no EPSG in its projection")
        gt = ds.GetGeoTransform()
        print(f"target lattice from {args.grid_like}: {ds.RasterYSize} x {ds.RasterXSize} "
              f"@ ({gt[1]:g} x {abs(gt[5]):g}) m, EPSG {int(code)}")
        return gt, ds.RasterXSize, ds.RasterYSize, int(code)
    if args.stack:
        import zarr
        at = dict(zarr.open_group(args.stack, mode="r").attrs)
        missing = [k for k in ("geotransform", "shape", "epsg") if k not in at]
        if missing:
            raise SystemExit(f"--stack {args.stack} has no {', '.join(missing)} attribute")
        gt = tuple(float(v) for v in at["geotransform"])
        ny, nx = (int(v) for v in at["shape"])
        print(f"target lattice from {args.stack}: {ny} x {nx} @ "
              f"({gt[1]:g} x {abs(gt[5]):g}) m, EPSG {int(at['epsg'])}")
        return gt, nx, ny, int(at["epsg"])
    return None


def crop_to_offsets(gg, off_rg, orbit, demI):
    """Window `gg` down to the offsets' own footprint, staying exactly on its lattice."""
    dx, dy = gg.spacing_x, gg.spacing_y
    gf = isce3.product.bbox_to_geogrid(off_rg, orbit, zero_lut, abs(dx), -abs(dy),
                                       gg.epsg, min_height=demI.min_height,
                                       max_height=demI.max_height)
    x0, y0 = gf.start_x, gf.start_y
    x1, y1 = gf.start_x + gf.width * gf.spacing_x, gf.start_y + gf.length * gf.spacing_y
    c0 = max(0, int(np.floor((x0 - gg.start_x) / dx)))
    c1 = min(gg.width, int(np.ceil((x1 - gg.start_x) / dx)) + 1)
    r0 = max(0, int(np.floor((y0 - gg.start_y) / dy)))
    r1 = min(gg.length, int(np.ceil((y1 - gg.start_y) / dy)) + 1)
    if c1 <= c0 or r1 <= r0:
        raise SystemExit("the offsets fall outside the target lattice "
                         "(--grid-like / --stack); check the DEM and the window")
    if (c0, r0, c1, r1) == (0, 0, gg.width, gg.length):
        return gg
    print(f"cropped to the offsets' footprint: {r1 - r0} x {c1 - c0} cells at row {r0}, "
          f"col {c0} of {gg.length} x {gg.width} (--no-crop-to-data for the whole lattice)")
    return isce3.product.GeoGridParameters(gg.start_x + c0 * dx, gg.start_y + r0 * dy,
                                           dx, dy, c1 - c0, r1 - r0, gg.epsg)


def target_geogrid(args, off_rg, orbit, demI):
    """The output GeoGridParameters: --grid-like / --stack, else fitted to the offsets."""
    ext = external_grid(args)
    if ext is not None:
        gt, nx, ny, epsg = ext
        if args.epsg and int(args.epsg) != epsg:
            print(f"  NOTE: --epsg {args.epsg} does not apply; the target lattice is in "
                  f"EPSG {epsg} and is used as it stands")
        gg = isce3.product.GeoGridParameters(gt[0], gt[3], gt[1], gt[5], nx, ny, epsg)
        crop = args.crop_to_data if args.crop_to_data is not None else bool(args.stack)
        return crop_to_offsets(gg, off_rg, orbit, demI) if crop else gg

    epsg = args.epsg
    if epsg is None:
        # NISAR L2 projection: polar stereographic
        # beyond |lat| 60, else the UTM zone of the scene centre
        from nisar.workflows.dumpconfig import point_to_epsg
        c = isce3.geometry.rdr2geo_bracket(
            off_rg.sensing_mid, 0.5 * (off_rg.starting_range + off_rg.end_range), orbit,
            off_rg.lookside, 0.0, off_rg.wavelength, dem=demI)
        lon, lat = np.degrees(ellip.xyz_to_lon_lat(c)[:2])
        epsg = int(point_to_epsg(float(lon), float(lat)))
        print(f"scene centre ({lat:.4f}, {lon:.4f}) -> EPSG {epsg}")
    px, py = args.posting if args.posting else (
        # posting from the offsets' own ground sampling: finer would only interpolate,
        # coarser would throw measurements away
        round(off_rg.range_pixel_spacing / 10) * 10.0,
        round(args.skip[0] * args.az_spacing / 10) * 10.0)
    gg = isce3.product.bbox_to_geogrid(off_rg, orbit, zero_lut, px, -py, int(epsg),
                                       min_height=demI.min_height,
                                       max_height=demI.max_height)
    print(f"target grid fitted to the offsets: {gg.length} x {gg.width} @ "
          f"({px} x {py}) m, EPSG {epsg}")
    return gg


def geocode_layers(layers, off_rg, orbit, dem_raster, gg, args):
    """Geocode each radar-grid offset layer onto `gg`; returns arrays on the map grid."""
    g = isce3.geocode.GeocodeFloat32()
    g.orbit = orbit
    g.ellipsoid = ellip
    g.doppler = zero_lut                  # the focused SLCs (hence the offsets) are zero-Doppler
    g.threshold_geo2rdr = 1.0e-8
    g.numiter_geo2rdr = 25
    g.data_interpolator = args.interp
    g.geogrid(gg.start_x, gg.start_y, gg.spacing_x, gg.spacing_y,
              gg.width, gg.length, gg.epsg)

    out = {}
    for name, arr in layers.items():
        src = os.path.join(args.scratch, f"rdr_{name}.tif")
        dst = os.path.join(args.scratch, f"geo_{name}.tif")
        _write_f32(src, arr)
        # NaN-initialised, so cells the geocoder never maps stay nodata instead of reading
        # as a real zero offset
        _write_f32(dst, np.full((gg.length, gg.width), np.nan, np.float32))
        print(f"[geocode] {name} ...", flush=True)
        g.geocode(radar_grid=off_rg, input_raster=isce3.io.Raster(src),
                  output_raster=isce3.io.Raster(dst, update=True),
                  dem_raster=dem_raster,
                  output_mode=isce3.geocode.GeocodeOutputMode.INTERP)
        out[name] = gdal.Open(dst).ReadAsArray().astype(np.float32)
        v = out[name]
        print(f"[geocode] {name}: {100 * np.isfinite(v).mean():.1f} % of cells filled")
    return out


def _write_f32(path, arr):
    ds = gdal.GetDriverByName("GTiff").Create(path, int(arr.shape[1]), int(arr.shape[0]),
                                              1, gdal.GDT_Float32)
    ds.GetRasterBand(1).WriteArray(np.asarray(arr, np.float32))
    ds.FlushCache()


# ---------------------------------------------------- radar-lattice geolocation

def lattice_map_coords(args, ref_meta, demI, win, lat_shape, epsg, need, key="offsets"):
    """Map coordinates (X, Y) in `epsg` of every correlation-window centre, NaN elsewhere."""
    from pyproj import Transformer

    nwd, nwa = lat_shape
    ii = win[0] + args.search[0] + args.winsize[0] // 2 + np.arange(nwd) * args.skip[0]
    jj = win[1] + args.search[1] + args.winsize[1] // 2 + np.arange(nwa) * args.skip[1]
    X = np.full(lat_shape, np.nan)
    Y = np.full(lat_shape, np.nan)

    topo = os.path.join(args.scratch, "rdr2geo")
    xp, yp = os.path.join(topo, "x.rdr"), os.path.join(topo, "y.rdr")
    dsx = gdal.Open(xp) if os.path.exists(xp) else None
    dsy = gdal.Open(yp) if os.path.exists(yp) else None
    if dsx is not None and dsy is not None:
        if (dsx.RasterYSize, dsx.RasterXSize) != (win[2], win[3]):
            print(f"[geoloc] {topo} is {dsx.RasterYSize} x {dsx.RasterXSize}, not the "
                  f"{win[2]} x {win[3]} window; falling back to per-window rdr2geo")
        else:
            # Rdr2Geo writes x/y in its epsg_out, which coregister_isce3 leaves at the
            # isce3 default (4326, so degrees); read it back rather than assuming.
            src_epsg = 4326
            vrt = os.path.join(topo, "topo.vrt")
            if os.path.exists(vrt):
                from osgeo import osr
                code = osr.SpatialReference(
                    wkt=gdal.Open(vrt).GetProjection()).GetAuthorityCode(None)
                if code:
                    src_epsg = int(code)
            print(f"[geoloc] window centres from {topo} (EPSG {src_epsg}), the rdr2geo "
                  f"the coregistration already ran", flush=True)
            bx, by = dsx.GetRasterBand(1), dsy.GetRasterBand(1)
            c0, span, step = int(jj[0] - win[1]), int(jj[-1] - jj[0]) + 1, int(args.skip[1])
            xs = np.empty(lat_shape)
            ys = np.empty(lat_shape)
            for m, r in enumerate(ii - win[0]):
                xs[m] = bx.ReadAsArray(c0, int(r), span, 1)[0, ::step]
                ys[m] = by.ReadAsArray(c0, int(r), span, 1)[0, ::step]
            # rdr2geo leaves the pixels it could not solve at exactly zero, which is a real
            # coordinate in the Gulf of Guinea and must not reach the mask lookup
            ok = np.isfinite(xs) & np.isfinite(ys) & ~((xs == 0) & (ys == 0))
            tr = Transformer.from_crs(src_epsg, epsg, always_xy=True)
            X[ok], Y[ok] = tr.transform(xs[ok], ys[ok])
            print(f"[geoloc] {int(ok.sum())} of {ok.size} window centres located")
            return X, Y

    # keyed on the offsets file, not the run: the geolocation depends only on the lattice
    # and the geometry, so every run over one offset field shares it -- including the ones
    # the geocoded offset products use exactly this name
    cache = (os.path.join(args.cache_dir, f"geoloc_{key}_{nwd}x{nwa}_{epsg}.npz")
             if args.geoloc_cache else None)
    if cache and os.path.exists(cache):
        c = np.load(cache)
        if c["X"].shape == X.shape:
            X, Y = c["X"].copy(), c["Y"].copy()
            print(f"[geoloc] {int(np.isfinite(X).sum())} window centres from {cache}")
    todo = need & ~np.isfinite(X)
    if todo.any():
        print(f"[geoloc] rdr2geo for {int(todo.sum())} window centres "
              f"(no coregistration rasters to read them from)", flush=True)
        gi, gj = np.where(todo)
        xs, ys, ok = M.geolocate(gi, gj, ii, jj, ref_meta["rg"], ref_meta["orbit"],
                                 demI, epsg)
        X[gi[ok], gj[ok]] = xs[ok]
        Y[gi[ok], gj[ok]] = ys[ok]
        if cache:
            os.makedirs(args.cache_dir, exist_ok=True)
            M._save_atomic(cache, X, Y)
            print(f"[geoloc] cached {int(np.isfinite(X).sum())} centres in {cache}")
    return X, Y


def clean_lattice(args, ref_meta, demI, win, npz, gg, gt, az, rg, snr, smooth_out=None):
    """--snr-min gate -> radar-domain water/ice mask -> outlier rejection -> hole fill."""
    def applied_rubbersheet():
        """What an earlier pass -- or an earlier RUN -- already folded into <axis>.off."""
        g2r = os.path.join(args.scratch, "geo2rdr")
        out = {}
        for name, axis in (("azimuth", "az"), ("range", "rg")):
            f = os.path.join(g2r, f"rubbersheet_{axis}.npy")
            if not os.path.exists(f):
                continue
            a = np.load(f)
            if a.shape == az.shape and np.any(a):
                out[name] = a
        return out or None
    good = np.isfinite(az) & np.isfinite(rg) & (snr >= args.snr_min)
    print(f"windows with snr >= {args.snr_min}: {good.sum()} of {good.size} "
          f"({100 * good.mean():.1f} %)")
    if args.reject_at_search_limit:
        lim = good & ((np.abs(az) >= 0.9 * args.search[0]) |
                      (np.abs(rg) >= 0.9 * args.search[1]))
        print(f"windows at the search limit, dropped: {int(lim.sum())} "
              f"({100 * lim.mean():.2f} %)")
        good &= ~lim
    if not good.any():
        raise SystemExit("no windows survive --snr-min; lower it or widen --search")
    for name, v in (("azimuth", az), ("range", rg)):
        q = v[good]
        print(f"  {name:8s} median {np.median(q):+.3f} px  "
              f"p1..p99 {np.percentile(q, 1):+.3f} .. {np.percentile(q, 99):+.3f}  "
              f"std {q.std():.3f}")

    keep = good.copy()
    filled = np.zeros(az.shape, bool)
    azf = np.where(keep, az, np.nan).astype(np.float32)
    rgf = np.where(keep, rg, np.nan).astype(np.float32)
    snrf = np.where(keep, snr, np.nan).astype(np.float32)

    rdr_mask = (args.mask_where in ("rdr", "both")
                and (args.mask_water or args.mask_glacier))
    if rdr_mask or args.rdr_fill:
        if rdr_mask:
            # Mask BEFORE geocoding.  Masking afterwards cannot undo the damage: the
            # interpolation has already carried glacier and water offsets into the land
            # cells around them, and blanking the on-ice cells leaves that behind.
            with timed("locate window centres"):
                X, Y = lattice_map_coords(args, ref_meta, demI, win, az.shape, gg.epsg,
                                          good,
                                          key=os.path.splitext(os.path.basename(npz))[0])
            keep &= np.isfinite(X)
            with timed("radar-domain mask"):
                bad = M.rdr_mask_samples(X[keep], Y[keep], gt, gg.epsg, args)
            ki, kj = np.where(keep)
            print(f"[rdr-mask] {int(bad.sum())} of {ki.size} windows on water or ice "
                  f"({100 * bad.mean():.1f} %)")
            keep[ki[bad], kj[bad]] = False
        if args.outlier_mad > 0:
            with timed("outlier rejection"):
                keep &= ~M.reject_outliers(az, rg, keep, args.search[0], args.search[1],
                                           args.outlier_mad)
        if args.outlier_spike > 0 or args.outlier_min_neighbors > 0:
            with timed("spike rejection"):
                keep &= ~M.reject_spikes(az, rg, keep, args.outlier_spike,
                                         args.outlier_min_neighbors,
                                         box=args.outlier_spike_box)
        azf = np.where(keep, az, np.nan).astype(np.float32)
        rgf = np.where(keep, rg, np.nan).astype(np.float32)
        snrf = np.where(keep, snr, np.nan).astype(np.float32)
        if args.rdr_fill:
            cell = (args.skip[0] * args.az_spacing,
                    args.skip[1] * float(ref_meta["rg"].range_pixel_spacing))
            rb = applied_rubbersheet()
            prior_from = (None if rb is None else
                          {n: (azf if n == "azimuth" else rgf) + a for n, a in rb.items()})
            with timed("fill holes (radar lattice)"):
                azf, rgf, snrf, filled = M.fill_holes(azf, rgf, snrf, keep, args,
                                                      smooth_out=smooth_out, cell=cell,
                                                      prior_from=prior_from)
    return keep, azf, rgf, snrf, filled


def iono_screen(args, ref_meta, win, az, keep, off_rg, orbit, dem_raster, gg, filled=None):
    """Build the ionospheric phase screen and geocode it; returns the npz for the ifg."""
    phi, valid = ION.screen_from_offsets(args, ref_meta, win, az, keep, filled)
    path = ION.save_screen(args, phi, valid, win)
    with timed("geocode (ionospheric screen)"):
        geo = geocode_layers({"iono_screen": phi,
                              "iono_valid": valid.astype(np.float32)},
                             off_rg, orbit, dem_raster, gg, args)
    p = os.path.join(args.out_dir, f"iono_screen_{args.tag}.tif")
    B.save_gtiff(p, geo["iono_screen"], gg, nodata=np.nan)
    png_for(p, geo["iono_screen"])
    p = os.path.join(args.out_dir, f"iono_screen_valid_{args.tag}.tif")
    B.save_gtiff(p, geo["iono_valid"], gg, nodata=np.nan)
    png_for(p, geo["iono_valid"], gray=True, vlim=(0.0, 1.0))
    return path


def cleanup_scratch(args):
    """Delete the intermediate rasters in --scratch.  On by default; --keep-scratch keeps."""
    import glob
    import shutil

    if args.keep_scratch:
        print(f"keeping the intermediates in {args.scratch} (--keep-scratch)")
        return
    victims = [os.path.join(args.scratch, d) for d in ("rdr2geo", "geo2rdr")]
    # *.hdr and *.aux.xml because the ENVI driver names the sidecar after the stem, not the
    # file: sec_coreg.c8 gets sec_coreg.hdr, range.off gets range.hdr
    for pat in ("*.off", "*.hdr", "*.aux.xml", "rdr_*.tif", "geo_*.tif", "ml_*.tif",
                "offsets*.bin", "check*.bin"):
        victims += glob.glob(os.path.join(args.scratch, pat))
    binaries = ("ref.c8", "sec.c8", "sec_coreg.c8")
    if args.keep_binaries:
        keep = {os.path.join(args.scratch, os.path.splitext(f)[0] + s)
                for f in binaries for s in (".hdr", ".aux.xml")}
        victims = [p for p in victims if p not in keep]
    else:
        for f in binaries:
            victims += [os.path.join(args.scratch, f + s) for s in ("", ".vrt")]
    freed, n = 0, 0
    for p in victims:
        if os.path.isdir(p):
            freed += sum(os.path.getsize(os.path.join(r, f))
                         for r, _, fs in os.walk(p) for f in fs)
            shutil.rmtree(p, ignore_errors=True)
            n += 1
        elif os.path.exists(p):
            freed += os.path.getsize(p)
            os.remove(p)
            n += 1
    print(f"removed {n} intermediate file(s) from {args.scratch}, {freed / 1e9:.1f} GB "
          f"(--keep-scratch to keep them, --keep-binaries for just the SLC crops)")


# ---------------------------------------------------------------- plots

# Colour, stretch and colourbar all live in the plotting helpers now -- these two are the call sites
# this module has always had, kept so nothing below changes, but the decisions are made in
# one place for every script in the pipeline.
def png_for(path, arr, gray=False, vlim=None):
    """The single-raster quicklook for one GeoTIFF, coloured like its panel."""
    return PL.quicklook(path, arr, gray=gray, vlim=vlim)


def plot_panels(panels, ext, xlabel, ylabel, png, aspect="equal"):
    """Three-panel quicklook: azimuth offset, range offset, SNR."""
    return PL.plot_panels(panels, ext, xlabel, ylabel, png, aspect=aspect)


# ---------------------------------------------------------------- driver

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="reference focused SLC / RSLC (HDF5)")
    ap.add_argument("--sec", required=True, help="secondary focused SLC / RSLC (HDF5)")
    ap.add_argument("--dem", required=True,
                    help="DEM for the gross offset and the geocoding")
    ap.add_argument("--freq", default="A", choices=["A", "B"], help="NISAR RSLC input only")
    ap.add_argument("--pol", default="HH", help="NISAR RSLC input only")
    ap.add_argument("--tag", default=None,
                    help="output filename tag (default: t<track>_<refdate>_<secdate>, "
                         "read from the input filenames)")

    g = ap.add_argument_group("what to correlate")
    g.add_argument("--window", nargs=4, type=int, default=None,
                   metavar=("AZ0", "RG0", "NAZ", "NRG"),
                   help="sub-window of the REFERENCE grid (default: the whole frame, which "
                        "is ~45 GB of scratch per date at complex64)")

    g.add_argument("--winsize", nargs=2, type=int, default=(64, 64), metavar=("AZ", "RG"),
                   help="correlation window (default 64 64)")
    g.add_argument("--search", nargs=2, type=int, default=(32, 32), metavar=("AZ", "RG"),
                   help="half search range in px (default 32 32).  A search that is too "
                        "small does not report a large offset, it silently rejects the "
                        "window")
    g.add_argument("--skip", nargs=2, type=int, default=(32, 32), metavar=("AZ", "RG"),
                   help="step between windows (default 32 32); this, not --winsize, sets "
                        "the ground sampling of the product")
    g.add_argument("--gross", nargs=2, default=("geom", "geom"), metavar=("AZ", "RG"),
                   help="gross offset: 'geom' (default) predicts it from orbit + DEM at "
                        "the window centre, or give explicit integers")
    g.add_argument("--oversample", type=int, default=32,
                   help="correlation-surface oversampling for sub-pixel (default 32)")
    g.add_argument("--deramp", type=int, default=1,
                   help="0 = magnitude only, 1 = complex with deramp (default 1)")

    g = ap.add_argument_group("coregistration (runs BEFORE the residual offsets)")
    g.add_argument("--coreg", choices=["geometric", "none"], default="geometric",
                   help="geometric (default): rdr2geo on the reference, geo2rdr into the "
                        "secondary and resamp_slc, so the offsets follow the terrain.  "
                        "none: no coregistration, leaving the whole geometric field in "
                        "the product")

    g.add_argument("--coreg-winsize", nargs=2, type=int, default=(128, 128),
                   metavar=("AZ", "RG"), help="coarse-pass correlation window (default "
                                              "128 128: bigger windows, more robust peaks)")
    g.add_argument("--coreg-skip", nargs=2, type=int, default=(1024, 1024),
                   metavar=("AZ", "RG"),
                   help="coarse-pass step (default 1024 1024); the fit needs a few thousand "
                        "good windows spread over the frame, not a dense field")
    g.add_argument("--coreg-snr-min", type=float, default=8.0,
                   help="SNR gate for windows entering the fit (default 8, stricter than "
                        "--snr-min: a bad peak biases every pixel through the polynomial)")
    g.add_argument("--coreg-iters", type=int, default=3, metavar="N",
                   help="resample / verify / absorb-the-leftover cycles (default 3).  The "
                        "check is what catches a resampling that did not land where it was "
                        "told -- see coregister()")
    g.add_argument("--coreg-tol", type=float, default=0.05, metavar="PX",
                   help="stop iterating once the leftover constant is below this (default "
                        "0.05 px).  Do not set it below 1/--oversample: the correlator "
                        "reports offsets quantised to that step (1/32 = 0.031 px), so a "
                        "tighter tolerance can never be met and the loop just burns "
                        "iterations against its own resolution floor")
    g.add_argument("--coreg-lines-per-tile", type=int, default=4096,
                   help="rdr2geo / geo2rdr block height (default 4096)")
    g.add_argument("--resamp-lines-per-tile", type=int, default=0, metavar="N",
                   help="ResampSlc tile height; 0 (default) picks it from the frame width, "
                        "since GPU memory scales with tile_lines * width * chip_size^2")
    g.add_argument("--force-coreg", action="store_true",
                   help="redo the resampling even if sec_coreg.c8 is already there")
    g.add_argument("--rubbersheet-az", type=int, default=0, metavar="N",
                   help="fold the DENSE azimuth offset field back into geo2rdr/azimuth.off "
                        "and resample again, N times (default 0 = off).  The loop above "
                        "removes only a CONSTANT, so a per-pixel azimuth shift -- an "
                        "along-track TEC gradient -- survives it and decorrelates the "
                        "interferogram wherever it approaches the azimuth resolution.  "
                        "Costs one blockwise pass over the ~24 GB offset raster plus a "
                        "resamp per iteration.  Needs --coreg geometric, and the scratch "
                        "from the run that made it (--keep-scratch)")


    g.add_argument("--rubbersheet-az-redense", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="re-measure the dense offsets against the rubbersheeted secondary "
                        "(default on).  The azimuth product stays relative to GEOMETRY: "
                        "the applied field is added back to what the correlator then "
                        "reports, so the signal is removed from the interferogram and not "
                        "from the measurement")
    g.add_argument("--refocus-km", default="0", metavar="KM|auto",
                   help="Doppler-dependent azimuth refocus of the rubbersheeted secondary "
                        "run once after the last rubbersheet resample "
                        "and its redense pass, so the offsets products are measured on the "
                        "resampled secondary as before and only the interferogram and the "
                        "routes see the refocused one.  A target's synthetic aperture pierces "
                        "~9 km of ionosphere, so where the applied field changes by pixels "
                        "over a few km each Doppler sub-aperture still needs a DIFFERENT "
                        "shift and no single resampling registers the band, which leaves "
                        "a dark band on a streak's steep flanks.  The "
                        "refocus applies, per Doppler component, the applied field seen at "
                        "that sub-aperture's pierce point, displaced along track by "
                        "KM * f / bandwidth.  'auto' calibrates KM on the frame's steepest "
                        "blocks (8 km there, an ionosphere near 275 km) and does nothing "
                        "when that gains under 0.01; a number uses it as is.  Measured on "
                        "the frame: steep mask 0.41 -> 0.53, rim 0.50 -> 0.55, quiet ground "
                        "unchanged.  Default 0 = off, so T121/T135 runs are untouched; "
                        "turn it on per pair through COREG_EXTRA")
    g.add_argument("--refocus-gain", default="auto", metavar="auto|G",
                   help="'auto' (default) tests a local gain 0..1 on the refocus phase in "
                        "every --refocus-block x 2048 block whose predicted spread exceeds "
                        "--refocus-gain-min-spread px and keeps the one with the best "
                        "coherence, so a feature that is NOT Doppler-dependent -- on T087 a "
                        "-4.2 px plateau 2.4 km long with all quarter-band coherences ~0.3 -- "
                        "is left alone instead of being made worse; a number applies that "
                        "gain everywhere")
    g.add_argument("--refocus-gain-min-spread", type=float, default=1.0, metavar="PX")
    g.add_argument("--refocus-block", type=int, default=240, metavar="LINES",
                   help="azimuth block of the Doppler-domain filter (default 240)")
    g.add_argument("--refocus-hop", type=int, default=120, metavar="LINES",
                   help="its hop; Hann overlap-add when smaller than the block (default 120)")
    g.add_argument("--refocus-calib-blocks", type=int, default=24,
                   help="steepest blocks the 'auto' aperture is calibrated on (default 24)")
    g.add_argument("--refocus-workers", type=int, default=6,
                   help="CPU workers for the gain search and the filter (default 6)")
    g.add_argument("--refocus-keep-prefocus", action="store_true",
                   help="keep the resampled-only secondary as sec_coreg_prefocus.c8 (23.5 GB)")

    g = ap.add_argument_group("interferogram from the coregistered pair (the interferogram stage)")
    g.add_argument("--ifg", action="store_true",
                   help="after coregistering, also form the interferogram from the same "
                        "pair.  Flattened with the geo2rdr offsets (orbit + DEM) that the "
                        "coregistration itself used")
    g.add_argument("--ifg-looks", nargs=2, type=int, default=(24, 16), metavar=("AZ", "RG"),
                   help="multilook factors (default 24 16: 107 m along track x 50 m slant range, 68-92 m "
                        "on the ground, on the 40 MHz modes; the 20 MHz mode 2005 uses 24 8 for the same cell)")
    g.add_argument("--ifg-filter", action="store_true", help="Goldstein filter")
    g.add_argument("--ifg-topo-phase", action="store_true",
                   help="fit and remove a phase term proportional to elevation "
                        "(tropospheric stratification); see the interferogram stage's --topo-phase, "
                        "which also explains why it is off by default")
    g.add_argument("--ifg-unwrap", action="store_true",
                   help="unwrap in radar coordinates, before any geocoding")
    g.add_argument("--ifg-unwrap-method", default="snaphu",
                   choices=["snaphu", "phass", "icu"])
    g.add_argument("--ifg-unwrap-coh-thresh", type=float, default=0.2)
    g.add_argument("--ifg-geocode", action="store_true", help="geocode the interferogram")
    g.add_argument("--ifg-keep-rdr-npz", action="store_true",
                   help="also keep the interferogram on its own RADAR multilook grid "
                        "(the interferogram stage's --keep-rdr-npz).  The geocoded product has been "
                        "through a resample; the radar one has not, and it is the only grid "
                        "on which the phase can be compared per-pixel against a field that "
                        "lives on the correlation lattice")
    g.add_argument("--ifg-posting", type=float, default=90.0,
                   help="interferogram output posting in m (default 90)")
    g.add_argument("--ifg-coh-min", type=float, default=0.2,
                   help="coherence below this is blanked in the quicklooks (default 0.2)")
    g.add_argument("--iono-screen", action="store_true",
                   help="build the ionospheric phase screen by integrating the azimuth "
                        "offsets along track, and write a corrected interferogram beside "
                        "the uncorrected one.  Independent of --rubbersheet-az: that fixes "
                        "the registration, this removes the phase")
    g.add_argument("--iono-screen-km", type=float, default=2.0, metavar="KM",
                   help="low-pass (Gaussian sigma) applied to the azimuth field before "
                        "integrating, in km (default 2).  It keeps the integration from "
                        "random-walking on per-window noise, but it is not free: a longer "
                        "sigma leaves the screen progressively short of the interferogram's "
                        "own azimuth phase gradient, while below ~1 km the offset noise "
                        "starts to win it back")
    g.add_argument("--iono-integrate-km", type=float, default=0.0, metavar="KM",
                   help="damp the along-track integration above this wavelength "
                        "(default 0 = a plain cumsum).  The cumsum inverts the difference "
                        "operator exactly, including at zero frequency where it has "
                        "unbounded gain, so it turns the integrand's noise into a RANDOM "
                        "WALK: past some along-track wavelength the screen adds structure "
                        "rather than removing it.")
    g.add_argument("--iono-screen-passes", type=int, default=1, metavar="N",
                   help="residual-correction passes for that low-pass (default 1, the "
                        "plain Gaussian).  Iterating gives response 1-(1-H)^n, flattening "
                        "the passband while the stopband stays shut.  Leave this at 1 "
                        "unless there is a measured reason not to")
    g.add_argument("--iono-screen-gain", type=float, default=1.0, metavar="ALPHA",
                   help="gain on the along-track integration: the derivative of the "
                        "interferogram is fitted as alpha * (azimuth-shift observable) + "
                        "beta, and alpha rather than the theoretical constant is what gets "
                        "integrated.  Default 1.0 is the theoretical constant alone.  Alpha "
                        "is frame dependent, so measure it before setting it "
                        "(--iono-screen-calibrate prints it)")
    g.add_argument("--iono-screen-calibrate", action="store_true",
                   help="have the interferogram stage fit the gain "
                        "against this pair and PRINT the gain the screen is short "
                        "by.  Diagnostic only; set --iono-screen-gain to act on it")
    g.add_argument("--iono-shell-km", type=float, default=350.0, metavar="H",
                   help="effective ionospheric shell height, km (default 350).  MUST match "
                        "what the route driver uses, because screen_offsets prefers the "
                        "screen this run writes over rebuilding its own -- so a mismatch "
                        "here silently delivers the wrong screen.  0 puts the screen on "
                        "the ground instead.  See the offsets screen's pierce_sweep")
    g.add_argument("--iono-screen-max-gap", type=float, default=8.0, metavar="KM",
                   help="cells further than this from any surviving measurement are "
                        "flagged invalid in the screen (default 8 km).  The largest glacier "
                        "holes on this frame are 10-28 km across, the same scale as the "
                        "signal, so beyond a few km the screen is extrapolation and is "
                        "marked rather than quietly applied")

    g = ap.add_argument_group("GPU")
    g.add_argument("--gpus", default="0", metavar="LIST",
                   help="CUDA devices to shard the offset lattice over: a list such as "
                        "0,1,2,3, or 'all' (default 0).  PyCuAmpcor is single-device, so "
                        "each GPU gets its own process and its own bands of window rows")
    g.add_argument("--bands-per-gpu", type=int, default=4, metavar="N",
                   help="bands each GPU processes in sequence (default 4).  More bands "
                        "means finer progress reporting -- a band reports when it lands, "
                        "and one whole-frame run reports nothing until it is finished")
    g.add_argument("--streams", type=int, default=2)
    g.add_argument("--chunk", nargs=2, type=int, default=(16, 16), metavar=("D", "A"),
                   help="windows per GPU chunk (default 16 16); lower it if the card runs "
                        "out of memory")

    g = ap.add_argument_group("output")
    g.add_argument("--snr-min", type=float, default=5.0,
                   help="drop windows below this correlation SNR before geocoding "
                        "(default 5); failed matches must not be interpolated in")
    g.add_argument("--reject-at-search-limit", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="also drop matches sitting at the edge of the search window "
                        "(default on): those are the correlator running out of room to "
                        "look, not measurements of a large offset")
    g.add_argument("--units", choices=["px", "m", "both"], default="both",
                   help="write offsets in pixels, metres, or both (default both)")
    g.add_argument("--posting", nargs=2, type=float, default=None, metavar=("X", "Y"),
                   help="output posting in m (default: the offsets' own ground sampling)")
    g.add_argument("--epsg", type=int, default=None,
                   help="output projection (default: UTM zone of the scene centre, "
                        "polar stereographic beyond |lat| 60)")
    g.add_argument("--grid-like", default=None, metavar="TIF",
                   help="geocode onto the grid of this GeoTIFF, e.g. an "
                        "insar_l0_output_*/coh_*.tif, so the products are pixel-aligned "
                        "with the interferogram")
    g.add_argument("--stack", default=None, metavar="ZARR",
                   help="same, taking the grid from a zarr stack of geocoded SLCs "
                        "(its geotransform/shape/epsg attrs), e.g. "
                        "stack_t087_f057_A_HH_120m_c32.zarr")
    g.add_argument("--crop-to-data", action=argparse.BooleanOptionalAction, default=None,
                   help="crop the output to the offsets' own footprint.  Default: on for "
                        "--stack, whose grid covers a whole frame stack and would be almost "
                        "all nodata, off for --grid-like, where the point is usually to "
                        "match another product cell for cell.  Either way the crop stays on "
                        "the lattice -- only the origin moves, by a whole number of cells -- "
                        "so the products remain pixel-aligned with it")
    g.add_argument("--interp", default="BILINEAR",
                   choices=["SINC", "BILINEAR", "BICUBIC", "NEAREST", "BIQUINTIC"],
                   help="interpolation of the offset field during geocoding "
                        "(default BILINEAR)")
    g.add_argument("--out-dir", default="outputs_offsets")
    g.add_argument("--cache-dir", default="cache",
                   help="WorldCover tiles, the RGI region index and the rdr2geo "
                        "geolocation cache live here")
    g.add_argument("--scratch", default="offsets_scratch")
    g.add_argument("--keep-scratch", action="store_true",
                   help="keep every intermediate in --scratch (rdr2geo/geo2rdr layers, "
                        "offset rasters, exported SLC crops).  They are deleted by "
                        "default: on a full frame they are hundreds of GB.  Needed for a "
                        "later --reuse-offsets --ifg, which re-reads the coregistered "
                        "pair and the geo2rdr offsets it flattens with")
    g.add_argument("--keep-binaries", action="store_true",
                   help="keep just the exported flat SLC crops, not the rest of --scratch")

    g = ap.add_argument_group("masking (water and ice) and hole filling")
    g.add_argument("--mask-water", action=argparse.BooleanOptionalAction, default=False,
                   help="mask ESA WorldCover permanent water.  Water decorrelates, and "
                        "the interpolation smears those failed matches into the land "
                        "around them")
    g.add_argument("--water-year", type=int, default=2021, choices=[2020, 2021])
    g.add_argument("--mask-glacier", action=argparse.BooleanOptionalAction, default=False,
                   help="mask RGI 7.0 glacier complexes, streamed from NSIDC (needs "
                        "EarthData credentials in ~/.netrc).  Ice genuinely moves metres "
                        "between passes, so on a frame hunting anything else it is signal "
                        "from the wrong process")
    g.add_argument("--rgi-region", default=None, metavar="NN_name",
                   help="force an RGI region, e.g. 01_alaska; default picks every region "
                        "whose extent covers the product")
    g.add_argument("--mask-buffer", type=float, default=0.0, metavar="M",
                   help="grow the glacier polygons by this many metres before masking; "
                        "~half a correlation window (160 m at --winsize 64) also rejects "
                        "windows straddling a terminus")
    g.add_argument("--mask-where", choices=["geo", "rdr", "both"], default="geo",
                   help="geo (default): mask the geocoded products, leaving the unmasked "
                        "ones alongside.  rdr: drop the masked windows BEFORE geocoding, "
                        "so nothing from water or ice reaches the map grid and the masked "
                        "ground stays empty -- masking afterwards is too late to be clean, "
                        "the interpolation has already carried those offsets into the land "
                        "around them.  both: do that and write the geocoded masked set too")
    g.add_argument("--mask-glacier-in-products",
                   action=argparse.BooleanOptionalAction, default=None,
                   help="also blank the ice in the geocoded products, not just in the "
                        "measurement (default: only with --mask-where geo, where the "
                        "products are the only place it can happen).  Under --mask-where "
                        "rdr/both the on-ice windows are already out of the field, and "
                        "blanking the cells as well throws away the one thing the product "
                        "can honestly say there -- that the correction applied over the "
                        "ice is the one its surroundings got.  offset_filled_<tag>.tif "
                        "marks those cells as inferred")
    g.add_argument("--geoloc-cache", action=argparse.BooleanOptionalAction, default=True,
                   help="cache the window-centre geolocation --mask-where rdr needs "
                        "(default on; free and unused when --coreg geometric has already "
                        "written the rdr2geo rasters)")
    g.add_argument("--rdr-fill", action=argparse.BooleanOptionalAction, default=False,
                   help="interpolate the holes left by --mask-where rdr/both (default "
                        "OFF, so masked ground reads as nodata and is visible as such).  "
                        "A filled product looks complete and shows no sign of where the "
                        "ice was, so it has to be asked for; offset_filled_<tag>.tif marks "
                        "every cell that came out of a fill rather than a measurement")
    g.add_argument("--outlier-mad", type=float, default=5.0, metavar="K",
                   help="before any filling, drop measured windows further than K robust "
                        "deviations from their local level (default 5; 0 disables).  These "
                        "are scattered failed matches, and leaving them in drags the fill "
                        "across whole basins")
    g.add_argument("--outlier-spike", type=float, default=0.0, metavar="PX",
                   help="also drop a kept window more than PX px from the median of its "
                        "kept --outlier-spike-box neighbours (default 0 = off).  An "
                        "ABSOLUTE threshold, "
                        "for the failed matches --outlier-mad's scaled block test lets "
                        "through inside a steep streak and in a block of their own; they "
                        "only matter with --fill-robust 0, which honours them at full "
                        "weight.  The masks module's reject_spikes has the measurement (3 px on "
                        "a frame carrying a narrow, steep streak)")
    g.add_argument("--outlier-min-neighbors", type=int, default=0, metavar="N",
                   help="also drop a kept window with fewer than N kept neighbours in the "
                        "--outlier-spike-box (default 0 = off): nothing can check it.  "
                        "6 on a 64 x 64 window lattice")
    g.add_argument("--outlier-spike-box", type=int, default=7, metavar="W",
                   help="neighbourhood of the two tests above, W x W cells (default 7).  "
                        "Both iterate until nothing more falls, so clusters of failed "
                        "matches peel from the outside")
    g.add_argument("--fill-reject", action=argparse.BooleanOptionalAction, default=True,
                   help="treat measured windows the robust fit drives to zero weight as "
                        "holes and fill them too (default on)")
    g.add_argument("--fill-method", default="pls", metavar="NAME",
                   help="pls (default), pls_mg, biharmonic, laplace, spring, idw, local, "
                        "griddata, griddata_cubic -- the fill module documents them and "
                        "scores them against withheld truth if run directly")
    g.add_argument("--fill-robust", type=int, default=3, metavar="N",
                   help="bisquare re-weighting passes in the fill (default 3, 0 disables)")
    g.add_argument("--fill-cutoff", type=float, default=100.0, metavar="N",
                   help="fill smoothing scale as a cutoff wavelength in offset samples "
                        "(default 100).  Set it from the signal you are after, not from "
                        "the data: cross-validation tunes to the noise here")
    g.add_argument("--fill-s", type=float, default=None, metavar="S",
                   help="raw penalty, overriding --fill-cutoff")
    g.add_argument("--fill-hole-cutoff", default="auto", metavar="N",
                   type=lambda v: v if str(v).lower() == "auto" else float(v),
                   help="fill the HOLES from a second fit at this cutoff (with "
                        "--fill-hole-robust), keeping --fill-cutoff's surface within "
                        "--fill-hole-km of the kept windows.  For a --fill-cutoff short "
                        "enough to follow a steep streak, which is too short to hold a "
                        "glacier basin: the masks module's fill_holes has the measurement (100 on "
                        "a steep streak, i.e. the default fill inside the "
                        "holes).  `auto` (default) is the BULB CHECK: with the default fill "
                        "it does nothing; otherwise the robust fit is compared with the "
                        "data surface deeper than 0.5 km into the holes and the holes are "
                        "handed over only if they differ by more than --fill-bulb-px.  0 "
                        "disables the check")
    g.add_argument("--fill-bulb-px", type=float, default=2.0, metavar="PX",
                   help="the bulb check's threshold (default 2 px): the smallest deep-hole "
                        "departure of the data surface from the robust one that hands the "
                        "holes over.  A frame with deep glacier holes can read tens of "
                        "px here; a clean "
                        "short-cutoff fill reads under 1")
    g.add_argument("--fill-hole-robust", type=int, default=3, metavar="N",
                   help="bisquare passes of the --fill-hole-cutoff fit (default 3)")
    g.add_argument("--fill-hole-km", type=float, default=0.25, metavar="KM",
                   help="how far from the nearest kept window --fill-cutoff's surface "
                        "reaches before fading into the --fill-hole-cutoff one (default "
                        "0.25 km)")
    g.add_argument("--fill-aniso", default=1.0, metavar="W",
                   type=lambda v: v if str(v).lower() == "auto" else float(v),
                   help="weight on the RANGE axis of the fill's roughness penalty "
                        "(default 1 = isotropic).  An ionospheric offset field is a "
                        "range-elongated band, not a blob: the "
                        "azimuth field takes 1.90 km along track and 24.55 km across it "
                        "to change by 0.10 px, 12.9 : 1, and an isotropic fill returns "
                        "3.0 : 1 inside a hole.  W=16 brings the filled surface to "
                        "13.5 : 1 and, on withheld discs, takes the azimuth rms from "
                        "0.1447 to 0.0555 px in a 14 km hole and from 0.2034 to 0.0584 px "
                        "in its deep interior.  It does nothing for the RANGE channel "
                        "(0.0325 -> 0.0322 px), which has no anisotropic signal because "
                        "at L-band over this TEC it has no signal.  The masks module's fill_holes "
                        "has the full table, including why --fill-trust-km should be "
                        "raised once this is on.  W=auto measures the elongation from the "
                        "pair's own field and uses it -- the portable setting, because the "
                        "number is NOT a constant: five pairs over four track/frames span "
                        "frame to frame, and two pairs on the same frame differ "
                        "from each other, so it is a property of the day's ionosphere, not "
                        "of the geometry.  The elongation helper prints it for any offsets "
                        "npz without re-running anything.  W=auto is measured PER CHANNEL: "
                        "the azimuth field is 11.9 : 1 here and the range field 1.5 : 1, and "
                        "handing the azimuth number to the range channel is what put an "
                        "invented lobe in the middle of the icefields"),
    g.add_argument("--fill-aniso-direction", default="auto",
                   choices=("auto", "range", "azimuth"), metavar="DIR",
                   help="clip the --fill-aniso auto estimate to one sign: `range` forces "
                        "W >= 1 (range-elongated bands, the usual case), `azimuth` forces "
                        "W <= 1, `auto` (default) leaves it free.  Set it from the pair's "
                        "D_RA in the streak-index table -- the streak index defines "
                        "D_RA = R cos(2 theta) as +1 range streaking, -1 azimuth streaking, "
                        "and its SIGN is reliable where its magnitude saturates by SNR 0.2.  "
                        "positive means `range`.  A measured "
                        "elongation that contradicts DIR is clipped and reported loudly, "
                        "never silently reversed"),
    g.add_argument("--fill-trust-km", type=float, default=1.5, metavar="KM",
                   help="how deep into a hole the fill is an interpolation rather than an "
                        "extrapolation (default 1.5 km; 0 restores the old unbounded "
                        "fill).  Beyond it the surface is faded into the level the "
                        "surrounding measurements define, which by construction cannot "
                        "leave their range.  The masks module's fill_holes has the measurement: "
                        "the unbounded fill put +-2.5 px into the middle of the icefields "
                        "in the RANGE channel, where the measured field is +-0.17 px and "
                        "there is no ionosphere to find.  The DEPTH is measured with the "
                        "ruler each channel's own anisotropy implies, so this stays a "
                        "distance in the direction that channel's fill is weakest in")
    g.add_argument("--fill-trust-km-flat", type=float, default=1.5, metavar="KM",
                   help="the trust radius for a channel with NO measured anisotropy "
                        "(default 1.5 km; only consulted with --fill-aniso auto).  8 km is "
                        "calibrated to how deep a withheld-disc test still validates the "
                        "ANISOTROPIC azimuth fill; a channel measuring 1.5 : 1 has no "
                        "direction to extrapolate along and inherits no such licence.  "
                        "The p90 |deviation| of the "
                        "filled RANGE surface more than 6 km into a hole, in radians of "
                        "interferogram phase: 34.6 at 8 km, 29.2 at 4, 24.2 at 3, 23.5 at "
                        "1.5.  The masks module's fill_priors has the full table")
    g.add_argument("--reuse-offsets", action="store_true",
                   help="skip the correlation and re-geocode an existing offsets npz "
                        "(for changing --snr-min, --posting, --grid-like, ...)")
    g.add_argument("--offsets", default=None, metavar="NPZ",
                   help="where the raw ampcor field is written / read back with "
                        "--reuse-offsets (default <out-dir>/offsets_<tag>.npz)")
    a = ap.parse_args(argv)
    if a.window and a.aoi:
        raise SystemExit("--window and --aoi are alternatives")
    if a.grid_like and a.stack:
        raise SystemExit("--grid-like and --stack are alternatives")
    if a.mask_glacier_in_products is None:
        a.mask_glacier_in_products = a.mask_where == "geo"
    if a.rubbersheet_az and a.coreg != "geometric":
        raise SystemExit("--rubbersheet-az edits geo2rdr/azimuth.off, so it needs "
                         "--coreg geometric")
    if (a.rubbersheet_az or a.iono_screen) and not a.rdr_fill:
        # A hole in the field would freeze the geometric offset there and put a step across
        # every mask edge; the screen would integrate across it.  Both want the smoother's
        # surface, which is what --rdr-fill computes.
        a.rdr_fill = True
        print("--rubbersheet-az / --iono-screen imply --rdr-fill")
    a.gpus = resolve_gpus(a.gpus)
    return a


def default_tag(ref, sec):
    """The pair tag from the two filenames; 'ref_sec' if they say nothing."""
    def date(p):
        m = re.search(r"(\d{4})-?(\d{2})-?(\d{2})", os.path.basename(p))
        return "".join(m.groups()) if m else None
    m = re.search(r"_t0*(\d+)[_-]", os.path.basename(ref))
    trk = f"t{int(m.group(1)):03d}_" if m else ""
    d1, d2 = date(ref), date(sec)
    return f"{trk}{d1}_{d2}" if d1 and d2 else "ref_sec"


def main(argv=None):
    global _T0
    _T0 = time.time()
    args = parse_args(argv)
    args.tag = args.tag or default_tag(args.ref, args.sec)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.scratch, exist_ok=True)

    with timed("read SLC metadata + DEM"):
        ref = load_slc(args.ref, args.freq, args.pol)
        sec = load_slc(args.sec, args.freq, args.pol)
        dem_raster, demI = B.load_dem(args.dem)

    win = tuple(args.window) if args.window else None
    full_frame = win is None                  # implicit whole frame: may be inset below
    if full_frame:
        win = (0, 0, ref["shape"][0], ref["shape"][1])
    if win[0] + win[2] > ref["shape"][0] or win[1] + win[3] > ref["shape"][1]:
        raise SystemExit(f"window {win} exceeds the reference image {ref['shape']}")

    centre = (win[0] + win[2] // 2, win[1] + win[3] // 2)
    if args.epsg is None:
        args.epsg = scene_epsg(ref, demI, centre)
    npz = args.offsets or os.path.join(args.out_dir, f"offsets_{args.tag}.npz")

    if args.reuse_offsets:
        d = np.load(npz)
        az, rg, snr = d["azimuth"], d["range"], d["snr"]
        win = tuple(int(v) for v in d["window"])
        gross = tuple(int(v) for v in d["gross"])
        args.az_spacing = azimuth_ground_spacing(ref, demI, centre)
        # an offsets npz may carry the lattice but not the spacing
        if "az_spacing" in d.files and abs(args.az_spacing - float(d["az_spacing"])) > 0.01:
            print(f"  NOTE: the npz was written with {float(d['az_spacing']):.4f} m/px; "
                  f"using the freshly measured value instead")
        # the lattice the npz was measured on is what geocoding must be told about, whatever
        # the command line now says
        args.winsize = tuple(int(v) for v in d["winsize"])
        args.search = tuple(int(v) for v in d["search"])
        args.skip = tuple(int(v) for v in d["skip"])
        print(f"reusing {npz}: {az.shape[0]} x {az.shape[1]} windows, window {win}, "
              f"gross {gross}")
        # The pair from the run that made this npz, if it is still on disk.  pair0 is the
        # RAW one -- resamp_slc always starts from sec.c8, never from its own output -- so
        # --rubbersheet-az needs it as well as the coregistered product.
        pad = (args.search[0] + args.winsize[0], args.search[1] + args.winsize[1])
        if "sec_origin" in d.files:
            so = tuple(int(v) for v in d["sec_origin"])
            ss = tuple(int(v) for v in d["sec_shape"])
        else:
            # An npz from before those keys existed.  `gross` was zeroed by the
            # coregistration before it was written, so rebuilding the raw crop origin from
            # it is only right when the gross offset really was (0, 0) -- on this pair the
            # azimuth one was +624 px.  gross0, if present, is the value before zeroing.
            g0 = tuple(int(v) for v in d["gross0"]) if "gross0" in d.files else gross
            so = (win[0] + g0[0] - pad[0], win[1] + g0[1] - pad[1])
            ss = (win[2] + 2 * pad[0], win[3] + 2 * pad[1])
            if "gross0" not in d.files:
                print(f"  NOTE: this npz predates gross0, so the raw secondary crop origin "
                      f"is rebuilt from gross {g0} and is only right if that was the true "
                      f"gross offset -- re-run without --reuse-offsets if it was not")
        pair0 = dict(fref=os.path.join(args.scratch, "ref.c8"),
                     ref_shape=(win[2], win[3]),
                     fsec=os.path.join(args.scratch, "sec.c8"),
                     sec_shape=ss, sec_origin=so, pad=pad)
        pair = dict(pair0, fsec=os.path.join(args.scratch, "sec_coreg.c8"),
                    sec_shape=(win[2], win[3]), sec_origin=(win[0], win[1]),
                    sec_origin0=so)
        need = [pair["fref"], pair["fsec"]]
        if args.rubbersheet_az:
            need += [pair0["fsec"], os.path.join(args.scratch, "geo2rdr", "azimuth.off")]
        missing = [f for f in need if not os.path.exists(f)]
        if missing:
            print(f"[reuse] {', '.join(os.path.basename(f) for f in missing)} not in "
                  f"{args.scratch}: no interferogram and no rubbersheet from this npz "
                  f"(the run that writes them has to keep them with --keep-scratch)")
            pair = pair0 = None
    else:
        print("gross offset from orbit + DEM:")
        gaz, grg = gross_from_geometry(ref, sec, demI, centre)
        aaz, arg_ = gross_from_annotation(ref, sec)
        print(f"  the annotation difference would have said azimuth {aaz:+d}, range "
              f"{arg_:+d} px (off by {gaz - aaz:+d}, {grg - arg_:+d}) -- see "
              f"gross_from_annotation()")
        if args.gross[0] != "geom":
            gaz = int(args.gross[0])
        if args.gross[1] != "geom":
            grg = int(args.gross[1])
        gross = (gaz, grg)
        print(f"gross offset (secondary relative to reference): "
              f"azimuth {gaz:+d} px, range {grg:+d} px")
        args.az_spacing = azimuth_ground_spacing(ref, demI, centre)
        if full_frame:
            win = inset_full_frame(ref, sec, win, gross, args)

        with timed("export SLC pair to flat binary"):
            pair = export_pair(args, ref, sec, win, gross)
        pair0 = pair                # the RAW pair: every resample starts here, never from
                                    # a previous resample's output
        gross0 = gross              # coregistration zeroes `gross` below, but the raw crop
                                    # origin has to survive into the npz for --reuse-offsets
        tag = os.path.join(args.scratch, "offsets")

        if args.coreg == "geometric":
            # isce3 computes the offsets (rdr2geo -> geo2rdr) and applies them (resamp_slc);
            # then measure what is left and rubbersheet it away
            with timed("coregistration (rdr2geo + geo2rdr + resamp)"):
                pair = coregister_isce3(args, pair0, ref, sec, win, dem_raster)
            gross = (0, 0)
            coarse = (tuple(args.coreg_winsize), tuple(args.search), tuple(args.coreg_skip))
            g2r_dir = os.path.join(args.scratch, "geo2rdr")
            for it in range(1, args.coreg_iters + 1):
                with timed(f"coregistration check {it}"):
                    d_az, d_rg = rubbersheet(args, pair, sec, g2r_dir, coarse, it)
                if max(abs(d_az), abs(d_rg)) <= args.coreg_tol:
                    print(f"[coreg] within --coreg-tol {args.coreg_tol} px: coregistered")
                    break
                if it == args.coreg_iters:
                    print(f"[coreg] WARNING: {d_az:+.3f} / {d_rg:+.3f} px still out after "
                          f"{it} iteration(s); the residual offsets carry it as a constant")
                    break
                _add_constant(os.path.join(g2r_dir, "azimuth.off"), d_az)
                _add_constant(os.path.join(g2r_dir, "range.off"), d_rg)
                # resample_only: the offsets on disk now carry the correction and
                # must NOT be recomputed by geo2rdr
                with timed(f"rubbersheet resample {it}"):
                    pair = coregister_isce3(args, pair0, ref, sec, win, dem_raster,
                                            resample_only=True)

        # --- 4. dense offsets.  After coregistration these are RESIDUAL offsets: the
        # geometric field is already in the resampling, so what is measured is displacement
        # plus noise.  gross_in_crop is (0, 0) for a coregistered pair, `pad` for a raw one.
        with timed("dense ampcor"):
            az, rg, snr = run_ampcor(args, pair, tag,
                                     gross_in_crop=(0, 0) if args.coreg != "none" else None)
        # WHAT THIS npz MEANS, on every path: THE OFFSET RELATIVE TO GEOMETRY.  Every
        # ionospheric route integrates it, so the meaning cannot depend on how the run was
        # started.  A RESUMED run reuses a sec_coreg.c8 that an earlier run already
        # rubbersheeted, so the dense pass above measured the RESIDUAL; add back whatever is
        # already applied.  A fresh run reads zeros here and nothing moves.
        g2r0 = os.path.join(args.scratch, "geo2rdr")
        why = ("the dense pass measured the residual on top of it, so the npz gets that "
               "field plus this one -- the offset relative to geometry")
        rb0_az = rubbersheet_state(g2r0, az.shape, "az", why=why)
        np.savez_compressed(npz, azimuth=az + rb0_az, range=rg, snr=snr,
                            window=np.array(win), gross=np.array(gross),
                            # gross before coregistration zeroed it, and the raw secondary
                            # crop, so --reuse-offsets can rebuild the pair resamp_slc needs
                            gross0=np.array(gross0),
                            sec_origin=np.array(pair0["sec_origin"]),
                            sec_shape=np.array(pair0["sec_shape"]),
                            skip=np.array(args.skip), winsize=np.array(args.winsize),
                            search=np.array(args.search),
                            az_spacing=np.array(args.az_spacing),
                            rg_spacing=np.array(ref["rg"].range_pixel_spacing),
                            coreg=np.array(args.coreg))
        print(f"wrote {npz}")
        # the interferogram is made further down, after the offsets have been cleaned and
        # the target lattice is known: --rubbersheet-az resamples the secondary again and
        # --ifg-geocode needs the offsets' own grid to land on

    # --- radar-domain view, before anything is masked: where correlation fails is the
    # first thing to look at, and --snr-min drops it from everything downstream
    i0 = win[0] + args.search[0] + args.winsize[0] // 2
    j0 = win[1] + args.search[1] + args.winsize[1] // 2
    ext = [j0, j0 + az.shape[1] * args.skip[1], i0 + az.shape[0] * args.skip[0], i0]
    with timed("plot (radar domain)"):
        plot_panels([(az, "azimuth offset", "px", False), (rg, "range offset", "px", False),
                     (snr, "correlation SNR", "", True)],
                    ext, "range [px]", "azimuth [px]",
                    os.path.join(args.out_dir, f"offsets_rdr_{args.tag}.png"), aspect="auto")

    # The target grid is settled before anything is masked, because masking on the RADAR
    # lattice rasterises water and ice on that same grid -- one rasterisation, and the mask
    # a window centre sees is bit-for-bit the one the geocoded stage would have applied.
    off_rg = offset_radar_grid(ref["rg"], win, args, az.shape)
    gg = target_geogrid(args, off_rg, ref["orbit"], demI)
    gt = (gg.start_x, gg.spacing_x, 0.0, gg.start_y, 0.0, gg.spacing_y)

    smooth = {}
    keep, azf, rgf, snrf, filled = clean_lattice(args, ref, demI, win, npz, gg, gt,
                                                 az, rg, snr, smooth_out=smooth)

    # --- azimuth rubbersheet.  The coregistration loop above can only take out a CONSTANT,
    # so a per-pixel azimuth shift -- which is what an along-track TEC gradient produces --
    # survives it and is still in sec_coreg.c8 when the interferogram is formed.  It costs
    # coherence directly, as 1 - |shift|/resolution.  Folding the measured field back into
    # geo2rdr's azimuth.off and resampling again is what recovers it.
    #
    # This DOES touch the interferometric phase: the SLCs carry an azimuth carrier and
    # resamp_slc re-ramps about the INPUT position, so every pixel of shift puts 2*pi*fd/PRF
    # into the interferogram.  make_interferogram takes it back out (--az-carrier); the
    # ionospheric PHASE screen is a third, separate correction (--iono-screen).
    if args.rubbersheet_az and pair is not None:
        if pair0 is None:
            print(f"[rbsheet] skipped: the raw secondary is not in {args.scratch} -- "
                  f"the rubbersheet needs a run kept with --keep-scratch")
        else:
            g2r_dir = os.path.join(args.scratch, "geo2rdr")
            applied = rubbersheet_state(g2r_dir, az.shape)
            refocus_state = None
            # Turning either flag off does not undo what a previous run baked into that
            # raster, and nothing else would ever take it out again -- the run would
            # silently inherit a corrupted resampling (azimuth) or flattening (range).
            # Un-apply it here, the same increment bookkeeping as everywhere else.  This
            # is what makes a SWEEP over rubbersheet configurations inside one scratch
            # exact rather than cumulative.
            if applied.any() and args.reuse_offsets:
                print("[rbsheet] WARNING: azimuth.off already carries a rubbersheet, but "
                      "these offsets came out of an npz and were measured against a "
                      "different secondary.  Re-run without --reuse-offsets, or with "
                      "--force-coreg, if that was not intended")
            for it in range(1, args.rubbersheet_az + 1):
                # The smoother's own surface, not the measured field: handing resamp_slc
                # per-window correlation noise would degrade the registration it is meant
                # to improve.  What the correlator reports is always the residual on top of
                # whatever azimuth.off already carries, so this IS the increment -- and
                # `applied` is the running total relative to pure geometry.
                inc = np.nan_to_num(smooth.get("azimuth", azf)).astype(np.float32)
                with timed(f"azimuth rubbersheet {it}"):
                    upsample_add(os.path.join(g2r_dir, "azimuth.off"), inc, args)
                applied = applied + inc
                # written even when azimuth is off, as the zeros it then is: downstream
                # (the route driver stage_prerb) subtracts this file to rebuild the
                # pre-rubbersheet secondary, and a missing file is not the same statement
                # as a measured zero.
                save_rubbersheet_state(g2r_dir, applied)
                with timed(f"rubbersheet resample {it}"):
                    pair = coregister_isce3(args, pair0, ref, sec, win, dem_raster,
                                            resample_only=True)
                if not args.rubbersheet_az_redense:
                    break
                with timed(f"dense ampcor (after rubbersheet {it})"):
                    az, rg, snr = run_ampcor(args, pair,
                                             os.path.join(args.scratch, "offsets"),
                                             gross_in_crop=(0, 0))
                smooth = {}
                keep, azf, rgf, snrf, filled = clean_lattice(args, ref, demI, win, npz,
                                                             gg, gt, az, rg, snr,
                                                             smooth_out=smooth)

            # --refocus-km: the resample registered the band centre; this registers the
            # rest of the Doppler band, AFTER the redense pass and not before it.  Running
            # the correlator on the refocused secondary makes its sub-pixel peak react to
            # the reshaped impulse response rather than to a shift, so the redense field
            # stays measured on the resampled secondary and the offsets products are
            # unchanged; the interferogram and everything reading sec_coreg.c8 get the
            # refocused one.
            if str(args.refocus_km).strip().lower() not in ("0", "0.0", "", "off", "none"):
                if not args.rubbersheet_az:
                    print("[refocus] skipped: --refocus-km needs the azimuth rubbersheet "
                          "(it steers by the applied field)")
                else:
                    with timed("refocus (Doppler-dependent azimuth)"):
                        refocus_state = refocus_secondary(args, pair, ref, sec, win, applied)

            # The product stays what it has always been: the offset RELATIVE TO GEOMETRY.
            # After a redense pass the correlator was measuring against the rubbersheeted
            # secondary, so what it reported is the residual on top of `applied` and the
            # applied field has to go back in.  Without this the azimuth product would read
            # ~0 everywhere and the ionospheric signal would have been deleted from the
            # measurement rather than from the interferogram.  Without --redense nothing is
            # re-measured, so `az` is still the field relative to geometry and is left be.
            if args.rubbersheet_az_redense:
                az = az + applied
                azf = azf + applied
            rb = npz.replace(".npz", "_rbsheet.npz")
            extra = {}
            if refocus_state is not None:
                extra["refocus_km"] = np.array(float(refocus_state["A_km"]))
            np.savez_compressed(rb, applied=applied, azimuth=azf, range=rgf, snr=snrf,
                                filled=filled, window=np.array(win),
                                skip=np.array(args.skip), winsize=np.array(args.winsize),
                                search=np.array(args.search),
                                az_spacing=np.array(args.az_spacing), **extra)
            print(f"[rbsheet] wrote {rb} (the applied field, and the offsets it leaves "
                  f"behind"
                  f"{'; refocus_km ' + str(extra['refocus_km']) if 'refocus_km' in extra else ''})")

    # built whether or not there is an interferogram to apply it to: the screen and its
    # validity layer are products in their own right, on the offsets' own grid
    screen = (iono_screen(args, ref, win, az, keep, off_rg, ref["orbit"], dem_raster, gg,
                          filled)
              if args.iono_screen else None)

    if args.ifg and pair is not None:
        if args.coreg == "none":
            print("[ifg] --ifg needs a coregistered secondary; skipped with --coreg none")
        else:
            with timed("interferogram"):
                make_interferogram(args, pair, ref, sec, win, gg=gg, screen=screen)

    # --- radar-domain view of the CLEANED field, plus the axis self-certification.
    #
    # The geocoded quicklook is on EPSG:3413, which at this longitude rotates the grid by
    # about -105 deg: the range axis lands near-vertical on the page and along-track
    # near-horizontal.  So range-elongated bands in the AZIMUTH component read as vertical
    # stripes and look like a range field.  They are not.  In radar geometry there is no
    # rotation -- axis 0 IS azimuth, axis 1 IS range -- and a band running along this
    # figure's x axis is range-elongated by definition.  Look HERE when the labels are
    # doubted; the raw figure written earlier cannot settle it, because it is plotted
    # before --snr-min and its p98 stretch is set by the failed matches, leaving the +-1 px
    # signal a uniform wash.
    _meas = np.where(keep, np.float32(1.0), np.float32(np.nan))
    with timed("plot (radar domain, cleaned)"):
        plot_panels([(azf * _meas, "azimuth offset (measured cells)", "px", False),
                     (rgf * _meas, "range offset (measured cells)", "px", False),
                     (np.where(filled, 1.0, 0.0).astype(np.float32), "filled flag", "", True)],
                    ext, "range [px]", "azimuth [px]",
                    os.path.join(args.out_dir, f"offsets_rdr_clean_{args.tag}.png"),
                    aspect="auto")
    try:
        from ._utils.elongation import reach, structure_function

        _el = {}
        for _n, _v in (("azimuth", azf), ("range", rgf)):
            _ra = reach(structure_function(_v, keep, 0), 0.10)
            _rr = reach(structure_function(_v, keep, 1), 0.10)
            _el[_n] = (_rr / _ra if np.isfinite(_rr) and np.isfinite(_ra) and _ra > 0
                       else np.inf)
        _rgsp = float(ref["rg"].range_pixel_spacing)
        print(f"[axes] azimuth {args.az_spacing:.4f} m/px ground, elongation "
              f"{_el['azimuth']:.1f} : 1   |   range {_rgsp:.4f} m/px slant, elongation "
              f"{_el['range']:.1f} : 1")
        print("[axes] the BANDED channel is the one scaled by the azimuth spacing, and its "
              "bands lie ACROSS track -- an along-track TEC gradient shifts targets in "
              "azimuth and puts the |dTEC/dx_az| contours perpendicular to that shift")
        if abs(args.az_spacing / _rgsp - 1.0) < 0.10:
            print("[axes] WARNING: the two pixel spacings are within 10 % of each other on "
                  "this frame, so the m/px ratio alone no longer identifies a channel")
    except Exception as _exc:                       # a diagnostic must never fail a run
        print(f"[axes] elongation check skipped: {_exc}")

    layers = {"azimuth": azf, "range": rgf, "snr": snrf}
    if filled.any():
        # Geocoded like any other layer, so the flag lands on the same grid as the
        # products: 0 where every contributing sample was measured, 1 where all of them
        # came out of a fill, fractional in between.  NaN where the offsets are NaN, so the
        # flag never claims "measured" for a cell that has no value at all.
        layers["filled"] = np.where(np.isfinite(azf), filled, np.nan).astype(np.float32)

    with timed("geocode"):
        geo = geocode_layers(layers, off_rg, ref["orbit"], dem_raster, gg, args)
    if args.rdr_fill:
        # a gap-free radar lattice is not a gap-free map grid -- the geocoding leaves its
        # own specks wherever the samples bunch up over steep terrain
        with timed("fill holes (map grid)"):
            M.fill_interior(geo, args, keys=("azimuth", "range", "snr"))

    # --- products.  Pixels are the measurement; metres are the same thing scaled by the
    # grid, along-track on the ground for azimuth and SLANT range for range.
    scale = {"azimuth": args.az_spacing, "range": float(ref["rg"].range_pixel_spacing)}
    units = ["px", "m"] if args.units == "both" else [args.units]
    tif = {}                                 # (layer, unit) -> GeoTIFF path
    with timed("write GeoTIFFs"):
        for name in ("azimuth", "range"):
            for unit in units:
                a = geo[name] * (scale[name] if unit == "m" else 1.0)
                p = os.path.join(args.out_dir, f"offset_{name}_{args.tag}_{unit}.tif")
                B.save_gtiff(p, a, gg, nodata=np.nan)
                png_for(p, a)
                v = a[np.isfinite(a)]
                print(f"  median {np.median(v):+.4f} {unit}  p1..p99 "
                      f"{np.percentile(v, 1):+.4f} .. {np.percentile(v, 99):+.4f}")
                tif[(name, unit)] = p
        p = os.path.join(args.out_dir, f"offset_snr_{args.tag}.tif")
        B.save_gtiff(p, geo["snr"], gg, nodata=np.nan)
        png_for(p, geo["snr"], gray=True)
        tif[("snr", None)] = p
        if "filled" in geo:
            # without this there is no way to tell a measurement from an inference
            p = os.path.join(args.out_dir, f"offset_filled_{args.tag}.tif")
            B.save_gtiff(p, geo["filled"], gg, nodata=np.nan)
            png_for(p, geo["filled"], gray=True, vlim=(0.0, 1.0))
            f = geo["filled"][np.isfinite(geo["filled"])]
            print(f"  {100 * (f > 0.5).mean():.1f} % of valid cells are mostly "
                  f"interpolated rather than measured")

    geo_ext = [gg.start_x, gg.start_x + gg.width * gg.spacing_x,
               gg.start_y + gg.length * gg.spacing_y, gg.start_y]

    def geo_png(read, suffix):
        """One three-panel quicklook per unit, so every GeoTIFF written has a picture."""
        for unit in units:
            plot_panels([(read(("azimuth", unit)), "azimuth (along-track) offset", unit, False),
                         (read(("range", unit)), "range (slant) offset", unit, False),
                         (read(("snr", None)), "correlation SNR", "", True)],
                        geo_ext, "easting [m]", "northing [m]",
                        os.path.join(args.out_dir,
                                     f"offsets_geo_{args.tag}_{unit}{suffix}.png"))

    with timed("plot (geocoded)"):
        geo_png(lambda k: (geo[k[0]] * (scale[k[0]] if k[1] == "m" else 1.0)
                           if k[0] != "snr" else geo["snr"]), "")

    if args.mask_where in ("geo", "both") and (args.mask_water or args.mask_glacier):
        # "_watermasked" when water is all that was masked, so the filenames older runs
        # and scripts expect are unchanged; "_masked" once ice is in there too
        suffix = ("_masked" if args.mask_glacier and args.mask_glacier_in_products
                  else "_watermasked")
        masked, gl = {}, None
        with timed("mask (map grid)"):
            for k, p in tif.items():
                dst = p.replace(".tif", f"{suffix}.tif")
                a, gl = M.mask_geotiff(p, dst, args, gl)     # RGI rasterised once
                B.save_gtiff(dst, a, gg, nodata=np.nan)      # same format as the rest
                png_for(dst, a, gray=k[0] == "snr")
                masked[k] = a
        with timed("plot (masked)"):
            geo_png(lambda k: masked[k], suffix)

    cleanup_scratch(args)
    print_timings()
    return 0


if __name__ == "__main__":
    sys.exit(main())
